#!/usr/bin/env python3
"""P3 lab: predict, then measure, a real local LLM service (gRPC_llm).

The loop every P-series lab follows:
  1. PREDICT on paper from two facts: how many bytes the model reads per token,
     and how fast this machine can move bytes / do math.
  2. MEASURE the running service over its own gRPC API.
  3. EXPLAIN the gap between the two.

Run it with the llm-tiers virtualenv (it has grpcio and tokenizers); the tiers live in the
sibling llm-tiers package since 2026-11-05 (previously gRPC_llm/llm_service):

  TIERS=".../Software/AI/llm-tiers"
  "$TIERS/.venv/bin/python" labs/p3/llm_service_lab.py --root "$TIERS" \
      --addr localhost:50051,localhost:50061 --seconds 30

Nothing is written inside llm-tiers. The service must already be running
(`make up TIER=heavy` there). Each request is a normal Generate call.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import struct
import sys
import time
import uuid

# --------------------------------------------------------------------------- model files

GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # type id -> bytes


def read_gguf(path: pathlib.Path) -> dict:
    """Minimal GGUF reader: metadata keys plus every tensor's shape and byte size.

    Format: magic 'GGUF', version, tensor count, metadata count, metadata
    key/values, tensor infos (name, dims, type, offset), then aligned data.
    Tensor byte sizes come from the gaps between data offsets, so no table of
    quantization block sizes is needed.
    """
    f = open(path, "rb")

    def u32(): return struct.unpack("<I", f.read(4))[0]
    def u64(): return struct.unpack("<Q", f.read(8))[0]
    def string(): return f.read(u64()).decode("utf-8", "replace")

    def value(t):
        if t == 8:
            return string()
        if t == 9:  # array: element type, count, elements
            et, n = u32(), u64()
            if et == 8:
                for _ in range(n):
                    f.seek(u64(), 1)
                return f"<{n} strings>"
            size = GGUF_SCALAR[et]
            if n > 64:
                f.seek(size * n, 1)
                return f"<{n} values>"
            return [value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
               12: "<d"}[t]
        return struct.unpack(fmt, f.read(GGUF_SCALAR[t]))[0]

    if f.read(4) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    version, n_tensors, n_kv = u32(), u64(), u64()
    meta = {}
    for _ in range(n_kv):
        k = string()
        meta[k] = value(u32())
    tensors = []
    for _ in range(n_tensors):
        name = string()
        dims = [u64() for _ in range(u32())]
        ttype, off = u32(), u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = int(meta.get("general.alignment", 32))
    data_start = (f.tell() + align - 1) // align * align
    file_size = path.stat().st_size
    tensors.sort(key=lambda t: t["offset"])
    for i, t in enumerate(tensors):
        end = tensors[i + 1]["offset"] if i + 1 < len(tensors) else file_size - data_start
        t["bytes"] = end - t["offset"]
        n = 1
        for d in t["dims"]:
            n *= d
        t["elements"] = n
    arch = meta.get("general.architecture", "?")
    g = lambda k, d=0: meta.get(f"{arch}.{k}", d)
    return {
        "format": f"GGUF v{version}", "arch": arch, "file_bytes": file_size, "tensors": tensors,
        "layers": g("block_count"), "hidden": g("embedding_length"), "heads": g("attention.head_count"),
        "kv_heads": g("attention.head_count_kv", g("attention.head_count")), "ctx_train": g("context_length"),
        "file_type": meta.get("general.file_type"),
    }


def read_mlx_dir(d: pathlib.Path) -> dict:
    """MLX / Hugging Face folder: config.json + safetensors (header = JSON with shapes and offsets)."""
    cfg = json.loads((d / "config.json").read_text())
    tensors = []
    total = 0
    for st in sorted(d.glob("*.safetensors")):
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            el = 1
            for s in info["shape"]:
                el *= s
            tensors.append({"name": name, "dims": info["shape"], "bytes": b - a, "elements": el, "dtype": info["dtype"]})
        total += st.stat().st_size
    q = cfg.get("quantization", {})
    return {
        "format": f"safetensors (MLX {q.get('bits', '?')}-bit, group {q.get('group_size', '?')})", "arch": cfg.get("model_type"),
        "file_bytes": total, "tensors": tensors, "layers": cfg["num_hidden_layers"], "hidden": cfg["hidden_size"],
        "heads": cfg["num_attention_heads"], "kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "ctx_train": cfg.get("max_position_embeddings"), "tied": cfg.get("tie_word_embeddings", False),
        "q_bits": q.get("bits"),
    }


def analyse(m: dict) -> dict:
    """Bytes read per generated token, parameter count, KV cache per token."""
    tensors = m["tensors"]
    names = {t["name"] for t in tensors}
    is_embed = lambda n: n in ("token_embd.weight",) or n.endswith("embed_tokens.weight") or \
        n.endswith("embed_tokens.scales") or n.endswith("embed_tokens.biases")
    has_head = any(n in ("output.weight",) or n.endswith("lm_head.weight") for n in names)
    weight_bytes = sum(t["bytes"] for t in tensors)
    embed_bytes = sum(t["bytes"] for t in tensors if is_embed(t["name"]))
    # Decode reads every weight once per token, except the embedding table, from
    # which it reads one row. If the output head is tied to the embedding, the
    # whole table is read anyway (to score every vocabulary entry).
    per_token = weight_bytes - (embed_bytes if has_head else 0)
    # Parameter count. For MLX 4-bit, a uint32 holds 8 weights; count logical weights.
    params = 0
    for t in tensors:
        if t.get("dtype") == "U32" and m.get("q_bits"):
            params += t["elements"] * (32 // m["q_bits"])
        elif t.get("dtype") in ("F16", "BF16", "F32") and (t["name"].endswith(".scales") or t["name"].endswith(".biases")):
            continue
        else:
            params += t["elements"]
    head_dim = m["hidden"] // m["heads"] if m["heads"] else 0
    kv_per_token = 2 * m["layers"] * m["kv_heads"] * head_dim * 2  # K and V, fp16
    return {"weight_bytes": weight_bytes, "embed_bytes": embed_bytes, "per_token_bytes": per_token,
            "params": params, "head_dim": head_dim, "kv_bytes_per_token": kv_per_token,
            "bits_per_weight": 8 * weight_bytes / params if params else 0, "tied": not has_head}


def models_dir(root: pathlib.Path) -> pathlib.Path:
    """Shared weight store: $LLM_MODELS_DIR, else <root>/../models.nosync (llm-tiers layout),
    else the pre-2026-11 gRPC_llm symlink <root>/llm_service/models."""
    env = os.environ.get("LLM_MODELS_DIR")
    if env:
        return pathlib.Path(env)
    shared = root.parent / "models.nosync"
    return shared if shared.exists() else root / "llm_service" / "models"


def locate_weights(root: pathlib.Path, filename: str, name: str) -> pathlib.Path | None:
    models = models_dir(root)
    if filename.endswith(".gguf") and (models / filename).exists():
        return models / filename
    for snap in (models / "hf" / "hub").glob(f"models--*--{filename}/snapshots/*"):
        return snap
    for snap in (models / "hf" / "hub").glob("models--*/snapshots/*"):
        if filename.lower() in str(snap).lower() or name.lower() in str(snap).lower():
            return snap
    return None


def find_tokenizer(root: pathlib.Path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    for tj in (models_dir(root) / "hf" / "hub").glob("models--*Qwen2.5*/snapshots/*/tokenizer.json"):
        return Tokenizer.from_file(str(tj))
    return None


# --------------------------------------------------------------------------- measuring

def load_stubs(root: pathlib.Path):
    """Import only the generated protobuf stubs (stable), nothing else. --root is the llm-tiers
    checkout; a pre-2026-11 gRPC_llm checkout (llm_service package) still works."""
    sys.path.insert(0, str(root))
    try:
        from llm_tiers import llm_pb2, llm_pb2_grpc
    except ModuleNotFoundError:
        from llm_service import llm_pb2, llm_pb2_grpc
    return llm_pb2, llm_pb2_grpc


def generate(stub, pb, prompt: str, max_tokens: int, timeout: float = 600) -> dict:
    """One streamed Generate call, timed on the client.

    If the server sends a Usage message on the final chunk (newer gRPC_llm),
    its exact token counts and server-side timings are returned as well.
    """
    t0 = time.perf_counter()
    first, chunks, usage = None, 0, None
    for chunk in stub.Generate(pb.GenerateRequest(prompt=prompt, max_tokens=max_tokens, temperature=0.0),
                               timeout=timeout):
        if chunk.token:
            if first is None:
                first = time.perf_counter()
            chunks += 1
        if hasattr(chunk, "HasField") and "usage" in [f.name for f in chunk.DESCRIPTOR.fields] and chunk.HasField("usage"):
            u = chunk.usage
            usage = {f.name: getattr(u, f.name) for f in u.DESCRIPTOR.fields}
    end = time.perf_counter()
    ttft = (first or end) - t0
    window = end - first if first else 0
    return {"ttft_s": ttft, "chunks": chunks, "total_s": end - t0,
            "decode_tok_s": (chunks - 1) / window if window > 0 and chunks > 1 else 0.0, "usage": usage}


CONTEXT = ("The orchestrator routes each user request to a tier. The heavy tier handles coding, reasoning, and "
           "analysis. The standard tier handles routing, classification, and extraction. Every tier speaks the "
           "same gRPC contract and streams tokens back. ")
DECODE_PROMPT = ("Explain in detail, step by step, how a CPU cache works, including cache lines, associativity, "
                 "and why access order matters for performance.")


def gpu_busy_pct(samples: int = 3) -> float | None:
    """GPU 'Device Utilization %' from ioreg (no sudo). Median of a few 0.5 s samples."""
    import re
    import subprocess
    vals = []
    for _ in range(samples):
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            vals.append(int(m.group(1)))
        time.sleep(0.5)
    return statistics.median(vals) if vals else None


def unique(prompt: str) -> str:
    # A unique first line defeats llama.cpp's prompt-prefix reuse, so every
    # request really processes its whole prompt (a cold, fair measurement).
    return f"[request {uuid.uuid4().hex[:12]}]\n{prompt}"


def fmt_bytes(b: float) -> str:
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="path to the gRPC_llm checkout")
    ap.add_argument("--addr", default="localhost:50051,localhost:50061")
    ap.add_argument("--runs", type=int, default=3, help="repeats of each measured prompt")
    ap.add_argument("--seconds", type=float, default=0, help="extra: keep decoding for N s, one line per request")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-repeat", type=int, default=40, help="copies of the context paragraph in the long prompt")
    ap.add_argument("--bw-gpu", type=float, default=410, help="GB/s the GPU can stream (spec sheet)")
    ap.add_argument("--bw-cpu", type=float, default=224, help="GB/s the CPU cores can stream (bandwidth lab)")
    ap.add_argument("--gpu-tflops", type=float, default=14.0,
                    help="GPU dense fp16/fp32 TFLOP/s; Apple does not publish this (third-party estimate, 32-core M4 Max)")
    ap.add_argument("--cpu-tflops", type=float, default=0.77, help="CPU NEON fp32 TFLOP/s (peak lab)")
    ap.add_argument("--json", help="append records here (JSON lines)")
    a = ap.parse_args()

    root = pathlib.Path(a.root).expanduser()
    pb, pbg = load_stubs(root)
    import grpc
    tok = find_tokenizer(root)
    count = (lambda s: len(tok.encode(s).ids)) if tok else (lambda s: max(1, len(s) // 4))
    out = open(a.json, "a") if a.json else None
    fits = []  # (bytes per token, seconds per token) for the cost-model fit

    idle = gpu_busy_pct()
    if idle is not None:
        print(f"GPU utilisation before any request: {idle:.0f}%", end="")
        if idle > 15:
            print("  <- something else is using the GPU (a game engine, video, another model).")
            print("   On Apple silicon the GPU is one shared device: expect low and unstable numbers until it is idle.")
        else:
            print("  (idle: good)")
    for addr in [x for x in a.addr.split(",") if x]:
        stub = pbg.LLMServiceStub(grpc.insecure_channel(addr))
        try:
            info = stub.GetActiveModel(pb.GetActiveModelRequest(), timeout=5)
        except grpc.RpcError as e:
            print(f"\n== {addr}: not reachable ({e.code().name}). Start it with `make llm-up` in gRPC_llm.")
            continue
        device = info.device or "cpu"
        print(f"\n== P3.service  {addr}  {info.model_name}  tier={info.tier} engine={info.engine} device={device}")
        path = locate_weights(root, info.model_filename, info.model_name)
        if not path:
            print(f"   could not find weights for {info.model_filename}; skipping")
            continue
        m = read_gguf(path) if path.suffix == ".gguf" else read_mlx_dir(path)
        z = analyse(m)

        # ---------------- 1. predict
        bw = a.bw_gpu if device in ("metal", "cuda") else a.bw_cpu
        tflops = a.gpu_tflops if device in ("metal", "cuda") else a.cpu_tflops
        decode_ceiling = bw * 1e9 / z["per_token_bytes"]
        flops_per_token = 2 * z["params"]
        prefill_ceiling = tflops * 1e12 / flops_per_token
        long_prompt = CONTEXT * a.prefill_repeat + "\n\nSummarise the text above in two sentences."
        n_long, n_short = count(long_prompt), count(DECODE_PROMPT)
        ctx = info.context_window or m["ctx_train"] or 0
        print(f"   weights: {m['format']}, {fmt_bytes(z['weight_bytes'])}, {z['params'] / 1e9:.2f} B parameters, "
              f"{z['bits_per_weight']:.2f} bits/weight")
        print(f"   shape: {m['layers']} layers, hidden {m['hidden']}, {m['heads']} query heads / {m['kv_heads']} KV heads "
              f"(head dim {z['head_dim']}), output head {'tied to embedding' if z['tied'] else 'separate'}")
        print("\n   1. PREDICT (paper, before running)")
        print(f"      bytes read per new token  = all weights{'' if z['tied'] else ' minus the embedding table'} "
              f"= {fmt_bytes(z['per_token_bytes'])}")
        print(f"      decode ceiling            = {bw:.0f} GB/s / {fmt_bytes(z['per_token_bytes'])} = {decode_ceiling:6.1f} tok/s"
              f"   ({'GPU' if bw == a.bw_gpu else 'CPU'} bandwidth)")
        print(f"      prefill ceiling           = {tflops:.2f} TFLOP/s / (2 x {z['params'] / 1e9:.2f} B) = "
              f"{prefill_ceiling:6.0f} tok/s   (compute roof; assumption: {tflops} TFLOP/s)")
        print(f"      long prompt               = {n_long} tokens -> TTFT >= {n_long / prefill_ceiling:5.2f} s")
        print(f"      KV cache                  = 2 x {m['layers']} layers x {m['kv_heads']} KV heads x {z['head_dim']} x 2 B"
              f" = {z['kv_bytes_per_token'] / 1024:.0f} KiB per token; full {ctx}-token context = "
              f"{fmt_bytes(z['kv_bytes_per_token'] * ctx)}")
        if m["kv_heads"] and m["heads"] != m["kv_heads"]:
            print(f"                                  (GQA: {m['heads'] // m['kv_heads']}x smaller than one KV head per query head)")

        # ---------------- 2. measure
        print("\n   2. MEASURE (over gRPC, cold prompts: a unique first line defeats prompt caching)")
        generate(stub, pb, unique("Say OK"), 4, 120)  # warm-up: load kernels, wake the GPU clock
        dec, ttft_short, ttft_long, server = [], [], [], []
        for r in range(a.runs):
            d = generate(stub, pb, unique(DECODE_PROMPT), a.decode_tokens, 600)
            p = generate(stub, pb, unique(long_prompt), 8, 600)
            dec.append(d["decode_tok_s"])
            ttft_short.append(d["ttft_s"])
            ttft_long.append(p["ttft_s"])
            print(f"      run {r + 1}: decode {d['decode_tok_s']:6.1f} tok/s ({d['chunks']} tokens)   "
                  f"TTFT short {d['ttft_s'] * 1000:6.0f} ms   TTFT long ({n_long} tok) {p['ttft_s']:5.2f} s")
            if d["usage"] and p["usage"]:
                du, pu = d["usage"], p["usage"]
                server.append((du, pu))
                print(f"             server says: decode {du.get('tokens_per_sec', 0):6.1f} tok/s; long prompt "
                      f"{pu.get('prompt_tokens', 0)} tokens, prefill {pu.get('prefill_ms', 0):7.0f} ms "
                      f"({pu.get('prefill_tokens_per_sec', 0):6.0f} tok/s), queued {pu.get('queue_ms', 0):5.0f} ms")
        # Prefill rate from the DIFFERENCE between long and short prompts:
        # fixed costs (gRPC, Python, first decode step) cancel out.
        s_dec = s_pre = 0.0
        if server:
            # Server-side numbers exclude gRPC, Python and client overhead and use exact token counts.
            s_dec = statistics.median(u.get("tokens_per_sec", 0) for u, _ in server)
            s_pre = statistics.median(p_.get("prefill_tokens_per_sec", 0) for _, p_ in server)
            print(f"      client vs server decode: {statistics.median(dec):.1f} vs {s_dec:.1f} tok/s; "
                  f"the difference is streaming/gRPC/Python overhead per token.")
        dt = statistics.median(ttft_long) - statistics.median(ttft_short)
        prefill = (n_long - n_short) / dt if dt > 0 else float("nan")
        if server and s_pre > 0:
            prefill = s_pre  # exact: server-timed, real token counts
        decode = statistics.median(dec)
        unstable = max(dec) / max(min(dec), 1e-9) > 1.3
        if unstable:
            print(f"      ! decode varied {min(dec):.1f}-{max(dec):.1f} tok/s between runs: the GPU or service is shared.")
        # Same long prompt twice in a row: llama.cpp can reuse the cached prefix.
        same = unique(long_prompt)
        generate(stub, pb, same, 4, 600)
        warm = generate(stub, pb, same, 4, 600)["ttft_s"]
        warm_u = generate(stub, pb, same, 4, 600)["usage"]
        if warm_u:
            print(f"      server: {warm_u.get('cached_prompt_tokens', 0)} of {warm_u.get('prompt_tokens', 0)} prompt "
                  f"tokens reused from the KV cache")
        print(f"      repeat of an identical long prompt: TTFT {warm:5.2f} s "
              f"(vs {statistics.median(ttft_long):.2f} s cold){'  <- prefix cache hit' if warm < 0.5 * statistics.median(ttft_long) else ''}")

        sustained = []
        if a.seconds > 0:
            print(f"\n      sustained decode for {a.seconds:.0f} s (one line per request):")
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < a.seconds:
                d = generate(stub, pb, unique(DECODE_PROMPT), a.decode_tokens, 600)
                sustained.append(d["decode_tok_s"])
                # Server queue time > 0 means another client was using this tier at the same time.
                q = f"  queued {d['usage'].get('queue_ms', 0):5.0f} ms" if d["usage"] else ""
                print(f"        t={time.perf_counter() - t0:6.1f}s  decode {d['decode_tok_s']:6.1f} tok/s  "
                      f"({100 * d['decode_tok_s'] / sustained[0]:5.1f}% of first){q}")

        # ---------------- 3. explain
        eff_bw = decode * z["per_token_bytes"] / 1e9
        print("\n   3. EXPLAIN THE GAP")
        print(f"      decode : measured {decode:6.1f} vs ceiling {decode_ceiling:6.1f} tok/s -> {100 * decode / decode_ceiling:4.0f}%"
              f"  (effective {eff_bw:.0f} GB/s)")
        print(f"      prefill: measured {prefill:6.0f} vs ceiling {prefill_ceiling:6.0f} tok/s -> {100 * prefill / prefill_ceiling:4.0f}%")
        print(f"      prefill is {prefill / decode:.0f}x faster per token than decode: it batches all prompt tokens through")
        print("      each weight once (compute-bound); decode re-reads every weight for each new token (bandwidth-bound).")
        reasons = [
            "The spec bandwidth is a peak; GPUs typically sustain 70-90% of it on a pure stream.",
            "Each token also reads the KV cache (grows with context) and runs attention, norms and sampling.",
            "Per-token fixed costs: GPU kernel launches/command buffers, Python in the engine, gRPC streaming.",
        ]
        if z["params"] < 5e9:
            reasons.append("Small model: each token's streaming time is short, so fixed per-token costs are a larger share.")
        if m["format"].startswith("GGUF") and "K" in str(info.model_filename).upper():
            reasons.append("K-quants (Q4_K/Q5_K) unpack 6-bit sub-block scales per 256 weights: extra ALU work per byte.")
        if warm >= 0.5 * statistics.median(ttft_long):
            reasons.append("No prompt-prefix reuse on this engine: a repeated long prompt (agent system prompts) pays "
                           "full prefill every time.")
        if device == "cpu":
            reasons.append("CPU path: only the CPU clusters' share of bandwidth (~55% of spec) is reachable at all.")
        for r in reasons:
            print(f"      - {r}")
        rec = {"lab": "p3.service", "addr": addr, "model": info.model_name, "engine": info.engine, "device": device,
               "format": m["format"], "params": z["params"], "per_token_bytes": z["per_token_bytes"],
               "kv_bytes_per_token": z["kv_bytes_per_token"], "decode_pred": decode_ceiling, "decode_meas": decode,
               "prefill_pred": prefill_ceiling, "prefill_meas": prefill, "eff_bw_gbs": eff_bw,
               "ttft_short_ms": 1000 * statistics.median(ttft_short), "ttft_long_s": statistics.median(ttft_long),
               "ttft_long_warm_s": warm, "prompt_tokens": n_long, "sustained": sustained,
               "tokens_counted_with": "Qwen2.5 tokenizer" if tok else "4 chars/token estimate"}
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        if device in ("metal", "cuda") and not unstable:
            fits.append((z["per_token_bytes"], 1.0 / decode, info.model_name, info.engine))

    # Two or more GPU tiers: fit  seconds/token = bytes/token / B + c,
    # the same two-parameter model the ANE R1 work used (per-byte + fixed cost).
    if len(fits) >= 2:
        (b1, t1, n1, e1), (b2, t2, n2, e2) = sorted(fits)[0], sorted(fits)[-1]
        B = (b2 - b1) / (t2 - t1) / 1e9 if t2 != t1 else float("nan")
        c = t1 - b1 / (B * 1e9)
        print("\n== Cost model from the measured tiers:  time per token = bytes per token / B + c")
        print(f"   {n1} ({e1}): {fmt_bytes(b1)} -> {t1 * 1000:.2f} ms/token;  {n2} ({e2}): {fmt_bytes(b2)} -> {t2 * 1000:.2f} ms/token")
        if not (0 < B <= a.bw_gpu * 1.05) or c < 0:
            print(f"   Fit rejected: B = {B:.0f} GB/s, c = {c * 1000:.2f} ms is physically impossible (c must be >= 0,")
            print("   B <= spec bandwidth). The runs were too noisy or contended; rerun on an idle machine.")
            return 0
        print(f"   B = {B:.0f} GB/s achievable streaming rate,  c = {c * 1000:.2f} ms fixed cost per token")
        print("   Caveat: two points from two different engines; treat B and c as a first estimate, then")
        print("   validate by predicting a third model and checking (that is how you validate a cost model).")
        if out:
            out.write(json.dumps({"lab": "p3.service", "case": "cost_model", "B_gbs": B, "c_ms": c * 1000}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
