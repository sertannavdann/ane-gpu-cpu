"""Manual trial: a GPU model (MLX) and an ANE model (ane.cpp serve) finishing one task together.

Task: extract one commitment from each of 8 notes (structured JSON), then write a briefing.
Conditions (all produce the same final briefing shape):
  gpu_serial      GPU extracts all notes, then drafts priorities, then composes.
  ane_gpu_parallel  ANE extracts notes while GPU drafts priorities; then GPU composes.
  ane_gpu_serial    ANE extracts notes, then GPU drafts, then composes (no overlap; isolates contention).
Records wall time, extraction accuracy per device, GPU decode rate with/without ANE activity,
ANE rates, memory and swap. This is a feasibility trial, not the plan's gated R3/R4 measurement.

Usage: python -m prototype.scenario_parallel --ane-model <dir> [--gpu-model <hf id>] [--conditions a,b,c]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKERS = int(os.environ.get("ANE_WORKERS", "1"))   # >1 uses batched serve sessions, which corrupt lanes >0 on this build (see results)
os.environ.setdefault("HF_HOME", str(ROOT / "cache.nosync" / "hf"))
ANE_BIN = ROOT / "prototype" / "ane" / "ane.cpp-bin"
FIX = json.loads((ROOT / "prototype" / "scenario_notes.json").read_text())

EXTRACT_PROMPT = ("Extract one commitment from the note. Reply with ONLY a JSON object with keys "
                  "\"who\", \"when\", \"where\", \"action\" (action under 8 words). Use the words from the note.\n\nNote: {note}\n\nJSON:")
DRAFT_PROMPT = ("Write one short paragraph (3 sentences) listing these priorities for tomorrow in a calm, direct tone:\n{pri}")
COMPOSE_PROMPT = ("Write tomorrow's briefing. Start with this priorities paragraph verbatim:\n\n{draft}\n\n"
                  "Then add a section 'Commitments' with one bullet per item below, in the form 'when — who — action (where)'. "
                  "Do not invent items.\n\n{items}")


# ----------------------------------------------------------------------------- helpers
def vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    free = int(re.search(r"Pages free:\s+(\d+)", out).group(1)) * 16384
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    used = float(re.search(r"used = ([\d.]+)M", swap).group(1)) if "used" in swap else None
    return {"free_gb": round(free / 2**30, 2), "swap_used_mb": used}


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s or "").lower()).strip()


def score(pred: dict | None, label: dict) -> dict:
    if not pred:
        return {"valid_json": False, "fields_ok": 0, "who": False, "when": False, "where": False, "action": False}
    r = {"valid_json": True}
    for k in ("who", "when", "where"):
        r[k] = norm(label[k]) in norm(pred.get(k)) or norm(pred.get(k)) in norm(label[k]) and bool(norm(pred.get(k)))
    la, pa = norm(label["action"]).split(), norm(pred.get("action")).split()
    r["action"] = len(set(la) & set(pa)) >= max(1, len(la) // 2)
    r["fields_ok"] = sum(int(r[k]) for k in ("who", "when", "where", "action"))
    return r


# ----------------------------------------------------------------------------- ANE side
class AneServer:
    def __init__(self, model_dir: str, port: int, sessions: int, log: Path):
        self.port, self.log = port, log
        self.proc = subprocess.Popen([str(ANE_BIN), "serve", "--model", model_dir, "--port", str(port), "--sessions", str(sessions),
                                      "--max-tokens", "96", "--temp", "0"], stdout=log.open("w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ane.cpp serve exited {self.proc.returncode}; see {log}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        self.startup_s = time.time() - t0

    def rss_gb(self) -> float:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True, text=True).stdout.strip()
        return round(int(out or 0) / 2**20, 2)

    def request(self, rid: int, prompt: str, max_tokens: int = 96) -> dict:
        t0 = time.perf_counter()
        with socket.create_connection(("127.0.0.1", self.port), timeout=600) as s:
            s.sendall((json.dumps({"id": rid, "prompt": prompt, "max_tokens": max_tokens, "enable_thinking": False, "temp": 0.0}) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        rec = json.loads(buf.decode() or "{}")
        rec["wall_s"] = time.perf_counter() - t0
        return rec

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ----------------------------------------------------------------------------- GPU side
class Gpu:
    def __init__(self, model_id: str):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        t0 = time.time()
        self.model, self.tok = load(model_id)
        mx.eval(self.model.parameters())
        self.load_s = time.time() - t0
        self.active_gb = round(mx.get_active_memory() / 2**30, 2)

    def generate(self, user: str, max_tokens: int) -> dict:
        from mlx_lm import stream_generate
        prompt = self.tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t0 = time.perf_counter(); first = None; text = []; n = 0
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens):
            if first is None:
                first = time.perf_counter() - t0
            text.append(r.text); n = r.generation_tokens
        dt = time.perf_counter() - t0
        return {"text": "".join(text), "ttft_s": first, "tokens": n, "decode_tps": (n - 1) / (dt - first) if n > 1 and dt > first else None,
                "wall_s": dt, "peak_gb": round(self.mx.get_peak_memory() / 2**30, 2)}


# ----------------------------------------------------------------------------- task phases
def extract_all(dev: str, fn, notes, workers: int) -> list[dict]:
    def one(i_note):
        i, note = i_note
        rec = fn(i, EXTRACT_PROMPT.format(note=note["text"]))
        pred = parse_json(rec.get("text", ""))
        return {"id": note["id"], "device": dev, "pred": pred, "score": score(pred, note["label"]), "raw": rec}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, enumerate(notes)))
    for r in results:
        r["phase_wall_s"] = time.perf_counter() - t0
    return results


def run_condition(name: str, gpu: Gpu, ane: AneServer | None, notes) -> dict:
    mem0 = vm(); t0 = time.perf_counter(); rec = {"condition": name, "mem_before": mem0}
    if name == "gpu_serial":
        ext = extract_all("gpu", lambda i, p: gpu.generate(p, 96), notes, workers=1)
        draft = gpu.generate(DRAFT_PROMPT.format(pri=FIX["priorities_note"]), 120)
        rec["overlap"] = False
    elif name == "ane_gpu_parallel":
        holder = {}
        th = threading.Thread(target=lambda: holder.update(ext=extract_all("ane", lambda i, p: ane.request(i, p), notes, workers=WORKERS)))
        th.start()
        t_draft = time.perf_counter()
        draft = gpu.generate(DRAFT_PROMPT.format(pri=FIX["priorities_note"]), 120)
        draft["ended_before_ane_finished"] = th.is_alive()
        th.join()
        ext = holder["ext"]; rec["overlap"] = True
        rec["draft_started_at_s"] = t_draft - t0
    elif name == "ane_gpu_serial":
        ext = extract_all("ane", lambda i, p: ane.request(i, p), notes, workers=WORKERS)
        draft = gpu.generate(DRAFT_PROMPT.format(pri=FIX["priorities_note"]), 120)
        rec["overlap"] = False
    else:
        raise ValueError(name)
    items = "\n".join(f"- {json.dumps(r['pred'])}" for r in ext if r["pred"]) or "- (no items extracted)"
    compose = gpu.generate(COMPOSE_PROMPT.format(draft=draft["text"].strip(), items=items), 320)
    rec.update({"wall_s": time.perf_counter() - t0, "extraction_phase_s": ext[0]["phase_wall_s"],
                "extraction": [{k: v for k, v in r.items() if k != "raw"} | {"raw_text": (r["raw"].get("text") or "")[:600], "gen_tokens": r["raw"].get("generation_tokens") or r["raw"].get("tokens"),
                               "tps": r["raw"].get("generation_tps") or r["raw"].get("decode_tps"), "item_wall_s": r["raw"].get("wall_s")} for r in ext],
                "fields_correct": sum(r["score"]["fields_ok"] for r in ext), "fields_total": 4 * len(ext),
                "valid_json": sum(r["score"]["valid_json"] for r in ext),
                "draft": {k: v for k, v in draft.items() if k != "text"}, "compose": {k: v for k, v in compose.items() if k != "text"},
                "briefing_text": compose["text"], "mem_after": vm(), "ane_rss_gb": ane.rss_gb() if ane else None})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ane-model", required=True); ap.add_argument("--gpu-model", default="mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit")
    ap.add_argument("--conditions", default="gpu_serial,ane_gpu_parallel,ane_gpu_serial"); ap.add_argument("--port", type=int, default=55171)
    ap.add_argument("--sessions", type=int, default=4); ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    out = ROOT / "results" / (a.run_id or time.strftime("scenario-%Y%m%d-%H%M%S")); out.mkdir(parents=True, exist_ok=True)
    conds = a.conditions.split(","); notes = FIX["notes"]
    log = {"gpu_model": a.gpu_model, "ane_model": a.ane_model, "mem_start": vm(), "docker_vm": bool(subprocess.run(["pgrep", "-f", "com.docker"], capture_output=True).stdout)}
    gpu = Gpu(a.gpu_model); log["gpu_load_s"] = gpu.load_s; log["gpu_active_gb"] = gpu.active_gb; log["mem_after_gpu"] = vm()
    ane = None
    if any(c.startswith("ane") for c in conds):
        ane = AneServer(a.ane_model, a.port, a.sessions, out / "ane-serve.log")
        log["ane_startup_s"] = ane.startup_s; log["ane_rss_gb"] = ane.rss_gb(); log["mem_after_both"] = vm()
        warm = ane.request(999, EXTRACT_PROMPT.format(note=notes[0]["text"]))
        log["ane_warmup"] = {k: warm.get(k) for k in ("ok", "prompt_tokens", "prompt_tps", "generation_tokens", "generation_tps", "wall_s", "error")}
    results = []
    try:
        for c in conds:
            print(f"== {c}", flush=True)
            r = run_condition(c, gpu, ane, notes); results.append(r)
            print(json.dumps({k: r[k] for k in ("wall_s", "extraction_phase_s", "fields_correct", "fields_total", "valid_json")}), flush=True)
    finally:
        if ane:
            ane.close()
    log["conditions"] = results
    (out / "summary.json").write_text(json.dumps(log, indent=2, default=str))
    lines = [f"# Scenario: GPU + ANE finishing one task ({out.name})", "",
             f"GPU: `{a.gpu_model}` (load {gpu.load_s:.0f} s, {gpu.active_gb} GB active). ANE: `{Path(a.ane_model).name}`"
             + (f" (startup {log['ane_startup_s']:.0f} s, RSS {log['ane_rss_gb']} GB)." if ane else " (not started)."),
             f"Docker VM resident: {log['docker_vm']}. Free memory at start {log['mem_start']['free_gb']} GB; with both models {log.get('mem_after_both', log['mem_after_gpu'])}.", "",
             "| condition | overlap | wall s | extraction phase s | fields correct | valid JSON | draft TTFT s | draft tok/s | compose tok/s | swap used MB after |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        d, c = r["draft"], r["compose"]
        lines.append(f"| {r['condition']} | {r['overlap']} | {r['wall_s']:.1f} | {r['extraction_phase_s']:.1f} | {r['fields_correct']}/{r['fields_total']} | {r['valid_json']}/{len(r['extraction'])} | "
                     f"{d['ttft_s']:.2f} | {d['decode_tps'] or 0:.1f} | {c['decode_tps'] or 0:.1f} | {r['mem_after']['swap_used_mb']} |")
    lines += ["", "Per-note extraction rates (tok/s) by device:"]
    for r in results:
        lines.append(f"- {r['condition']}: " + ", ".join(f"{e['id']}={e['tps'] or 0:.1f}" for e in r["extraction"]))
    lines += ["", "Feasibility trial, one run per condition, fixed prompts and greedy decoding. Not the plan's gated R3/R4 measurement (no repeated windows, no energy)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    for r in results:
        (out / f"briefing-{r['condition']}.md").write_text(r["briefing_text"])
    print((out / "report.md").read_text()); print(f"results in {out}")


if __name__ == "__main__":
    main()
