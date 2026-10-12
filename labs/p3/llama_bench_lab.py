#!/usr/bin/env python3
"""P3 lab: llama.cpp on the CPU, predicted from P1 and measured with llama-bench.

  python3 labs/p3/llama_bench_lab.py                       # 1.5B in F16/Q8_0/Q4_K_M/Q4_0
  python3 labs/p3/llama_bench_lab.py --threads 1,4,8 --reps 5
  python3 labs/p3/llama_bench_lab.py --models path/a.gguf,path/b.gguf

Needs the pinned CPU build (labs/p3/vendor_llama.sh) and the models
(labs/p3/models.sh). Same loop as every lab:

  1. PREDICT  decode  = min(bandwidth available to t threads, roof) / bytes per token
              prefill = t x per-core fp16 FLOP/s / (2 x parameters)
  2. MEASURE  llama-bench, prompt (pp) and generation (tg) separately, per thread count
  3. EXPLAIN  where each format sits against its roof, and where thread scaling stops
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from llm_service_lab import analyse, fmt_bytes, read_gguf  # noqa: E402

BENCH = HERE / "llama.nosync" / "build" / "bin" / "llama-bench"
MODELS = HERE / "models.nosync"
DEFAULT = ["f16", "q8_0", "q4_k_m", "q4_0"]

# Ceilings measured by the P1 labs on this machine (quiet run, 2026-10-12).
P1 = {
    "bw_1core": 90.0,      # GB/s, one P-core streaming from DRAM
    "bw_all": 253.0,       # GB/s, best CPU DRAM read
    "fp16_1core": 270.0,   # GFLOP/s, NEON fp16 FMA peak on one P-core
}


def bw_for(threads: int) -> float:
    """Bandwidth t threads can pull: each core adds ~one core's worth until the CPU roof."""
    return min(threads * P1["bw_1core"], P1["bw_all"])


def run_bench(models: list[pathlib.Path], threads: list[int], pp: int, tg: int, reps: int) -> list[dict]:
    cmd = [str(BENCH), "-o", "json", "-r", str(reps), "-p", str(pp), "-n", str(tg),
           "-t", ",".join(map(str, threads))]
    for m in models:
        cmd += ["-m", str(m)]
    print("   $ " + " ".join(c if " " not in c else repr(c) for c in cmd[:10]) + " ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr[-2000:])
    return json.loads(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="comma list of GGUF paths (default: the four 1.5B formats)")
    ap.add_argument("--threads", default="1,2,4,6,8,10")
    ap.add_argument("--pp", type=int, default=512, help="prompt tokens for the prefill test")
    ap.add_argument("--tg", type=int, default=128, help="generated tokens for the decode test")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--bw", type=float, default=None, help="override the CPU bandwidth roof (GB/s)")
    ap.add_argument("--json", help="append records here (JSON lines)")
    a = ap.parse_args()
    if a.bw:
        P1["bw_all"] = a.bw
    if not BENCH.exists():
        sys.exit("missing llama-bench: run  bash labs/p3/vendor_llama.sh")
    models = [pathlib.Path(p) for p in a.models.split(",") if p] or \
        [MODELS / f"qwen2.5-1.5b-instruct-{t}.gguf" for t in DEFAULT]
    missing = [str(m) for m in models if not m.exists()]
    if missing:
        sys.exit("missing models (run  bash labs/p3/models.sh): " + ", ".join(missing))
    threads = [int(t) for t in a.threads.split(",")]

    print("== P3.llamacpp  llama.cpp CPU backend: predict, measure, explain")
    info = {}
    print("\n   1. PREDICT (from the model files and the P1 ceilings, before running)")
    print(f"      bandwidth for t threads = min(t x {P1['bw_1core']:.0f}, {P1['bw_all']:.0f}) GB/s;  "
          f"prefill compute = t x {P1['fp16_1core']:.0f} GFLOP/s (NEON fp16)")
    print(f"      {'format':<10} {'bytes/token':>12} {'bits/wt':>8} {'decode @1T':>11} {'decode roof':>12} "
          f"{'knee':>6} {'prefill @1T':>12}")
    for m in models:
        z = analyse(read_gguf(m))
        name = m.stem.split("instruct-")[-1].upper()
        info[m.name] = (name, z)
        roof = P1["bw_all"] * 1e9 / z["per_token_bytes"]
        one = bw_for(1) * 1e9 / z["per_token_bytes"]
        knee = P1["bw_all"] / P1["bw_1core"]
        pre1 = P1["fp16_1core"] * 1e9 / (2 * z["params"])
        print(f"      {name:<10} {fmt_bytes(z['per_token_bytes']):>12} {z['bits_per_weight']:8.2f} "
              f"{one:9.0f}/s {roof:10.0f}/s {knee:5.1f}T {pre1:10.0f}/s")
    print(f"      Decode should scale until ~{P1['bw_all'] / P1['bw_1core']:.1f} threads, then flatten at the "
          "bandwidth roof;\n      prefill should keep scaling with threads, because it is compute-bound.")

    print(f"\n   2. MEASURE (llama-bench: pp{a.pp} = prefill, tg{a.tg} = decode; {a.reps} reps per cell)")
    rows = run_bench(models, threads, a.pp, a.tg, a.reps)
    out = open(a.json, "a") if a.json else None
    table = {}  # (model, test, threads) -> (avg, sd)
    for r in rows:
        test = "pp" if r["n_prompt"] > 0 and r["n_gen"] == 0 else "tg"
        key = (pathlib.Path(r["model_filename"]).name, test, r["n_threads"])
        table[key] = (r["avg_ts"], r["stddev_ts"])
    commit = rows[0].get("build_commit", "?") if rows else "?"
    print(f"      build {commit}, cpu: {rows[0].get('cpu_info', '?') if rows else '?'}")

    for test, label in (("tg", "decode (tok/s)"), ("pp", "prefill (tok/s)")):
        print(f"\n      {label:<16}" + "".join(f"{t:>9}T" for t in threads))
        for m in models:
            name, z = info[m.name]
            cells = []
            for t in threads:
                v = table.get((m.name, test, t))
                cells.append(f"{v[0]:10.1f}" if v else f"{'-':>10}")
            print(f"      {name:<16}" + "".join(cells))

    print("\n   3. EXPLAIN (measured vs predicted)")
    print(f"      {'format':<8} {'best decode':>12} {'at':>4} {'% roof':>7} {'eff GB/s':>9} "
          f"{'1T decode %pred':>16} {'best prefill':>13} {'at':>4} {'prefill %pred':>14}")
    summary = []
    for m in models:
        name, z = info[m.name]
        tg = {t: table[(m.name, "tg", t)][0] for t in threads if (m.name, "tg", t) in table}
        pp = {t: table[(m.name, "pp", t)][0] for t in threads if (m.name, "pp", t) in table}
        if not tg or not pp:
            continue
        tb, pb = max(tg, key=tg.get), max(pp, key=pp.get)
        roof = P1["bw_all"] * 1e9 / z["per_token_bytes"]
        eff = tg[tb] * z["per_token_bytes"] / 1e9
        one_pred = bw_for(1) * 1e9 / z["per_token_bytes"]
        pre_pred = pb * P1["fp16_1core"] * 1e9 / (2 * z["params"])
        print(f"      {name:<8} {tg[tb]:10.1f}/s {tb:>3}T {100 * tg[tb] / roof:6.0f}% {eff:9.0f} "
              f"{100 * tg.get(1, 0) / one_pred:15.0f}% {pp[pb]:11.0f}/s {pb:>3}T {100 * pp[pb] / pre_pred:13.0f}%")
        rec = {"lab": "p3.llamacpp", "format": name, "per_token_bytes": z["per_token_bytes"], "params": z["params"],
               "decode": tg, "prefill": pp, "decode_roof": roof, "eff_gbs": eff, "build": commit}
        summary.append(rec)
        if out:
            out.write(json.dumps(rec) + "\n")

    if len(summary) >= 2:
        base = next((s for s in summary if s["format"] == "F16"), summary[0])
        print(f"\n      Speedup over {base['format']} at each format's best thread count, vs the byte ratio:")
        bdec = max(base["decode"].values())
        for s in summary:
            if s is base:
                continue
            ratio = base["per_token_bytes"] / s["per_token_bytes"]
            sp = max(s["decode"].values()) / bdec
            print(f"        {s['format']:<8} decode {sp:4.2f}x  (bytes predict {ratio:4.2f}x)  "
                  f"prefill {max(s['prefill'].values()) / max(base['prefill'].values()):4.2f}x")
    print("\n      Reading guide:")
    print("      - decode flattening near the bandwidth knee = memory-bound (P1 bandwidth lab).")
    print("      - decode below the roof at every thread count = per-token overhead or unpacking cost (P2).")
    print("      - prefill above the fp16 prediction = int8 kernels (sdot / i8mm) beat fp16 FMA on Q8/Q4.")
    print("      - prefill flattening = compute shared with fewer free cores, or thread synchronisation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
