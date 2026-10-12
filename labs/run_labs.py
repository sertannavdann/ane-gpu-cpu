#!/usr/bin/env python3
"""Build, test and run the P-series hardware labs; explain and save the results.

Examples
  python3 labs/run_labs.py --quick                 # every lab, ~3 minutes
  python3 labs/run_labs.py --seconds 2             # every lab, 2 s per case
  python3 labs/run_labs.py --only gemv,quant --watch --interval 0.25
  python3 labs/run_labs.py --soak 60               # 60 s sustained load, one line per second
  python3 labs/run_labs.py --only llm              # predict + measure the running gRPC_llm tiers

Each lab is a standalone binary in labs/build/ and can be run directly; this
script adds the order (ceilings measured first and fed to later labs), the
interpretation after each lab, and a results folder under results/labs-<time>/.
Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import predictions as pred  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
LABS = ["peak", "bandwidth", "latency", "gemm", "gemv", "quant"]
OPT_IN = ["llm"]  # needs the gRPC_llm services running; only with --only llm
GRPC_LLM = pathlib.Path.home() / "Documents" / "Documents - Sertan’s MacBook Pro" / "Software" / "AI" / "gRPC_llm"
BINARY = {"peak": "p1_peak", "bandwidth": "p1_bandwidth", "latency": "p1_latency", "gemm": "p1_gemm",
          "gemv": "p1_gemv", "quant": "p2_quant"}
# Processes from the R/L-series that must not share the machine with a benchmark.
RSERIES = ("run_r4", "ane.cpp-bin", "mlx_lm")


def sh(cmd: list[str], cwd: pathlib.Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        return ""


# --------------------------------------------------------------------------- preflight

def preflight() -> dict:
    """What else is running? Benchmarks are only fair on a quiet machine."""
    rseries, busy = [], []
    me = os.getpid()
    for line in sh(["ps", "-Ao", "pid=,pcpu=,command="]).splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, pcpu, cmd = int(parts[0]), float(parts[1]), parts[2]
        if pid == me or "/labs/build/" in cmd:
            continue
        if any(k in cmd for k in RSERIES):
            rseries.append(cmd[:120])
        if pcpu >= 20:
            busy.append({"pcpu": pcpu, "command": os.path.basename(cmd.split()[0])[:60]})
    busy.sort(key=lambda b: -b["pcpu"])
    p_cores = int(sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]) or 1)
    busy_cores = math.ceil(sum(b["pcpu"] for b in busy) / 100.0)
    gpu = None
    m = re.search(r'"Device Utilization %"=(\d+)', sh(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]))
    if m:
        gpu = int(m.group(1))
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "gpu_busy_pct": gpu,
        "git": sh(["git", "rev-parse", "--short", "HEAD"], ROOT),
        "macos": sh(["sw_vers", "-productVersion"]),
        "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "p_cores": sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
        "e_cores": sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"]),
        "load_avg": [round(x, 2) for x in os.getloadavg()],
        "power": (sh(["pmset", "-g", "batt"]).splitlines() or ["?"])[0],
        "busy_processes": busy[:8],
        "busy_cores": busy_cores,
        "work_threads": max(1, p_cores - busy_cores),
        "rseries_processes": rseries,
    }


def print_preflight(env: dict) -> None:
    print(f"Machine: {env['cpu']} ({env['p_cores']}P + {env['e_cores']}E), macOS {env['macos']}, git {env['git']}")
    print(f"Power: {env['power']}")
    print(f"Load average (1/5/15 min): {env['load_avg']}")
    if env.get("gpu_busy_pct") is not None:
        g = env["gpu_busy_pct"]
        print(f"GPU utilisation now: {g}%" + ("  <- another app is using the GPU; the llm lab will read low" if g > 15 else ""))
    if env["busy_processes"]:
        print("Other processes using >= 20% of a core (they compete for cores and memory bandwidth):")
        for b in env["busy_processes"]:
            print(f"   {b['pcpu']:6.1f}%  {b['command']}")
        print(f"   They hold about {env['busy_cores']} core(s). GEMM/GEMV/quant labs will use {env['work_threads']} threads")
        print("   so no thread has to share a core (see the barrier effect in the peak lab).")
        print("   Results stay valid for learning, but expect noise and lower all-core numbers.")
        print("   Close them for numbers you intend to quote.")
    print()


# --------------------------------------------------------------------------- build / run

def build() -> None:
    if not (BUILD / "build.ninja").exists():
        env = dict(os.environ, CXX="/usr/bin/clang++")
        subprocess.run(["cmake", "-S", str(HERE), "-B", str(BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"],
                       check=True, env=env, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", str(BUILD)], check=True, stdout=subprocess.DEVNULL)


def run_tests() -> None:
    print("Correctness first: a fast wrong kernel is worthless.")
    r = subprocess.run([str(BUILD / "test_kernels")], capture_output=True, text=True)
    print("  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode != 0:
        sys.exit("Kernel tests failed; not running benchmarks.")


def run_lab(name: str, args: list[str], log, cmd: list[str] | None = None) -> None:
    cmd = (cmd or [str(BUILD / BINARY[name])]) + args
    log.write(f"\n$ {' '.join(cmd)}\n")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        if p.wait() != 0:
            print(f"  [lab {name} exited with code {p.returncode}]")


def records(path: pathlib.Path, lab: str, case: str | None = None) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("lab") == lab and (case is None or r.get("case") == case):
                out.append(r)
    return out


# --------------------------------------------------------------------------- interpretation

LOG = None  # console.log of the current run; notices are written there too


def notice(lines: list[str]) -> None:
    text = "\n  >> What to notice\n" + "".join(f"     - {l}\n" for l in lines) + "\n"
    sys.stdout.write(text)
    if LOG:
        LOG.write(text)


def explain_peak(j: pathlib.Path, ceil: dict) -> None:
    ch = {r["chains"]: r["median"] for r in records(j, "p1.peak", "chains_f32")}
    clk = records(j, "p1.peak", "clock_estimate")
    thr = records(j, "p1.peak", "threads_f32")
    ecs = records(j, "p1.peak", "ecores_f32")
    f16 = records(j, "p1.peak", "f16_1core")
    if ch:
        ceil["peak_1core"] = max(ch.values())
    if thr:
        ceil["peak_all"] = max(r["median"] for r in thr)
    if clk:
        ceil["ghz"] = clk[0]["ghz"]
    lines = []
    if 1 in ch and ch:
        best_c = max(ch, key=ch.get)
        lines.append(f"1 chain: {ch[1]:.1f} GFLOP/s; {int(best_c)} chains: {ch[best_c]:.1f} GFLOP/s "
                     f"({ch[best_c] / ch[1]:.0f}x). Same instructions, same core: only the dependencies changed.")
    if clk:
        lines.append(f"Inferred clock ≈ {clk[0]['ghz']:.2f} GHz (assumes 4-cycle FMA latency). "
                     "Every other lab's 'per cycle' reasoning uses this number.")
    if f16 and ch:
        lines.append(f"fp16 peak {f16[0]['median']:.0f} vs fp32 {max(ch.values()):.0f} GFLOP/s: "
                     "same instruction rate, twice the lanes per 128-bit register.")
    if thr:
        one = min(thr, key=lambda r: r["threads"])
        best = max(thr, key=lambda r: r["median"])
        lines.append(f"Scaling: {one['median']:.0f} GFLOP/s on 1 thread -> {best['median']:.0f} at "
                     f"{int(best['threads'])} threads ({best['median'] / one['median']:.1f}x).")
        per = sorted(thr, key=lambda r: r["threads"])
        lines.append("Per-thread rate: " + ", ".join(f"{int(r['threads'])}T {r['median'] / r['threads']:.0f}" for r in per) +
                     " GFLOP/s. A lower all-core clock explains a modest drop. A collapse once threads >= free "
                     "cores is the barrier effect: every pool.run step waits for its slowest thread, and a thread "
                     "sharing a core with another busy app is always slowest.")
    if ecs and thr:
        e = ecs[0]
        lines.append(f"{int(e['threads'])} E-cores: {e['median']:.0f} GFLOP/s, about "
                     f"{e['median'] / e['threads'] / one['median'] * 100:.0f}% of a P-core each. "
                     "E-cores are narrower and clocked lower; they exist for efficiency, not peak.")
    notice(lines)


def explain_bandwidth(j: pathlib.Path, ceil: dict) -> None:
    sweep = records(j, "p1.bandwidth", "sweep")
    reads = records(j, "p1.bandwidth", "dram_read")
    ecs = records(j, "p1.bandwidth", "dram_read_ecores")
    c = records(j, "p1.bandwidth", "ceiling")
    lines = []
    if sweep:
        by = {}
        for r in sweep:
            by.setdefault(r["level"], []).append(r["median"])
        parts = [f"{lvl} ≈ {sum(v) / len(v):.0f} GB/s" for lvl, v in by.items()]
        lines.append("One core, by level: " + ", ".join(parts) + ". Each cache edge is a step down.")
    if reads:
        one = min(reads, key=lambda r: r["threads"])
        best = max(reads, key=lambda r: r["median"])
        ceil["bw"] = best["median"]
        lines.append(f"DRAM read: {one['median']:.0f} GB/s on 1 core, best {best['median']:.0f} GB/s at "
                     f"{int(best['threads'])} threads. One core cannot keep enough misses in flight; "
                     "a few cores saturate what the CPU clusters can draw.")
    if c and c[0].get("spec_gbs"):
        lines.append(f"That is {100 * c[0]['dram_read_gbs'] / c[0]['spec_gbs']:.0f}% of the {c[0]['spec_gbs']:.0f} GB/s "
                     "spec figure. The rest is reachable by the GPU, which is why GPU decode is faster than CPU decode.")
        lines.append("P0 cross-check: the ANE streamed weights at ≈90 GB/s, about the same as ONE CPU core here.")
    if ecs:
        lines.append(f"E-cores alone: {ecs[0]['median']:.0f} GB/s. Background work on E-cores leaves bandwidth "
                     "for the foreground, but it is slow.")
    notice(lines)


def explain_latency(j: pathlib.Path, ceil: dict) -> None:
    rnd = sorted(records(j, "p1.latency", "random"), key=lambda r: r["bytes"])
    seq = sorted(records(j, "p1.latency", "sequential"), key=lambda r: r["bytes"])
    if not rnd:
        return
    ghz = ceil.get("ghz", 0)
    first, last = rnd[0], rnd[-1]
    mid = min(rnd, key=lambda r: abs(r["bytes"] - 2 * 2**20))
    cyc = (lambda ns: f" ≈ {ns * ghz:.{1 if ns * ghz < 10 else 0}f} cycles") if ghz else (lambda ns: "")
    lines = [
        f"L1 hit {first['ns']:.2f} ns{cyc(first['ns'])}; L2 (2 MiB) {mid['ns']:.1f} ns{cyc(mid['ns'])}; "
        f"DRAM {last['ns']:.0f} ns{cyc(last['ns'])}.",
        f"A DRAM miss costs ~{last['ns'] / first['ns']:.0f} L1 hits. Kernels win by reusing data in cache "
        "(blocking) or by keeping many misses in flight (streaming).",
    ]
    if seq:
        lines.append(f"Sequential order at the largest set: {seq[-1]['ns']:.1f} ns per line, "
                     f"{last['ns'] / seq[-1]['ns']:.0f}x faster than random: the prefetcher at work.")
    notice(lines)


def explain_gemm(j: pathlib.Path, ceil: dict) -> None:
    steps = records(j, "p1.gemm")
    st = [r for r in steps if r["case"].startswith("steps_")]
    lines = []
    if st:
        base = st[0]["median"]
        lines.append("Step speedups over " + st[0]["case"][6:] + ": " +
                     ", ".join(f"{r['case'][6:]} {r['median'] / base:.0f}x" for r in st[1:]))
        lines.append("ikj vs naive is pure memory access order: same FLOPs, unit-stride instead of column walks.")
        lines.append("If 'blocked' is not faster than ikj at this size, the matrices already fit in L2; "
                     "rerun with --only gemm and a larger n (gemm binary: --n 2048) to see blocking pay off.")
    sp = records(j, "p1.gemm", "shape_packed")
    sa = records(j, "p1.gemm", "shape_accel")
    if sa:
        m1 = min(sa, key=lambda r: r["m"])
        mx = max(sa, key=lambda r: r["m"])
        lines.append(f"Accelerate: M={int(m1['m'])} -> {m1['median']:.0f} GFLOP/s, M={int(mx['m'])} -> "
                     f"{mx['median']:.0f} GFLOP/s. Same weights; batching tokens raises FLOPs per byte.")
        lines.append("Decode (M=1) is memory-bound; prefill (M>=32) is compute-bound. That is P0's ANE story: "
                     "its runtime ran prefill one token at a time, i.e. as decode.")
    if sp and sa:
        lines.append("The packed kernel is slow at M=1: it pads to 8 rows and repacks all weights every call. "
                     "Real runtimes pre-pack weights once and use a dedicated GEMV for decode.")
    notice(lines)


def explain_gemv(j: pathlib.Path, ceil: dict) -> None:
    rs = records(j, "p1.gemv")
    f32 = [r for r in rs if r["case"] == "NEON fp32"]
    f16 = [r for r in rs if r["case"].startswith("NEON fp16")]
    lines = []
    bw = ceil.get("bw")
    if f32:
        b = max(f32, key=lambda r: r["median"])
        pct = f" = {100 * b['median'] / bw:.0f}% of the measured DRAM read roof" if bw else ""
        lines.append(f"Best fp32 GEMV {b['median']:.0f} GB/s at {int(b['threads'])} threads{pct}. "
                     "Decode is a streaming read; nothing else matters once the roof is reached.")
        one = [r for r in f32 if r["threads"] == 1]
        if one:
            ghz = ceil.get("ghz", 4.5)
            lim = 4 * ghz * 4  # 4 accumulator chains x 4 lanes, one FMA per chain per 4-cycle latency
            lines.append(f"1 thread: {one[0]['median']:.0f} GB/s. The kernel has 4 accumulator chains of 4 lanes; with "
                         f"4-cycle FMA latency that is at most 4 lanes x {ghz:.1f} GHz = {lim / 4:.0f} G weights/s "
                         f"= {lim:.0f} GB/s of fp32 per core. The single-thread number is an instruction-latency "
                         "limit, not DRAM. Try 8 accumulators and measure again.")
    if f16:
        b16 = max(f16, key=lambda r: r["median"])
        g16 = b16["median"]
        lines.append(f"Best fp16 {g16:.0f} GB/s = {g16 / 2:.0f} G weights/s: "
                     f"a 4B-parameter fp16 model (8 GB) would decode at ≈{g16 / 8:.0f} tok/s on these CPU cores; "
                     f"Q4_0 (2.25 GB) at ≈{g16 / 2.25:.0f} tok/s if it reached the same GB/s.")
        lines.append("P0 cross-check: the ANE decoded the 4B model at 11.3 tok/s and the GPU at 40 tok/s. "
                     "tok/s ≈ bandwidth / weight bytes explains both.")
    notice(lines)


def explain_quant(j: pathlib.Path, ceil: dict) -> None:
    g = [r for r in records(j, "p2.quant") if r["case"].startswith("gemv_")]
    lines = []
    sweep = records(j, "p2.quant", "block_sweep")
    if sweep:
        q32 = [r for r in sweep if r["bits"] == 4 and r["block"] == 32]
        q4k = [r for r in sweep if r["bits"] == 4 and r["block"] == 4096]
        if q32 and q4k:
            lines.append(f"Q4 with outliers: block 32 error {q32[0]['err_outlier']:.3f} vs block 4096 "
                         f"{q4k[0]['err_outlier']:.3f}, for 0.5 extra bits per weight.")
    sq = records(j, "p2.quant", "smoothquant")
    if sq:
        b = min(sq, key=lambda r: r["err"])
        lines.append(f"SmoothQuant: W8A8 error {sq[0]['err']:.3f} -> {b['err']:.3f} at alpha {b['alpha']:.2f}.")
    aw = records(j, "p2.quant", "awq")
    if aw:
        b = min(aw, key=lambda r: r["err"])
        lines.append(f"AWQ-style scaling: W4 error {aw[0]['err']:.3f} -> {b['err']:.3f} at alpha {b['alpha']:.2f}.")
    if g:
        tmax = max(r["threads"] for r in g)
        at = {r["case"][5:]: r for r in g if r["threads"] == tmax}
        if "fp16" in at and "Q4_0" in at:
            sp = at["Q4_0"]["median"] / at["fp16"]["median"]
            lines.append(f"At {int(tmax)} threads Q4_0 decodes {sp:.1f}x faster than fp16 (byte ratio predicts 3.6x). "
                         f"Q4_0 moved {at['Q4_0']['gbs']:.0f} GB/s vs fp16 {at['fp16']['gbs']:.0f} GB/s: "
                         "if Q4 moves fewer GB/s, unpacking is the bottleneck, not memory.")
        one = {r["case"][5:]: r for r in g if r["threads"] == 1}
        if "Q8_0" in one and "fp16" in one:
            lines.append(f"1 thread: Q8_0 {one['Q8_0']['median']:.0f} G weights/s vs fp16 {one['fp16']['median']:.0f}: "
                         "integer sdot does 16 multiply-adds per instruction, so low-bit can be faster even "
                         "before bandwidth matters.")
    notice(lines)


EXPLAIN = {"peak": explain_peak, "bandwidth": explain_bandwidth, "latency": explain_latency,
           "gemm": explain_gemm, "gemv": explain_gemv, "quant": explain_quant}


# --------------------------------------------------------------------------- roofline

def ascii_roofline(j: pathlib.Path, ceil: dict) -> str:
    peak, bw = ceil.get("peak_all"), ceil.get("bw")
    if not peak or not bw:
        return "(roofline needs the peak and bandwidth labs in the same run)\n"
    W, H = 66, 20
    x0, x1, y0, y1 = -1.0, 3.0, 0.0, 4.0  # log10 FLOP/byte, log10 GFLOP/s

    def col(ai):
        return int(round((math.log10(ai) - x0) / (x1 - x0) * (W - 1)))

    def row(g):
        return H - 1 - int(round((math.log10(max(g, 1)) - y0) / (y1 - y0) * (H - 1)))

    grid = [[" "] * W for _ in range(H)]
    for c in range(W):
        ai = 10 ** (x0 + c / (W - 1) * (x1 - x0))
        r = row(min(peak, ai * bw))
        if 0 <= r < H:
            grid[r][c] = "-" if ai * bw >= peak else "/"
    pts = []
    for r in records(j, "p1.gemm", "shape_accel"):
        pts.append((r["ai"], r["median"], "a"))
    for r in records(j, "p1.gemm", "shape_packed"):
        pts.append((r["ai"], r["median"], "p"))
    for r in records(j, "p1.gemv"):
        if "median" not in r:
            continue
        bpw = r.get("bytes_per_weight", 4)
        pts.append((2 / bpw, r["median"] * 2 / bpw, "v"))
    for r in records(j, "p2.quant"):
        if r["case"].startswith("gemv_"):
            pts.append((2 / r["bytes_per_weight"], r["median"] * 2, "q"))
    for ai, g, ch in pts:
        c, rr = col(ai), row(g)
        if 0 <= c < W and 0 <= rr < H:
            grid[rr][c] = ch
    lines = []
    for i, gr in enumerate(grid):
        gval = 10 ** (y1 - i / (H - 1) * (y1 - y0))
        label = f"{gval:7.0f} |" if i % 3 == 0 else "        |"
        lines.append(label + "".join(gr))
    lines.append("        +" + "-" * W)
    axis = [" "] * (W + 4)
    for v in (0.1, 1, 10, 100, 1000):
        s, c = f"{v:g}", col(v)
        for k, chx in enumerate(s):
            axis[c + k] = chx
    lines.append("         " + "".join(axis) + "   FLOP/byte")
    lines.append(f"  roof: '/' = {bw:.0f} GB/s x intensity, '-' = {peak:.0f} GFLOP/s NEON peak (y axis GFLOP/s, log)")
    lines.append("  points: a = Accelerate GEMM (M = 1..512), p = packed NEON GEMM, v = GEMV, q = quantized GEMV")
    lines.append("  'a' above the '-' roof: Accelerate uses the SME matrix unit, which has its own, higher roof.")
    lines.append("  GEMM points at low intensity above '/': the 12 MiB weights were cache-resident, not DRAM.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- predict -> measure -> gap

TABLE: list[dict] = []  # every prediction of the run, for summary.md


def show_predictions(name: str, ceil: dict) -> list[dict]:
    P = pred.predictions(name, ceil)
    if P:
        text = "\n  1. PREDICT (on paper, before measuring)\n"
        for p in P:
            text += (f"     {p['label']:<40s} {pred.fmt(p['pred'], p['unit']):>10s} {pred.unit_label(p['unit']):<8s}"
                     f"  {p['how']}\n")
        sys.stdout.write(text)
        if LOG:
            LOG.write(text)
    return P


def show_gap(name: str, P: list[dict], jpath: pathlib.Path) -> None:
    rows = []
    for p in P:
        v = p["value"] if "value" in p else p["measure"](jpath)
        ratio = (v / p["pred"]) if (v is not None and p["pred"]) else None
        rows.append((p, v, ratio))
        TABLE.append({"lab": name, "label": p["label"], "pred": p["pred"], "meas": v, "unit": p["unit"],
                      "how": p["how"], "ratio": ratio})
    if not rows:
        return
    text = "\n  2-3. PREDICTED vs MEASURED (explain every gap; see 'What to notice')\n"
    text += f"     {'quantity':<40s} {'predicted':>10s} {'measured':>10s} {'meas/pred':>10s}\n"
    for p, v, ratio in rows:
        text += (f"     {p['label']:<40s} {pred.fmt(p['pred'], p['unit']):>10s} {pred.fmt(v, p['unit']):>10s} "
                 f"{(f'{ratio:.2f}' if ratio is not None else '-'):>10s}\n")
    sys.stdout.write(text)
    if LOG:
        LOG.write(text)


def run_llm(a, jpath: pathlib.Path, ceil: dict, log) -> None:
    root = pathlib.Path(a.llm_root)
    py = root / "llm_service" / ".venv" / "bin" / "python"
    if not py.exists():
        print(f"  gRPC_llm virtualenv not found at {py}; set --llm-root")
        return
    print("\n  The service lab prints its own PREDICT / MEASURE / EXPLAIN for each tier.")
    args = ["--root", str(root), "--addr", a.llm_addr, "--runs", str(a.llm_runs), "--json", str(jpath)]
    if a.llm_seconds:
        args += ["--seconds", str(a.llm_seconds)]
    if "bw" in ceil:
        args += ["--bw-cpu", f"{ceil['bw']:.0f}"]
    run_lab("llm", args, log, cmd=[str(py), str(HERE / "p3" / "llm_service_lab.py")])
    show_gap("llm", pred.p3_predictions(jpath), jpath)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=None, help="measurement time per case (default 1.0; --quick 0.3)")
    ap.add_argument("--quick", action="store_true", help="short budgets and fewer sizes (~3 minutes total)")
    ap.add_argument("--only", default="", help="comma list of labs: " + ",".join(LABS))
    ap.add_argument("--watch", action="store_true", help="print a live line every --interval while measuring")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--soak", type=float, default=0, help="run only a sustained all-P-core load for N seconds")
    ap.add_argument("--force", action="store_true", help="run even if R/L-series processes are active")
    ap.add_argument("--llm-root", default=str(GRPC_LLM), help="gRPC_llm checkout (for --only llm)")
    ap.add_argument("--llm-addr", default="localhost:50051,localhost:50061", help="tiers to measure")
    ap.add_argument("--llm-runs", type=int, default=3)
    ap.add_argument("--llm-seconds", type=float, default=0, help="extra sustained decode per tier, seconds")
    a = ap.parse_args()

    env = preflight()
    print_preflight(env)
    if env["rseries_processes"] and not a.force:
        sys.exit("R/L-series processes are running (see memory note): wait for them, or pass --force.")

    build()
    run_tests()

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ROOT / "results" / f"labs-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    global LOG
    jpath, log = out / "records.jsonl", open(out / "console.log", "w")
    LOG = log
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"Results folder: {out.relative_to(ROOT)}\n")

    if a.soak:
        run_lab("peak", ["--part", "soak", "--seconds", str(a.soak), "--interval", str(a.interval if a.watch else 1.0),
                         "--json", str(jpath)], log)
        return

    seconds = a.seconds if a.seconds is not None else (0.3 if a.quick else 1.0)
    common = ["--seconds", str(seconds), "--json", str(jpath), "--interval", str(a.interval)]
    if a.watch:
        common.append("--watch")
    only = [s for s in a.only.split(",") if s] or LABS
    unknown = set(only) - set(LABS) - set(OPT_IN)
    if unknown:
        sys.exit(f"unknown labs: {', '.join(sorted(unknown))}")
    quick = {
        "bandwidth": ["--dram-mib", "256", "--threads", "1,4,8,10", "--sizes-kib", "16,96,1024,8192,65536,524288"],
        "latency": ["--sizes-kib", "32,512,8192,24576,131072"],
        "peak": ["--threads", "1,4,8,10,14"],
        "gemm": ["--n", "768", "--m", "1,16,64,256,512"],
        "gemv": ["--footprint-mib", "512"],
        "quant": ["--footprint-mweights", "128"],
    }
    wt = env["work_threads"]
    tlist = ",".join(str(t) for t in sorted({1, min(4, wt), wt}))
    ceil: dict = {}
    for name in only:
        if name == "llm":
            run_llm(a, jpath, ceil, log)
            continue
        extra = list(quick[name]) if a.quick else []
        if name in ("gemv", "quant"):
            extra += ["--threads", tlist]
        if name == "gemm":
            extra += ["--threads", str(wt)]
            for k, flag in (("peak_1core", "--peak-1core"), ("peak_all", "--peak-all"), ("bw", "--bw")):
                if k in ceil:
                    extra += [flag, f"{ceil[k]:.1f}"]
        if name == "gemv" and "bw" in ceil:
            extra += ["--bw", f"{ceil['bw']:.1f}"]
        P = show_predictions(name, ceil)
        run_lab(name, common + extra, log)
        show_gap(name, P, jpath)
        EXPLAIN[name](jpath, ceil)

    roof = ascii_roofline(jpath, ceil)
    print("== Roofline (this run) ==\n")
    print(roof)
    summary = [f"# Lab run {stamp}", "", "## Machine", "", "```", json.dumps(env, indent=2), "```", "",
               "## Ceilings measured", ""]
    for k, v in ceil.items():
        summary.append(f"- {k}: {v:.1f}")
    summary += ["", "## Predicted vs measured", "",
                "| lab | quantity | predicted | measured | meas/pred | how predicted |",
                "| --- | --- | ---: | ---: | ---: | --- |"]
    for t in TABLE:
        r = f"{t['ratio']:.2f}" if t["ratio"] is not None else "-"
        summary.append(f"| {t['lab']} | {t['label']} | {pred.fmt(t['pred'], t['unit'])} {pred.unit_label(t['unit'])} | "
                       f"{pred.fmt(t['meas'], t['unit'])} | {r} | {t['how']} |")
    summary += ["", "## Roofline", "", "```", roof, "```", "", "Full console output: `console.log`; "
                "every measurement: `records.jsonl`."]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    log.close()
    print(f"Saved: {out.relative_to(ROOT)}/summary.md, records.jsonl, console.log, env.json")


if __name__ == "__main__":
    main()
