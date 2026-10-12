"""Orchestrates R0 provenance and R1 projection experiments; writes results/ane-*/.
Usage: python -m prototype.ane.benchmark --r0 | --r1 [--quick]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import reference as R
from .shapes import export as export_weight

ROOT = Path(__file__).resolve().parents[2]
ANE = ROOT / "prototype" / "ane"
BUILD = ANE / "build"
FIX = ANE / "fixtures"
MARKERS = Path.home() / "Library/Caches/ane_lm/compiled_markers"


def sh(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw).stdout.strip()
    except Exception as e:
        return f"unavailable: {e!r}"


def host_manifest() -> dict:
    fw = "/System/Library/PrivateFrameworks"
    return {"chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]), "os": sh(["sw_vers", "-productVersion"]),
            "build": sh(["sw_vers", "-buildVersion"]), "xcode": sh(["xcodebuild", "-version"]).replace("\n", " "),
            "sdk": sh(["xcrun", "--show-sdk-version"]), "compiler": sh([sh(["xcrun", "-f", "clang++"]), "--version"]).split("\n")[0],
            "cmake": sh(["cmake", "--version"]).split("\n")[0], "python": sys.version.split()[0], "platform": platform.platform(),
            "frameworks": {n: os.path.realpath(f"{fw}/{n}.framework/Versions/Current") if os.path.exists(f"{fw}/{n}.framework") else None
                           for n in ("AppleNeuralEngine", "ANECompiler", "MilAneflow", "ANEServices")},
            "docker_vm_running": bool(sh(["pgrep", "-f", "com.docker"])), "resident_llm_procs": sh(["pgrep", "-fl", "llm_service|ane.cpp|mlx"]),
            "power_mode": sh(["pmset", "-g", "batt"]).split("\n")[0], "marker_cache_dir": str(MARKERS),
            "pin": json.loads((ANE / "vendor/PIN.json").read_text()), "cmake_cache_flags": {
                k: v for k, v in (l.split("=", 1) for l in (BUILD / "CMakeCache.txt").read_text().splitlines()
                                  if l.startswith(("CMAKE_CXX_COMPILER:", "CMAKE_BUILD_TYPE:", "CMAKE_OSX_DEPLOYMENT_TARGET:", "CMAKE_GENERATOR:")))}}


def r0(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    q = json.loads((FIX / "q_proj.json").read_text()) if (FIX / "q_proj.json").exists() else export_weight("q_proj", FIX)
    N, K = q["shape_NK"]
    runs = []
    for label in ("cold", "warm"):
        if label == "cold":   # remove only markers we are about to create: run once, capture model_id, delete, rerun
            pass
        p = subprocess.run([str(BUILD / "probe"), "--k", str(K), "--n", str(N), "--calls", "200", "--out-dir", str(out)],
                           capture_output=True, text=True)
        rec = json.loads(p.stdout.strip().splitlines()[-1]); rec["stderr_tail"] = p.stderr[-2000:]; rec["exit"] = p.returncode; rec["label"] = label
        if label == "cold" and rec.get("model_id"):
            marker = MARKERS / f"{rec['model_id']}.ok"
            rec["marker_existed_before_this_run"] = None  # unknown for first run; cleared below for a true cold rerun
            if marker.exists():
                marker.unlink()
                p2 = subprocess.run([str(BUILD / "probe"), "--k", str(K), "--n", str(N), "--calls", "200", "--out-dir", str(out)], capture_output=True, text=True)
                rec = json.loads(p2.stdout.strip().splitlines()[-1]); rec["label"] = "cold"; rec["exit"] = p2.returncode; rec["marker_cleared"] = True
        runs.append(rec)
    manifest = {"experiment": "ane-r0-feasibility", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": host_manifest(),
                "shape_source": {"model_id": q["model_id"], "tensor": q["tensor"], "shape_NK": q["shape_NK"]},
                "note": "probe weights are seeded synthetic bf16 at the model-derived shape; real weights enter in R1"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.json").write_text(json.dumps({"runs": runs}, indent=2))
    cold, warm = runs[0], runs[1]
    lines = [f"# R0 native ANE feasibility ({manifest['started']})", "",
             f"Host: {manifest['host']['chip']}, macOS {manifest['host']['os']} ({manifest['host']['build']}), {manifest['host']['xcode']}, SDK {manifest['host']['sdk']}.",
             f"Runtime: ane.cpp `{manifest['host']['pin']['commit'][:12]}` core/ane_runtime.cpp only, built with Ninja + Xcode clang (Make generator rejected: colons in the workspace path break its rules).", "",
             f"Shape: q_proj of {q['model_id']} layer 0, K={K}, N={N}, S_alloc=32 (upstream IOSurface width floor).", "",
             "| run | ANE available | compile ms | compiled | cache loads | first call ms | call µs p50 | p95 | max | rel L2 vs fp32 | nonfinite | pass |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in runs:
        c = r.get("call_us", {})
        lines.append(f"| {r['label']} | {r['ane_available']} | {r.get('compile_ms', 0):.1f} | {r.get('compile_count_delta')} | {r.get('cache_loads_delta')} | "
                     f"{r.get('first_call_ms', 0):.2f} | {c.get('p50', 0):.0f} | {c.get('p95', 0):.0f} | {c.get('max', 0):.0f} | {r.get('rel_l2', 0):.2e} | {r.get('nonfinite')} | {r.get('pass')} |")
    lines += ["", "Placement evidence: the in-memory model path (`_ANEInMemoryModel compileWithQoS/loadWithQoS/evaluateWithQoS`) succeeded and the runtime's model attributes report Float16 planes of Width 32 (see `probe-run1.json` model_info). "
              "This is the private ANE runtime executing; per-operator hardware placement by power rail was not captured (needs root powermetrics).",
              "", "`graph.mil` is the MIL emitted by upstream `mil_gen_matmul` for this shape. Call time is host-observed wall time around the synchronous evaluate, including input pack and output unpack; it is not a hardware cycle count.",
              "", f"Exit: {'compile and eval path works' if all(r.get('pass') for r in runs) else 'FAILED, see summary.json'}."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print((out / "report.md").read_text())


# ---------------------------------------------------------------- R1
def run_bench(spec: dict, spec_path: Path) -> dict:
    spec_path.write_text(json.dumps(spec, ensure_ascii=False))
    p = subprocess.run([str(BUILD / "bench_main"), str(spec_path)], capture_output=True, text=True)
    if p.returncode not in (0, 5):
        raise RuntimeError(f"bench_main exit {p.returncode}: {p.stderr[-800:]}")
    rec = json.loads(p.stdout.strip().splitlines()[-1]); rec["exit"] = p.returncode; rec["stderr"] = p.stderr[-500:]
    return rec


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def r1(out: Path, quick: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"; work.mkdir(exist_ok=True)
    seeds = [1, 2, 3, 4, 5] if not quick else [1, 2]
    S_values = [1, 8, 32]
    shapes = {}
    for name in ("q_proj", "k_proj", "gate_proj"):
        rec = json.loads((FIX / f"{name}.json").read_text()) if (FIX / f"{name}.json").exists() else export_weight(name, FIX)
        shapes[name] = rec
    real = {}
    if (FIX / "real_activations.json").exists():
        xa = np.load(FIX / "real_x_attn_in.npy"); xm = np.load(FIX / "real_x_mlp_in.npy")
        real = {"q_proj": xa, "k_proj": xa, "gate_proj": xm}
        real_meta = json.loads((FIX / "real_activations.json").read_text())
        act_scale = float(real_meta["x_attn_in"]["rms"])   # random fixtures scaled to the real activation RMS
    else:
        real_meta = None; act_scale = 1.0
    numerics, timing, mil_texts = [], [], {}

    for name, rec in shapes.items():
        N, K = rec["shape_NK"]; W = np.load(FIX / f"{name}.f32.npy"); wpath = str(FIX / f"{name}.bf16.bin")
        for S in S_values:
            for seed in seeds:
                cases = R.fixtures(S, K, seed, act_scale)
                if name in real and seed == seeds[0]:
                    cases["real"] = np.ascontiguousarray(real[name][:S], dtype=np.float32)
                spec_cases = []
                for cname, X in cases.items():
                    xp = work / f"{name}_S{S}_s{seed}_{cname}.x.bin"; X.tofile(xp)
                    spec_cases.append({"name": cname, "x": str(xp), "y_out": str(work / f"{name}_S{S}_s{seed}_{cname}.PATH.y.bin")})
                for path in ("visible", "upstream"):
                    sc = [dict(c, y_out=c["y_out"].replace("PATH", path)) for c in spec_cases]
                    spec = {"path": path, "K": K, "N": N, "S": S, "weight_bf16": wpath, "cases": sc, "warmup": 3, "calls": 5,
                            "sentinel": 3.0, "measurements_out": "", "mil_out": str(out / f"graph.{path}.mil"), "summary_out": ""}
                    summ = run_bench(spec, work / "spec.json")
                    for c in sc:
                        X = cases[c["name"]]
                        Yfull = np.fromfile(c["y_out"], dtype=np.float32).reshape(R.S_ALLOC, N)
                        Y = Yfull[:S]; ref = R.ref_fp32(X, W); ref16 = R.ref_fp16_inputs(X, W)
                        e = R.errors(Y, ref); e16 = R.errors(Y, ref16)
                        zero_ref = c["name"] == "zeros"
                        ok, why = R.gate(e, zero_reference=zero_ref)
                        # padding lanes: every padded lane saw the sentinel vector; contamination check on valid rows is the gate above,
                        # and padded rows must equal W @ (sentinel * ones) within fp16 tolerance.
                        with np.errstate(all="ignore"):
                            pad_ref = (np.full(K, 3.0) @ W.T.astype(np.float64)) if S < R.S_ALLOC else None
                        pad_err = R.errors(Yfull[S:], np.tile(pad_ref, (R.S_ALLOC - S, 1))) if pad_ref is not None else None
                        numerics.append({"proj": name, "path": path, "S": S, "seed": seed, "case": c["name"], "K": K, "N": N,
                                         "vs_fp32": e, "vs_fp16_inputs": e16, "gate_pass": ok, "gate_reasons": why,
                                         "padded_lanes_vs_sentinel_ref": pad_err, "compile_us": summ["compile_us"], "cache_loaded": summ["cache_loaded"]})
        # timing windows: three per path, randomized order, >= 10 s each, on seed 1 random case
        S_t = 32 if not quick else 8
        cases = R.fixtures(S_t, K, 1, act_scale); xp = work / f"{name}_timing.x.bin"; cases["random"].tofile(xp)
        order = [("visible", w) for w in range(3)] + [("upstream", w) for w in range(3)]
        random.Random(7).shuffle(order)
        calls = 200 if quick else 0   # 0 -> computed from a pilot to reach >= 10 s
        pilot = {}
        for path in ("visible", "upstream"):
            spec = {"path": path, "K": K, "N": N, "S": S_t, "weight_bf16": wpath, "cases": [{"name": "t", "x": str(xp), "y_out": ""}],
                    "warmup": 20, "calls": 200, "sentinel": 0.0, "measurements_out": str(work / "pilot.jsonl"), "mil_out": "", "summary_out": ""}
            run_bench(spec, work / "spec.json")
            us = [json.loads(l)["call_us"] for l in (work / "pilot.jsonl").read_text().splitlines()]
            pilot[path] = float(np.median(us))
        for path, w in order:
            n_calls = calls or max(500, int(10e6 / pilot[path]))
            mpath = work / f"{name}_{path}_w{w}.jsonl"
            spec = {"path": path, "K": K, "N": N, "S": S_t, "weight_bf16": wpath, "cases": [{"name": "t", "x": str(xp), "y_out": ""}],
                    "warmup": 50, "calls": n_calls, "sentinel": 0.0, "measurements_out": str(mpath), "mil_out": "", "summary_out": ""}
            summ = run_bench(spec, work / "spec.json")
            rows = [json.loads(l) for l in mpath.read_text().splitlines()]
            rec_t = {"proj": name, "path": path, "window": w, "S": S_t, "K": K, "N": N, "calls": len(rows), "failures": summ["failures"],
                     "checksum_drift": summ["checksum_drift"], "compile_us": summ["compile_us"], "cache_loaded": summ["cache_loaded"],
                     "first_call_us": summ["first_call_us"], "weight_bytes": summ["index_map"]["weight_bytes"],
                     "window_s": sum(r["call_us"] for r in rows) / 1e6}
            for ph in ("pack_us", "handoff_us", "eval_us", "read_us", "transpose_us", "unpack_us", "call_us"):
                v = [r[ph] for r in rows]; rec_t[ph] = {"p50": pct(v, 50), "p95": pct(v, 95), "min": min(v), "max": max(v)}
            timing.append(rec_t)
        # stability: 1000 calls, checksum drift
        spec = {"path": "visible", "K": K, "N": N, "S": S_t, "weight_bf16": wpath, "cases": [{"name": "t", "x": str(xp), "y_out": ""}],
                "warmup": 5, "calls": 1000, "sentinel": 0.0, "measurements_out": "", "mil_out": "", "summary_out": ""}
        summ = run_bench(spec, work / "spec.json")
        timing.append({"proj": name, "path": "visible", "window": "stability", "S": S_t, "K": K, "N": N, "calls": 1000,
                       "failures": summ["failures"], "checksum_drift": summ["checksum_drift"]})
        # upstream_matvec S=1 full-call reference and MLX fp16 GPU reference for the same S values
        x1 = work / f"{name}_S1.x.bin"; R.fixtures(1, K, 1, act_scale)["random"].tofile(x1)
        spec = {"path": "upstream_matvec", "K": K, "N": N, "S": 1, "weight_bf16": wpath, "cases": [{"name": "t", "x": str(x1), "y_out": ""}],
                "warmup": 50, "calls": 2000, "sentinel": 0.0, "measurements_out": str(work / "mv.jsonl"), "mil_out": "", "summary_out": ""}
        summ = run_bench(spec, work / "spec.json")
        v = [json.loads(l)["call_us"] for l in (work / "mv.jsonl").read_text().splitlines()]
        timing.append({"proj": name, "path": "upstream_matvec", "window": "S1", "S": 1, "K": K, "N": N, "calls": len(v),
                       "failures": summ["failures"], "call_us": {"p50": pct(v, 50), "p95": pct(v, 95), "min": min(v), "max": max(v)}})
        for S in S_values:
            X = R.fixtures(S, K, 1, act_scale)["random"]
            _, t = R.ref_mlx_fp16(X, W, repeats=300)
            timing.append({"proj": name, "path": "mlx_fp16_gpu", "window": "ref", "S": S, "K": K, "N": N, "calls": 300,
                           "call_us": {"p50": pct(t["call_us"], 50), "p95": pct(t["call_us"], 95), "min": min(t["call_us"]), "max": max(t["call_us"])}})
    for path in ("visible", "upstream"):
        mil_texts[path] = (out / f"graph.{path}.mil").read_text() if (out / f"graph.{path}.mil").exists() else ""
    mil_identical = mil_texts["visible"] == mil_texts["upstream"]

    fit = cost_fit(timing)

    with (out / "measurements.jsonl").open("w") as f:
        for r in numerics: f.write(json.dumps({"kind": "numerics", **r}) + "\n")
        for r in timing: f.write(json.dumps({"kind": "timing", **r}) + "\n")
    gates_failed = [r for r in numerics if not r["gate_pass"] and r["case"] != "outlier"]
    outlier_nonfinite = [r for r in numerics if r["case"] == "outlier" and r["vs_fp32"]["nonfinite"]]
    summary = {"experiment": "ane-r1-projection", "seeds": seeds, "S_values": S_values, "shapes": {k: v["shape_NK"] for k, v in shapes.items()},
               "gates": R.GATES["projection"], "numerics_cases": len(numerics), "gate_failures_excluding_outlier": len(gates_failed),
               "outlier_cases_with_nonfinite": len(outlier_nonfinite), "mil_identical_visible_vs_upstream": mil_identical,
               "real_activation_case": (f"included: layer-0 activations captured with MLX from {real_meta['model']} on a {real_meta['S']}-token prompt "
                                        f"(x_attn_in rms {real_meta['x_attn_in']['rms']:.3f}, abs max {real_meta['x_attn_in']['abs_max']:.2f}); random fixtures scaled to rms {act_scale:.3f}")
                                       if real_meta else "deferred",
               "cost_fit": fit, "timing": timing}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    manifest = {"experiment": "ane-r1-projection", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": host_manifest(),
                "weights": shapes, "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIX.glob("*.bf16.bin")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "report.md").write_text(r1_report(summary, numerics, manifest))
    print((out / "report.md").read_text())


def cost_fit(timing):
    """Local linear fit of ANE evaluate p50 (µs) vs weight bytes over the measured shapes (visible path,
    numbered windows). Evaluate time, not total call time: pack/read/transpose are host-side costs."""
    pts = {}
    for t in timing:
        if t["path"] == "visible" and isinstance(t.get("window"), int) and isinstance(t.get("eval_us"), dict):
            pts.setdefault(t["weight_bytes"], []).append(t["eval_us"]["p50"])
    if len(pts) < 2:
        return None
    xs = np.array(sorted(pts), dtype=float); ys = np.array([np.mean(pts[x]) for x in xs])
    slope, icpt = np.polyfit(xs, ys, 1)
    return {"points_eval_p50_us": {int(x): float(y) for x, y in zip(xs, ys)}, "fixed_us": float(icpt), "us_per_MB": float(slope * 1e6),
            "implied_GBps": float(1e6 / slope / 1e9) if slope > 0 else None,
            "note": "three shapes (k_proj 2.1 MB, q_proj 4.2 MB, gate_proj 6.3 MB), S=32, evaluate time only; a local fit, not a device constant"}


def rerender(out: Path) -> None:
    rows = [json.loads(l) for l in (out / "measurements.jsonl").read_text().splitlines()]
    numerics = [{k: v for k, v in r.items() if k != "kind"} for r in rows if r["kind"] == "numerics"]
    timing = [{k: v for k, v in r.items() if k != "kind"} for r in rows if r["kind"] == "timing"]
    s = json.loads((out / "summary.json").read_text()); s["cost_fit"] = cost_fit(timing); s["timing"] = timing
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    m = json.loads((out / "manifest.json").read_text())
    (out / "report.md").write_text(r1_report(s, numerics, m)); print((out / "report.md").read_text())


def r1_report(s, numerics, m) -> str:
    L = [f"# R1 projection lowering ({m['started']})", "",
         f"Host: {m['host']['chip']}, macOS {m['host']['os']} ({m['host']['build']}); ane.cpp `{m['host']['pin']['commit'][:12]}`.",
         f"Weights: real BF16 tensors from {list(m['weights'].values())[0]['model_id']} layer 0: " + ", ".join(f"`{k}` {v}" for k, v in s['shapes'].items()) + " ([N,K]; full sha256 in manifest).",
         f"Visible-path MIL byte-identical to upstream `mil_gen_matmul`: **{s['mil_identical_visible_vs_upstream']}**.", "",
         "## Numerics (gates frozen: rel L2 <= 0.002, normalized max <= 0.02, zero-ref abs <= 0.001)", "",
         f"{s['numerics_cases']} case evaluations over seeds {s['seeds']}, S in {s['S_values']}, both paths, all shapes. "
         f"Gate failures excluding the outlier probe: **{s['gate_failures_excluding_outlier']}**. Outlier cases with nonfinite output: {s['outlier_cases_with_nonfinite']}.", ""]
    L += ["| proj | path | case | worst rel L2 (fp32 ref) | worst norm max | worst rel L2 (fp16-input ref) | nonfinite | pad-lane worst rel L2 |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    import itertools
    key = lambda r: (r["proj"], r["path"], r["case"])
    for k, grp in itertools.groupby(sorted(numerics, key=key), key=key):
        g = list(grp)
        w = lambda f: max((f(r) for r in g if f(r) is not None), default=None)
        fmt = lambda v: "n/a" if v is None else f"{v:.2e}"
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(w(lambda r: r['vs_fp32']['rel_l2']))} | {fmt(w(lambda r: r['vs_fp32']['norm_max_err']))} | "
                 f"{fmt(w(lambda r: r['vs_fp16_inputs']['rel_l2']))} | {sum(r['vs_fp32']['nonfinite'] for r in g)} | "
                 f"{fmt(w(lambda r: (r['padded_lanes_vs_sentinel_ref'] or {}).get('rel_l2')))} |")
    L += ["", "## Timing (host wall time, µs; evaluate is synchronous so device sync is inside eval; read = IOSurface lock + fp16→fp32 of the whole output surface; transpose = host [N,32]→[32,N])", "",
          "| proj | path | window | S | calls | fail | drift | compile µs | cache | pack p50 | handoff p50 | eval p50 | read p50 | transpose p50 | call p50 | call p95 |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in s["timing"]:
        def g(k):
            v = t.get(k)
            return f"{v['p50']:.0f}" if isinstance(v, dict) and v.get("p50") is not None else "—"
        p95 = f"{t['call_us']['p95']:.0f}" if isinstance(t.get("call_us"), dict) else "—"
        comp = f"{t['compile_us']:.0f}" if t.get("compile_us") else "—"
        L.append(f"| {t['proj']} | {t['path']} | {t['window']} | {t['S']} | {t.get('calls', '')} | {t.get('failures', '')} | "
                 f"{t.get('checksum_drift', '')} | {comp} | {t.get('cache_loaded', '')} | {g('pack_us')} | {g('handoff_us')} | "
                 f"{g('eval_us')} | {g('read_us')} | {g('transpose_us')} | {g('call_us')} | {p95} |")
    if s["cost_fit"]:
        f = s["cost_fit"]
        pts = ", ".join(f"{b/1e6:.1f} MB → {u:.0f} µs" for b, u in sorted((int(k), v) for k, v in f["points_eval_p50_us"].items()))
        L += ["", f"Evaluate p50 by weight size (mean of three windows): {pts}."]
        L += ["", f"Cost fit (visible path, ANE evaluate p50 µs vs weight bytes): fixed {f['fixed_us']:.0f} µs + {f['us_per_MB']:.1f} µs/MB "
              f"(≈ {f['implied_GBps']:.1f} GB/s effective weight streaming). {f['note']}."]
    L += ["", f"Real activation case: {s['real_activation_case']}.",
          "", "Not measured: energy and per-operator hardware placement (root powermetrics not run). MLX rows are same-shape fp16 GPU matmuls for scale, not a same-runtime comparison."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r0", action="store_true"); ap.add_argument("--r1", action="store_true"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--rerender", action="store_true")
    a = ap.parse_args()
    if a.r0: r0(ROOT / "results" / "ane-r0-feasibility")
    if a.rerender: rerender(ROOT / "results" / "ane-r1-projection"); return
    if a.r1: r1(ROOT / "results" / ("ane-r1-projection-quick" if a.quick else "ane-r1-projection"), a.quick)


if __name__ == "__main__":
    main()
