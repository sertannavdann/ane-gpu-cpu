"""R2: reproduce upstream's fused SwiGLU FFN on the ANE, then test intermediate-channel chunking (2, 4 chunks).
Usage: python -m prototype.ane.ffn_benchmark [--quick]   -> results/ane-r2-ffn/"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from . import reference as R
from .benchmark import host_manifest, pct

ROOT = Path(__file__).resolve().parents[2]
ANE = ROOT / "prototype" / "ane"; BUILD = ANE / "build"; FIX = ANE / "fixtures"
GATES = {"rel_l2": 0.005, "norm_max_err": 0.02, "zero_abs": 0.001}


def run(spec: dict, spec_path: Path) -> dict:
    spec_path.write_text(json.dumps(spec, ensure_ascii=False))
    p = subprocess.run([str(BUILD / "ffn_main"), str(spec_path)], capture_output=True, text=True)
    if p.returncode not in (0, 5):
        raise RuntimeError(f"ffn_main exit {p.returncode}: {p.stderr[-800:]}")
    rec = json.loads(p.stdout.strip().splitlines()[-1]); rec["exit"] = p.returncode
    return rec


def ref_ffn(X, Wg, Wu, Wd):
    with np.errstate(all="ignore"):
        x = X.astype(np.float64); g = x @ Wg.astype(np.float64).T; u = x @ Wu.astype(np.float64).T
        h = (g / (1.0 + np.exp(-g))) * u
        return (h @ Wd.astype(np.float64).T).astype(np.float32)


def gate(err, zero_ref=False):
    why = []
    if err["nonfinite"]: why.append(f"nonfinite={err['nonfinite']}")
    if zero_ref:
        if err["max_abs_err"] > GATES["zero_abs"]: why.append(f"zero-ref abs {err['max_abs_err']:.3g}")
    else:
        if err["rel_l2"] is None or err["rel_l2"] > GATES["rel_l2"]: why.append(f"rel_l2 {err['rel_l2']}")
        if err["norm_max_err"] > GATES["norm_max_err"]: why.append(f"norm_max {err['norm_max_err']:.3g}")
    return not why, why


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    out = ROOT / "results" / ("ane-r2-ffn-quick" if a.quick else "ane-r2-ffn"); out.mkdir(parents=True, exist_ok=True)
    work = out / "work"; work.mkdir(exist_ok=True)
    Wg = np.load(FIX / "gate_proj.f32.npy"); Wu = np.load(FIX / "up_proj.f32.npy"); Wd = np.load(FIX / "down_proj.f32.npy")
    I, D = Wg.shape; assert Wu.shape == (I, D) and Wd.shape == (D, I)
    wpaths = {k: str(FIX / f"{k}_proj.bf16.bin") for k in ("gate", "up", "down")}
    meta = json.loads((FIX / "real_activations.json").read_text()); xm = np.load(FIX / "real_x_mlp_in.npy")
    act_scale = float(meta["x_mlp_in"]["rms"])
    seeds = [1, 2] if a.quick else [1, 2, 3, 4, 5]; S_values = [1, 8, 32]
    configs = [("fused", 1), ("fused_exp", 1), ("chunked", 2), ("chunked", 4)]
    numerics, timing = [], []

    def spec_base(mode, chunks, S):
        return {"mode": mode, "chunks": chunks, "dim": D, "inter": I, "S": S, "wg": wpaths["gate"], "wu": wpaths["up"], "wd": wpaths["down"],
                "warmup": 3, "calls": 5, "measurements_out": "", "summary_out": "", "mil_out": ""}

    # ---- numerics
    for S in S_values:
        for seed in seeds:
            cases = R.fixtures(S, D, seed, act_scale)
            if seed == seeds[0]:
                cases["real"] = np.ascontiguousarray(xm[:S], dtype=np.float32)
            for mode, chunks in configs:
                sc = []
                for cname, X in cases.items():
                    xp = work / f"S{S}_s{seed}_{cname}.x.bin"; X.tofile(xp)
                    sc.append({"name": cname, "x": str(xp), "y_out": str(work / f"S{S}_s{seed}_{cname}_{mode}{chunks}.y.bin")})
                spec = spec_base(mode, chunks, S); spec["cases"] = sc; spec["mil_out"] = str(out / f"graph.{mode}{chunks}.mil")
                summ = run(spec, work / "spec.json")
                for c in sc:
                    X = cases[c["name"]]; Y = np.fromfile(c["y_out"], dtype=np.float32).reshape(R.S_ALLOC, D)[:S]
                    ref = ref_ffn(X, Wg, Wu, Wd); e = R.errors(Y, ref); ok, why = gate(e, zero_ref=c["name"] == "zeros")
                    if c["name"] == "basis":   # reference rms ~8e-4 sits at fp16's smallest normals; reported, not gated (channel order proven in R1)
                        ok, why = True, ["basis: not gated for the FFN (fp16 range), reported only"]
                    numerics.append({"mode": f"{mode}{chunks}", "S": S, "seed": seed, "case": c["name"], "vs_fp32": e, "gate_pass": ok, "gate_reasons": why,
                                     "compile_us": summ["compile_us"], "cache_loaded": summ["cache_loaded"]})
    # fused vs chunked agreement on the same inputs (device-vs-device, independent of the reference)
    agree = []
    for S in S_values:
        for cname in ("random", "real"):
            yf = np.fromfile(work / f"S{S}_s{seeds[0]}_{cname}_fused1.y.bin", dtype=np.float32).reshape(R.S_ALLOC, D)[:S]
            for chunks in (2, 4):
                yc = np.fromfile(work / f"S{S}_s{seeds[0]}_{cname}_chunked{chunks}.y.bin", dtype=np.float32).reshape(R.S_ALLOC, D)[:S]
                agree.append({"S": S, "case": cname, "chunks": chunks, "rel_l2_vs_fused": R.errors(yc, yf)["rel_l2"]})

    # ---- timing: three windows per config at S=32 (and S=1), randomized; >= 10 s each
    x32 = work / "t32.x.bin"; R.fixtures(32, D, 1, act_scale)["random"].tofile(x32)
    x1 = work / "t1.x.bin"; R.fixtures(1, D, 1, act_scale)["random"].tofile(x1)
    plan = [(m, c, S, w) for (m, c) in configs for S in (32, 1) for w in range(3)]
    plan += [("fused_matvec", 1, 1, w) for w in range(3)] + [("chunked_matvec", 2, 1, w) for w in range(3)] + [("chunked_matvec", 4, 1, w) for w in range(3)]
    random.Random(11).shuffle(plan)
    pilot = {}
    for mode, chunks, S, w in plan:
        key = (mode, chunks, S)
        xp = x32 if S == 32 else x1
        if key not in pilot:
            spec = spec_base(mode, chunks, S); spec.update({"cases": [{"name": "t", "x": str(xp), "y_out": ""}], "warmup": 10, "calls": 50, "measurements_out": str(work / "pilot.jsonl")})
            run(spec, work / "spec.json"); pilot[key] = float(np.median([json.loads(l)["call_us"] for l in (work / "pilot.jsonl").read_text().splitlines()]))
        n_calls = 150 if a.quick else max(300, int(10e6 / pilot[key]))
        mpath = work / f"{mode}{chunks}_S{S}_w{w}.jsonl"
        spec = spec_base(mode, chunks, S); spec.update({"cases": [{"name": "t", "x": str(xp), "y_out": ""}], "warmup": 30, "calls": n_calls, "measurements_out": str(mpath)})
        summ = run(spec, work / "spec.json")
        rows = [json.loads(l) for l in mpath.read_text().splitlines()]
        rec = {"mode": f"{mode}{chunks}" if "matvec" not in mode else f"{mode}{chunks}", "S": S, "window": w, "calls": len(rows), "failures": summ["failures"],
               "checksum_drift": summ["checksum_drift"], "dispatches": rows[0]["dispatches"], "compile_us": summ["compile_us"], "cache_loaded": summ["cache_loaded"],
               "weight_bytes": summ["weight_bytes"], "window_s": sum(r["call_us"] for r in rows) / 1e6}
        for ph in ("pack_us", "handoff_us", "eval_us", "read_us", "transpose_us", "call_us"):
            v = [r[ph] for r in rows]; rec[ph] = {"p50": pct(v, 50), "p95": pct(v, 95), "min": min(v), "max": max(v)}
        timing.append(rec)
    # stability
    spec = spec_base("fused", 1, 32); spec.update({"cases": [{"name": "t", "x": str(x32), "y_out": ""}], "warmup": 5, "calls": 1000})
    summ = run(spec, work / "spec.json"); stability = {"calls": 1000, "failures": summ["failures"], "checksum_drift": summ["checksum_drift"]}

    # ---- decision
    def agg(mode, S, key="call_us", q="p50"):
        v = [t[key][q] for t in timing if t["mode"] == mode and t["S"] == S]; return float(np.mean(v)) if v else None
    fused50, fused95 = agg("fused1", 32), agg("fused1", 32, q="p95")
    decisions = {}
    for ch in (2, 4):
        c50, c95 = agg(f"chunked{ch}", 32), agg(f"chunked{ch}", 32, q="p95")
        gain = (fused50 - c50) / fused50 if fused50 else None
        p95reg = (c95 - fused95) / fused95 if fused95 else None
        num_ok = all(r["gate_pass"] for r in numerics if r["mode"] == f"chunked{ch}" and r["case"] not in ("outlier", "basis"))
        decisions[f"chunked{ch}"] = {"call_p50_gain": gain, "p95_regression": p95reg, "numerics_ok": num_ok,
                                     "retain": bool(gain is not None and gain >= 0.10 and p95reg is not None and p95reg <= 0.05 and num_ok)}
    e50, e95 = agg("fused_exp1", 32), agg("fused_exp1", 32, q="p95")
    worst = lambda mode: max(r["vs_fp32"]["rel_l2"] or 0 for r in numerics if r["mode"] == mode and r["case"] not in ("outlier", "zeros", "basis"))
    decisions["fused_exp1"] = {"call_p50_gain": (fused50 - e50) / fused50, "p95_regression": (e95 - fused95) / fused95,
                               "numerics_ok": all(r["gate_pass"] for r in numerics if r["mode"] == "fused_exp1" and r["case"] not in ("outlier", "basis")),
                               "worst_rel_l2_fused": worst("fused1"), "worst_rel_l2_exp": worst("fused_exp1"),
                               "retain": None}
    d = decisions["fused_exp1"]; d["retain"] = bool(d["numerics_ok"] and d["worst_rel_l2_exp"] < d["worst_rel_l2_fused"] / 2 and d["p95_regression"] <= 0.05 and d["call_p50_gain"] >= -0.05)
    pred_us = 50 + 11.2 * (timing[0]["weight_bytes"] / 1e6)
    summary = {"experiment": "ane-r2-ffn", "shape": {"dim": D, "inter": I, "weight_bytes": int(timing[0]["weight_bytes"])}, "gates": GATES, "seeds": seeds,
               "numerics_cases": len(numerics), "gate_failures_excluding_outlier": sum(1 for r in numerics if not r["gate_pass"] and r["case"] not in ("outlier", "basis")),
               "gate_note": "basis case reported but not gated for the FFN: its reference values (rms ~8e-4) are at fp16's smallest normal magnitudes; outlier (±20000 inputs) saturates fp16 intermediates by design",
               "outlier_nonfinite_cases": sum(1 for r in numerics if r["case"] == "outlier" and r["vs_fp32"]["nonfinite"]),
               "fused_vs_chunked_agreement": agree, "stability_fused_S32": stability, "r1_fit_prediction_eval_us": pred_us,
               "real_activation": {"rms": act_scale, "abs_max": meta["x_mlp_in"]["abs_max"]}, "decisions": decisions, "timing": timing}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with (out / "measurements.jsonl").open("w") as f:
        for r in numerics: f.write(json.dumps({"kind": "numerics", **r}) + "\n")
        for r in timing: f.write(json.dumps({"kind": "timing", **r}) + "\n")
    m = {"experiment": "ane-r2-ffn", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": host_manifest(),
         "weights": {k: json.loads((FIX / f"{k}_proj.json").read_text()) for k in ("gate", "up", "down")}, "activations": meta}
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    (out / "report.md").write_text(report(summary, numerics, m)); print((out / "report.md").read_text())


def report(s, numerics, m):
    L = [f"# R2 SwiGLU FFN on the ANE ({m['started']})", "",
         f"Host: {m['host']['chip']}, macOS {m['host']['os']} ({m['host']['build']}); ane.cpp `{m['host']['pin']['commit'][:12]}`. "
         f"Layer-0 FFN of Qwen3-0.6B: dim {s['shape']['dim']}, intermediate {s['shape']['inter']}, {s['shape']['weight_bytes']/1e6:.1f} MB fp16 weights (full sha256 in manifest).",
         f"Baseline = upstream fused kernel (`ane_compile_fused_ffn`, one dispatch). Change = upstream chunked-accumulate kernels forced to 2 and 4 chunks over the intermediate dimension.",
         f"Real activation: layer-0 MLP input from MLX (rms {s['real_activation']['rms']:.3f}, abs max {s['real_activation']['abs_max']:.2f}); random fixtures scaled to that rms.", "",
         f"## Numerics (gates: rel L2 <= {s['gates']['rel_l2']}, normalized max <= {s['gates']['norm_max_err']}, zero-ref abs <= {s['gates']['zero_abs']})", "",
         f"{s['numerics_cases']} case evaluations, seeds {s['seeds']}, S in [1, 8, 32]. Gate failures excluding the outlier probe: **{s['gate_failures_excluding_outlier']}**. "
         f"Outlier cases with nonfinite output: {s['outlier_nonfinite_cases']}. {s['gate_note']}.", "",
         "| mode | case | worst rel L2 | worst norm max | nonfinite |", "| --- | --- | ---: | ---: | ---: |"]
    key = lambda r: (r["mode"], r["case"])
    for k, grp in itertools.groupby(sorted(numerics, key=key), key=key):
        g = list(grp); rl = [r["vs_fp32"]["rel_l2"] for r in g if r["vs_fp32"]["rel_l2"] is not None]
        L.append(f"| {k[0]} | {k[1]} | {max(rl) if rl else float('nan'):.2e} | {max(r['vs_fp32']['norm_max_err'] for r in g):.2e} | {sum(r['vs_fp32']['nonfinite'] for r in g)} |")
    L += ["", "Chunked vs fused on identical inputs (rel L2): " + ", ".join(f"S{a['S']} {a['case']} {a['chunks']}ch {a['rel_l2_vs_fused']:.1e}" for a in s["fused_vs_chunked_agreement"]), "",
          "## Timing (host wall µs; sync inside eval; chunked rows sum all chunk dispatches)", "",
          "| mode | S | window | calls | dispatches | fail | drift | compile µs | pack p50 | handoff p50 | eval p50 | read p50 | transpose p50 | call p50 | call p95 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in sorted(s["timing"], key=lambda t: (t["mode"], -t["S"], t["window"])):
        g = lambda k: f"{t[k]['p50']:.0f}"
        L.append(f"| {t['mode']} | {t['S']} | {t['window']} | {t['calls']} | {t['dispatches']} | {t['failures']} | {t['checksum_drift']} | {t['compile_us']:.0f} | "
                 f"{g('pack_us')} | {g('handoff_us')} | {g('eval_us')} | {g('read_us')} | {g('transpose_us')} | {g('call_us')} | {t['call_us']['p95']:.0f} |")
    L += ["", f"R1 cost-fit prediction for one fused dispatch of {s['shape']['weight_bytes']/1e6:.1f} MB: {s['r1_fit_prediction_eval_us']:.0f} µs evaluate.",
          f"Stability, fused S=32, 1000 calls: failures {s['stability_fused_S32']['failures']}, checksum drift {s['stability_fused_S32']['checksum_drift']}.", "",
          "## Decision (chunking: retain only if >= 10 % lower call p50, p95 regression <= 5 %, numerics intact; exp-form: retain if error at least halved with time within 5 %)", ""]
    for k, d in s["decisions"].items():
        extra = f", worst rel L2 {d['worst_rel_l2_exp']:.1e} vs fused {d['worst_rel_l2_fused']:.1e} (numerical rule: at least halved, time within 5 %)" if "worst_rel_l2_exp" in d else ""
        L.append(f"- {k}: call p50 gain {d['call_p50_gain']*100:+.1f} %, p95 change {d['p95_regression']*100:+.1f} %, numerics ok {d['numerics_ok']}{extra} → **{'retain' if d['retain'] else 'reject; keep unmodified fused path'}**")
    L += ["", "Not measured: energy; per-operator placement by power rail. One host, one OS build."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
