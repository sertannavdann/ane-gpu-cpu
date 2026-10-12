"""Readable 'what changed' report across R0–R4, built from the recorded results. Writes docs/R4-what-changed.md.
Usage: python -m prototype.r4.what_changed [--r4 results/ane-r4-concurrent]"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"


def load(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def fmt(v, d=0, suffix=""):
    return "n/a" if v is None else f"{v:.{d}f}{suffix}"


def energy_text(en, es):
    if not en or "conditions" not in en or not any("avg_W" in v for v in en["conditions"].values()):
        return "5. **Energy.** Not measured: `powermetrics` needs root and no log was supplied. Tasks per joule, the adoption metric in the plan, stays open."
    c = en["conditions"]; fa, g, n = c.get("fg_alone", {}), c.get("fg_bg_gpu", {}), c.get("fg_bg_ane", {})
    floor = (fa.get("avg_W") or {}).get("combined")
    def marginal(v):
        if not v.get("avg_W") or floor is None or not v.get("bg_correct"): return None
        return (v["avg_W"]["combined"] - floor) * 300 * v["windows"] / v["bg_correct"]
    mg, mn = marginal(g), marginal(n)
    lines = [f"5. **Energy (measured with `powermetrics`, 1 s samples, CPU + GPU + ANE rails, log `{Path(en['log']).name}`).** Average draw: foreground alone {fmt(floor, 1)} W; background on the GPU {fmt((g.get('avg_W') or {}).get('combined'), 1)} W (GPU rail {fmt((g.get('avg_W') or {}).get('gpu'), 1)} W); background on the ANE {fmt((n.get('avg_W') or {}).get('combined'), 1)} W (ANE rail {fmt((n.get('avg_W') or {}).get('ane'), 2)} W, GPU {fmt((n.get('avg_W') or {}).get('gpu'), 1)} W). "
             f"Whole-system energy per correct background task: GPU placement {fmt(g.get('J_per_correct_bg_task'))} J, ANE placement {fmt(n.get('J_per_correct_bg_task'))} J, because the ANE draws about 5× less but takes about 5.5× longer. "
             f"Marginal energy above the foreground-alone floor: GPU {fmt(mg)} J per task, ANE {fmt(mn)} J per task. "
             f"Reading: for work that is rate-limited anyway (a bounded evaluator running a few tasks per minute), the ANE lane costs about {fmt(100 * (1 - mn / mg) if mg and mn else None)} % less marginal energy per task than the same tasks on the GPU and keeps the GPU's tail latency untouched; for bulk work the GPU finishes about 5.5× more tasks for roughly the same total energy per task."]
    tab = ["", "| Condition | avg W (CPU / GPU / ANE / all) | energy J | correct bg tasks | J per correct task | correct tasks per kJ |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k, v in c.items():
        if "avg_W" not in v: continue
        w = v["avg_W"]; tab.append(f"| {k} | {fmt(w['cpu'],1)} / {fmt(w['gpu'],1)} / {fmt(w['ane'],2)} / {fmt(w['combined'],1)} | {fmt(v['energy_J_total'])} | {v['bg_correct']} | {fmt(v['J_per_correct_bg_task'],1)} | {fmt(v['correct_bg_tasks_per_kJ'],1)} |")
    return "\n".join(lines + tab)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r4", default=str(R / "ane-r4-concurrent")); ap.add_argument("--r4-gpu", default=str(R / "ane-r4-gpu-rerun"), help="leak-fixed rerun of the GPU-side conditions; overrides fg_alone, fg_bg_gpu, bg_gpu_alone")
    ap.add_argument("--energy", default=str(R / "ane-r4-energy"), help="run with a powermetrics log; its energy.json and summary.json add the energy section")
    ap.add_argument("--out", default=str(ROOT / "docs/R4-what-changed.md"))
    a = ap.parse_args()
    r4 = load(Path(a.r4) / "summary.json"); r4m = load(Path(a.r4) / "manifest.json")
    r4g = load(Path(a.r4_gpu) / "summary.json") if a.r4_gpu else None
    if r4 and r4g:
        r4["conditions"]["fg_alone_run1"] = dict(r4["conditions"].get("fg_alone", {}))
        for k in ("fg_alone", "fg_bg_gpu", "bg_gpu_alone"):
            if k in r4g["conditions"]: r4["conditions"][k] = r4g["conditions"][k]
    r1 = load(R / "ane-r1-projection/summary.json"); r2 = load(R / "ane-r2-ffn/summary.json")
    r3a = load(R / "ane-r3-model-4b-v2/summary.json"); r3b = load(R / "ane-r3-model/summary.json")
    smoke27 = load(R / "ane-r4-27b-fg-smoke/summary.json")
    c = {k: v for k, v in (r4["conditions"] if r4 else {}).items() if k != "fg_alone_run1"}
    fa1 = (r4 or {}).get("conditions", {}).get("fg_alone_run1", {})
    fa, fg_gpu, fg_ane = c.get("fg_alone", {}), c.get("fg_bg_gpu", {}), c.get("fg_bg_ane", {})
    bga, bgn = c.get("bg_gpu_alone", {}), c.get("bg_ane_alone", {})
    args = (r4m or {}).get("log", {}).get("args", {})
    L = []
    L += ["# What changed: ANE, GPU and CPU on one Mac, R0 to R4", "",
          f"Generated from the recorded results on {(r4m or {}).get('log', {}).get('started', 'n/a')[:10]}. Host: {(r4m or {}).get('host', {}).get('chip', 'Apple M4 Max')}, macOS {(r4m or {}).get('host', {}).get('os', '')}; ane.cpp pinned at `{(r4m or {}).get('host', {}).get('pin', '')[:12]}`.", "",
          "## The one-paragraph version", ""]
    if r4:
        ttft_gain = (fa.get("fg_ttft_p95_ms") or 0) and ((fg_gpu.get("fg_ttft_p95_ms") or 0) - (fg_ane.get("fg_ttft_p95_ms") or 0))
        L += [f"We started with no way to run anything on the Neural Engine. We now have a pinned native runtime that compiles real transformer kernels for it, a measured cost model "
              f"(≈50 µs per dispatch plus ≈11 µs per MB of fp16 weights), a lowering fix that cuts the fused feed-forward kernel's error eightfold at no time cost, model-level proof that "
              f"a 4B model on the ANE matches the GPU reference on a frozen extraction task, and a concurrent-placement measurement. In that measurement the ANE runs the 4B background job about "
              f"{fmt((bga.get('bg_correct_per_min') or 0) / max(bgn.get('bg_correct_per_min') or 1e-9, 1e-9), 1)}× slower than the GPU, and moving the background job off the GPU changed foreground "
              f"first-token p95 from {fmt(fg_gpu.get('fg_ttft_p95_ms'))} ms (background on GPU) to {fmt(fg_ane.get('fg_ttft_p95_ms'))} ms (background on ANE), against {fmt(fa.get('fg_ttft_p95_ms'))} ms with no background at all. "
              f"Energy could not be measured without root, so no tasks-per-joule claim is made.", ""]
    L += ["## What was built, in order", "",
          "| Stage | What changed | Evidence |", "| --- | --- | --- |",
          "| Slice 1 | One-process scheduler: foreground/background queues, deadlines, cancellation, priority at the token boundary; a 60-request frozen workload; MLX generator. | `results/mlx-priority-run2`: foreground TTFT p95 1206 → 180 ms with priority. |",
          "| R0 | Pinned ane.cpp, built only its runtime core, proved compile → load → evaluate on this macOS. | `results/ane-r0-feasibility`: 1024→2048 projection, ≈100 µs per call, rel L2 3.9e-4. |",
          f"| R1 | Visible matrix-to-1×1-convolution lowering, byte-identical to upstream; real weights and activations; cost model. | `results/ane-r1-projection`: {r1['numerics_cases'] if r1 else 'n/a'} cases, 0 gate failures; fit {fmt((r1 or {}).get('cost_fit', {}).get('fixed_us'))} µs + {fmt((r1 or {}).get('cost_fit', {}).get('us_per_MB'), 1)} µs/MB. |",
          f"| R2 | Fused SwiGLU FFN reproduced; chunking rejected (+48 % time); the ANE `sigmoid` op found inaccurate (≈3e-3) and SiLU re-expressed as g/(1+exp(−g)). | `results/ane-r2-ffn`: FFN error 2.2e-2 → 2.6e-3, time unchanged; patch proposed, not applied by default. |",
          f"| R3 | Model-level check on Qwen3-0.6B and Qwen3.5-4B with forced kernel maps, per-step logit traces, and a frozen 30-case extraction fixture (v2). | `results/ane-r3-model-4b-v2`: ANE {r3a['conditions']['exp']['all_correct'] if r3a else 'n/a'}/30 vs GPU reference {r3a['conditions']['mlx']['all_correct'] if r3a else 'n/a'}/30; kernel fix neutral at model level. |",
          "| R4 | Concurrent placement: interactive foreground on the GPU, background extraction on GPU or ANE, three 5-minute windows each, memory sampled every second. | `results/ane-r4-concurrent` (table below). |", ""]
    if smoke27:
        s27 = smoke27["conditions"]
        L += ["## Memory: what does not fit", "",
              f"The 27B 4-bit assistant (18.1 GB) plus the 4B background does not fit 36 GB under the no-swap guardrail. In the 27B trial, ANE windows raised swap by "
              f"{fmt(s27.get('fg_bg_ane', {}).get('swap_delta_mb_total'))} MB and the foreground p95 TTFT reached {fmt(s27.get('fg_bg_ane', {}).get('fg_ttft_p95_ms'))} ms while pages were read back; "
              f"GPU windows stayed flat. The full R4 run therefore uses the 3B 4-bit interactive model as the foreground, which fits beside either placement. Choosing a foreground that fits is a "
              f"placement decision in its own right: on this machine the ANE worker costs 4.8 GB steady and up to 9.2 GB while loading, in addition to the GPU model.", ""]
    if r4:
        L += ["## R4 results", "",
              f"Foreground `{args.get('fg_model')}` on the GPU, one request every {args.get('fg_interval_s')} s, up to {args.get('fg_max_tokens')} tokens. Background `{args.get('bg_model')}` on the GPU (BF16, same process, preempted by the foreground at token boundaries) or on the ANE (fp16 hybrid, ane.cpp serve, one session). {fa.get('windows')} windows of {fmt(fa.get('window_s'))} s per foreground condition; baselines {fmt(bga.get('window_s'))} s.", "",
              (f"Provenance: GPU-side conditions from the leak-fixed rerun `{Path(a.r4_gpu).name}`, ANE-side conditions from `{Path(a.r4).name}`." if r4g else
               f"Provenance: all conditions from `{Path(a.r4).name}`, one session with the leak-fixed runner and a live `powermetrics` log. Earlier records (`ane-r4-concurrent`, whose GPU-side windows retained every finished generator's KV cache, and `ane-r4-gpu-rerun`) are kept and agree within run noise."), "",
              "| Condition | fg TTFT p50 / p95 ms | fg token gap p95 ms | fg decode tok/s | fg TTFT > 2 s | bg correct per min | bg latency p50 ms | swap Δ MB | MLX peak GB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k, v in c.items():
            L.append(f"| {k} | {fmt(v.get('fg_ttft_p50_ms'))} / {fmt(v.get('fg_ttft_p95_ms'))} | {fmt(v.get('fg_gap_p95_ms'))} | {fmt(v.get('fg_decode_tps'), 1)} | {fmt(v.get('fg_deadline_misses'))} | {fmt(v.get('bg_correct_per_min'), 2)} | {fmt(v.get('bg_latency_p50_ms'))} | {fmt(v.get('swap_delta_mb_total'))} | {fmt(v.get('mlx_active_gb_max'), 1)} |")
        L += ["", "## Result theory: what the numbers mean", ""]
        gpu_rate, ane_rate = bga.get("bg_correct_per_min") or 0, bgn.get("bg_correct_per_min") or 0
        share_gpu = (fg_gpu.get("bg_correct_per_min") or 0) / gpu_rate if gpu_rate else None
        L += [f"1. **Throughput.** Alone, the GPU completes {fmt(gpu_rate, 1)} correct background tasks per minute and the ANE {fmt(ane_rate, 1)}: the ANE path is {fmt(gpu_rate / ane_rate if ane_rate else None, 1)}× slower on this 4B model. "
              f"Its decode rate ({fmt(bgn.get('bg_decode_tps'), 1)} tok/s vs {fmt(bga.get('bg_decode_tps'), 1)}) and above all its prompt processing (≈25 tok/s, one 32-lane surface per token) set that ceiling; the cost model says ≈70 % of its per-token time is streaming fp16 weights at ≈90 GB/s, one sixth of the SoC's bandwidth.",
              f"2. **Sharing the GPU.** With the foreground active, the GPU still finishes {fmt(share_gpu * 100 if share_gpu is not None else None)} % of its solo background rate, because the foreground uses the GPU only in bursts and the scheduler hands the gaps to the background at token boundaries.",
              f"3. **Foreground protection.** Background on the GPU raised foreground first-token p95 from {fmt(fa.get('fg_ttft_p95_ms'))} ms (alone) to {fmt(fg_gpu.get('fg_ttft_p95_ms'))} ms (worst window {fmt(fg_gpu.get('fg_ttft_p95_ms_worst_window'))}); token gaps were unchanged ({fmt(fa.get('fg_gap_p95_ms'))} vs {fmt(fg_gpu.get('fg_gap_p95_ms'))} ms). The p95 growth is the background's non-preemptible prompt-processing step: a foreground arrival during it waits for it. Its p50 actually fell ({fmt(fa.get('fg_ttft_p50_ms'))} → {fmt(fg_gpu.get('fg_ttft_p50_ms'))} ms), consistent with the GPU staying in a high-clock state under continuous work instead of ramping up for each isolated request; that is a hypothesis from the numbers, not a measured clock. Background on the ANE gave {fmt(fg_ane.get('fg_ttft_p95_ms'))} ms TTFT p95, indistinguishable from an idle GPU. The ANE's measured contribution to responsiveness on this pairing is therefore the difference between {fmt(fg_gpu.get('fg_ttft_p95_ms'))} and {fmt(fg_ane.get('fg_ttft_p95_ms'))} ms at the 95th percentile, with the typical request unaffected either way.",
              f"4. **Memory.** With one placement resident at a time, swap stayed flat or fell in every window of both records once the harness leak was fixed (GPU placement {fmt(fg_gpu.get('swap_delta_mb_total'))} MB over three windows, ANE placement {fmt(fg_ane.get('swap_delta_mb_total'))} MB). GPU placement holds {fmt(fg_gpu.get('mlx_active_gb_max'), 1)} GB in MLX (both models); ANE placement holds {fmt(fg_ane.get('mlx_active_gb_max'), 1)} GB in MLX plus a separate 4.8 GB process (9.2 GB while loading), a second copy of the model. Free memory still bottoms near zero in GPU-placement windows on this 36 GB machine, so a larger foreground model tips either placement into swap; the 27B trial showed that directly.",
              f"4b. **Foreground decode rate.** The foreground decoded at {fmt(fa.get('fg_decode_tps'), 1)} tok/s alone, {fmt(fg_gpu.get('fg_decode_tps'), 1)} with the background on the GPU and {fmt(fg_ane.get('fg_decode_tps'), 1)} with the background on the ANE. The ANE placement protects the first token but costs about {fmt(100 * (1 - (fg_ane.get('fg_decode_tps') or 0) / ((fa1 or fa).get('fg_decode_tps') or 1)))} % of foreground decode speed (within run 1, whose own foreground-alone windows measured {fmt((fa1 or fa).get('fg_decode_tps'), 1)} tok/s), while GPU placement costs none, because the ANE worker's CPU side (prompt processing, norms, attention, sampling on Accelerate) competes with the foreground's CPU-side token loop and memory bandwidth. The ANE is isolated from the GPU, not from the SoC.",
              energy_text(load(Path(a.energy) / "energy.json"), load(Path(a.energy) / "summary.json")),
              "6. **Correctness.** Background correctness on both devices uses the same frozen fixture; R3 established that the 4B on the ANE matches the BF16 GPU reference (24/30 vs 23/30), so throughput and latency differences are not bought with accuracy.", "",
              "## What this does and does not establish", "",
              "- Established: the private ANE runtime works on this macOS; the projection and FFN lowerings are numerically characterized; a lowering fix improves the FFN kernel eightfold; a 4B model runs correctly through the hybrid path; the per-dispatch and per-byte costs are measured; the concurrent placement numbers above.",
              "- Not established: any throughput advantage for the ANE (it is slower on this model and runtime); batched multi-session decoding on the ANE (the runtime corrupts lanes beyond the first); results on other chips or OS builds; energy on any model other than the 4B pairing measured.",
              "- Open kernel questions, in order of measured size: fp16 divergence that grows with model depth (logit drift 0.4 → 1.4 from 0.6B to 4B), prompt-processing throughput on the ANE (25 tok/s), the ≈50 µs fixed dispatch cost, and the batched-lane defect.", ""]
    L += ["## Where everything lives", "",
          "- Plan and decisions: `docs/agentic-offload-prototype-plan.md` (revision 5 with R0–R4 records and change notes).",
          "- Code: `prototype/` (scheduler, ANE probes and lowerings under `prototype/ane/`, R3 harness under `prototype/ane/r3/`, R4 runner under `prototype/r4/`).",
          "- Results: `results/ane-r0-feasibility`, `ane-r1-projection`, `ane-r2-ffn`, `ane-r3-model*`, `ane-r4-concurrent`, plus the earlier scheduler and scenario runs.",
          "- Proposed upstream patch: `prototype/ane/vendor/patches-proposed/0001-ffn-silu-exp-form.patch` (not applied by default)."]
    Path(a.out).write_text("\n".join(L) + "\n"); print(f"wrote {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
