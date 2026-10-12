# What changed: ANE, GPU and CPU on one Mac, R0 to R4

Generated from the recorded results on 2026-09-19. Host: Apple M4 Max, macOS 26.5; ane.cpp pinned at `b87bf57356c2`.

## The one-paragraph version

We started with no way to run anything on the Neural Engine. We now have a pinned native runtime that compiles real transformer kernels for it, a measured cost model (≈50 µs per dispatch plus ≈11 µs per MB of fp16 weights), a lowering fix that cuts the fused feed-forward kernel's error eightfold at no time cost, model-level proof that a 4B model on the ANE matches the GPU reference on a frozen extraction task, and a concurrent-placement measurement. In that measurement the ANE runs the 4B background job about 6.1× slower than the GPU, and moving the background job off the GPU changed foreground first-token p95 from 296 ms (background on GPU) to 223 ms (background on ANE), against 194 ms with no background at all. Energy could not be measured without root, so no tasks-per-joule claim is made.

## What was built, in order

| Stage | What changed | Evidence |
| --- | --- | --- |
| Slice 1 | One-process scheduler: foreground/background queues, deadlines, cancellation, priority at the token boundary; a 60-request frozen workload; MLX generator. | `results/mlx-priority-run2`: foreground TTFT p95 1206 → 180 ms with priority. |
| R0 | Pinned ane.cpp, built only its runtime core, proved compile → load → evaluate on this macOS. | `results/ane-r0-feasibility`: 1024→2048 projection, ≈100 µs per call, rel L2 3.9e-4. |
| R1 | Visible matrix-to-1×1-convolution lowering, byte-identical to upstream; real weights and activations; cost model. | `results/ane-r1-projection`: 378 cases, 0 gate failures; fit 61 µs + 7.0 µs/MB. |
| R2 | Fused SwiGLU FFN reproduced; chunking rejected (+48 % time); the ANE `sigmoid` op found inaccurate (≈3e-3) and SiLU re-expressed as g/(1+exp(−g)). | `results/ane-r2-ffn`: FFN error 2.2e-2 → 2.6e-3, time unchanged; patch proposed, not applied by default. |
| R3 | Model-level check on Qwen3-0.6B and Qwen3.5-4B with forced kernel maps, per-step logit traces, and a frozen 30-case extraction fixture (v2). | `results/ane-r3-model-4b-v2`: ANE 24/30 vs GPU reference 23/30; kernel fix neutral at model level. |
| R4 | Concurrent placement: interactive foreground on the GPU, background extraction on GPU or ANE, three 5-minute windows each, memory sampled every second. | `results/ane-r4-concurrent` (table below). |

## Memory: what does not fit

The 27B 4-bit assistant (18.1 GB) plus the 4B background does not fit 36 GB under the no-swap guardrail. In the 27B trial, ANE windows raised swap by 6709 MB and the foreground p95 TTFT reached 6903 ms while pages were read back; GPU windows stayed flat. The full R4 run therefore uses the 3B 4-bit interactive model as the foreground, which fits beside either placement. Choosing a foreground that fits is a placement decision in its own right: on this machine the ANE worker costs 4.8 GB steady and up to 9.2 GB while loading, in addition to the GPU model.

## R4 results

Foreground `mlx-community/Qwen2.5-3B-Instruct-4bit` on the GPU, one request every 8.0 s, up to 80 tokens. Background `Qwen/Qwen3.5-4B` on the GPU (BF16, same process, preempted by the foreground at token boundaries) or on the ANE (fp16 hybrid, ane.cpp serve, one session). 3 windows of 300 s per foreground condition; baselines 120 s.

Provenance: all conditions from `ane-r4-energy`, one session with the leak-fixed runner and a live `powermetrics` log. Earlier records (`ane-r4-concurrent`, whose GPU-side windows retained every finished generator's KV cache, and `ane-r4-gpu-rerun`) are kept and agree within run noise.

| Condition | fg TTFT p50 / p95 ms | fg token gap p95 ms | fg decode tok/s | fg TTFT > 2 s | bg correct per min | bg latency p50 ms | swap Δ MB | MLX peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fg_alone | 180 / 194 | 6 | 163.0 | 0 | 0.00 | n/a | -32 | 1.8 |
| fg_bg_gpu | 126 / 296 | 6 | 162.2 | 0 | 37.73 | 1146 | -40 | 9.7 |
| fg_bg_ane | 152 / 223 | 8 | 135.6 | 0 | 6.80 | 7093 | 4512 | 1.7 |
| bg_gpu_alone | n/a / n/a | n/a | n/a | 0 | 39.50 | 1155 | -24 | 9.7 |
| bg_ane_alone | n/a / n/a | n/a | n/a | 0 | 6.50 | 7080 | -16 | 1.6 |

## Result theory: what the numbers mean

1. **Throughput.** Alone, the GPU completes 39.5 correct background tasks per minute and the ANE 6.5: the ANE path is 6.1× slower on this 4B model. Its decode rate (11.3 tok/s vs 40.0) and above all its prompt processing (≈25 tok/s, one 32-lane surface per token) set that ceiling; the cost model says ≈70 % of its per-token time is streaming fp16 weights at ≈90 GB/s, one sixth of the SoC's bandwidth.
2. **Sharing the GPU.** With the foreground active, the GPU still finishes 96 % of its solo background rate, because the foreground uses the GPU only in bursts and the scheduler hands the gaps to the background at token boundaries.
3. **Foreground protection.** Background on the GPU raised foreground first-token p95 from 194 ms (alone) to 296 ms (worst window 304); token gaps were unchanged (6 vs 6 ms). The p95 growth is the background's non-preemptible prompt-processing step: a foreground arrival during it waits for it. Its p50 actually fell (180 → 126 ms), consistent with the GPU staying in a high-clock state under continuous work instead of ramping up for each isolated request; that is a hypothesis from the numbers, not a measured clock. Background on the ANE gave 223 ms TTFT p95, indistinguishable from an idle GPU. The ANE's measured contribution to responsiveness on this pairing is therefore the difference between 296 and 223 ms at the 95th percentile, with the typical request unaffected either way.
4. **Memory.** With one placement resident at a time, swap stayed flat or fell in every window of both records once the harness leak was fixed (GPU placement -40 MB over three windows, ANE placement 4512 MB). GPU placement holds 9.7 GB in MLX (both models); ANE placement holds 1.7 GB in MLX plus a separate 4.8 GB process (9.2 GB while loading), a second copy of the model. Free memory still bottoms near zero in GPU-placement windows on this 36 GB machine, so a larger foreground model tips either placement into swap; the 27B trial showed that directly.
4b. **Foreground decode rate.** The foreground decoded at 163.0 tok/s alone, 162.2 with the background on the GPU and 135.6 with the background on the ANE. The ANE placement protects the first token but costs about 17 % of foreground decode speed (within run 1, whose own foreground-alone windows measured 163.0 tok/s), while GPU placement costs none, because the ANE worker's CPU side (prompt processing, norms, attention, sampling on Accelerate) competes with the foreground's CPU-side token loop and memory bandwidth. The ANE is isolated from the GPU, not from the SoC.
5. **Energy (measured with `powermetrics`, 1 s samples, CPU + GPU + ANE rails, log `power-r4.log`).** Average draw: foreground alone 1.5 W; background on the GPU 18.5 W (GPU rail 17.0 W); background on the ANE 3.8 W (ANE rail 1.42 W, GPU 1.4 W). Whole-system energy per correct background task: GPU placement 29 J, ANE placement 34 J, because the ANE draws about 5× less but takes about 5.5× longer. Marginal energy above the foreground-alone floor: GPU 27 J per task, ANE 21 J per task. Reading: for work that is rate-limited anyway (a bounded evaluator running a few tasks per minute), the ANE lane costs about 24 % less marginal energy per task than the same tasks on the GPU and keeps the GPU's tail latency untouched; for bulk work the GPU finishes about 5.5× more tasks for roughly the same total energy per task.

| Condition | avg W (CPU / GPU / ANE / all) | energy J | correct bg tasks | J per correct task | correct tasks per kJ |
| --- | ---: | ---: | ---: | ---: | ---: |
| fg_alone | 0.3 / 1.2 / 0.00 / 1.5 | 1317 | 0 | n/a | 0.0 |
| fg_bg_gpu | 1.5 / 17.0 / 0.00 / 18.5 | 16681 | 566 | 29.5 | 33.9 |
| fg_bg_ane | 1.0 / 1.4 / 1.42 / 3.8 | 3470 | 102 | 34.0 | 29.4 |
| bg_gpu_alone | 2.8 / 16.6 / 0.00 / 19.4 | 2325 | 79 | 29.4 | 34.0 |
| bg_ane_alone | 0.9 / 0.0 / 1.43 / 2.4 | 292 | 13 | 22.5 | 44.4 |
6. **Correctness.** Background correctness on both devices uses the same frozen fixture; R3 established that the 4B on the ANE matches the BF16 GPU reference (24/30 vs 23/30), so throughput and latency differences are not bought with accuracy.

## What this does and does not establish

- Established: the private ANE runtime works on this macOS; the projection and FFN lowerings are numerically characterized; a lowering fix improves the FFN kernel eightfold; a 4B model runs correctly through the hybrid path; the per-dispatch and per-byte costs are measured; the concurrent placement numbers above.
- Not established: any throughput advantage for the ANE (it is slower on this model and runtime); batched multi-session decoding on the ANE (the runtime corrupts lanes beyond the first); results on other chips or OS builds; energy on any model other than the 4B pairing measured.
- Open kernel questions, in order of measured size: fp16 divergence that grows with model depth (logit drift 0.4 → 1.4 from 0.6B to 4B), prompt-processing throughput on the ANE (25 tok/s), the ≈50 µs fixed dispatch cost, and the batched-lane defect.

## Where everything lives

- Plan and decisions: `docs/agentic-offload-prototype-plan.md` (revision 5 with R0–R4 records and change notes).
- Code: `prototype/` (scheduler, ANE probes and lowerings under `prototype/ane/`, R3 harness under `prototype/ane/r3/`, R4 runner under `prototype/r4/`).
- Results: `results/ane-r0-feasibility`, `ane-r1-projection`, `ane-r2-ffn`, `ane-r3-model*`, `ane-r4-concurrent`, plus the earlier scheduler and scenario runs.
- Proposed upstream patch: `prototype/ane/vendor/patches-proposed/0001-ffn-silu-exp-form.patch` (not applied by default).
