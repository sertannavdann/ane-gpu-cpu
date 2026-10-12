# P0: ANE project story

Draft, 2026-10-11. Sources: `docs/R4-what-changed.md` and `docs/agentic-offload-prototype-plan.md` (R1 to R4 records). Result directories are in the main checkout at `../ANE:GPU:CPU/results/`. Tags: **[M]** measured, **[D]** derived from measured numbers, **[H]** hypothesis not yet tested.

## The 30-second version

I put a 4B background model on the Apple Neural Engine (ANE) next to an interactive model on the GPU of an M4 Max. It worked and matched the GPU's accuracy, but it ran 6.1× slower. That gap has two causes, and both can be measured. Decode is limited by memory bandwidth: the ANE streams fp16 weights at about 90 GB/s, a fraction of what the GPU gets. Prefill is worse: the runtime processes the prompt one token at a time, so a 94-token prompt costs as much as decoding 94 tokens, while the GPU handles the whole prompt in one batched pass. What the ANE did buy is a protected GPU tail latency (foreground p95 TTFT 223 ms instead of 296 ms) at about 24 % lower marginal energy per task.

## "Why was the ANE 6.1× slower?" (the 2-minute version)

1. **Headline [M].** Running alone, the GPU completed 39.5 correct tasks per minute and the ANE 6.5. Accuracy was the same (R3: 24/30 on the ANE vs 23/30 on GPU BF16), so the gap is not bought with correctness.
2. **Split the per-task latency [M/D].** ANE p50 latency is 7.09 s per task. Prompt processing runs at ≈25 tok/s, so a 94-token note costs ≈3.7 s before the first output token [M]. Decode runs at 11.3 tok/s, so the ≈48-token answer costs about the same again [D]. On the GPU, 1.16 s p50 is almost entirely decode at 40 tok/s, which means the GPU's prefill is close to free [D].
3. **Decode is bandwidth-bound [M/D].** The R1 cost model, refit across runs, is ≈50–61 µs per dispatch plus 7.0–11.8 µs per MB of fp16 weights. At the per-token level about 70 % of ANE time is weight streaming at ≈90 GB/s [D]. Decode (M=1) is a matrix-vector product: about one FLOP per weight byte at fp16, far left on the roofline, so only bandwidth matters. The 11.3 vs 40 tok/s ratio (3.5×) is a bandwidth ratio, not a compute ratio [D, checked in P1].
4. **Prefill is not batched [M/H].** The GPU runs the prompt as a GEMM (M = 94), which moves to the compute-bound side of the roofline: each weight byte loaded is used 94 times. The ANE runtime pushes one token through per pass, using one lane of its 32-wide surface. It repeats the decode cost for every prompt token [M: 25 tok/s ≈ decode rate order]. Multi-session batching on this runtime corrupts every lane after the first (R3 Finding 1) [M], so the obvious fix was blocked by a correctness defect, not by the hardware [H: the ANE can batch prefill across the surface width].
5. **Fixed dispatch cost [M].** ≈50 µs per dispatch, several dispatches per layer. For Qwen3-0.6B that is ≈30 % of per-layer time [D]. Splitting kernels to stream weights faster was tried and rejected: each extra chunk added ≈50–60 µs and the streaming rate did not rise (R2) [M].
6. **The ANE is isolated from the GPU, not from the SoC [M].** With the background on the ANE, foreground decode dropped from 163 to 136 tok/s (−17 %). The ANE worker's CPU-side work (norms, attention, sampling) and its memory traffic compete with the foreground's token loop. It also holds a second 4.8 GB copy of the model (9.2 GB while loading).

**One sentence:** decode is memory-bound and the ANE gets a small slice of the bandwidth; prefill is compute-bound on the GPU but bandwidth-bound on the ANE, because the runtime cannot batch tokens.

### Numbers to check before quoting

- **Bandwidth share (settled by P1, 2026-10-12).** The R1 record calls 89 GB/s "about one sixth of the M4 Max DRAM bandwidth", which implies ≈540 GB/s, the 16-core M4 Max figure. This machine is the 14-core, 36 GB bin, specified at about 410 GB/s, so the ANE's share is **≈22 %**. Measured context from the P1 labs on a quiet machine: one CPU core streams **90 GB/s**, the same as the ANE; all CPU cores reach 253 GB/s (62 % of spec); and the GPU tiers of gRPC_llm imply ≈314 GB/s of usable streaming (cost-model fit, [results](labs/results-2026-10-12.md)). One-line version: **"the ANE got one CPU core's worth of bandwidth; the GPU got three and a half."**
- **Bytes per token.** 70 % of an ≈88 ms token at 90 GB/s means ≈5.6 GB streamed per token [D]. Reconcile this with the model's actual fp16 parameter bytes on the ANE side of the hybrid split before stating a GPU effective bandwidth.
- **GPU prefill "close to free"** is inferred from latency arithmetic, not from a measured TTFT split for the background model [H].

## "What would you change in the next chip?" (10-minute version)

Hardware asks are ranked by how much of the measured gap they address. Each asks for a capability, not a micro-architecture, which keeps it honest.

| # | Ask | Gap it attacks | Evidence | Kind |
| --- | --- | --- | --- | --- |
| 1 | **More DRAM bandwidth for the ANE**, or a bandwidth share that can be raised when the GPU is idle | ≈70 % of decode time is weight streaming at ≈90 GB/s | R1 fit, R4 decode rates | HW |
| 2 | **Native low-bit weight decode** (int4/int8 weights expanded to fp16 in the load path, per-block scales) | Same term: Q4 cuts weight bytes 4× vs fp16. The runtime had no low-bit path, so the ANE streamed fp16 while the GPU foreground ran 4-bit | R1 bytes term, P2 will quantify | HW + compiler |
| 3 | **Cheaper dispatch / chained graphs** (queue many layers in one submission, ideally device-side sequencing) | ≈50 µs × dispatches per token, ≈30 % on small models | R1, R2 chunking result | HW + driver |
| 4 | **Weight sharing between GPU and ANE** (same physical pages, one copy) | 4.8–9.2 GB duplicate model, swap in the 27B trial | R4 memory | OS / driver |
| 5 | **More accurate transcendentals** (sigmoid error ≈3e-3) | SwiGLU error 2.2e-2 before the exp-form rewrite | R2 | HW |
| 6 | **QoS for memory bandwidth across ANE, CPU and GPU** | −17 % foreground decode with the ANE active | R4 | HW / OS |

Software-only fixes come first, because they cost nothing in silicon and two of them may close most of the gap:

- **Batch prefill across the 32-wide surface.** Fix or replace the multi-lane decode defect. [H] This turns prefill into a GEMM on the ANE too.
- **Chunk the background's prefill on the GPU.** A scheduler change that would likely remove most of the 296 ms p95 penalty without using the ANE at all [H]. This is the cheapest competitor to the whole ANE lane and should be stated up front.
- **fp16 drift with depth** (logit drift 0.4 → 1.4 from 0.6B to 4B) is the open numerical question before larger models [M].

## Follow-up questions to prepare

- *"Why not just use the GPU?"* For bulk work, use the GPU: 5.5× more tasks at roughly the same energy per task. The ANE wins only for a rate-limited background lane, where the comparison is marginal energy (21 vs 27 J/task) and GPU tail latency.
- *"How do you know it actually ran on the ANE?"* The private `_ANE*` compile, load and evaluate path is in use, and the ANE power rail reads 1.42 W in `powermetrics` during ANE windows and 0 W otherwise [M]. Per-operator placement was not traced.
- *"What would you measure next?"* P1 (bandwidth ceiling), then per-layer drift, then batched prefill once lanes are correct.
- *"What did you get wrong?"* A harness transpose loop (176 µs) was briefly mistaken for ANE cost until the timer was split. The GPU-side runner leaked KV caches in the first R4 record. The fixture was revised twice, then frozen to avoid fitting it to the model.
