# P-series: inference systems prototypes

Branch `p-series`, worktree `../ANE-GPU-CPU-p-series`. Started 2026-10-11.

The R- and L-series asked whether the ANE helps on this Mac. The P-series builds the background those results rest on: kernels, quantization, runtimes, compilers and evaluation. Each prototype ends with numbers you can defend in an architect or runtime interview. Every prototype follows `docs/instructions.md`: one narrow end-to-end slice, a baseline first, and acceptance thresholds written down before any optimisation.

**Source of truth:** the owner's [interview prep plan](interview-prep-plan.md) defines the questions, the [A]/[R] role tags, and the order **P1 → P2 → P3 → P5 → P4 → P6 → P7** (P8 optional). Every prototype follows its loop: **predict on paper, measure, explain the gap.** This file records how each prototype is built on this Mac and where the hardware forces a change. The runnable guide is [labs/README.md](labs/README.md); cross-platform differences are in [labs/other-chips.md](labs/other-chips.md).

## Where this Mac forces a change to the plan

| Plan says | On this Mac | What we do instead |
| --- | --- | --- |
| `perf stat`, `-M TopdownL1`, VTune | no `perf` on macOS; the Docker VM sees no counters | Instruments / `xctrace` CPU Counters (to wire in); clock inferred from latency until then |
| AVX2 / AVX-512 / AVX-VNNI kernels | arm64 only | NEON + `sdot` kernels; `labs/portable/portable_lab.cpp` carries an AVX2 path (compiles, correct under Rosetta); real x86 timings need an x86 box |
| P3: profile llama.cpp on CPU | done first against the running gRPC_llm tiers (Metal) | `labs/p3/llm_service_lab.py` (predict → measure → explain, cost model); a pinned CPU llama.cpp build with `llama-bench` is the next P3 slice |
| P5: IREE on x86 | no native x86 | IREE `llvm-cpu` on arm64 (timed) + x86_64 under emulation (correctness only) + riscv64 under QEMU |
| Thread affinity experiments | macOS has no pinning, only QoS | QoS (P vs E cores); the runner picks thread counts from free cores and the peak lab demonstrates the barrier effect |

## Host facts and constraints

| Item | Value | Consequence |
| --- | --- | --- |
| SoC | Apple M4 Max, 14 CPU cores, 36 GB unified memory | Peak DRAM bandwidth for this bin is about 410 GB/s according to Apple's specs. The R4 note says the ANE's ≈90 GB/s is "one sixth" of it, which implies ≈540 GB/s. **P1 measures the real figure and corrects one of the two.** |
| CPU ISA | arm64, NEON, plus SME on M4 | Kernels in P1–P3 target NEON first. SME/AMX is a stretch goal, not a baseline. |
| Perf counters | No Linux `perf` on macOS. Docker VMs get no PMU access | Use `xctrace` (Instruments CPU Counters) or `kpc` through a small wrapper. Record which counters are available before promising any of them. |
| x86 | None native. Docker `linux/amd64` runs under emulation | P5's x86 target checks **correctness only**. Its timings mean nothing. arm64 macOS is the native performance target. |
| RISC-V | Homebrew clang 20 targets riscv64. No QEMU installed yet | Cross-compile with clang, then run under `qemu-riscv64` (user mode) inside an arm64 Linux container. Use `-cpu rv64,v=true,vlen=128/256/512` to test VLA code at several vector lengths. |
| Shared state | R/L-series results and venv are gitignored and not present in this worktree | Read earlier results from the main checkout at `../ANE:GPU:CPU/results/`. Never write there. P-series outputs go to `results/p*/` in this worktree. |
| Memory rule | Check `ps` for `run_r4`, `ane.cpp-bin` and MLX processes before running benchmarks | Concurrent runs distort bandwidth numbers in both series. |

## Order and dependencies

```
P0 ─ story (from existing R4 data, no new code)
P1 ─ matmul + roofline ──┬─> P2 quantized kernels ──> P3 llama.cpp profile
                         │                              │
P7 ─ harness (thin v0 with P1, grows with each one) ────┤
                         │                              v
                         └─> P6 RVV + perf model     P4 mini runtime ──> P8 (optional)
                                    │
                                    └─> P5 IREE x86 / RISC-V
```

P1 comes first because its measured bandwidth and compute ceilings are what P2, P3, P4 and P6 are judged against. A minimal P7 (fixed seeds, warmup, repeated trials, median plus spread, JSON records) is built at the same time as P1, so no later prototype rolls its own timing code.

## Prototype briefs

Each brief gives the outcome, the smallest slice that delivers it, and the acceptance check. Effort caps are guesses, to be revised after P1.

### P0 — ANE project story (both roles)
- **Outcome:** a 2-minute and a 10-minute answer to "why was the ANE 6.1× slower?" and "what would you change in the next chip?", every number traced to a result file.
- **Slice:** `docs/p0-ane-story.md` (drafted). Its decode arithmetic is checked against P1's measured bandwidth once P1 exists.
- **Accept:** every claim is marked *measured*, *derived* or *hypothesis*, and points to a file under `results/`.
- **Cap:** 0.5 day.

### P1 — Matmul from scratch plus roofline (architect)
- **Outcome:** explain with your own measurements why decode (GEMV, ~1 FLOP/byte at fp16) is bandwidth-bound and prefill (GEMM) is compute-bound.
- **Slice:** `p1-matmul/` in C++17 with CMake. Steps in order: (1) a STREAM-style bandwidth probe, single-threaded and all-core; (2) a naive GEMM; (3) loop reordering; (4) cache blocking; (5) NEON micro-kernel; (6) threads (P-cores and E-cores, pinned via QoS); (7) GEMV for the same shapes. Check against Accelerate `cblas_sgemm`/BNNS. A Python script plots the roofline from the JSON records.
- **Shapes:** Qwen-class projections, for example 1024×3072 and 4096×11008, with M=1 (decode) and M=32/128/512 (prefill).
- **Accept:** max relative error ≤1e-5 (fp32) against Accelerate on every shape. Each optimisation step is a point on the roofline. The final GEMV reaches ≥70 % of measured bandwidth. GEMM's fraction of Accelerate is reported, with no target.
- **Cap:** 4 days.

### P2 — 4-bit and 8-bit quantized kernels, llama.cpp style (both)
- **Outcome:** know why Q4 decode is faster (fewer bytes per weight), what block size costs in accuracy and in bytes, and how to decide whether a quantized model is "good enough".
- **Slice:** `p2-quant/`. Q8_0 and Q4_0 block formats (32 weights + fp16 scale), plus a Q4_K-like two-level scale. Scalar reference, then NEON dot-product kernels (`sdot`). Weight error sweeps over block size 16/32/64/128. Run RTN against an AWQ-style activation-aware scale and a SmoothQuant-style migration on real Qwen3-0.6B weights and calibration activations. GPTQ in NumPy on one layer only.
- **Accept:** bit-exact with a scalar reference. Q4 GEMV speedup over fp16 lands within 25 % of the byte-ratio prediction from P1's roofline. Perplexity on a fixed text slice plus the R3 30-case fixture for 0.6B: fp16 vs Q8 vs Q4, with a predefined ceiling on the quality drop.
- **Cap:** 4 days.

### P3 — Profile and modify llama.cpp on CPU (architect)
- **Outcome:** say where llama.cpp's time goes, why thread scaling flattens, and how close it gets to the bandwidth limit.
- **Slice:** pinned llama.cpp checkout (vendored like `ane.cpp`, gitignored, rebuilt by script), CPU backend only, Metal off. `llama-bench` sweeps threads 1–14 × {Q4_0, Q8_0, F16} × {pp512, tg128}. Per-op timing via `GGML_PERF` or a patched graph-compute hook. One modification: swap in the P2 Q4 kernel, or change the thread-split policy, measured with the P7 harness.
- **Accept:** tg tokens/s × bytes per token as a fraction of P1 bandwidth, per thread count. An explanation of the knee (P-core vs E-core, bandwidth saturation, barrier cost) backed by counters. The modification is kept only if it beats the baseline by more than run-to-run spread.
- **Cap:** 4 days.

### P4 — Small inference runtime in C++ (runtime)
- **Outcome:** working knowledge of memory planning, KV cache (contiguous, then paged), continuous batching, operator fusion and numerical regression tests.
- **Slice:** `p4-runtime/` runs Qwen3-0.6B fp32/Q8 on CPU using P1/P2 kernels. Static memory planner over the graph (liveness, then buffer reuse). Contiguous KV cache first, then a paged block table. Continuous-batching scheduler reuses the Slice-1 scheduler ideas (`prototype/policy.py`) in C++. One fusion (RMSNorm+matmul or SwiGLU).
- **Accept:** per-layer activations match an MLX/PyTorch reference within tolerances recorded before the run. Greedy tokens identical to the reference for N steps on the R3 prompts. Paged and contiguous KV produce bit-identical outputs. Batching raises aggregate tok/s at a stated p95 per-token latency limit.
- **Cap:** 7 days.

### P5 — IREE end to end, x86 and RISC-V under QEMU (runtime)
- **Outcome:** explain IREE's stages (input → flow → stream → HAL → target codegen), how dispatches are formed, and how to debug a wrong output on one target.
- **Slice:** pinned IREE release wheels (`iree-base-compiler`, `iree-base-runtime`). Model: one transformer block exported from PyTorch via `iree-turbine` (or StableHLO). Targets: `llvm-cpu` host arm64 (native, timed), `x86_64` in an amd64 container (correctness), `riscv64` with `+v` cross-compiled and run with `iree-run-module` under `qemu-riscv64`. Use `--mlir-print-ir-after-all` and `--iree-hal-dump-executable-*` to inspect dispatches.
- **Accept:** all three targets match the PyTorch reference within tolerance. Dispatch count and grouping are documented. One deliberately injected miscompile or bad flag is found by bisecting dispatches (`--iree-flow-trace-dispatch-tensors`), with the steps written down.
- **Cap:** 5 days. Risk: the RISC-V runtime build. Fall back to the `riscv64` Linux cross-build documented by IREE.

### P6 — RISC-V vector kernels plus analytical performance model (architect)
- **Outcome:** write vector-length-agnostic kernels, and argue for one custom instruction with numbers. Explain memory ordering (RVWMO) and coherence costs as they apply to the kernels.
- **Slice:** `p6-rvv/`. RVV 1.0 intrinsics for GEMV, Q8 dot product and softmax, cross-compiled with clang. Checked under QEMU at VLEN 128/256/512 (correctness only; QEMU timing is not performance). An analytical model in Python: instruction counts from the kernels (or from a QEMU plugin) × an assumed microarchitecture (VLEN, lanes, issue width, L1/L2/DRAM bandwidth), validated against P1's measured arm64 results for the same kernels.
- **Accept:** correct at every VLEN. The model predicts the measured arm64 NEON kernel within 30 % before it is used to argue for hardware. One proposed instruction (for example a fused int4 unpack-and-dot) with its modelled speedup on decode and the area and latency assumptions behind it.
- **Cap:** 5 days.

### P7 — Benchmark and evaluation harness (runtime)
- **Outcome:** fair benchmarking and quality testing that catches regressions without flaking.
- **Slice:** `p7-harness/` Python package. v0 (with P1): warmup, N trials, median/IQR, CPU frequency and thermal notes, environment capture (git SHA, OS, power source), JSON records. v1: statistical comparison (bootstrap CI or Mann-Whitney) for "is B faster than A". Quality gates for non-deterministic outputs: exact match at temperature 0, tolerance bands on logits, distributional tests at T>0. Reuses the R3 frozen fixture and `prototype/bench/metrics.py` where they fit.
- **Accept:** an A/A test (same binary twice) never reports a difference at the chosen alpha over 20 runs. A planted 5 % slowdown and a planted quality regression are both detected.
- **Cap:** 1 day for v0, 3 days for v1.

### P8 — Parallel inference on one machine (optional, runtime)
- **Outcome:** reason about tensor, pipeline and expert parallelism and the cost of all-reduce. Mixture-of-experts load balance.
- **Slice:** only after P4. Simulate N "devices" as processes or threads with shared-memory all-reduce on CPU, or split CPU+GPU. Measure all-reduce cost against message size and compare with the α–β model. Toy MoE routing with skewed load.
- **Accept:** decided when P4 finishes. Deferred by the deletion test until then.

## Repository layout

```
docs/p-series-plan.md      this file
docs/p0-ane-story.md       P0
docs/interview-prep-plan.md  owner's plan: questions, [A]/[R] tags, order (source of truth)
docs/labs/README.md        plain-language runnable guide: predict -> measure -> explain, per lab
docs/labs/other-chips.md   Apple vs generic ARM64, Qualcomm, x86/Windows, FPGA, RISC-V
labs/                      C++ labs + run_labs.py + predictions.py (P1 p1_*, P2 p2_quant, P7 v0 common/harness.h)
labs/p3/                   llm_service_lab.py: predict and measure the gRPC_llm tiers
labs/portable/             portable_lab.cpp (NEON / AVX2 / scalar) + macOS-vs-Linux comparison
results/labs-<time>/       gitignored run outputs: summary.md, records.jsonl, console.log, env.json
```

Status 2026-10-12: P1, P2 (accuracy and speed parts), P7 v0 (harness + predict/measure tables), a first P3 slice against gRPC_llm, and a portable cross-platform lab are built and run. Still open in P1/P2: hardware counters via `xctrace`, GPTQ on a real layer, perplexity and fixture evaluation of Q8/Q4 on Qwen3-0.6B. P3 has two slices: the gRPC_llm service lab and the pinned CPU llama.cpp build (`b11562`) with a 4-format × 1–10-thread bench ([results](labs/results-p3-llamacpp-2026-10-12.md)). Next in P3: a time profile, one measured modification, and perplexity per format. P4–P8 are not started.

## Open decisions for the owner

1. **Q4 baseline:** match llama.cpp's exact `block_q4_0` layout (so P2 and P3 share kernels) or design your own (more to explain, no reuse). Recommended: match llama.cpp.
2. **P5 model:** one transformer block (fast to debug) or full Qwen3-0.6B (closer to real, much slower under QEMU). Recommended: one block.
