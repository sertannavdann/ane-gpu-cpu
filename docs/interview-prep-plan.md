# IMachines interview prep: questions → prototypes

Owner's plan, saved 2026-10-12. This is the source of truth for the P-series; [p-series-plan.md](p-series-plan.md) records how it is built on this Mac and [labs/README.md](labs/README.md) is the runnable guide.

Tags: **[A]** = AI/ML Performance Architect, **[R]** = Sr. AI Runtime Engineer.
Each prototype is built to answer a cluster of likely questions. Rule for every prototype: **predict the number on paper first, then measure it, then explain the gap.** That loop is exactly what both interviews test.

Suggested order (about 5–6 weeks part-time): P1 → P2 → P3 → P5 → P4 → P6 → P7. P8 is optional.

---

## P0. Your own project (they *will* ask, prepare first) [A][R]
Questions:
- "Walk me through the ANE project. Why was the ANE 6.1× slower than the GPU?" → Weight streaming at ~90 GB/s versus SoC bandwidth; prefill at 25 tok/s; fixed dispatch cost of 50 µs.
- "How did you build the cost model? How would you validate it on new hardware?"
- "Your fp16 drift grows with depth (0.4 → 1.4). Why, and how would you fix it?" → Accumulation precision, norm/residual in higher precision, per-layer scaling.
- "If you designed the next NPU, what would you change based on what you measured?" → This is a co-design question; have three concrete answers ready (dispatch overhead, batched lanes, on-chip weight reuse).

No build needed. Write a one-page summary and practise saying it in 3 minutes and again in 10.

---

## P1. CPU GEMM/GEMV from scratch + roofline [A] (core), [R]
**Build:** In C++, write matmul (prefill-like, M×K·K×N) and matvec (decode-like) kernels in stages: naive → loop reorder → cache blocking (L1/L2 tiles) → register micro-kernel → SIMD (AVX2/AVX-512 or NEON) → multithreaded (OpenMP or your own thread pool). Measure GFLOP/s and GB/s at every stage with `perf stat` (cycles, IPC, L1/LLC misses). Plot all stages on a roofline for your machine. Measure peak bandwidth yourself with a STREAM-like microbenchmark.

**Questions it answers:**
- Why is LLM decode memory-bound and prefill compute-bound? Compute the arithmetic intensity of each. (fp16 GEMV ≈ 1 FLOP/byte; GEMM grows with batch/sequence length.)
- What's the theoretical max tok/s for a 7B Q4 model (~4 GB) on a machine with 80 GB/s? (~20 tok/s.) What does a batch of 8 change?
- How do you choose tile sizes? What's a register-blocked micro-kernel and why does it exist?
- Your kernel runs at 30 % of peak. Walk me through finding out why.
- Threading: scaling stops at 8 cores. Why? (Bandwidth saturation, false sharing, NUMA, SMT, frequency drop under AVX-512.)
- What do IPC, front-end-bound and back-end-bound tell you? (Use `perf stat -M TopdownL1` or VTune.)
- How does hardware prefetching interact with your access pattern?

---

## P2. Quantized dot-product kernels (llama.cpp style) [A][R]
**Build:** Implement Q8_0 and Q4_0 block quantization (32-element blocks + fp16 scale). Then write a dequantize-in-register dot product: first scalar, then SIMD, then integer dot instructions (AVX-VNNI `vpdpbusd` or ARM `sdot`). Compare speed and error against fp32. Add a simple per-channel INT8 version and an outlier experiment (inject large channels, watch the error blow up).

**Questions it answers:**
- Block vs per-tensor vs per-channel quantization: trade-offs in accuracy, metadata overhead and kernel complexity.
- Why does 4-bit decode run faster even though you do extra dequantization work? (Fewer bytes moved, and decode is bandwidth-bound.)
- What are GPTQ, AWQ and SmoothQuant? Why do activation outliers make W8A8 hard?
- Weight-only vs weight+activation quantization: when does each pay off on a CPU?
- KV-cache quantization: what breaks?
- How do you decide a quantized model is "good enough"? (Perplexity, KL vs the fp16 model, task accuracy, top-k agreement.) This is the JD's "quantifying model success".
- Fine-tuning: what are QAT and LoRA/QLoRA, and how does fine-tuning interact with quantization?

---

## P3. Profile and modify llama.cpp on CPU [A] (core), [R]
**Build:** Build llama.cpp, run a small model (Qwen2.5-0.5B/1.5B or TinyLlama) at several quant formats (Q4_0, Q4_K_M, Q8_0, F16) and thread counts 1…N. Record prefill and decode tok/s separately (`llama-bench`). Run `perf record` / flame graphs to find the hottest ggml ops. Compare decode tok/s to your P1 bandwidth ceiling. Then **change something**: optimize one kernel, try a different thread/affinity layout, or add a counter. Show before/after.

**Questions it answers:**
- How does llama.cpp work? (GGUF + mmap, ggml compute graph, backend ops, thread pool, KV cache layout.)
- Where does time go in prefill vs decode? Which ops dominate (matmul vs attention vs norm/rope)?
- Why does decode stop scaling past N threads, while prefill keeps scaling?
- How close is llama.cpp to the hardware bound on your machine, and what accounts for the rest?
- How would you add a new CPU backend or ISA (for example RISC-V RVV) to ggml?
- What does `mmap` buy you, and what does it cost (page faults, first-token latency)?

---

## P4. Mini inference runtime in C++ [R] (core), [A]
**Build:** A small C++ runtime for one model (GPT-2 small or a ~100M Llama-style model) built from your P1/P2 kernels:
- weight loader (GGUF or safetensors)
- static memory planner (liveness analysis → reuse one arena for activations)
- KV cache: first contiguous, then **paged** (fixed-size blocks + block table)
- operator fusion (RMSNorm+matmul, SiLU·mul)
- thread pool
- a request scheduler with **continuous batching** and token-boundary preemption (reuse the design from your ANE repo's scheduler).

Verify numerics against a PyTorch reference using per-token logit diffs.

**Questions it answers:**
- Design an inference runtime for a new AI processor. What are the components, and what does the hardware abstraction look like?
- How much memory does the KV cache need? Do the per-token math: 2 × layers × kv_heads × head_dim × bytes. Llama-3-8B with GQA is ≈128 KB/token in fp16. How many concurrent 8k-context users fit in 64 GB?
- Paged attention: what problem does it solve? What about prefix caching?
- Continuous vs static batching; the trade-off between TTFT and throughput; chunked prefill.
- Speculative decoding: when does it help on a bandwidth-bound device?
- How do you plan memory for a static graph vs dynamic shapes?
- How do you test a runtime so numerical regressions are caught? (Golden logits, tolerances, fixed-seed traces.)

---

## P5. IREE end-to-end, x86 and RISC-V [R] (core, the JD says "especially"), [A]
**Build:**
1. Export a small model (MLP → one transformer block → small LM) from PyTorch with `iree-turbine` (or ONNX → `iree-import-onnx`).
2. `iree-compile` for `llvm-cpu` on x86. Run with `iree-run-module` and benchmark with `iree-benchmark-module`.
3. Cross-compile for `riscv64` with RVV (`--iree-llvmcpu-target-triple=riscv64`, `+v` CPU features) and run under QEMU user mode.
4. Dump intermediate IR (`--mlir-print-ir-after-all` or `--compile-to=` stages). Follow one matmul through: torch → linalg → flow dispatch regions → stream → HAL executables → vectorized LLVM.
5. Compare the same model against llama.cpp, and against your P4 runtime.

**Questions it answers:**
- Explain IREE's architecture: the compiler phases (Input → Flow → Stream → HAL → VM), the runtime (HAL drivers, devices, executables, VM/bytecode), and async execution.
- What is a dispatch region and how does IREE form them (fusion decisions)?
- How would you bring up IREE on a new RISC-V AI processor? Would you use a new HAL driver or the CPU backend with custom codegen? How would you add custom instructions via microkernels (`ukernels`)?
- MLIR basics: dialects, progressive lowering, linalg tiling and vectorization.
- How does IREE handle dynamic shapes? What's the cost?
- IREE vs llama.cpp vs ONNX Runtime vs TVM: when would you choose each?
- Debugging: an IREE-compiled model gives wrong outputs on RISC-V but not on x86. How do you bisect it? (Per-dispatch output comparison, disabling fusions, checking the vector-length-agnostic tail handling.)

---

## P6. RISC-V vector kernels + an analytical performance model [A] (preferred quals), [R]
**Build:** Write the P2 quantized dot product and a GEMV with **RVV intrinsics**. Run them on QEMU, plus Spike if you want an instruction count. Write them vector-length agnostic (`vsetvli`, tail handling, try LMUL = 1/2/4/8). Since QEMU timing is meaningless, build a simple **analytical model** in Python instead: count instructions and bytes per kernel, then predict cycles for an assumed core (VLEN, issue width, L1/L2 bandwidth, DRAM BW). Compare its predictions against your measured x86/ARM numbers to calibrate. Optional stretch goal: a tiny trace-driven cache simulator (set-associative, LRU) fed by your GEMM's address stream.

**Questions it answers:**
- RVV vs fixed-width SIMD (AVX/NEON): vector-length-agnostic programming, LMUL, tail/mask policy. Why is this good for a chip vendor?
- What custom extensions would you add to a RISC-V core for LLM inference? (Matrix extension, int4 dot, dequant support, bigger L2/scratchpad, prefetch hints.) Justify each one with numbers from your model.
- Memory consistency: RVWMO vs x86-TSO; what fences you need in a lock-free ring buffer (tie to your zero-copy IPC work at Ford).
- Cache coherence (MESI/MOESI), false sharing, and what happens to a multi-core KV-cache update.
- Pipeline basics: hazards, out-of-order execution, why gather/scatter is slow, branch misprediction in sampling code.
- "Estimate tok/s for model X on a hypothetical chip with Y cores, Z GB/s and W TOPS." Your model gives the answer, and that's what the Architect role does day to day.

---

## P7. Benchmark + evaluation harness, production style [R] (core), [A]
**Build:** A harness that runs P3/P4/P5 backends on a frozen prompt set and records:
- TTFT, inter-token latency (p50/p95/p99) and throughput
- memory peak
- quality metrics: perplexity, KL vs reference, task accuracy
- performance counters

Add **regression detection** for CI: statistically compare against a baseline using repeated runs, confidence intervals and a noise threshold, and flag flaky runs. Your ANE repo's evidence registry is a good starting point.

**Questions it answers:**
- How do you benchmark inference fairly? (Warmup, CPU frequency pinning, thread affinity, cold vs warm cache, separating prefill from decode.)
- How do you handle non-deterministic outputs in tests? (Greedy/fixed seed, logit-level tolerances, distribution metrics instead of exact string match.)
- A perf test is flaky by ±8 %. What do you do? (This is in the Runtime JD: "drive improvements to test suite efficiency".)
- What do you monitor in production inference, and what alerts would you set up?
- How do you prove an optimization didn't silently hurt quality?

---

## P8. Parallelism on one box (optional) [R]
**Build:** In your P4 runtime, split matmuls tensor-parallel across 2 processes using shared memory, with a hand-written all-reduce. Simulate a small MoE layer with expert parallelism and router load imbalance.

**Questions it answers:**
- Data vs tensor vs pipeline vs expert parallelism: what gets communicated, and when is each worth it?
- All-reduce cost model (ring: 2(n−1)/n × data / bandwidth).
- MoE inference: why it's bandwidth-friendly per token but hard on memory capacity; expert load imbalance.
- Pruning: structured vs unstructured, and why unstructured sparsity rarely speeds up CPUs.

---

## Quick-fire questions to rehearse (no build needed)
- Draw the transformer decoder block and label FLOPs and bytes for each op. What does GQA/MQA change? FlashAttention: why does tiling help?
- RMSNorm vs LayerNorm; RoPE; why softmax needs max-subtraction.
- Little's law applied to inference serving.
- Explain the cache hierarchy of the CPU you used, with numbers (sizes, latencies, bandwidth).
- What's the difference between latency-bound, bandwidth-bound and compute-bound, and how do you tell from counters?
- Behavioural: a time you disagreed with a hardware team; a performance bug that took weeks; a time a measurement disproved your hypothesis (use the ANE chunking result, +48 %, or the sigmoid finding).

## Reuse what you build
Put P1–P7 in one public repo with one README table of predicted vs measured numbers. That table becomes a CV line ("built a CPU/RISC-V LLM inference stack; decode within X % of the bandwidth bound") and a ready answer to "show me something you've built".
