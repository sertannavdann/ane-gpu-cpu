# Glossary: every term in the labs, in plain words

You can read the lab guide without running anything. When a word or abbreviation stops you, look it up here. Entries are grouped by topic, and each group builds on the one before it, so reading straight through also works as a short course.

---

## 1. The chip and its parts

**SoC (system on a chip).** One piece of silicon holding the CPU, GPU, Neural Engine, memory controllers and more. Apple's M-series, Qualcomm Snapdragon and phone chips are SoCs. A desktop PC instead spreads these over separate chips and cards.

**CPU (central processing unit).** The general-purpose processor that runs programs instruction by instruction. It is made of several **cores**, each able to run its own thread.

**Core.** One independent processor inside the CPU. The M4 Max here has 14.

**P-core / E-core (performance / efficiency core).** Apple's two kinds of core. P-cores are big and fast; E-cores are small, slower and use much less power. This Mac has 10 P-cores (in two groups of 5) and 4 E-cores.

**Thread.** A sequence of instructions the operating system can schedule onto a core. Running a job on 8 threads means splitting it into 8 pieces that run at the same time.

**Clock / GHz (gigahertz).** How many steps per second a core takes: 4.5 GHz = 4.5 billion steps ("cycles") per second.

**Cycle.** One tick of the clock. "4 cycles" at 4.5 GHz is about 0.9 nanoseconds.

**GPU (graphics processing unit).** A processor with thousands of simple arithmetic units, built for doing the same operation on lots of data at once. Originally for graphics, now the main engine for running AI models.

**NPU (neural processing unit) / ANE (Apple Neural Engine).** A processor designed only for neural-network arithmetic, usually low-precision numbers. Very power-efficient, but less flexible than a GPU. Apple's is the ANE; Qualcomm's is the Hexagon NPU.

**Unified memory.** One pool of RAM shared by the CPU, GPU and NPU on an SoC. No copying between "CPU memory" and "graphics memory", but everyone competes for the same bandwidth.

**VRAM (video RAM).** The separate, fast memory on a PC graphics card. A model must fit in it to run fully on that GPU.

**PCIe (Peripheral Component Interconnect Express).** The slot and link that connect a graphics card to a PC. Much slower than VRAM or main memory, so moving model data over it is a bottleneck.

**HBM (high-bandwidth memory).** Memory stacked right next to a chip, giving very high bandwidth. Used on data-centre GPUs and high-end FPGA cards.

**FPGA (field-programmable gate array).** A chip whose circuits you configure yourself after manufacture. No fixed instruction set: you design the hardware. Used to prototype new chips.

**DSP slice / BRAM (block RAM).** Inside an FPGA: DSP slices are small built-in multiply-add units; BRAM is fast on-chip memory blocks.

**RISC-V.** An open, royalty-free instruction set that anyone can build chips for, and extend with their own instructions. The target of prototypes P5 and P6.

**ISA (instruction set architecture).** The vocabulary of instructions a processor understands: ARM64, x86-64 and RISC-V are three different ISAs. Software compiled for one doesn't run natively on another.

**ARM64 / AArch64.** The 64-bit ARM instruction set, used by Apple silicon, Qualcomm laptop chips, phones and many servers.

**x86-64 / win64.** The instruction set of Intel and AMD PC processors. "win64" means 64-bit Windows on such a processor.

---

## 2. Memory: why data location matters

**Cache.** A small, fast memory close to a core that keeps recently used data, so the core doesn't wait for main memory. There are several levels:

- **L1 (level 1).** The smallest and fastest, private to each core (128 KB per P-core here). About 3 cycles away.
- **L2 (level 2).** Larger and slower, shared by a group of cores (16 MB per P-group here). About 30 cycles away.
- **SLC (system-level cache).** A last cache shared by the whole SoC, including the GPU.

**DRAM (dynamic random-access memory) / RAM.** Main memory: large (36 GB here) but slow to reach, about 110 nanoseconds (~500 cycles).

**LPDDR / LPDDR5X (low-power double data rate).** The type of DRAM used in laptops and phones. The number is the generation.

**Cache line.** The chunk of memory moved between memory and cache in one go: 128 bytes on Apple chips, 64 bytes on most others. Asking for one byte brings in the whole line.

**Page.** The chunk of memory the operating system manages and maps: 16 KB on macOS for Apple silicon, 4 KB on most other systems.

**TLB (translation lookaside buffer).** A small cache of page addresses. Programs use "virtual" addresses that must be translated to physical ones; bigger pages mean fewer translations to cache.

**Latency.** How long one request takes from asking to answer (e.g. 110 ns for DRAM).

**Bandwidth.** How many bytes per second can flow when many requests are in flight. Measured in **GB/s** (gigabytes per second). Latency is how long one delivery takes; bandwidth is how many deliveries arrive per second.

**Prefetcher.** Hardware that spots a pattern in your memory accesses (for example "reading upwards in order") and fetches the next data before you ask. That is why reading in order is fast and jumping around is slow.

**Little's law.** *Amount in flight = rate × waiting time.* To get 90 GB/s from memory that takes 110 ns to answer, about 10 KB must be on its way at any moment. A core can only track so many outstanding requests, which caps one core's bandwidth.

**False sharing.** Two threads writing different variables that happen to sit in the same cache line. The line bounces between their cores and both slow down.

**Write-allocate.** When a program writes to memory not in cache, many CPUs first *read* that line, then modify it. A "write" can therefore move twice the bytes you expect.

**NUMA (non-uniform memory access).** On big multi-socket servers, each processor has "near" memory and "far" memory; far memory is slower. Not an issue on a single Mac chip.

**Memory controller.** The part of the chip that talks to DRAM. Its capacity sets the bandwidth limit.

---

## 3. Doing arithmetic fast

**FLOP (floating-point operation).** One addition or multiplication on decimal numbers. **GFLOP/s** = billion per second; **TFLOP/s** = trillion per second.

**FMA (fused multiply-add).** One instruction that computes `a × b + c`, i.e. 2 FLOPs. The basic building block of every AI kernel.

**Dependency chain.** A sequence where each step needs the previous step's result, so they can't overlap. One chain runs at the speed of the instruction's *latency*; many independent chains run at the core's *throughput*.

**Pipe / pipeline / execution unit.** The hardware that executes instructions. A core with 4 FMA pipes can start 4 FMAs every cycle, if there is independent work to give them.

**SIMD (single instruction, multiple data).** One instruction that works on several numbers at once (a "vector"). SIMD families: **NEON** (ARM, 128 bits = 4 × 32-bit numbers), **SVE** (Scalable Vector Extension, a newer ARM vector set with variable width, not on Apple chips), **AVX2 / AVX-512** (Advanced Vector Extensions, x86, 256 or 512 bits), **RVV** (RISC-V Vector extension).

**Register.** A tiny storage slot inside the core where arithmetic happens. NEON has 32 vector registers of 128 bits each.

**Intrinsic.** A C/C++ function that maps directly to one SIMD instruction, e.g. `vfmaq_f32` (NEON FMA on 4 floats).

**Scalar.** One number at a time; the opposite of SIMD.

**Matrix unit: SME / AMX.** Extra hardware that multiplies whole small matrices in one go. **SME** (Scalable Matrix Extension) is ARM's, present on the M4 and used by Apple's Accelerate library. **AMX** (Advanced Matrix Extensions) is Intel's (and also the name of Apple's earlier, undocumented unit).

**int8 dot-product instructions: sdot / i8mm / VNNI.** Instructions that multiply many 8-bit integers and add them up in one go. `sdot` (ARM) does 16 multiply-adds per instruction; `i8mm` (ARM int8 matrix multiply, instruction `smmla`) does a small 8-bit matrix product; **VNNI** (Vector Neural Network Instructions, x86, `vpdpbusd`) is Intel's equivalent. The reason low-bit models are cheap to compute.

**Accelerate / BLAS.** **BLAS** (Basic Linear Algebra Subprograms) is the standard interface for fast matrix maths. **Accelerate** is Apple's library implementing it, among other things, and it uses the SME unit. Others: OpenBLAS, Intel MKL.

**Metal.** Apple's programming interface for the GPU. **CUDA** is NVIDIA's; **Vulkan** and **DirectX** are cross-vendor or Windows equivalents.

**QoS (quality of service) class.** macOS's way of saying how important a thread is. It cannot pin a thread to a core; it only hints P-core or E-core.

**Affinity / pinning.** On Linux and Windows, binding a thread to a specific core. Not available on macOS.

**SMT (simultaneous multithreading) / Hyper-Threading.** Two threads sharing one core's units. Common on x86, absent on Apple chips.

**Barrier.** A point where all threads wait until every thread has arrived. The step takes as long as the *slowest* thread, which is the "barrier effect" seen in the labs.

**Thread pool.** Threads created once and reused, so each task doesn't pay the cost of starting new threads.

**Thermal throttling.** A chip slowing its clock when it gets too hot. It doesn't recover while the heavy load continues.

---

## 4. Measuring and modelling performance

**Roofline.** A chart with two ceilings: a flat one (the compute limit, FLOP/s) and a slope (the memory limit, bandwidth × intensity). A program can't go above either. Where it sits tells you what limits it.

**Arithmetic intensity (FLOP/byte).** How much maths a program does per byte it fetches. Low intensity → limited by memory ("memory-bound" or "bandwidth-bound"); high intensity → limited by maths ("compute-bound").

**Memory-bound / compute-bound / latency-bound.** What's holding a program back: waiting for bytes, doing arithmetic, or waiting on one step at a time.

**Microbenchmark.** A tiny program that measures one thing (one instruction's speed, one cache's latency).

**Median / p10 / p50 / p90 / p95 / p99.** Ways to summarise many measurements. The median (p50) is the middle value. p95 is the value 95 % of measurements are below, which describes the slow "tail". p10–p90 is the range that excludes the most extreme 10 % at each end.

**Spread (%).** (p90 − p10) ÷ median: how noisy a measurement is. Above ~10 % something is interfering.

**A/A test.** Running the same thing twice and comparing. Any difference you see is pure noise, and tells you how big a "real" improvement must be to count.

**Warm-up.** Untimed runs before measuring, so first-time costs (loading pages, filling caches, raising the clock) don't pollute the result.

**Noisy neighbour.** Another program sharing the machine and stealing cores, bandwidth or GPU time.

**Hardware counters / PMU (performance monitoring unit).** Counters inside the CPU that count cycles, instructions, cache misses and so on. Read with **perf** on Linux, **VTune** or **uProf** on x86, **Instruments / xctrace** on macOS.

**IPC (instructions per cycle).** How many instructions a core completes per clock tick on average. Low IPC means the core is mostly waiting.

**Cost model.** A formula predicting time from a few measured constants, e.g. *time per token = bytes ÷ bandwidth + fixed cost*. Validated by predicting a case it wasn't fitted on.

**powermetrics.** macOS's tool for measuring power (watts) per CPU, GPU and ANE. Needs administrator rights (sudo).

**J (joule) / W (watt).** Energy and power: 1 watt = 1 joule per second. "1 J per token" at 36 W means about 36 tokens per second.

---

## 5. How language models run

**LLM (large language model).** A neural network that predicts the next word piece of text, one at a time.

**Token.** A word or piece of a word, the unit a model reads and writes. Roughly ¾ of an English word.

**Parameters / weights.** The numbers inside a model, learned in training. "1.5B" = 1.5 billion parameters. Every one is read for every generated token.

**Tokenizer.** The component that turns text into tokens and back.

**Prefill (prompt processing, "pp").** Reading the prompt. All prompt tokens are processed together, so each weight is fetched once and used many times: **compute-bound**.

**Decode (token generation, "tg").** Producing the answer one token at a time. Each new token needs every weight fetched again: **memory-bound**. tokens/s ≈ bandwidth ÷ bytes of weights.

**tok/s (tokens per second).** Generation or prompt-processing speed.

**TTFT (time to first token).** How long until the first word of the answer appears: mostly the prefill time.

**GEMM (general matrix-matrix multiply).** Multiplying two matrices. Prefill is mostly GEMMs. Shapes are written M × K times K × N, where M is the number of tokens processed together.

**GEMV (general matrix-vector multiply).** Multiplying a matrix by one vector. Decode is mostly GEMVs (M = 1).

**Micro-kernel / packing / tiling (blocking).** Techniques for fast GEMM. *Tiling* splits matrices into cache-sized tiles; *packing* copies a tile into a tidy contiguous layout; the *micro-kernel* is the innermost loop that keeps a small block of the result in registers.

**Batching.** Processing several tokens or requests together so each weight fetched is reused. **Continuous batching** adds and removes requests on the fly.

**KV cache (key-value cache).** Saved intermediate results for every token so far, so the model doesn't recompute the past for each new token. Size = 2 × layers × KV heads × head size × bytes per number, per token.

**Attention / heads.** The part of a model that looks back at earlier tokens. It is split into several **heads** that each look for different patterns.

**GQA (grouped-query attention).** Several attention heads share one key/value head, which shrinks the KV cache.

**Prefix cache.** Reusing the KV cache of a prompt beginning that was already processed. A repeated system prompt then costs almost nothing.

**Context (window).** How many tokens the model can see at once (e.g. 16,384).

**Dispatch.** Handing one piece of work to an accelerator (GPU/NPU). Each dispatch has a fixed cost, which hurts when the pieces are small.

**Perplexity.** How surprised a model is by real text, a standard quality score. Lower is better. Used to check that quantization didn't hurt the model.

**Logits.** The raw scores a model gives every possible next token, before choosing one. Comparing logits between two runs catches small numerical differences.

---

## 6. Making models smaller: quantization

**Floating-point formats: fp32 / F32, fp16 / F16, bf16 / BF16.** Ways to store decimal numbers in 32 or 16 bits. **bf16** ("brain float") is a 16-bit format with fp32's range but less precision.

**int8 / int4.** Whole numbers stored in 8 or 4 bits.

**Quantization.** Storing weights (and sometimes activations) with fewer bits, plus a **scale** to convert back. Fewer bytes → faster decode, at the cost of rounding error.

**Block quantization.** Giving each small group of weights (a block, e.g. 32) its own scale, so one large weight only distorts its own block.

**Q8_0, Q4_0, Q4_K_M, Q5_K_M.** llama.cpp format names. The number is the bits per weight (8, 4, 5). `_0` means a simple block of 32 with one scale. `_K` formats ("k-quants") use blocks of 256 with smaller sub-blocks and extra scales. `_M` / `_S` mean medium or small quality (some layers kept at higher precision). Real bits per weight are a little higher than the name because of the scales: Q8_0 ≈ 8.5, Q4_0 ≈ 4.5.

**RTN (round-to-nearest).** The simplest quantization: just round each weight.

**Outlier.** A value much larger than the rest. One outlier in a block forces a large scale and ruins the precision of its neighbours.

**Activations.** The intermediate values flowing through a model while it runs (as opposed to the fixed weights).

**W8A8 / W4A16.** Shorthand for weight bits and activation bits: W8A8 = 8-bit weights and 8-bit activations; W4A16 = 4-bit weights, 16-bit activations.

**SmoothQuant.** A method that moves the difficulty of activation outliers into the weights (divide activations by a scale, multiply weights by it), so both quantize well to 8 bits.

**AWQ (activation-aware weight quantization).** Scales up the weight columns that matter most (judged by activation size) before 4-bit quantization, giving them more precision.

**GPTQ.** Quantizes weights one column at a time and adjusts the remaining weights to cancel the error, using statistics of real activations.

**QAT (quantization-aware training) / LoRA / QLoRA.** Training with quantization in the loop (QAT), or training small add-on matrices (LoRA), including on top of a quantized model (QLoRA).

---

## 7. Software, files and tools in these labs

**llama.cpp / ggml.** A popular C/C++ engine for running language models on CPUs and GPUs; **ggml** is its tensor library. **llama-bench** measures its speed; **llama-quantize** converts formats.

**GGUF.** llama.cpp's model file format: weights plus metadata (architecture, tokenizer) in one file.

**safetensors.** A common model file format (JSON header plus raw tensor data), used by MLX and Hugging Face.

**MLX.** Apple's machine-learning framework for Apple silicon, running on the GPU through Metal.

**Hugging Face.** A website that hosts models; the labs download Qwen's official files from it.

**Qwen2.5.** The model family used in the labs (by Alibaba), in sizes 0.5B to 14B.

**gRPC.** A way for programs to call each other over the network with typed messages. gRPC_llm's services talk this way; **protobuf** is its message format, and the **stubs** are generated code for calling it.

**Pinned (version / commit).** Fixed to one exact version, so results can be reproduced later. A **commit** is one saved state of a code repository, identified by a hash like `b2f036c`.

**Build / CMake / Ninja.** Compiling source code into programs. **CMake** generates the build instructions; **Ninja** runs them.

**Docker / container / VM (virtual machine).** Docker runs programs in isolated **containers**; on a Mac they run inside a Linux **VM**, a simulated computer that adds a layer between the program and the hardware.

**Rosetta.** Apple's translator that runs x86-64 programs on Apple silicon. Good for correctness checks, meaningless for timing.

**QEMU.** An emulator that runs programs for another processor type (e.g. RISC-V) in software. Correctness only; timing is meaningless.

**IREE / MLIR / HAL.** **IREE** is a compiler and runtime that turns models into code for many chips. **MLIR** is the compiler framework it is built on. **HAL** (hardware abstraction layer) is the part that hides chip differences. These are the subject of P5.

**Memory ordering: TSO / RVWMO.** Rules for when one core sees another core's writes. x86 uses **TSO** (total store order: fairly strict); ARM and RISC-V (**RVWMO**, RISC-V weak memory ordering) are looser, so lock-free code needs explicit fences.

**Cache coherence / MESI.** How cores keep their caches consistent when they share data. **MESI** (Modified, Exclusive, Shared, Invalid) is the classic protocol.

**VLA (vector-length agnostic).** Code that works whatever the hardware's vector width, as RISC-V RVV and ARM SVE allow.

**HLS (high-level synthesis).** Writing FPGA hardware in C/C++ instead of a hardware language.

**sudo.** Running a command with administrator rights; it asks for your password.

**.nosync.** A folder-name suffix that tells iCloud Drive not to sync that folder. Used here for large downloads.
