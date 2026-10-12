# Hardware labs: a plain-language guide

These labs answer one question from many angles: **what limits how fast a computer can run a language model?** You don't need to trust any claim here. Every number comes from code you can run on this Mac in under a minute, and every lab follows the same three steps:

1. **Predict.** Write the number down on paper first, from a few facts about the chip.
2. **Measure.** Run the lab.
3. **Explain the gap.** The difference between the guess and the measurement is where the learning is. It is also exactly what both interviews test ([interview plan](../interview-prep-plan.md); **[A]** = Performance Architect, **[R]** = Runtime Engineer).

How the labs differ on other chips (generic ARM64, Qualcomm, x86/Windows, FPGA) is covered in **[other-chips.md](other-chips.md)**.

---

## Start here: 60 seconds

```bash
cd "../ANE-GPU-CPU-p-series"            # the p-series worktree
python3 labs/run_labs.py --quick         # all CPU labs, about 35 seconds
```

It prints a **prediction** before each lab, the lab's own output, a **predicted vs measured** table, and a short **"What to notice"** explaining the result. Everything is saved to `results/labs-<time>/` (`summary.md` holds the table and a roofline chart).

| Recipe | What it does | Time |
| --- | --- | --- |
| `python3 labs/run_labs.py --quick` | every CPU lab, short budgets | ~35 s |
| `python3 labs/run_labs.py` | every CPU lab, 1 s per case | ~3.5 min |
| `python3 labs/run_labs.py --seconds 3` | longer runs, steadier numbers | ~10 min |
| `python3 labs/run_labs.py --only gemv --watch --interval 0.25` | one lab, a live line 4× per second | ~1 min |
| `python3 labs/run_labs.py --soak 120` | all P-cores flat out for 2 min, one line per second (does it slow down when hot?) | 2 min |
| `python3 labs/run_labs.py --only llm` | predict and measure your **gRPC_llm** model servers (start them with `make llm-up` in gRPC_llm) | ~2 min |
| `bash labs/portable/compare_native_vs_linux.sh` | same program, macOS vs Linux on the same chip | ~1 min |

**Before you trust any number**, look at the first lines the runner prints. It lists other apps using the CPU and the GPU. During the runs in this guide, Rider and Unreal Editor were each using a full CPU core, and the GPU was 47–86 % busy with no model request in flight (probably Unreal, which renders with Metal). The runner works around the busy CPU cores automatically, but it can't free the GPU. Close those apps for numbers you want to quote.

---

## The machine in one picture

```
 Apple M4 Max (this Mac): 10 performance cores in 2 groups of 5, 4 efficiency cores
 ┌──── P-group 0 ────┐ ┌──── P-group 1 ────┐ ┌─── E-group ───┐
 │ 5 cores           │ │ 5 cores           │ │ 4 cores       │  each core: its own 128 KB L1 data cache
 │ shared L2 16 MB   │ │ shared L2 16 MB   │ │ shared L2 4 MB│
 └─────────┬─────────┘ └─────────┬─────────┘ └──────┬────────┘
           └──────── system-level cache (SLC) ──────┘ ◄── GPU (32 cores) and Neural Engine use the same memory
                               │
                   36 GB unified memory, ~410 GB/s on paper
```

**Unified memory** means the CPU, GPU and Neural Engine all read the *same* RAM. There is no separate graphics card with its own memory. That's why a 9 GB model can run on the GPU without being copied, and also why every chip on the SoC competes for the same bandwidth.

What the labs measured (2026-10-12, with the busy apps noted above):

| | Paper prediction | Measured | Lab |
| --- | ---: | ---: | --- |
| One core, one chain of dependent maths | 9.0 GFLOP/s | 9.3 | peak |
| One core, best case | 144 GFLOP/s | 119 | peak |
| Independent chains needed to keep a core busy | 16 | 16 | peak |
| Inferred clock speed | 4.5 GHz (reported max) | ≈4.6 GHz | peak |
| Fetch from L1 cache | 0.67 ns | 0.77 ns | latency |
| Fetch from main memory (DRAM) | ~100 ns | 111–132 ns | latency |
| Memory bandwidth, all CPU cores | 410 GB/s (spec) | ~224 GB/s | bandwidth |
| 8-bit quantization error (block 32) | 0.0047 | 0.0053 | quant |
| 4-bit quantization error (block 32) | 0.0854 | 0.0969 | quant |
| Qwen2.5-14B decode on the GPU | 48 tok/s | 31.5 (66 %) | llm |
| Qwen2.5-3B (MLX) decode on the GPU | 236 tok/s | 125–135 (53–57 %) | llm |

The rest of this guide explains each row: why the guess was right, or why it missed.

---

## Lab 1: `peak`, how much maths can one core do?

**In one sentence:** a core can start several multiply-adds every clock tick, but only if they don't wait on each other.

**Interview questions it answers:** "Your kernel runs at 30 % of peak, why?" [A] · "What's a register-blocked micro-kernel and why does it exist?" [A] · "Why does scaling stop at 8 cores?" [A][R]

**Picture it.** A multiply-add (FMA) is like a dishwasher cycle: it takes 4 minutes (cycles), but the kitchen has 4 dishwashers. If each load needs the previous load's clean plates, you use one machine and wait 4 minutes each time. If you have 16 independent loads, all 4 machines run constantly and a load finishes every quarter-minute: **16× more work from the same kitchen.**

**Predict.**
- One dependent chain: one FMA every 4 cycles. One NEON FMA does 4 multiply-adds = 8 FLOPs. 8 FLOPs ÷ 4 cycles × 4.5 GHz = **9 GFLOP/s**.
- Best case: 4 pipes × 8 FLOPs × 4.5 GHz = **144 GFLOP/s**, reached at 4 pipes × 4 cycles = **16 chains**.

**Run.** `labs/build/p1_peak --part chains --seconds 1`

**What you see.** 1 chain: 9.3 GFLOP/s. 16 chains: 119 GFLOP/s, **13× faster with identical instructions**. Half precision (fp16) gives exactly 2×, because a 128-bit register holds 8 fp16 numbers instead of 4 fp32.

**Why the best case lands at 82 % of the guess:** the 4.5 GHz clock and "4 FMA pipes" are public estimates, not Apple specs. Sustaining all four every cycle also needs the loop overhead to fit around them. The 1-chain result (9.3 measured vs 9.0 predicted) implies a **4.6 GHz** clock, so the clock guess was right. The shortfall is in keeping all pipes fed.

**The thread surprise** (`--part threads`): per-core speed was 118 (1 thread), 112 (4), **96 (8)**, then collapsed to **44 (10)**. The clock drops a little when all cores are busy (118 → 96), but not by 60 %. The real reason is the **barrier effect**: every step waits for the slowest thread, and with two other apps each holding a core, the 9th and 10th threads had to share a core. One slow worker holds up the whole team. **Try it:** close Rider and Unreal and rerun. If 10 threads then beat 8, the explanation holds.

**A bug this lab caught.** The first version wrote the chain as `b + acc×a`. The compiler added a register copy to every step (`fmla` + `mov`), which made it look as though the clock were 2.6 GHz. Writing `acc + a×b` gives one instruction per step. Lesson: **look at the assembly before trusting a speed test**:
```bash
objdump -d --no-show-raw-insn labs/build/p1_peak | awk '/fma_chains_f32ILi1EE.*>:/,/ret/'
```

**What's missing on a Mac.** Linux's `perf stat` (cycle counts, instructions per cycle, cache misses) doesn't exist on macOS, and Docker's Linux VM can't see the hardware counters either. The Apple equivalent is Instruments → CPU Counters (`xctrace`). Until that's wired in, the clock is *inferred*, not read.

---

## Lab 2: `latency`, how far away is the data?

**In one sentence:** data in the nearest cache arrives about 150× faster than data from main memory.

**Interview questions it answers:** "Explain your CPU's cache hierarchy, with numbers" [A] · "How does prefetching interact with your access pattern?" [A] · "Little's law" [A][R]

**Picture it.** L1 is the desk, L2 the bookshelf, DRAM the library across town. The lab plays treasure hunt: each note says where the next note is, so you can't fetch two at once. The time per note is the distance to wherever the notes are kept.

**Predict.** L1 ≈ 3 cycles ÷ 4.5 GHz = **0.67 ns**. DRAM ≈ **100 ns** (typical for phone and laptop memory).

**Run.** `labs/build/p1_latency --seconds 0.5`

**What you see.** 0.77 ns up to 112 KB (L1 is 128 KB); ≈7 ns in L2; a big jump past ~12–16 MB, where the L2 runs out; **111 ns** in DRAM (132 ns when the machine was busier). The "sequential" column follows the same notes in order. It stays near 9 ns per line even at 512 MB, because the **prefetcher** sees the pattern and fetches ahead.

**Why it matters.** To stream 78 GB/s from memory that takes 111 ns to answer, a core must have about 8.7 KB (68 cache lines) **in flight at once**. That's Little's law: *amount in flight = rate × wait time*. A core can only track so many outstanding requests, which is why one core can't use the whole memory system.

**On other chips.** Apple uses 128-byte cache lines and 16 KB memory pages; almost every other CPU (x86, Qualcomm, generic ARM64) uses 64-byte lines and 4 KB pages. See [other-chips.md](other-chips.md).

---

## Lab 3: `bandwidth`, how fast can bytes flow?

**In one sentence:** one core can pull ~78 GB/s from memory; the CPU cores together top out at ~224 GB/s, about half of what the chip can deliver.

**Interview questions it answers:** "Measure your machine's peak bandwidth" [A] · "Why does scaling stop?" [A][R]

**Predict.** Apple's spec: **410 GB/s**. L1 per core: 3 loads × 16 bytes × 4.5 GHz ≈ **216 GB/s**.

**Run.** `labs/build/p1_bandwidth --seconds 1`

**What you see.** One core: L1 ≈160, L2 ≈106, DRAM ≈78 GB/s. Adding cores: 1 → 78, 4 → 214, 6 → **224**, then flat. The 4 efficiency cores alone manage ≈32 GB/s.

**Why the gap from 410.** The 410 GB/s is for the *whole chip*. The CPU clusters connect to memory through links that can't carry all of it. The GPU can. This one fact explains most of the "CPU vs GPU" difference in LLM speed later on.

**Tie to the ANE story (P0).** The Neural Engine streamed weights at ≈90 GB/s, about **one CPU core's worth**, or 22 % of the chip's bandwidth (the earlier R1 note said "one sixth", which assumed the 16-core chip). That is the main reason it decoded 3.5× slower than the GPU.

---

## Lab 4: `gemm`, multiplying matrices from scratch

**In one sentence:** the same multiplication gets **~1,900×** faster through access order, cache-sized tiles, vector instructions, threads, and finally Apple's matrix unit.

**Interview questions it answers:** "How do you choose tile sizes?" [A] · "Why is decode memory-bound and prefill compute-bound?" [A][R] · "What does a batch of 8 change?" [A]

| Step | What changed | Plain-language reason | Speed (1024×1024) |
| --- | --- | --- | ---: |
| naive | textbook three loops | walks down a column, so every multiply touches a new cache line | 1.5 GFLOP/s |
| ikj | swap two loops | walks along rows, so each fetched cache line is fully used | 21× |
| blocked | work on cache-sized tiles | reuse a tile while it's on the desk instead of going back to the library | 19× (no gain here: the matrix already fits in L2; try `--n 2048`) |
| packed + NEON | copy tiles into tidy strips; an 8×8 block of the answer lives in 16 registers | 16 independent multiply-adds per step, exactly Lab 1's "16 chains" | 69× |
| 10 threads | split the work | more cores | 349× |
| Accelerate | Apple's library | uses the **SME matrix unit**, a separate engine with a much higher ceiling than NEON | **1,929×** (≈2.9 TFLOP/s) |

**The most important result is Part 2 ("shapes").** The same weights (1024 → 3072, one Qwen3-0.6B layer) are multiplied by M tokens at once:
- **M = 1** (generating one token, "decode"): every weight is fetched for **one** multiply-add. That's 0.5 FLOP per byte, so the speed is set by memory: **213 GFLOP/s**.
- **M = 512** (reading a prompt, "prefill"): every weight fetched is used **512 times**. The speed is set by compute: **2,967 GFLOP/s**.

**Same weights, 14× faster, just by batching.** This is half of the ANE story: its runtime read the prompt one token at a time, which turned prefill into decode.

**Prediction that "failed" on purpose.** The paper prediction for M=1 from DRAM is 0.5 × 224 ≈ 112 GFLOP/s. Measured: 213. The 12 MB test matrix stays in the cache between repeats, so it never comes from DRAM. Lab 5 fixes that by using a model-sized set of weights.

---

## Lab 5: `gemv`, generating a token is a bandwidth problem

**In one sentence:** tokens per second ≈ memory bandwidth ÷ size of the weights.

**Interview questions it answers:** "Max tok/s for a 7B Q4 model (~4 GB) at 80 GB/s?" (≈20) [A] · "How close is your kernel to the bandwidth bound?" [A]

**Predict.** Best case = the measured memory roof (~224 GB/s). One core: 4 chains × 4 numbers ÷ 4 cycles × 4.5 GHz ≈ 18 G weights/s = **72 GB/s** (a maths limit, not a memory one).

**Run.** `labs/build/p1_gemv --seconds 1`

**What you see.**
- A plain C loop: 8.6 GB/s. The compiler is not allowed to reorder floating-point additions, so the loop stays one slow chain (Lab 1 again).
- NEON, one core: 67 GB/s, close to the 72 predicted. **One core is limited by maths, not memory.**
- NEON, 4 threads: **201 GB/s, 90 % of the memory roof**. Beyond this, nothing helps except reading fewer bytes.
- fp16 weights (half the bytes): one core is *slower* in GB/s (42), because converting fp16 to fp32 costs extra instructions. With 8 threads both formats hit the roof, and fp16 then gives **almost 2× the tokens per second**.

**Translated to real models:** at ~180 GB/s, a 4-billion-parameter fp16 model (8 GB) would run at ≈22 tok/s on these CPU cores, and a 4-bit version (2.25 GB) at up to ≈79. The same formula explains the ANE's 11.3 tok/s (90 GB/s) and the GPU's 40 tok/s in the R4 work.

---

## Lab 6: `quant`, making numbers smaller

**In one sentence:** storing weights in 4 or 8 bits instead of 16 or 32 means fewer bytes to move, at the cost of some rounding error.

**Interview questions it answers:** "Block vs per-tensor vs per-channel?" · "Why is 4-bit faster despite extra work?" · "GPTQ, AWQ, SmoothQuant?" · "Why do activation outliers make W8A8 hard?" · "Is the quantized model good enough?" [A][R]

**How the formats work (llama.cpp style).** Weights go in **blocks of 32**. Each block stores one scale number (16 bits) and 32 small integers. **Q8_0** = 8.5 bits per weight. **Q4_0** = 4.5 bits per weight (two weights per byte).

**Predict the error.** A block's step size = biggest value ÷ number of levels. Rounding error averages step ÷ √12. The biggest of 32 random normal numbers is about 2.07, so the predicted error is 2.07 ÷ 127 ÷ √12 = **0.0047** for 8-bit, and 2.07 ÷ 7 ÷ √12 = **0.085** for 4-bit. **Measured: 0.0053 and 0.097**, about 13 % higher, because the scale itself is stored in 16 bits and adds a little rounding of its own.

**Run.** `labs/build/p2_quant --seconds 1`

**What you see, and why.**
- **Block size:** with a few outliers (very large weights), 4-bit error is **0.13 with blocks of 32, but 0.69 with blocks of 4096**. One giant weight sets its block's scale and squashes every neighbour toward zero. Small blocks contain the damage, at a cost of 0.5 extra bits per weight. That's why llama.cpp uses 32.
- **SmoothQuant** (8-bit weights *and* 8-bit activations): real models have a few input channels with huge values. With one scale for the whole activation tensor, those channels take all the levels. Moving that range into the weights cuts the error from **0.069 to 0.016**.
- **AWQ-style scaling** (4-bit weights only): give important channels more of the 4-bit grid before rounding. Error falls from **0.099 to 0.047** at α = 0.25. Too much scaling makes it worse again; AWQ searches for the best value per layer.
- **Speed** (7 threads): Q8_0 is **1.65×** faster than fp16 (bytes predict 1.88×); Q4_0 is **2.31×** (bytes predict 3.56×). Q4 falls short because unpacking two 4-bit numbers from each byte costs instructions. Even at 8 threads it moves only 126 GB/s against a ~220 GB/s roof. **4-bit decode on these cores is limited by instructions.** That's the argument for a custom "unpack and dot product" instruction in P6.
- **One core:** Q8_0 is 2.5× faster than fp16. The `sdot` instruction does 16 small multiply-adds at once, so smaller numbers mean cheaper maths too, not just fewer bytes.

**What's missing.** "Good enough" can't be judged from random weights. It needs perplexity on fixed text and the R3 30-case task set, on a real model, with the acceptable loss **written down before** the run. GPTQ is also not built yet (planned in NumPy on one real layer).

---

## Lab 7: `llm`, your gRPC_llm project under the microscope

**In one sentence:** read the model file, predict its speed from bytes and FLOPs, then measure the real server and explain the difference.

**Interview questions it answers:** "How close is your runtime to the hardware bound, and what accounts for the rest?" [A] · "How much memory does the KV cache need?" [R] · "Prefix caching?" [R] · "How do you benchmark fairly?" [R] · "How did you build the cost model, and how would you validate it?" (P0) [A][R]

**Run** (with the tiers up: `make llm-up` in gRPC_llm):
```bash
python3 labs/run_labs.py --only llm                 # both tiers, 3 runs each
python3 labs/run_labs.py --only llm --llm-seconds 60  # plus 60 s of sustained decode
```
The lab only sends ordinary `Generate` calls and writes nothing inside gRPC_llm. It imports gRPC_llm's generated protobuf stubs and nothing else.

**Step 1, predict (from the model file alone).** The lab opens the GGUF or safetensors file, reads every tensor's size, and works out:

| | Qwen2.5-14B Q4_K_M (llama.cpp) | Qwen2.5-3B 4-bit (MLX) |
| --- | ---: | ---: |
| Bytes read per new token | 8.54 GB (all weights except the embedding table) | 1.74 GB (embedding is shared with the output, so all of it) |
| Decode ceiling = 410 GB/s ÷ bytes | **48 tok/s** | **236 tok/s** |
| Prefill ceiling = GPU FLOP/s ÷ (2 × parameters) | **474 tok/s** | **2,268 tok/s** |
| KV cache = 2 × layers × KV heads × 128 × 2 bytes | 192 KB per token, 3.2 GB at 16k context | 36 KB per token, 302 MB at 8k |

(The GPU's FLOP/s isn't published by Apple; the lab assumes 14 TFLOP/s for a 32-core M4 Max and says so. Both models use **GQA**, where several query heads share one key/value head. That shrinks the KV cache 5× and 8× respectively.)

**Step 2, measure.** Each prompt starts with a unique line so that no cache can help, which makes the measurement fair. The newer gRPC_llm server reports exact token counts and its own timings in a `Usage` message; the lab prints both client and server numbers.

| | Decode | % of ceiling | Prefill | % of ceiling |
| --- | ---: | ---: | ---: | ---: |
| 14B, llama.cpp/Metal | 31.5 tok/s | 66 % | 330 tok/s | 70 % |
| 3B, MLX/Metal | 125–135 tok/s | 53–57 % | 1,236–1,691 tok/s | 54–75 % |

**Step 3, explain the gap.**
- 410 GB/s is a peak. GPUs usually sustain 70–90 % of it on a pure stream.
- Each token also reads the KV cache and runs attention, normalisation and sampling.
- **Fixed costs per token** (launching GPU work, Python in the engine, streaming over gRPC) weigh more on the small model: its token takes ~8 ms, so 1 ms of overhead is 12 %. Client and server decode rates agree within ~1 tok/s, so gRPC itself adds almost nothing.
- **The GPU was shared.** It was 47–86 % busy before any request, and in one run the 14B model swung between 28 and 9 tok/s. The lab now checks GPU load first and refuses to fit a cost model from unstable runs.

**Two findings about gRPC_llm itself:**
1. **Prefix caching differs by engine.** Sending the same 1,851-token prompt twice: llama.cpp answered the repeat in **0.05 s instead of 5.7 s**, because it reused the cached prompt. MLX took 1.4 s both times; the server confirms **0 of 1,865 prompt tokens reused**. Agent loops that resend a long system prompt pay the full prompt cost on every turn on the MLX tier. That is an improvement worth making there.
2. **The same model on a CPU in Docker** (from gRPC_llm's earlier logs: Qwen2.5-3B Q5_K_M): 39 tok/s decode and 131 tok/s prefill, against **91 and 1,529** on Metal. Decode drops 2.3× (CPU bandwidth instead of GPU bandwidth); prefill drops **12×** (CPU maths instead of GPU maths). That's the decode/prefill split from Lab 4, on a real service.

**Cost model.** When two tiers give stable numbers, the lab fits *time per token = bytes ÷ B + c* (the same per-byte + fixed-cost model as the ANE R1 work) and rejects impossible fits (c < 0 or B above the spec). A stable run gave **B ≈ 265 GB/s** (about 65 % of the 410 GB/s spec actually used for streaming) and **c ≈ 1.7 ms** of fixed cost per token. That c is 6 % of a 14B token but 21 % of a 3B token, which is why the small model sits further below its ceiling. Caveats: two engines, a busy GPU. An earlier, unstable run produced a negative c and was rejected. Validating the model means predicting a *third* model, then checking.

---

## Lab 8: `portable`, same chip, different operating system

**In one sentence:** the same C++ file, run on macOS and on Linux on the *same* M4 Max, shows what the operating system adds or takes away.

**Run.** `bash labs/portable/compare_native_vs_linux.sh 0.5`

`portable_lab.cpp` is standard C++ with one fast path per instruction set (NEON for ARM, AVX2 for x86, plain C otherwise), which is how llama.cpp supports many chips. It compiles on macOS, Linux and Windows (instructions at the top of the file).

| | macOS native | Linux in Docker's VM |
| --- | ---: | ---: |
| Memory page size | 16 KB | 4 KB |
| Memory bandwidth (best) | 190 GB/s | 132 GB/s |
| GEMV (best) | 86 GB/s | 64 GB/s |
| int8 dot product, one core | 126 GOP/s | 121 GOP/s |
| CPU features visible | NEON, dotprod, i8mm, bf16, **SME** | NEON, dotprod, i8mm, bf16, *no SME* |

**Why.** The maths is identical (same cores). The **memory path** is 30 % slower in the VM: 4 KB pages mean four times as many address translations as 16 KB pages, and a virtual machine adds a second translation layer on top. The VM also hides SME (so no Accelerate-style matrix unit), Metal (no GPU) and the Neural Engine. That's why gRPC_llm's Docker tier runs on the CPU only.

**x86 too (correctness only).** `arch -x86_64` runs the AVX2 build through Rosetta: the results are correct (the int8 check passes) and the program sees 4 KB pages, but the timings are emulation, not x86 hardware. For real x86 numbers, see [other-chips.md](other-chips.md).

**Lesson inside the code.** This program creates threads for every layer, so with many threads its GEMV stalls (86 GB/s), while `p1_gemv`, which keeps a pool of threads alive, reaches 200 GB/s. Creating threads is not free.

---

## The roofline: one chart that ties it all together

```
speed (GFLOP/s, log)
       |                     a  a       a = Accelerate GEMM (Apple's matrix unit: its own higher ceiling)
   546 |       /--------------p------   - = compute ceiling (Lab 1)
       |     //   q       p             / = memory ceiling = bandwidth × FLOPs-per-byte (Lab 3)
   127 |   v/  q     p                  p = our GEMM, M = 1 … 512 tokens
    30 |  / q                           v = GEMV (decode), q = quantized GEMV
       +-------------------------------
         0.1   1    10   100   FLOPs per byte
```

Any program sits somewhere on this chart. **On the slope:** memory-bound, so only fewer bytes (quantization) or more reuse (batching) help. **On the flat:** compute-bound, so better kernels or a matrix unit help. **Below both lines:** you are wasting the machine, through dependency chains, extra instructions or threads waiting at barriers. The runner draws this chart from your run.

---

## What is missing (honest list)

| Missing | Why it matters | Plan |
| --- | --- | --- |
| Hardware counters (cycles, IPC, cache misses) | The plan asks for `perf stat`; macOS has no `perf`, and the Docker VM sees no counters | Instruments / `xctrace` CPU Counters wrapper |
| Real x86 timings | P1, P2 and P5 mention AVX2/AVX-512/VNNI | `portable_lab.cpp` already compiles for x86; run it on an x86 Linux/Windows box |
| GPTQ, perplexity, task-accuracy gate for Q8/Q4 | "Is it good enough?" | P2 next slice, on Qwen3-0.6B |
| llama.cpp profiling (P3 proper) | Op-level time breakdown, thread scaling | Build pinned llama.cpp, `llama-bench`, Instruments Time Profiler |
| IREE, RISC-V, runtime, CI harness (P4–P7) | See the interview plan | Order: P5 → P4 → P6 → P7 |
| Quiet-machine numbers | All results here were taken with Rider, Unreal and a busy GPU | Rerun with those closed before quoting |

## Exercises

1. Close Rider and Unreal, then run `python3 labs/run_labs.py --seconds 2`. Which predictions moved closer?
2. `labs/build/p1_bandwidth --part threads --threads 1,2,3,4,5,6,7,8`: does the 6th thread (the second P-group) help?
3. `labs/build/p1_gemm --part steps --n 2048`: does cache blocking finally beat the simple loop swap?
4. Give `gemv::f32` 8 accumulators: does one core reach the predicted ~72 GB/s?
5. Process 4 blocks per loop in `dot_q4_0_neon`: does single-core Q4 catch up with Q8?
6. Run the `llm` lab twice without changing anything. How different are the medians? That difference is the noise any "optimisation" must beat (P7).

## Where the code is

```
labs/run_labs.py            runner: checks, build, tests, predict → measure → explain, roofline, results
labs/predictions.py         every paper prediction and the facts it uses (edit SPEC to predict another chip)
labs/common/                timing harness, thread pool (QoS → P/E cores), sysctl hardware info
labs/p1/                    peak, latency, bandwidth, gemm, gemv (+ kernels in *_kernels.h)
labs/p2/                    Q8_0 / Q4_0 formats and the quant lab
labs/p3/llm_service_lab.py  gRPC_llm: GGUF/safetensors reader, predictions, gRPC measurement, cost model
labs/portable/              one-file portable lab (NEON / AVX2 / scalar) + macOS-vs-Linux script
labs/tests/                 55 correctness checks, run before every benchmark
```
