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

**Before you trust any number**, look at the first lines the runner prints. It lists other apps using the CPU and the GPU. This guide uses two sets of runs:

- **Quiet machine** (2026-10-12, 00:50–01:02): Rider and Unreal Editor closed, GPU 0–3 % busy. These are the numbers to quote. Full evidence: **[results-2026-10-12.md](results-2026-10-12.md)**.
- **Busy machine** (the evening before): Rider and Unreal Editor each held a full CPU core, and the GPU was 47–86 % busy with no model request in flight (Unreal renders with Metal). These numbers are kept wherever the difference teaches something, because "another app was running" is the most common reason a benchmark lies.

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

What the labs measured:

| | Paper prediction | Quiet machine | Busy machine | Lab |
| --- | ---: | ---: | ---: | --- |
| One core, one chain of dependent maths | 9.0 GFLOP/s | 10.5 | 9.3 | peak |
| One core, best case | 144 GFLOP/s | 133 (92 %) | 119 | peak |
| Independent chains needed to keep a core busy | 16 | 16 | 16 | peak |
| All 10 P-cores | 1,152 GFLOP/s (8 cores) | **1,150** (10 cores) | 438 | peak |
| Fetch from L1 cache | 0.67 ns | 0.67 | 0.77 | latency |
| Fetch from main memory (DRAM) | ~100 ns | 109 | 111–132 | latency |
| Memory bandwidth, one core | ? | 90 GB/s | 78 | bandwidth |
| Memory bandwidth, all CPU cores | 410 GB/s (spec) | 253 (62 %) | 224 | bandwidth |
| Decode-style GEMV, best | = bandwidth roof | 248 GB/s (98 %) | 201 | gemv |
| 8-bit quantization error (block 32) | 0.0047 | 0.0053 | 0.0053 | quant |
| 4-bit quantization error (block 32) | 0.0854 | 0.0969 | 0.0969 | quant |
| Q8_0 decode speedup vs fp16 | 1.88× | **1.88×** | 1.65× | quant |
| Q4_0 decode speedup vs fp16 | 3.56× | 2.76× | 2.31× | quant |
| Qwen2.5-14B decode on the GPU | 48 tok/s | **35.3 (74 %)** | 29–31 | llm |
| Qwen2.5-3B (MLX) decode on the GPU | 236 tok/s | **150 (64 %)** | 121–135 | llm |

Notice the pattern: the quiet machine moves *every* multi-core and GPU number up by 10–160 %, while single-core maths and quantization error don't move at all. Noisy neighbours steal shared resources (cores, bandwidth, the GPU), not the arithmetic inside one core.

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

**What you see (quiet machine).** 1 chain: 10.5 GFLOP/s. 16 chains: 133 GFLOP/s, **13× faster with identical instructions**, and 92 % of the paper peak. Half precision (fp16) gives exactly 2× (270 GFLOP/s), because a 128-bit register holds 8 fp16 numbers instead of 4 fp32.

**A prediction that broke, and what it taught.** The 1-chain result is *faster* than predicted: 10.5 vs 9.0 GFLOP/s. If an FMA really takes 4 cycles, that implies a 5.2 GHz clock, which is above the M4's reported 4.5 GHz. A second, independent estimate disagrees: Lab 2's L1 hit takes 0.67 ns, which is exactly 3 cycles at 4.5 GHz. Both assumptions can't be right. The likely one to drop is the 4-cycle FMA latency: at 4.5 GHz the measured chain is about **3.4 cycles per FMA**. (On the busy machine the FMA estimate came out at 4.6 GHz, close to the reported clock, so it looked right, and an earlier draft of this guide accepted it. The L1 estimate from the same evening, 3.9 GHz, already disagreed; nobody compared the two.) Only hardware cycle counters can settle it, which is why "add Instruments counters" is on the missing list. It is also a ready answer for "a time a measurement disproved your hypothesis".

**Threads: the barrier effect, confirmed.**

| Threads | 1 | 2 | 4 | 5 | 6 | 8 | 10 | 14 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quiet: GFLOP/s per thread | 122 | 120 | 118 | 116 | 116 | 116 | **115** | 41 |
| Busy (Rider + Unreal): per thread | 118 | — | 112 | — | — | 96 | **44** | 36 |

On the quiet machine, 10 threads scale almost perfectly (**1,150 GFLOP/s**, 94 % of 10× one core). On the busy machine the same 10 threads gave 438: every step waits for the slowest thread, and with two apps each holding a core, at least one of the 10 threads always shared a core. Even on the quiet machine, **14 threads collapse to 41 per thread**. Threads 11–14 land on the slow efficiency cores (≈15 GFLOP/s each), and every step waits for them. One slow worker holds up the whole team. The same mechanism explains "why does llama.cpp stop scaling" in P3.

**Over time** (`--soak 120`): ≈1,100 GFLOP/s held for two minutes with no thermal decline (last/first 96 %), except a 30-second dip to as low as 26 % that then **fully recovered**. Throttling from heat doesn't recover while the load continues, so something else briefly took cores (iCloud sync and VS Code were running). The barrier effect turns a small interruption into a big drop.

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

**What you see.** 0.67 ns up to 112 KB (L1 is 128 KB), exactly the predicted 3 cycles; ≈6 ns in L2; a big jump past ~12–16 MB, where the L2 runs out; **109 ns** in DRAM (111–132 ns on the busy machine: other apps' memory traffic makes everyone wait longer). The "sequential" column follows the same notes in order. It stays near 8 ns per line even at 512 MB, because the **prefetcher** sees the pattern and fetches ahead.

**Why it matters.** To stream 90 GB/s from memory that takes 109 ns to answer, a core must have about 9.8 KB (77 cache lines of 128 bytes) **in flight at once**. That's Little's law: *amount in flight = rate × wait time*. A core can only track so many outstanding requests, which is why one core can't use the whole memory system.

**On other chips.** Apple uses 128-byte cache lines and 16 KB memory pages; almost every other CPU (x86, Qualcomm, generic ARM64) uses 64-byte lines and 4 KB pages. See [other-chips.md](other-chips.md).

---

## Lab 3: `bandwidth`, how fast can bytes flow?

**In one sentence:** one core can pull ~90 GB/s from memory; the CPU cores together top out at ~253 GB/s, about 62 % of what the chip can deliver.

**Interview questions it answers:** "Measure your machine's peak bandwidth" [A] · "Why does scaling stop?" [A][R]

**Predict.** Apple's spec: **410 GB/s**. L1 per core: 3 loads × 16 bytes × 4.5 GHz ≈ **216 GB/s**.

**Run.** `labs/build/p1_bandwidth --seconds 1`

**What you see (quiet).** One core: L1 ≈175, L2 ≈113, DRAM ≈90 GB/s. Adding cores: 1 → 87, 2 → 166, 3 → 212, **4 → 252**, then flat at ≈253 up to 10 threads. **Four cores are enough to saturate the CPU side.** The 4 efficiency cores alone manage only ≈19 GB/s.

**An unexplained result, kept on purpose.** In the same sweep, the *write* kernel fell from ≈240 GB/s to ≈105 GB/s at 8 and 10 threads, while reads stayed at 253. One possible cause: a write to memory first fetches the line (write-allocate), so many writers double the real traffic and contend at the memory controller. That's a hypothesis only. Write-allocate and coherence traffic are exactly what hardware counters would show. Interviewers like candidates who keep anomalies instead of hiding them.

**Why the gap from 410.** The 410 GB/s is for the *whole chip*. The CPU clusters connect to memory through links that can't carry all of it. The GPU can. This one fact explains most of the "CPU vs GPU" difference in LLM speed later on.

**Tie to the ANE story (P0).** The Neural Engine streamed weights at ≈90 GB/s, **exactly one CPU core's worth** (90 GB/s measured here), or 22 % of the chip's bandwidth (the earlier R1 note said "one sixth", which assumed the 16-core chip). That is the main reason it decoded 3.5× slower than the GPU.

---

## Lab 4: `gemm`, multiplying matrices from scratch

**In one sentence:** the same multiplication gets **~2,000×** faster through access order, vector instructions in a register-blocked micro-kernel, threads, and finally Apple's matrix unit.

**Interview questions it answers:** "How do you choose tile sizes?" [A] · "Why is decode memory-bound and prefill compute-bound?" [A][R] · "What does a batch of 8 change?" [A]

| Step | What changed | Plain-language reason | Speed (1024×1024) |
| --- | --- | --- | ---: |
| naive | textbook three loops | walks down a column, so every multiply touches a new cache line | 1.5 GFLOP/s |
| ikj | swap two loops | walks along rows, so each fetched cache line is fully used | 21× |
| blocked | work on cache-sized tiles | reuse a tile while it's on the desk instead of going back to the library | 19× (**slower** than ikj, see below) |
| packed + NEON | copy tiles into tidy strips; an 8×8 block of the answer lives in 16 registers | 16 independent multiply-adds per step, exactly Lab 1's "16 chains" | 70× |
| 10 threads | split the work | more cores | 532× |
| Accelerate | Apple's library | uses the **SME matrix unit**, a separate engine with a much higher ceiling than NEON | **2,018×** (≈3.0 TFLOP/s) |

**Blocking alone didn't help, even when it "should" have.** An earlier version of this guide predicted that a larger matrix (2048, 16 MB per matrix) would make blocking win. Measured at 2048: ikj 31 GFLOP/s, blocked 26, packed 107, packed on 10 threads 832, Accelerate 2,841. This chip has a 16 MB L2 per group and aggressive prefetchers, so ikj's long unit-stride rows are already served well. Tiling the loops without also packing the data and keeping the result in registers just adds loop overhead. Blocking pays off here only as part of the full recipe (packing + register micro-kernel). On a chip with small caches and weak prefetching the plain blocked version would matter more. Memory hierarchy decides which textbook step pays.

**The most important result is Part 2 ("shapes").** The same weights (1024 → 3072, one Qwen3-0.6B layer) are multiplied by M tokens at once:
- **M = 1** (generating one token, "decode"): every weight is fetched for **one** multiply-add. That's 0.5 FLOP per byte, so the speed is set by memory: **253 GFLOP/s**.
- **M = 512** (reading a prompt, "prefill"): every weight fetched is used **512 times**. The speed is set by compute: **3,218 GFLOP/s**.

**Same weights, 13× faster, just by batching.** This is half of the ANE story: its runtime read the prompt one token at a time, which turned prefill into decode.

**Prediction that "failed" on purpose.** The paper prediction for M=1 from DRAM is 0.5 × 253 ≈ 127 GFLOP/s. Measured: 253, exactly 2×. The 12 MB test matrix stays in the cache between repeats, so it never comes from DRAM. Lab 5 fixes that by using a model-sized set of weights.

---

## Lab 5: `gemv`, generating a token is a bandwidth problem

**In one sentence:** tokens per second ≈ memory bandwidth ÷ size of the weights.

**Interview questions it answers:** "Max tok/s for a 7B Q4 model (~4 GB) at 80 GB/s?" (≈20) [A] · "How close is your kernel to the bandwidth bound?" [A]

**Predict.** Best case = the measured memory roof (~253 GB/s). One core: 4 chains × 4 numbers ÷ 4 cycles × 4.5 GHz ≈ 18 G weights/s = **72 GB/s** (a maths limit, not a memory one).

**Run.** `labs/build/p1_gemv --seconds 1`

**What you see.**
- A plain C loop: 8.6 GB/s. The compiler is not allowed to reorder floating-point additions, so the loop stays one slow chain (Lab 1 again).
- NEON, one core: 73–76 GB/s against the 72 predicted. **One core is limited by maths, not memory.** (Slightly *above* the prediction, the same clue as in Lab 1: with the measured ≈3.4-cycle FMA the limit is ≈85 GB/s.)
- NEON, all P-cores: **248 GB/s, 98 % of the memory roof** (201 GB/s, 90 %, on the busy machine). Beyond this, nothing helps except reading fewer bytes.
- fp16 weights (half the bytes): one core is *slower* in GB/s (42), because converting fp16 to fp32 costs extra instructions. With 8 threads both formats hit the roof, and fp16 then gives **almost 2× the tokens per second**.

**Translated to real models:** at the measured ~244 GB/s for fp16, a 4-billion-parameter fp16 model (8 GB) would run at ≈30 tok/s on these CPU cores, and a 4-bit version (2.25 GB) at up to ≈108. The same formula explains the ANE's 11.3 tok/s (90 GB/s) and the GPU's 40 tok/s in the R4 work.

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
- **Speed** (10 threads, quiet): Q8_0 is **1.88×** faster than fp16, *exactly* the byte-ratio prediction. Q4_0 is **2.76×** (bytes predict 3.56×) and moves only 187 GB/s against fp16's 241. Q4 falls short because unpacking two 4-bit numbers from each byte costs instructions. (Busy machine, 7 threads: 1.65× and 2.31×.) **4-bit decode on these cores is limited by instructions.** That's the argument for a custom "unpack and dot product" instruction in P6.
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

| | Decode | % of ceiling | Prefill | % of ceiling | Busy GPU (decode) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14B, llama.cpp/Metal | **35.3 tok/s** (5 runs within ±1 %) | 74 % | 339 tok/s | 71 % | 9–31 tok/s |
| 3B, MLX/Metal | **150 tok/s** | 64 % | 1,157 tok/s | 51 % | 101–135 tok/s |

**Over a minute of continuous use** (`--llm-seconds 60`), the 14B tier slowed **smoothly from 34.8 to 26.5 tok/s (76 %)**, with every request independent. A steady slide like that is the signature of the GPU settling from a short boost to a level it can sustain on power or temperature, not of a software effect. gRPC_llm's own 5-minute log fits: it started at 24.9 tok/s (already settled) and kept 96 %. The 3B tier dipped to ≈130 and recovered to ≈145 (≈90 %). **So quote two numbers for a model: burst and sustained.** A power measurement (`make llm-bench POWER=1` in gRPC_llm) shows whether GPU watts fall along with the speed.

**Step 3, explain the gap.**
- 410 GB/s is a peak. GPUs usually sustain 70–90 % of it on a pure stream.
- Each token also reads the KV cache and runs attention, normalisation and sampling.
- **Fixed costs per token** (launching GPU work, Python in the engine, streaming over gRPC) weigh more on the small model: its token takes ~8 ms, so 1 ms of overhead is 12 %. Client and server decode rates agree within ~1 tok/s, so gRPC itself adds almost nothing.
- **When the GPU is shared, everything moves.** On the busy machine (GPU 47–86 % busy with Unreal), the 14B model swung between 28 and 9 tok/s from one run to the next. The lab now checks GPU load first and refuses to fit a cost model from unstable runs.

**Two findings about gRPC_llm itself:**
1. **Prefix caching differs by engine.** Sending the same 1,851-token prompt twice: llama.cpp answered the repeat in **0.05 s instead of 5.7 s**, because it reused the cached prompt. MLX took 1.4 s both times; the server confirms **0 of 1,865 prompt tokens reused**. Agent loops that resend a long system prompt pay the full prompt cost on every turn on the MLX tier. That is an improvement worth making there.
2. **The same model on a CPU in Docker** (from gRPC_llm's earlier logs: Qwen2.5-3B Q5_K_M): 39 tok/s decode and 131 tok/s prefill, against **91 and 1,529** on Metal. Decode drops 2.3× (CPU bandwidth instead of GPU bandwidth); prefill drops **12×** (CPU maths instead of GPU maths). That's the decode/prefill split from Lab 4, on a real service.

**Cost model.** When two tiers give stable numbers, the lab fits *time per token = bytes ÷ B + c* (the same per-byte + fixed-cost model as the ANE R1 work) and rejects impossible fits (c < 0 or B above the spec). The quiet run gave **B ≈ 314 GB/s** (77 % of the 410 GB/s spec is actually usable for streaming) and **c ≈ 1.1 ms** of fixed cost per token. That c is 4 % of a 14B token (28 ms) but 17 % of a 3B token (6.7 ms), which is why the small model sits further below its ceiling. Caveat: two engines. The busy-GPU run gave B ≈ 265, c ≈ 1.7 ms, and an earlier unstable run produced a negative c, which was rejected. Validating the model means predicting a *third* model, then checking.

---

## Lab 8: `portable`, same chip, different operating system

**In one sentence:** the same C++ file, run on macOS and on Linux on the *same* M4 Max, shows what the operating system adds or takes away.

**Run.** `bash labs/portable/compare_native_vs_linux.sh 0.5`

`portable_lab.cpp` is standard C++ with one fast path per instruction set (NEON for ARM, AVX2 for x86, plain C otherwise), which is how llama.cpp supports many chips. It compiles on macOS, Linux and Windows (instructions at the top of the file).

| | macOS native | Linux in Docker's VM |
| --- | ---: | ---: |
| Memory page size | 16 KB | 4 KB |
| Memory bandwidth, triad (best) | 249 GB/s | 206 GB/s (−17 %) |
| GEMV (best) | 134 GB/s | 84 GB/s (−37 %) |
| int8 dot product, one core | 129 GOP/s | 129 GOP/s (identical) |
| CPU features visible | NEON, dotprod, i8mm, bf16, **SME** | NEON, dotprod, i8mm, bf16, *no SME* |

(Quiet machine. On the busy machine the gap was 190 vs 132 GB/s: the VM suffers more from noisy neighbours.)

**Why.** The maths is identical (same cores). The **memory path** is slower in the VM: 4 KB pages mean four times as many address translations as 16 KB pages, and a virtual machine adds a second translation layer on top. The VM also hides SME (so no Accelerate-style matrix unit), Metal (no GPU) and the Neural Engine. That's why gRPC_llm's Docker tier runs on the CPU only.

**x86 too (correctness only).** `arch -x86_64` runs the AVX2 build through Rosetta: the results are correct (the int8 check passes) and the program sees 4 KB pages, but the timings are emulation, not x86 hardware. For real x86 numbers, see [other-chips.md](other-chips.md).

**Lesson inside the code.** This program creates threads for every layer, so its GEMV stalls at 134 GB/s and gets *worse* with 14 threads (112), while `p1_gemv`, which keeps a pool of threads alive, reaches 248 GB/s. Creating threads is not free, and the VM's thread creation is more expensive still (84 → 39 GB/s from 4 to 14 threads).

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
| Power and energy per token | Tokens per joule; whether the 14B slowdown follows GPU watts | `make llm-bench POWER=1 PORT=50051` (and `50061`) in gRPC_llm, in your own Terminal (needs sudo) |

## Exercises

1. Open a heavy app (a game engine, a video export) and rerun `python3 labs/run_labs.py --quick`. Compare with [results-2026-10-12.md](results-2026-10-12.md): which numbers move, and which never do?
2. `labs/build/p1_bandwidth --part threads --threads 1,2,3,4,5,6,7,8`: does the 6th thread (the second P-group) help?
3. `labs/build/p1_gemm --part steps --n 2048` showed plain blocking losing to the loop swap. Try tile sizes 32 and 128 in `gemm::blocked`: can any tile size win on this chip?
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
