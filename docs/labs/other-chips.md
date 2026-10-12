# How Apple silicon differs from other chips

The labs run on an Apple M4 Max. The *ideas* (dependency chains, cache levels, bandwidth roofs, quantization) are the same on every chip; the *numbers and the tools* are not. This page explains what changes when you move to:

- a **generic ARM64** machine running Linux: AWS Graviton, Ampere, a Raspberry Pi, or Docker's Linux VM on this very Mac;
- a **Qualcomm Snapdragon X** laptop running Windows on Arm;
- an **x86-64** PC or server, under Windows ("win64") or Linux;
- an **FPGA**;
- **RISC-V**, the target of P5 and P6.

Abbreviations (SIMD, NEON, AVX, SME, NPU, NUMA, SMT, VNNI, RVV…) are explained in plain words in the [glossary](glossary.md).

Facts marked *(spec)* are vendor figures; *(measured)* means this repo measured it; *(approx.)* means a typical value to check on the real machine.

---

## The short version

| | Apple M4 Max (this Mac) | Generic ARM64 + Linux | Qualcomm Snapdragon X + Windows | x86-64 + Windows/Linux | FPGA |
| --- | --- | --- | --- | --- | --- |
| **Vector width** | NEON, 128-bit | NEON 128-bit; some have SVE (Graviton3: 256-bit) | NEON, 128-bit | AVX2 256-bit; AVX-512 on many servers and AMD Zen 4/5 | whatever you build |
| **Matrix unit on the CPU** | SME (used by Accelerate) | rare (newer Armv9 designs may have SME) | none known on Oryon | Intel AMX on recent Xeons | you build MAC arrays from DSP blocks |
| **int8 dot instruction** | `sdot` (16 multiply-adds per instruction) | `sdot` on Armv8.2+ | `sdot` | VNNI `vpdpbusd` (unsigned × signed!) | DSP slices / LUTs |
| **Cache line** | 128 bytes *(measured)* | 64 bytes | 64 bytes | 64 bytes | you choose (bus width) |
| **Memory page** | 16 KB *(measured)* | 4 KB (64 KB optional) | 4 KB | 4 KB (2 MB "large pages" on request) | none (physical addresses) |
| **Memory** | unified: CPU, GPU and NPU share one RAM | CPU RAM; a GPU, if any, is separate | unified LPDDR5X | CPU RAM, plus a **separate** GPU with its own VRAM over PCIe | off-chip DDR or HBM, plus on-chip block RAM |
| **Bandwidth** | 410 GB/s *(spec)*; CPU cores reach ~253 *(measured)* | Pi 5 ~17; Graviton3 ~300 *(approx.)* | ~135 GB/s *(spec, X Elite)* | desktop dual-channel DDR5 ~80–100; servers 300–500+ *(approx.)* | DDR boards tens of GB/s; HBM boards ~460 *(spec, e.g. Alveo U280)* |
| **NPU** | Neural Engine (Core ML; private APIs in the R-series) | usually none | Hexagon NPU (QNN SDK, ONNX Runtime QNN) | Intel NPU (OpenVINO), AMD XDNA (Ryzen AI) on laptops | the FPGA *is* the accelerator |
| **Choosing which core a thread runs on** | **not possible**; only a hint (QoS class) | yes: `taskset`, `pthread_setaffinity_np`, `numactl` | yes: `SetThreadAffinityMask`, CPU Sets | yes (same APIs as left); NUMA matters on servers | no threads: hardware runs in parallel by design |
| **Two threads per core (SMT)** | no | no (most Arm) | no | **yes, usually 2** | n/a |
| **Hardware counters** | Instruments / `xctrace` (no `perf`) | `perf stat`, `perf record` | Windows Performance Analyzer, Snapdragon Profiler | `perf` (Linux), VTune, AMD uProf, WPA (Windows) | simulation and on-chip logic analysers |

---

## 1. Apple silicon: what is special here

**Unified memory.** The GPU reads the model straight from system RAM. On a PC with a graphics card, a 9 GB model must fit in the card's VRAM, or layers spill back to the CPU over PCIe (~32 GB/s each way on PCIe 4.0 x16), which is far slower than either memory. On a Mac the limit is total RAM (36 GB here), and the price is that CPU, GPU and NPU all share the same 410 GB/s. The R4 work measured exactly that: the Neural Engine slowed the GPU foreground by 17 % without touching the GPU at all.

**You cannot pin threads.** Linux and Windows let you say "run this thread on core 3". macOS only lets you hint how important a thread is (its QoS class): `USER_INTERACTIVE` may use performance cores, `BACKGROUND` stays on efficiency cores. The labs use that hint (`labs/common/pool.h`). It's also why the barrier effect in the `peak` lab is hard to control on a Mac: you can't keep your threads away from Rider's core.

**Big cache lines and pages.** 128-byte cache lines mean "false sharing" (two threads writing data that sit in the same line) happens at a 128-byte distance, not 64. That's why the bandwidth lab spaces per-thread results 128 bytes apart. 16 KB pages mean four times fewer address translations than 4 KB pages; the `portable` lab measured macOS streaming **249 GB/s against 206 GB/s** for Linux on the same chip (part of that is the VM, see below).

**Hidden engines.** Accelerate's matrix multiply ran at ≈3.0–3.2 TFLOP/s (measured) because it uses the SME matrix unit. Plain NEON code peaks at ≈1.15 TFLOP/s on all 10 P-cores (measured, quiet machine). The Neural Engine is reachable officially only through Core ML. Many speeds on a Mac therefore depend on *which library* you call, not on how good your code is.

**No `perf`.** The interview plan's `perf stat -M TopdownL1` doesn't exist on macOS. Instruments (CPU Counters template, or `xctrace` on the command line) is the replacement; `powermetrics` (needs sudo) gives power per CPU, GPU and ANE.

---

## 2. Generic ARM64 with Linux, including Docker on this Mac

**Same instruction set, different system.** NEON code (all of `labs/p1`, `labs/p2`) is valid on any ARMv8.2+ Linux machine. What breaks is the Apple-only plumbing: `sysctl` names, QoS classes and Accelerate. `labs/portable/portable_lab.cpp` avoids all three and runs there unchanged.

**Measured on this Mac, in Docker's Linux arm64 VM** (`bash labs/portable/compare_native_vs_linux.sh`, quiet machine):

| | macOS | Linux VM |
| --- | ---: | ---: |
| bandwidth (triad) | 249 GB/s | 206 GB/s |
| GEMV, threads created per layer | 134 GB/s | 84 GB/s |
| int8 maths | 129 GOP/s | 129 GOP/s |
| page size | 16 KB | 4 KB |
| SME, Metal, Neural Engine | available | **hidden** |

The maths is identical; memory is 17 % slower, and code that creates many threads suffers most (−37 %). Two reasons: 4 KB pages need more translations, and a VM translates every address twice (guest, then host). This is why gRPC_llm's Docker tier is CPU-only and much slower than its Metal tiers (39 vs 91 tok/s decode for the same 3B model, from gRPC_llm's logs).

**On a real ARM64 server** you gain `perf stat`, thread pinning and often more memory channels. Some chips also have **SVE**, a vector extension whose width isn't fixed in the code: the same binary runs on 128-bit or 256-bit hardware. That is the same idea as RISC-V's vectors (section 6).

---

## 3. Qualcomm Snapdragon X with Windows on Arm

The closest thing to a Mac in the PC world: an Arm SoC with unified LPDDR5X memory (~135 GB/s, spec), a GPU (Adreno), and an NPU (Hexagon, rated around 45 TOPS INT8).

- **CPU code:** the NEON paths in `portable_lab.cpp` compile with MSVC (`cl /O2 /std:c++20 /EHsc portable_lab.cpp`). The same `sdot` int8 kernel applies. Expect roughly a third of the M4 Max's bandwidth, so roughly a third of the decode speed on bandwidth-bound work. *(Prediction from bandwidth only; not measured here.)*
- **NPU:** reached through Qualcomm's QNN SDK or ONNX Runtime's QNN execution provider. Like the Apple Neural Engine, it prefers int8/int4, fixed shapes and whole-graph compilation. The R-series lessons (fixed cost per dispatch, batching, bandwidth share) transfer directly.
- **Windows-specific:** x64 programs run under emulation (Prism) and lose most of their SIMD speed; build native arm64. You *can* pin threads (`SetThreadAffinityMask`, or the CPU Sets API). Profile with Windows Performance Analyzer.

---

## 4. x86-64 with Windows ("win64") or Linux

**Wider vectors, fewer of them.** AVX2 registers hold 8 floats (NEON holds 4), and AVX-512 holds 16. Many x86 cores have **2** FMA pipes rather than Apple's 4. Redo Lab 1's prediction: chains needed = 2 pipes × 4 cycles = **8**, and peak per core = 2 × 16 FLOPs (AVX2) × clock. To predict an x86 core, change `SPEC` in `labs/predictions.py`.

**The int8 trap.** x86's byte multiply-add (`vpmaddubsw`, and VNNI's `vpdpbusd`) multiplies **unsigned × signed** bytes. Quantized weights are signed, so llama.cpp uses a trick: multiply |a| by (b with a's sign). `portable_lab.cpp` does exactly this in its AVX2 path, and the check passes when run under Rosetta.

**Hyper-threading (SMT).** Two threads share one core's FMA units. For GEMM or GEMV, use one thread per *physical* core; a second thread per core mostly adds contention. Apple, Qualcomm and most Arm server cores have no SMT, so this question only comes up on x86.

**NUMA.** Big servers have several memory controllers; memory "belongs" to the socket (or chiplet) that first touched it. The bandwidth lab already initialises each thread's slice from that thread ("first touch") for this reason. On Linux use `numactl --interleave=all` or pin threads near their data.

**A discrete GPU changes the memory picture.** VRAM is fast (often ~1 TB/s on high-end cards, approx.), but the model must *fit*. Decode tokens/s = VRAM bandwidth ÷ weight bytes, until the model is too large; then layers spill to the CPU and PCIe becomes the bottleneck. llama.cpp's `-ngl` (number of GPU layers) is exactly this trade-off.

**Windows porting gotchas.**
- `long` is **32-bit** on Windows (64-bit on macOS and Linux). Byte counts above 2 GB overflow silently; use `size_t` or `int64_t`.
- Large pages need the "Lock pages in memory" privilege.
- Time with `std::chrono::steady_clock` (it wraps `QueryPerformanceCounter`).
- `cl /O2 /std:c++20 /arch:AVX2 /EHsc portable_lab.cpp` builds the portable lab; `/arch:AVX512` is available for AVX-512 machines.

**Counters, finally.** On Linux x86, `perf stat -e cycles,instructions,cache-misses ./portable_lab` and `perf stat -M TopdownL1 ./portable_lab` (Intel) give the IPC and front-end/back-end-bound numbers the interview plan asks for. On Windows, use VTune or AMD uProf.

**What this Mac can and can't tell you about x86.** `arch -x86_64` runs an x86 build through Rosetta. That's good for **correctness** (it passed here) and it shows that x86 programs see 4 KB pages. It's useless for **timing**, which measures the emulator, not an x86 chip.

---

## 5. FPGA: when you build the hardware yourself

An FPGA has no instruction set, no caches and no threads until you design them. You describe a datapath (in Verilog/VHDL, or C++ via high-level synthesis such as Vitis HLS), and the chip is wired to match.

**What stays the same:** the roofline and Little's law. Decode still reads every weight once per token, so tokens/s ≈ bandwidth ÷ bytes. A board with DDR4 (tens of GB/s) decodes slowly no matter how many multipliers you build. HBM boards (~460 GB/s, spec, on high-end cards) are in M4 Max territory. Little's law becomes "how many memory requests can my design keep outstanding".

**What changes:**
- **On-chip memory is the superpower.** Block RAM is small (a few MB up to tens of MB on the largest parts), but it's very fast and spread across the chip. A small model, or one layer at a time, can stay on chip and stop paying DRAM bandwidth per token. This is the "on-chip weight reuse" answer to the P0 question "what would you change in the next NPU?".
- **Low clock, huge parallelism.** 200–500 MHz instead of 4.5 GHz, but thousands of multiply-accumulate units working in parallel. int8 and int4 are cheap, since narrower numbers need fewer DSP and logic resources.
- **Deterministic latency.** No OS, no scheduler, no barrier effect, no noisy neighbours: the same input takes the same number of cycles. That is the opposite of every timing problem in this guide.
- **Slow iteration.** A build takes hours, not seconds; you verify in simulation first.

**Why it matters for a RISC-V AI chip company.** Before silicon exists, new AI cores are prototyped on FPGAs. That's where a runtime engineer first brings up a compiler such as IREE, and where an architect checks the performance model against real (if slow-clocked) hardware. The predict → measure → explain loop is the same job.

---

## 6. RISC-V (P5 and P6)

**Vectors without a fixed width (RVV).** NEON is always 128 bits and AVX2 always 256. A RISC-V vector loop asks the hardware "how many elements can you do?" (`vsetvli`) on each pass, so one binary runs on 128-bit or 1024-bit hardware. That's attractive for a chip vendor: one software stack across several chip sizes. Tail handling, the last partial chunk, is a classic source of bugs (it's on the P5 debugging question list).

**Custom instructions.** RISC-V lets a vendor add instructions. The `quant` lab measured where one would pay off: 4-bit decode spends its time *unpacking*, not waiting for memory (126 GB/s against a ~220 GB/s roof). An "unpack int4 and dot product" instruction targets exactly that. P6's job is to back such a proposal with numbers.

**Memory ordering.** x86 keeps stores in order for other cores (TSO). ARM and RISC-V (RVWMO) do not, so lock-free code needs explicit fences. Code that "works" on x86 can break on ARM or RISC-V.

---

## Translating the kernels: the same idea on three instruction sets

| Operation | ARM NEON (these labs) | x86 AVX2 / AVX-512 | RISC-V RVV 1.0 |
| --- | --- | --- | --- |
| load floats | `vld1q_f32` (4) | `_mm256_loadu_ps` (8) / `_mm512_loadu_ps` (16) | `__riscv_vle32_v_f32m1(p, vl)` (vl chosen at run time) |
| multiply-add | `vfmaq_f32` | `_mm256_fmadd_ps` | `__riscv_vfmacc_vv_f32m1` |
| sum a vector | `vaddvq_f32` | shuffles + `_mm_hadd_ps` | `__riscv_vfredusum_vs_f32m1_f32m1` |
| int8 dot product | `vdotq_s32` (signed × signed) | `_mm256_dpbusd_epi32` (VNNI, **unsigned × signed**) or `maddubs` + `madd` | widening multiply + reduction (no single dot instruction in base RVV) |
| fp16 → fp32 | `vcvt_f32_f16` | `_mm256_cvtph_ps` (F16C) | `__riscv_vfwcvt_f_f_v_f32m2` |
| matrix unit | SME (Apple M4; Accelerate uses it) | AMX tiles (recent Xeon) | vendor matrix extensions |

## What each lab needs to run elsewhere

| Lab | Apple-only parts | Replacement on Linux/Windows |
| --- | --- | --- |
| harness, `run_labs.py` | `sysctl`, `pmset`, `ioreg` in the preflight checks | `lscpu` and `/sys/devices/system/cpu`; `GetLogicalProcessorInformationEx` |
| thread pool | QoS classes | real affinity (`pthread_setaffinity_np`, `SetThreadAffinityMask`) |
| `peak`, `gemv` | NEON intrinsics | AVX2/AVX-512 versions (see the table above); `portable_lab.cpp` shows the pattern |
| `latency` | 128 B stride default | `--stride 64` |
| `gemm` | Accelerate | OpenBLAS, BLIS or Intel MKL |
| `quant` | `sdot` | VNNI or `maddubs` + `madd` (sign trick) |
| `llm` | Metal tiers, `ioreg` GPU load | CUDA/ROCm/Vulkan builds of llama.cpp; `nvidia-smi` for GPU load |
| `portable` | none | runs as is (`g++` / `cl`) |
