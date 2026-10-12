# P3 results: llama.cpp on the CPU, 2026-10-12

Measured 01:47–01:53 EDT on the Apple M4 Max, quiet machine (Unreal Editor and Rider closed, load average ≈5).
An earlier run at 01:39 overlapped a busy iCloud sync process (`fileproviderd` at 100 % of a core). It agreed within a few percent at 1–8 threads and dipped at 10 threads. That dip was the background process, not llama.cpp.

Command: `python3 labs/p3/llama_bench_lab.py` (llama-bench, 3 repetitions per cell, `pp512` = 512-token prefill, `tg128` = 128-token decode).

Build (`labs/p3/vendor_llama.sh`):

```
tag=b11562
commit=b2f036c0f138414dedc987446f1d9d802f920275
built=2026-10-12T05:37:55Z
compiler=Apple clang version 17.0.0 (clang-1700.6.3.2)
flags=GGML_NATIVE=ON GGML_METAL=OFF GGML_BLAS=OFF GGML_ACCELERATE=OFF
```

Models (`labs/p3/models.sh`, all quantized from the same F16 file):

```
fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649  qwen2.5-1.5b-instruct-f16.gguf
00d711b346cb68f2b2e460deed533e55913b52a7793f3050f38d223870c135b8  qwen2.5-1.5b-instruct-q4_0.gguf
93bca95df5e97f277ffabf7a7dbd075c35aa5d7b8e4780dc7e13acc119ce144f  qwen2.5-1.5b-instruct-q4_k_m.gguf
bcbdf6b481ca6cd7197c80516b621da3238dc51d703062e0667b71c6aa08676a  qwen2.5-1.5b-instruct-q8_0.gguf
```

## Full output

```
== P3.llamacpp  llama.cpp CPU backend: predict, measure, explain

   1. PREDICT (from the model files and the P1 ceilings, before running)
      bandwidth for t threads = min(t x 90, 253) GB/s;  prefill compute = t x 270 GFLOP/s (NEON fp16)
      format      bytes/token  bits/wt  decode @1T  decode roof   knee  prefill @1T
      F16             3.09 GB    16.00        29/s         82/s   2.8T         76/s
      Q8_0            1.64 GB     8.50        55/s        154/s   2.8T         76/s
      Q4_K_M           980 MB     5.00        92/s        258/s   2.8T         76/s
      Q4_0             929 MB     4.77        97/s        272/s   2.8T         76/s
      Decode should scale until ~2.8 threads, then flatten at the bandwidth roof;
      prefill should keep scaling with threads, because it is compute-bound.

   2. MEASURE (llama-bench: pp512 = prefill, tg128 = decode; 3 reps per cell)
      build b2f036c, cpu: Apple M4 Max

      decode (tok/s)          1T        2T        4T        6T        8T       10T
      F16                   22.6      37.7      65.3      72.3      73.5      73.1
      Q8_0                  44.7      79.4     126.0     130.6     131.6     126.1
      Q4_K_M                37.2      69.1     120.3     149.3     164.0     177.0
      Q4_0                  42.2      79.9     142.3     167.8     195.2     196.9

      prefill (tok/s)         1T        2T        4T        6T        8T       10T
      F16                   30.4      59.1     116.3     174.5     223.3     265.0
      Q8_0                 110.6     216.8     426.8     623.4     818.4     941.2
      Q4_K_M                80.5     158.2     312.7     455.6     607.7     689.7
      Q4_0                 110.1     215.2     424.6     622.9     825.8     983.5

   3. EXPLAIN (measured vs predicted)
      format    best decode   at  % roof  eff GB/s  1T decode %pred  best prefill   at  prefill %pred
      F16            73.5/s   8T     90%       227              77%         265/s  10T            35%
      Q8_0          131.6/s   8T     85%       216              81%         941/s  10T           124%
      Q4_K_M        177.0/s  10T     69%       173              41%         690/s  10T            91%
      Q4_0          196.9/s  10T     72%       183              44%         983/s  10T           129%

      Speedup over F16 at each format's best thread count, vs the byte ratio:
        Q8_0     decode 1.79x  (bytes predict 1.88x)  prefill 3.55x
        Q4_K_M   decode 2.41x  (bytes predict 3.15x)  prefill 2.60x
        Q4_0     decode 2.68x  (bytes predict 3.32x)  prefill 3.71x

      Reading guide:
      - decode flattening near the bandwidth knee = memory-bound (P1 bandwidth lab).
      - decode below the roof at every thread count = per-token overhead or unpacking cost (P2).
      - prefill above the fp16 prediction = int8 kernels (sdot / i8mm) beat fp16 FMA on Q8/Q4.
      - prefill flattening = compute shared with fewer free cores, or thread synchronisation.
```
