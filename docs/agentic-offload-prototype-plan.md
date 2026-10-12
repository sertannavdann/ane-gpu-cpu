# NEXUS ANE kernel and small-model acceleration — revision 5

Revised 2026-09-19. Governed by [instructions.md](instructions.md). This is a bounded research plan plus conditional integration steps. Existing CPU/MLX prototype code and measurements are retained; new ANE work is not yet implemented.

## Scope correction and decision

The user's objective is to develop useful ANE computation: inspect and modify tensor lowering, execute transformer subgraphs, and run a small model alongside MLX within the local agent system. Replacing a fast CPU classifier was too narrow.

**ANE research starts directly. It does not depend on a CPU routing bottleneck or failure of the MLX responsiveness target.** Revision 4's conditional classifier-only ANE gate is superseded. Its scheduler remains a control workload.

The development guideline's foreground/background example is illustrative. Its applicable requirements are a concrete outcome, minimal experiment, existing-component baseline, fixed effort cap, correctness and an explicit decision. Here the immediate uncertainty is whether compiler-directed ANE lowering can yield useful computation on this specific M4 Max. We will reproduce existing kernels before changing them, and only integrate changes supported by measurements.

Two levels of progress are distinct:

- **Research complete:** reproducible numerical and timing evidence identifies working shapes and explains bottlenecks, including a negative result.
- **Useful backend validated:** a small model completes a defined task correctly while the GPU serves another request, with an end-to-end benefit and guardrails met.

## Outcome brief

| Field | Commitment |
| --- | --- |
| Customer and situation | Sertan, developing a local NEXUS agent system, wants small-model work to run on the ANE while the GPU remains available for interactive generation. |
| Problem and evidence | The current implementation exercises CPU routing and MLX generation, but contains no ANE kernel execution. Available short scheduler replays do not measure ANE feasibility, numerical fidelity or energy. |
| Outcome | Determine which real transformer operations can execute correctly and usefully on this M4 Max, then demonstrate one useful small-model task through the isolated runtime before a gRPC integration trial. |
| Immediate question | Can a selected linear projection and one fused FFN subgraph run correctly on ANE, and does an examined lowering improve their complete invocation cost over the reproduced upstream baseline? |
| Smallest complete experiment | Load one frozen projection weight and activation fixture, transform it to convolution layout, compile, execute repeatedly on ANE, unpack, compare with reference, and produce a reproducible correctness/timing record. Expand to one FFN only after that path works. |
| Included now | One device, one pinned native runtime, one model family's shapes, FP16 baseline, model-derived inputs, full cost measurement and one selected lowering experiment. |
| Deferred | General compiler/backend framework, arbitrary model conversion, automatic operator placement, whole-model cross-device sharding, training, speculative decoding, broad quantization search, mandatory Python binding, production NEXUS changes. |
| Owners | Outcome owner: Sertan. Delivery owner for the prototype work: Codex, with experiment decisions recorded here for Sertan's review. |
| Effort cap and review | Initial research cap: five working days of implementation effort, partitioned as R0/R1/R2 below. Review upon exit or cap, whichever occurs first. This is a planning cap, not a performance or delivery promise. Further stages have separate caps and entry gates. |
| Decision | Keep a correct measured improvement; continue to a small-model proof when the runtime is viable; otherwise document the failed constraint and stop or explicitly replan. No silent move to another framework or larger model. |

Target: locally observed M4 Max, 14 CPU cores, 32 GPU cores, 36 GB unified memory, macOS 26.5 build 25F5068a. Re-record the environment before hardware trials.

## Existing evidence and its limits

The source of truth is the existing artifacts, not the prior narrative:

| Evidence | Observed value | Interpretation |
| --- | --- | --- |
| [MLX baseline summary](../results/mlx-baseline-run2/summary.json) | 60 requests; foreground TTFT p95 1206 ms; foreground queue p95 1103 ms; wall time about 48.4 s | Short scheduling baseline, not a five-minute sustained run. |
| [MLX priority summary](../results/mlx-priority-run2/summary.json) | Foreground TTFT p95 180 ms; background completion 11/11; background completion-time p95 4849 ms vs 1899 ms baseline | Foreground improvement has a background latency cost; one run per condition. |
| Same summaries | Foreground router p95 about 3–4 ms; disposition accuracy 0.735; foreground rubric passes 13/49 | A fast route is not proof of correct useful completion. Audit rubrics/dispositions before task-efficiency claims. |
| [MLX generator](../prototype/generator_mlx.py) | Python calls MLX-LM directly; streaming and MLX allocation counters exist | GPU comparison runtime already available. No ANE code is present here. |
| Same generator | Model fingerprint hashes only the first MiB of each weight shard plus metadata | Retain historical IDs, but use full artifact checksums for new numerical experiments. |
| [Run manifest](../results/mlx-priority-run2/manifest.json) | Docker VM resident; MLX allocations recorded | No captured energy or system-wide swap evidence establishes ANE value or total memory use. |

These findings justify preserving the scheduler, not expanding its tuning in the active kernel slice. A yielded text event is not automatically a device synchronization or hardware preemption boundary; future interleaving claims require backend-specific validation.

## What we can program

The proposed work is **graph/operator lowering and native runtime engineering**. We can alter equations, tensor layouts, constant packing, graph fusion, shape specialization and dispatch boundaries. We are not assuming access to a CUDA-like thread/block ISA, explicit SRAM management or a public ANE assembly toolchain.

The inspected [ane.cpp runtime source](https://github.com/skyfallsin/ane.cpp/blob/main/core/ane_runtime.cpp) includes MIL generation, convolution-based projections/FFN graphs, private Objective-C runtime calls and IOSurface allocation. It is a concrete starting point for an isolated native probe. Names and interfaces must be rechecked at the pinned checkout; web snapshots are not build pins.

The [ane.cpp project](https://github.com/skyfallsin/ane.cpp) describes hybrid CPU/ANE generation. A small model through that path is genuine ANE acceleration even when normalization or attention remains on CPU; report the operation placement explicitly.

[ANEForge](https://github.com/sbryngelson/ANEForge) documents Python graph construction, direct private-runtime execution and compressed-weight paths. It is a reference for alternative lowerings, not a mandatory second dependency. Python can drive native ANE execution; C++ does not make an in-process binding impossible.

Initial roles:

| Component | Role |
| --- | --- |
| Python harness | Produce fixtures, run reference equations, launch experiments, compare results and record metrics. |
| Native C++/Objective-C runtime probe | Compile/dispatch ANE graphs and own surfaces/program lifetimes. Reuse the pinned ane.cpp path before writing new plumbing. |
| MLX | Matched GPU equations and later same-small-model comparison; existing Qwen2.5-3B answerer supplies concurrent load. |
| CPU reference | FP32 numerical reference, plus matched-precision timing where appropriate. |
| gRPC | Later request-level serving boundary; never one RPC per matrix operation. |

Start the probe as an isolated executable because private-runtime crashes and stuck evaluation are material current risks. This is one justified experimental process boundary, not a microservice architecture. A C ABI/pybind11 binding is an option only if a measured need justifies it. Do not treat subprocess overhead as ANE kernel time.

## First lowering: linear projection to 1×1 convolution

For a row-oriented input and stored projection weights:

```text
X: [S, K]             S = tokens or independent samples
W: [N, K]             K = input channels, N = output channels
Y = X Wᵀ + b          Y: [S, N]

A[0, k, 0, s] = X[s, k]             A: [1, K, 1, S]
F[n, k, 0, 0] = W[n, k]             F: [N, K, 1, 1]
Z = conv2d(A, F, bias=b,
           stride=1, padding=0, dilation=1, groups=1)
Y[s, n] = Z[0, n, 0, s]
```

This is an algebraic identity before finite-precision effects. It does not assert the compiler's physical memory layout or that every shape is supported. Emit MIL for the chosen native path and inspect the produced metadata/logs.

The probe must make the transformation visible: input/weight shapes, logical-to-physical index map, strides, padded dimensions, weight bytes, emitted MIL, compile result, dispatch count and output comparison. Inspect compiled artifacts/operation metadata where accessible; do not promise complete reconstruction of proprietary instructions.

Implementation choices to test:

- First use one real projection shape from a pinned small Qwen3 candidate's config, then a small synthetic case that exposes transposition mistakes. Never guess head/channel dimensions.
- Test useful token counts 1, 8 and 32 initially. If the runtime requires a wider surface, report logical S and allocated S separately, zero padding and trim outputs. A required surface width does not imply 32 independent requests.
- Zero-pad both inputs and weights if channels need alignment. For a pointwise projection padded token columns are independent; that fact does not make padding safe for attention.
- Separate one-time weight preparation and compile/load from per-call input packing, execution and unpacking.
- Reuse allocated buffers and compiled programs only after the repeated-call baseline shows the cost. Reuse ownership must prevent races or stale outputs.
- If a projection must be partitioned, distinguish output-channel tiling (concatenate) from input-channel tiling (sum partial products, bias once). The latter introduces additional accumulation error and dispatches.

First deliverable: a runnable native projection probe plus a Python numerical comparison that reports correct output and complete invocation latency. No Python model accessor alone satisfies this milestone.

## Second lowering: one useful transformer subgraph

After projection reproduction, select one FFN block from the same checkpoint. For a bias-free SwiGLU example:

```text
G = X Wgᵀ
U = X Wuᵀ
H = (G * sigmoid(G)) ⊙ U
Y = H Wdᵀ + R
```

Map the three linears to pointwise convolutions. Compare the unmodified upstream path, an intentionally separate-operation reference, and one proposed change. A fused graph in source is not proof of one physical dispatch; count what the runtime actually submits.

Choose **one** measured bottleneck: projection packing/alignment, fusion/residual placement, channel chunk size, or repeated surface transfer. Do not run a full combinatorial sweep. If upstream already implements the candidate fusion, reproduce it and choose a remaining measured cost; renaming an existing kernel is not an optimization.

QKV concatenation is a possible next experiment if FFN measurements point elsewhere: concatenate output channels, preserve distinct Q/K/V widths for grouped-query attention, then split correctly. Do not assume all three matrices are square or equal.

**RMSNorm is a separate numerical investigation**, not a prerequisite for calling the FFN ANE-accelerated:

```text
RMSNorm(x) = gamma ⊙ x / sqrt(mean(x²) + epsilon)
```

Keep normalization on the CPU reference initially. If moving it would remove a material boundary, test the channel reduction, accumulation precision and scaled/two-pass alternatives while preserving epsilon. For positive scaling c, normalizing x/c requires epsilon/c² for algebraic equivalence; rounding, underflow and the cost of finding c still require tests. Do not silently change the norm to make a benchmark pass. The [field guide](https://github.com/skyfallsin/apple-neural-engine-field-guide) reports norm-related numerical problems on its tested M3 Max path; those are a test hypothesis for this M4 Max, not a universal impossibility.

## Static shapes, KV state and quantization

These are staged engineering questions, not blanket exclusions:

| Constraint | Smallest useful treatment |
| --- | --- |
| Static graph/weights | Compile the selected frozen weights and a small shape set outside the timed request loop. Record compilation and cache cold starts separately. Avoid compilation per token. |
| Variable prompt length | Use bounded token buckets only when the small-model stage needs them; distinguish valid tokens, padding and compilation count. |
| Attention and KV cache | Retain the upstream proven placement first. Verify valid length, causal masks, rotary positions, cache growth and resets. Do not conflate projection batching with independent attention sessions. |
| FP16 range/accumulation | Test measured activation ranges, outliers, near-zero inputs and long reductions against a higher-precision reference. |
| Compressed weights | Begin FP16. Then, only if bandwidth is the measured limit, test one supported compressed format. |
| GGUF source weights | A GGUF quantization format is not automatically an ANE execution format. Decode/repack once when needed; quantify disk, compiled, resident and streamed representations separately. |
| Private API drift | Pin chip/OS/toolchain/upstream commit. Keep symbol/probe failures typed, isolate experiments, and rerun correctness after OS/runtime changes. |

The claim that any ANE quantization requires expanding all weights and losing all bandwidth savings is too broad: current ANEForge documentation reports compressed paths. Conversely, those reports do not establish Q4_K_M compatibility or a win on this host. Quantization support, activation arithmetic and accumulation precision are separate properties.

## Correctness and measurement protocol

All comparisons use the same input bytes, weight revision, bias/epsilon, shape and semantics. Keep two comparison tracks: matched precision for placement claims; optimized deployable formats for system comparisons. The existing Qwen2.5-3B 4-bit GPU model is a load generator, not a same-model reference for a different ANE Qwen3.

For R1/R2:

1. Create deterministic random, zero, basis-vector, outlier and real activation cases. Basis inputs verify channel ordering; sentinel/padding tests verify no output contamination.
2. Compare against FP32 reference. Record relative L2 error, RMS error, maximum absolute error and nonfinite counts. Cosine similarity alone cannot certify correctness.
3. Freeze tolerances before testing modified ANE lowerings. Initial engineering ceilings: relative L2 error <= 0.002 for a projection and <= 0.005 for the FFN, with normalized max error <= 0.02, where max error is divided by max(1, reference RMS). Zero-reference cases use an explicit absolute tolerance of 0.001. These are proposed test budgets, not hardware guarantees; reference-rounding analysis may justify a documented change before ANE comparisons.
4. Compare the modified path against both the FP32 reference and the reproduced native path. Repeat across at least five seeds and the three useful S values; run 1,000 repeated calls on the selected shape to test buffer ownership and stability.
5. Time cold compilation/load, warm dispatch/evaluation, packing/unpacking, synchronization, and complete invocation separately. Label host-observed evaluation time accurately; it is not a hardware cycle counter.
6. Evaluate/synchronize lazy MLX work before stopping its timer; equalize warmup and exclude fixture generation from timed calls.
7. Use three independent warm timing windows, randomized order and enough iterations for at least ten seconds per window. Record p50/p95, count, failures, useful operations and padding overhead. Treat sub-window repetitions as correlated for uncertainty.
8. Capture full weight/source hashes, compiler flags, model config/tokenizer revisions, chip, OS build, shape, dtype, dispatch count and memory observations. The native placement trace plus a successful ANE probe is primary execution evidence; aggregate power activity alone is not proof of per-operator placement.

Cost accounting:

```text
T_call = T_pack + T_handoff + T_eval + T_sync + T_unpack
T_model = sum(actual block costs and boundaries) + CPU remainder + sampling
```

Do not sum overlapping timestamps twice. For serial dispatch models, derive a local fit from measurements rather than assuming a universal 70–90 microsecond cost. Use a measured baseline fraction f to bound possible speedup by 1 / ((1-f) + f/s), then subtract new transfer/synchronization costs. Reject an isolated fast kernel that makes the complete model slower unless it delivers a separately validated energy benefit.

Energy is required for an efficiency claim. Discover supported powermetrics samplers on this machine, keep inference unprivileged, integrate a consistent package/rail scope across repeated workloads and include CPU work and ANE where exposed. Avoid summing overlapping power categories. Use long enough runs to resolve small differences and report idle reference, sampling limits and uncertainty. Report missing energy as unavailable, never zero.

## Bounded slices and entry gates

One active slice at a time. Proposed effort caps are recorded here; at the cap, report the result or replan rather than silently broadening scope.

| Slice | Work and evidence | Exit / decision | Effort cap |
| --- | --- | --- | --- |
| R0: native feasibility | Pin ane.cpp, inspect its actual headers/build, run one existing native operation on this Mac; produce environment/provenance record. | Known compile/eval path with observed placement, or reproducible incompatibility finding. | 1 day |
| R1: projection lowering | Implement the visible matrix-to-convolution probe, use one model-derived shape and S=1/8/32, compare CPU/MLX references and full timing. | Numerical gate passes, repeated execution stable, cost breakdown identifies the next uncertainty. | 2 days |
| R2: one subgraph experiment | Reproduce the existing FFN path and change only the measured bottleneck. Record MIL/layout/dispatch difference and before/after results. | Retain optimization only if >=10% lower complete-invocation p50 with no p95 regression above 5%, outside observed noise, and correctness intact. Otherwise retain the unmodified path and finding. | 2 days |
| R3: useful small model | Run a locally verified small dense Qwen3 candidate through the hybrid native path; select bounded structured extraction as the first non-classifier task. Compare same-model CPU/GPU controls as feasible. | Correct task completion and device map demonstrated; sustained resource/efficiency comparison supports or rejects model placement. | 3 days, after R0–R2 review |
| R4: concurrent GPU + ANE | Keep MLX foreground generation fixed; move the small-model background extraction job between matched GPU and ANE paths. Use actual overlap traces. | System adoption gate below holds, or record why offload does not help. | 2 days, after model correctness |
| R5: gRPC demonstration | Serve the validated model through an isolated host-native Generate endpoint and the existing NEXUS wire schema. | A client completes the selected task, observes real backend metadata, and handles cancellation/failure correctly. | 1 day, only for a useful backend |

R2's optimization threshold does not gate the feasibility of running a small model on an already adequate upstream backend. If reproduction works but our change loses, R3 may use the original path after the recorded review. A private-API failure is not a reason to invent bindings; an ANEForge substitution needs an explicit scope trade-off and new reproducibility record.

For R3 choose a small candidate such as Qwen3-0.6B only after the pinned runtime and artifact pass compatibility checks; “Qwen-family support” is insufficient. Small-model decode is in scope, but a hybrid operation map is expected. Native attention and an entirely ANE-resident model are later hypotheses, not required branding.

## Model/task and concurrent-system gates

Use a versioned set of 30 bounded extraction requests over supplied synthetic text: 20 clear cases, five missing/ambiguous-field cases and five cases with misleading quoted instructions. Score schema validity and exact normalized fields against fixed labels. No real tools or external data are needed. Keep a separate development set and freeze evaluation before comparisons.

A fixed small-model prompt and output cap must be shared across device placements. This task proves useful generation beyond routing without making model capability differences look like accelerator performance.

Before R4, require at least 27/30 correct task outcomes, with no unexplained regression against the matched reference; inspect all failures. This is a local engineering gate, not a population-wide accuracy estimate. For generative fidelity also compare fixed teacher-forced logits/loss or token rankings where the runtimes expose them; matching a few decoded strings is insufficient to validate all numerical rewrites.

For R4 run at least three five-minute windows per configuration, alternating order:

- MLX foreground alone.
- MLX foreground plus same-small-model background on GPU.
- MLX foreground plus same-small-model background on ANE/CPU.
- Record both-resident/idle condition if model residency itself changes the result.

Freeze offered foreground/background arrivals and output bounds. Count correctly completed background tasks / offered background tasks; count correctly completed tasks / joules over the same measured interval, including energy spent on failed attempts. Report foreground submission-to-first-visible-token p50/p95, visible-chunk gaps (token gaps only if actual token timing is available), deadline misses, background completion latency and throughput. No improvement may be obtained by discarding difficult requests.

System adoption target: >=15% higher correct background tasks/J than the same-model GPU placement, while foreground p95 TTFT regresses by no more than 5%, correct-background completion rate stays >=90%, and quality/resource guards hold. If valid energy data are unavailable, record a feasibility/latency result and leave the efficiency/adoption decision open. CPU comparison is added where needed to distinguish model/runtime changes from accelerator placement.

Retain the 26 GB combined worker planning ceiling on the 36 GB host, reducing it for other resident consumers. Collect MLX allocation, native process/IOSurface-related allocation where measurable, system pressure and swap; avoid double counting shared pages. No sustained swap growth, cross-session cache leakage or unbounded compile cache. Keep budgets conservative during compile/load peaks.

## Implementation boundaries and integration

Reuse the existing flat prototype and results convention. Add only files required by each slice; this tree is planned work:

```text
prototype/
  ane/
    probe.cpp             # native compile/eval experiment, plain machine-readable results
    lowerings.cpp         # projection and the one selected FFN lowering
    reference.py          # CPU/MLX equations, fixtures, error checks
    benchmark.py          # cold/warm timing and cost comparison
  generator_ane.py         # R3/R4 only: model-level adapter
  grpc_demo.py             # R5 only: request-level serving
tests/
  test_ane_numerics.py     # hardware-marked; explicit skip when unavailable
  test_ane_lifecycle.py    # only once model adapter exists
results/ane-<experiment>/
  manifest.json
  measurements.jsonl
  summary.json
  graph.mil
```

Upstream source/build lives in a pinned prototype-owned vendor area when implementation starts. Keep modifications as a small reviewable patch. Do not fork MLX/llama.cpp or build a general ggml-ANE backend to obtain the first measurement.

The model worker owns weights, compiled programs, surface lifetimes and KV state. Python chooses a model-level request, not an RPC route for every operator. A future in-process mixed MLX/ANE block is justified only if explicit buffer interop and synchronization beat the separate-model path. Unified physical memory alone does not establish zero-copy interoperability.

R5 uses one prototype-owned loopback endpoint (for example :55171 after checking availability). Keep tensor execution and IPC inside the native worker. Existing NEXUS proto provides Generate, GenerateBatch, GetActiveModel and ListModels; implement only demonstrated capabilities, returning UNIMPLEMENTED for unsupported RPCs. Field 7 already carries backend; do not introduce the conflicting proposed field-9 extensions. Preserve token/is_final semantics, validate claimed JSON output and expose truthful hybrid placement via prototype metadata until a production schema decision.

Cancellation is cooperative at verified safe boundaries. A Python timeout is not evidence a native evaluation stopped; retain buffer ownership until completion or confirmed worker exit. Mark unhealthy workers unavailable, bound restarts/admission, and never replay a partially streamed answer silently. Clean session state before reuse. Recover by disabling the ANE adapter and using the existing MLX path; preserve experiment evidence.

NEXUS remains read-only throughout these slices. Its [source manifest](nexus-source-manifest.json) identifies the earlier inspected working tree. Production deployment, framework-wide routing changes and model-tier migration require a separate integration decision.

## R0 and R1 results (2026-09-19)

Code: `prototype/ane/` (vendor script and pin, CMake for the upstream runtime core only, `probe.cpp`, `lowerings.{h,cpp}`, `bench_main.cpp`, `shapes.py`, `reference.py`, `benchmark.py`), test `tests/test_ane_numerics.py` (hardware-marked). Records: `results/ane-r0-feasibility/`, `results/ane-r1-projection/`. Setup notes in `prototype/README.md`.

**R0 — passed.** ane.cpp pinned at `b87bf57356c2` (2026-04-25, MIT); only `core/ane_runtime.cpp` is built, with two `Timer` statics supplied in `compat.cpp`; Ninja generator (Make fails on the colons in the workspace path). On macOS 26.5 / Xcode 26.2 the private classes `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject` resolve and the compile → load → evaluate path works. For a 1024→2048 fp16 projection: cold compile 33–64 ms, marker-cache load 9–10 ms, evaluate 102–105 µs p50 (host wall), rel L2 3.9e-4 against an fp32 reference. The runtime's model attributes report Float16 planes, Width 32, PlaneStride 64 B, which is the `[1,C,1,32]` layout upstream assumes. Placement by power rail was not captured (needs root).

**R1 — passed.** Real BF16 weights from `Qwen/Qwen3-0.6B` layer 0 (`k_proj` 1024×1024, `q_proj` 2048×1024, `gate_proj` 3072×1024; full tensor and shard sha256 in the manifest). Our visible lowering emits MIL byte-identical to upstream `mil_gen_matmul`, so the upstream path is reproduced, not approximated.

| Result | Value |
| --- | --- |
| Numerics | 360 case evaluations (3 shapes × S ∈ {1, 8, 32} × 5 seeds × 2 paths × 4 fixtures); 0 gate failures. Worst rel L2 3.5e-4 vs fp32, 2.7e-4 vs fp16-rounded-input reference; worst normalized max error 3.7e-3; 0 nonfinite. Basis vectors confirm channel order; sentinel-filled padding lanes compute independently and never leak into valid rows. |
| Stability | 1000 repeated calls per shape, 0 failures, 0 output-checksum drift. |
| Per-call breakdown, S = 32, p50 µs | pack 10, surface write 2, evaluate 76 / 93 / 123 (k / q / gate), output read 3–7, transpose 12–31. Three ≥10 s windows per path, randomized order; upstream and visible agree within 2 µs. |
| Local cost fit (evaluate vs fp16 weight bytes) | **≈ 50 µs fixed + 11.2 µs/MB (≈ 89 GB/s effective streaming)**. Refits across runs ranged 50–61 µs fixed and 7.8–11.8 µs/MB; treat both as ±20 %. Compile-cache load 8–15 ms per kernel. |
| Upstream `ane_matvec`, S = 1, full call | 81 / 101 / 136 µs p50 (k / q / gate). |
| MLX fp16 GPU, same matmul | 143–215 µs p50: launch-bound at these sizes, a scale reference only. |

Caveats recorded: the outlier fixture (±20 000 inputs) produced no nonfinite output because these weights keep products far below the fp16 output saturation ANEForge documents near 32 752, so saturation was not exercised; the real-activation fixture (MLX hidden-state capture) was deferred within the cap; energy and per-operator placement remain unmeasured. One harness defect was found and fixed: the first output transpose loop cost 176 µs for N = 2048 and was briefly mistaken for an ANE cost until the timer was split (surface read is 3–7 µs). The tiled transpose is 12–31 µs.

**What the fit implies for a small model.** Per Qwen3-0.6B layer on the upstream kernel map (fused QKV 8.4 MB, o_proj 4.2 MB, SwiGLU FFN 18.9 MB): predicted ANE evaluate ≈ 3 × 50 µs + 31.5 MB × 11.2 µs/MB ≈ 500 µs, of which ≈ 70 % is weight streaming at ≈ 89 GB/s (about one sixth of the M4 Max DRAM bandwidth) and ≈ 30 % fixed dispatch. Over 28 layers that is ≈ 14 ms per decoded token before CPU norms, attention and sampling. Those are the two measured bottlenecks: effective streaming rate for large kernels, fixed dispatch cost for small ones.

**R2 choice (written before R2 code, per plan).** Reproduce the upstream fused SwiGLU FFN (`ane_compile_fused_ffn`, layer-0 shapes, 18.9 MB) as the baseline; verify against fp32 with the FFN gates (rel L2 ≤ 0.005, normalized max ≤ 0.02) and check the evaluate time against the fit's 261 µs prediction. The one change: **intermediate-channel chunking** via upstream's chunked-accumulate path forced to 2 and 4 chunks, to determine whether smaller per-dispatch weight sets stream at a higher effective rate or only add ≈ 50 µs per extra dispatch. Prediction from the fit: chunking loses by ≈ 50 µs per added chunk unless the per-chunk streaming rate rises by more than that. Retention rule unchanged: keep only ≥ 10 % lower complete-invocation p50 with no p95 regression above 5 % and correctness intact; otherwise record the negative result and retain the unmodified path. If the chunking result is unambiguous within one day, the remaining R2 day goes to measuring the fixed dispatch cost against the runtime's exposed options (QoS, direct evaluate, queue depth), since that cost bounds small-kernel decode.

### Change note (instructions.md §6)

1. **Outcome:** revision 5 outcome brief; immediate question answered for the projection: it runs correctly on this M4 Max through the pinned native runtime with a measured cost model.
2. **What fails without it / deferred:** without R0/R1 no ANE execution or cost evidence existed. Deferred as planned: FFN, RMSNorm, quantization, Python binding, model loading, gRPC, energy.
3. **Mechanisms and constraints:** one native executable per experiment (fault containment of a private runtime; not a service). Ninja instead of Make: constraint is the workspace path. `compat.cpp`: two statics, because the upstream TU that defines them needs Rust and network.
4. **Verified:** numerics gates, lane isolation, 1000-call stability, MIL identity, hardware-marked test; `pytest tests -q` passes with the ANE test skipped when the build is absent.
5. **Baseline and result:** upstream kernel is the baseline and the visible path matches it; no speedup is claimed. The cost fit is the deliverable.
6. **Release / recovery:** results are self-describing under `results/ane-*`; recovery is deleting `prototype/ane/build`, `prototype/ane/vendor`, and the marker cache under `~/Library/Caches/ane_lm/`. NEXUS untouched.

## Trial: 27B on the GPU and Qwen3.5-4B on the ANE, one task (2026-09-19)

Requested by the outcome owner ahead of R2; recorded as a manual feasibility trial (guideline §1 rule 4), not the gated R3/R4 measurement. Script `prototype/scenario_parallel.py`, fixture `prototype/scenario_notes.json`, results `results/scenario-27b-gpu-4b-ane-run1/` (batched ANE sessions) and `results/scenario-27b-gpu-4b-ane-run2-seq/` (one ANE session).

Task: extract one commitment (who/when/where/action as JSON) from each of 8 synthetic notes, draft a priorities paragraph, compose a briefing. Conditions: GPU does everything serially; ANE extracts while the GPU drafts, then the GPU composes; ANE extracts, then GPU drafts and composes (no overlap).

**Environment facts established on the way.**

- The full ane.cpp binary needs Rust (tokenizers-cpp); Rust was installed with Homebrew. CMake's dependency fetcher splits paths on colons, so the full binary is built outside the workspace in `~/.cache/ane-cpp-build-full/` and linked as `prototype/ane/ane.cpp-bin`.
- Model loads failed at arbitrary layers with "ANE compile failed" and, once, "Compiler internal error: ZinObjectGenWriteSymbolData failed". Root cause: the data volume had 5.4 GB free. The runtime writes an fp16 copy of every kernel's weights under `$TMPDIR` and keeps the ANE compile cache, so a 9.3 GB model needs well over 10 GB free. After space was freed the same models loaded. Qwen3.5-4B's first load took 41 s and 8 GB of disk; warm serve startup is 14–37 s.
- The workspace moved into iCloud Drive during cleanup. The 30 GB model cache is now `cache.nosync/` (the suffix keeps it off iCloud); code defaults and the README were updated. Native builds were reconfigured at the new path; 29 tests pass.
- Both checkpoints are `Qwen3_5ForConditionalGeneration` with the text model under `text_config`; ane.cpp's Qwen3.5 loader handles that layout.

**Standalone numbers (no other model loaded).**

| Model / device | Load | Memory | Prompt tok/s | Generation tok/s | Extraction JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B, ane.cpp (hybrid ANE+CPU), one session | 14–41 s | 4.8 GB steady, 9.2 GB peak | 25 | 11.5 | valid and correct on every sequential request |
| Qwen3-0.6B, ane.cpp | 7 s | 2.0 GB | 165 | 24.7 | valid |
| Qwen3.8-27B 4-bit, MLX GPU | 5–110 s | 18.1 GB active | (TTFT 0.55 s short prompt) | 17.7 | valid 8/8 |

The ANE path's prompt processing (25 tok/s) is its bottleneck: a 94-token note costs ≈3.7 s before the first output token, about as long as the 48-token answer.

**Finding 1 — batched serve sessions corrupt output.** With `serve --sessions 4` and concurrent requests, only the first lane of each batch decodes correctly; lanes 2–4 produce garbage (`{ "<think>": "... " "" - 2, ""...`). Reproduced in isolation without the GPU model loaded (`results/…run1` extraction: 1/8 valid; diagnostic: 4 sequential requests 4/4 valid, two batches of 4 concurrent requests 1/4 valid each). Sequential requests through the same server are always valid. This is a correctness defect in upstream's multi-session batched decode for Qwen3.5 on this build and OS, not in the single-projection lane isolation verified in R1. Consequence: the ANE worker must run one request at a time until this is fixed, and its throughput is then 11.5 tok/s with no batching gain. Candidate for a later slice; not pursued in this trial.

**Finding 2 — memory guardrail breached with the Docker VM resident.** With the 27B (18.1 GB), the ANE server (4.8 GB) and the Docker VM (3.8 GB) resident, swap grew from 0.4 GB to 12.8 GB during run 1. The 26 GB worker budget was met on paper (23 GB) but the host total was not. Run 2 was made after the VM was stopped; see its swap column.

**Run 1 (batched sessions, invalid).** GPU-only 98.8 s wall (extraction 80.8 s, 16/32 fields by the strict rubric, 8/8 valid JSON); parallel 57.4 s wall but 1/8 valid JSON, so the speed is meaningless; GPU decode while the ANE was active 15.3 tok/s vs 17.7 idle (≈14 % contention during the draft).

**Run 2 (one ANE session).** All 8/8 extractions valid on every condition. Docker VM was still resident; swap again reached 12–13 GB.

| Condition | Wall s | Extraction phase s | Fields correct | GPU decode tok/s during draft | Compose tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU only, serial | 116.3 | 98.3 | 16/32 | 17.7 | 17.7 |
| ANE extracts while GPU drafts, GPU composes | 83.4 | 66.7 | 18/32 | 16.9 | 17.9 |
| ANE extracts, then GPU drafts and composes | 82.4 | 63.4 | 18/32 | 17.6 | 17.8 |

Reading: moving extraction to the ANE finished the task 28 % sooner than the 27B doing everything, with equal or slightly better field accuracy, because a 4B model at 11.5 tok/s beats a 27B at 17.7 tok/s once prompt cost is included and the GPU is freed. The overlap itself bought almost nothing (83.4 vs 82.4 s) because the GPU's independent work (an 8 s draft) is tiny next to 67 s of ANE extraction; GPU decode under ANE load dropped ≈5 % (16.9 vs 17.7). The GPU-only extraction was slower here than in run 1 (98 vs 81 s) with swap at 6–13 GB, so run-to-run noise under memory pressure is large. One run per condition; no energy.

Decision for the next slice: (1) the batched-session defect (Finding 1) is the blocking item for any ANE throughput claim and is the natural R3 question; (2) rerun this trial with the Docker VM stopped and three repetitions before quoting any number; (3) ANE prompt processing at 25 tok/s is the cost to attack if extraction stays on the ANE.

## R1 closed and R2 results (2026-09-19)

**R1 closed.** The deferred real-activation fixture is in: 32 tokens through layer 0 of Qwen3-0.6B captured with MLX (`prototype/ane/activations.py`; attention input rms 0.211, abs max 3.05). Rerun with 378 cases, 0 gate failures; the real-activation rows are worst rel L2 2.8e-4 to 4.8e-4. Random fixtures are now scaled to the real rms. Cost refit: 61 µs + 7.0 µs/MB this run (range over all runs: 50–61 µs fixed, 7.0–11.8 µs/MB); the fixed dispatch cost is the stable number, the slope is ±35 % between runs. Record: `results/ane-r1-projection/`.

**R2 — SwiGLU FFN, layer 0 of Qwen3-0.6B (dim 1024, intermediate 3072, 18.9 MB fp16).** Code: `prototype/ane/ffn_main.cpp`, `ffn_benchmark.py`, `mil_main.cpp` (generic MIL probe), `silu_probe.py`; record `results/ane-r2-ffn/`. Baseline reproduced: upstream `ane_compile_fused_ffn`, one dispatch, evaluate 217–223 µs p50 at S = 32 (fit predicted 261), call 242–251 µs, 1000 calls with 0 drift.

| Variant | Dispatches | Evaluate p50 µs | Call p50 / p95 µs | Worst rel L2 (random, real) | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| upstream fused | 1 | 217–223 | 242–251 / 256–258 | 2.2e-2 | baseline; **fails the frozen 0.005 gate** |
| chunked ×2 (change under test) | 2 | 284 | 366 / 391 | 2.2e-2 | reject: +48 % call p50 |
| chunked ×4 | 4 | 399 | 545 / 584 | 2.2e-2 | reject: +121 % |
| fused, SiLU as g/(1+exp(−g)) | 1 | 219–222 | 243–250 / 257 | 2.6e-3 | **retain**: error ÷8.5, time within 0.2 % |

Chunking answered the R2 question as the fit predicted: each extra dispatch costs ≈50–60 µs and the per-chunk streaming rate does not rise, so splitting the intermediate dimension only adds fixed cost. Rejected; the unmodified single-dispatch path stays.

The numerical result was not planned and is the more useful one. Upstream's fused FFN misses the frozen FFN gate (2.2e-2 vs 0.005) while the 2- and 4-chunk variants agree with it to 4e-4, so the error is not in the reduction. Sub-graph probes (`mil_main`) localized it: gate conv 2.7e-4, down conv 2.5e-3, but the ANE `sigmoid` op alone is 3.2e-3 with an absolute error up to 3.4e-3 across the whole input range, and the `silu` op lowers identically. SwiGLU amplifies that in the negative tail where g·σ(g) is small. Rewriting the activation as g/(1+exp(−g)) (MIL `exp`, `add`, `real_div`) gives 2.6e-3 on real activations at the same dispatch count and time; the tanh form gives 2.9e-3. A reference-rounding analysis was done first and ruled out fp16 intermediate rounding and fp16 accumulation (both explain < 6e-4), so the gate was kept, not loosened.

Gating notes recorded in the summary: the basis-vector case is reported but not gated for the FFN (its reference rms ≈ 8e-4 sits at fp16's smallest normal magnitudes, which is why R1 used it only for channel order); the ±20 000 outlier case saturates fp16 intermediates by design and shows nonfinite outputs on every variant.

**Retained change and how it is carried.** The exp-form lowering lives in `prototype/ane/lowerings.cpp` (`emit_ffn_mil`, `compile_ffn`) and is the FFN path the prototype uses. A proposed upstream patch for `mil_gen_fused_ffn` and `mil_gen_ffn_chunk_accum` is in `prototype/ane/vendor/patches-proposed/0001-ffn-silu-exp-form.patch`; it is **not** applied by `vendor.sh`. Applying it to the model path (`ane.cpp-bin`) is an R3 decision that needs a model-level check first, because the two other generators containing the same pair (residual-add and O-proj-fused variants) were not measured, and the model-level effect on token outputs has not been tested.

### Change note (instructions.md §6)

1. **Outcome:** R2 of revision 5; the immediate question (does an examined lowering improve the complete invocation) is answered for the FFN: not in time, yes in accuracy.
2. **Without it / deferred:** without R2 the FFN error floor would have gone into R3 unnoticed. Deferred: RMSNorm on ANE, quantized weights, QKV concatenation, energy, model-level validation of the patch.
3. **Mechanisms:** one new generic probe executable (`mil_main`) with the same constraints as before; a patch file, unapplied. No new runtime dependency.
4. **Verified:** 252 FFN cases over 5 seeds × 3 S values × 4 variants against float64; sub-graph localization; 1000-call stability; 3 timing windows per variant, randomized order; 29 tests pass.
5. **Baseline and result:** upstream fused kernel is the baseline for both time and error; results table above; measurements in `results/ane-r2-ffn/measurements.jsonl`.
6. **Recovery:** delete `results/ane-r2-ffn`, the build directory and the marker cache; the vendor tree is unmodified.

**Next (R3 entry, cap 3 days):** run a small model through the hybrid path with the retained FFN lowering applied as a patch, and compare token-level outputs against the unpatched binary and an MLX reference on the frozen extraction set before any placement claim. The batched-session defect found in the trial is the first R3 check, since it decides whether the ANE worker can batch at all.

## R3 (part 1): does the FFN kernel correction survive integration? (2026-09-19)

Entry corrections adopted from review: Qwen3 compiles the residual-add FFN first and uses the plain fused FFN only as a fallback, so an unforced patched/unpatched comparison would have run identical kernels; and `serve` batching is Qwen3.5-only, so its defect stays a separate check (one session until verified) and does not bear on Qwen3-0.6B.

**Setup.** `prototype/ane/r3/`: frozen 30-case extraction set (`extraction_eval.json`: 20 clear, 5 missing-field, 5 misleading-instruction; verbatim-span labels, JSON null for missing, normalized exact match), `score.py`, two instrumented builds from the pinned tree (`build_variants.sh`, patch records in `r3/patches/`): `trace` = upstream + passive per-step top-16 logit trace (`ANE_TRACE_PATH`) + FFN kernel-map log + `ANE_FORCE_PLAIN_FFN` runtime switch; `exp` = trace + the proposed exp-form patch. Conditions share weights (Qwen3-0.6B BF16), prompt tokens (chat template, thinking off) and greedy decoding; `mlx_reference.py` is the GPU reference with the same top-16 trace. Record: `results/ane-r3-model/`.

| Condition | FFN kernels actually run | Schema valid | All fields | Fields | Gen tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| upstream kernel map (trace build) | resadd = 28 | 30/30 | 3/30 | 75/120 | 24.5 |
| control: plain fused, sigmoid form | fused = 28 | 30/30 | 3/30 | 74/120 | 23.8 |
| experimental: plain fused, g/(1+exp(−g)) | fused = 28 | 30/30 | 3/30 | 75/120 | 23.1 |
| MLX GPU reference (BF16) | — | 30/30 | 3/30 | 68/120 | — |

| Pair | Identical token sequences | Top-1 agreement on identical prefix | Mean / max centered top-16 logit delta |
| --- | ---: | ---: | ---: |
| control vs experimental | 26/30 | 0.967 | 0.105 / 0.448 |
| upstream map vs control | 29/30 | 0.967 | 0.034 / 0.133 |
| control vs MLX | 22/30 | 0.915 | 0.424 / 3.83 |
| experimental vs MLX | 21/30 | 0.948 | 0.393 / 3.68 |
| upstream map vs MLX | 23/30 | 0.948 | 0.409 / 3.78 |

**Reading.**

1. The kernel correction survives integration: the experimental build runs the corrected FFN in all 28 layers with no schema, field or speed regression, and its logits sit slightly closer to the GPU reference (top-1 agreement 0.948 vs 0.915; mean centered delta 0.39 vs 0.42). It does not produce a model-level accuracy gain on this set.
2. The FFN activation error is not the dominant source of ANE-vs-GPU divergence. Swapping the activation moves the logits by 0.1 on average; the ANE path as a whole differs from MLX by 0.4, so most of the gap comes from elsewhere in the fp16 pipeline (other kernels, the residual-add variant, norms and attention on the CPU, fp16 activations everywhere). Localizing that is the next kernel question.
3. The three control-vs-experimental divergences all occur at steps where the two leading candidates are within about 1.4 logits; net field change is zero.
4. Task accuracy is flat at 3/30 for every condition, including the BF16 GPU reference: the 0.6B model drops the clock time in 15/30 "when" fields ("Tuesday" for "Tuesday 09:30"). Under the guideline's decision table this is "technical behaviour works but value is absent": the model and prompt, not the kernels, limit the task. The R3 correctness gate (≥ 27/30) is not met by any device, so R4 (concurrent placement, tasks per joule) is not entered on this model.

**Decision.** Keep the exp-form lowering as the prototype's FFN path and keep the upstream patch proposed, now with model-level evidence of no regression; still not applied to the model binary by default (the residual-add and O-proj-fused generators carry the same sigmoid pair and were not measured). Before R4, change the model or prompt, not the kernels: candidates are Qwen3-1.7B (same family, same tooling) or a Qwen3.5 dense model (enables the batched serve path once its lane defect is verified). Re-run this exact harness first; the R3 gate must be met by the MLX reference before any ANE placement claim is attempted. The batched-session lane defect remains the first Qwen3.5 check.

Qualification carried forward: the retained FFN lowering is validated on random and real activations; the basis (fp16-range) and extreme-input cases remain excluded from its gate.

## R3 (part 2): the same harness on Qwen3.5-4B (2026-09-19)

Requested rerun on the larger model. Harness changes: the Qwen3.5 loader received the same instrumentation as Qwen3 (force-plain switch, kernel-map log); a `--prompts-file` batch mode runs all 30 cases in one process with a model reset between cases (one 4B load per condition instead of thirty); the MLX reference loads the same BF16 checkpoint directly. Record: `results/ane-r3-model-4b/`; patches in `prototype/ane/r3/patches/`.

| Condition | FFN kernels actually run | Schema valid | All fields (strict) | Fields (strict) | Clear / missing / misleading | Gen tok/s |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| upstream kernel map | resadd = 32 | 30/30 | 10/30 | 87/120 | 4 / 2 / 4 | 11.3 |
| control: plain fused, sigmoid | fused = 32 | 30/30 | 10/30 | 87/120 | 4 / 2 / 4 | 11.3 |
| experimental: plain fused, exp-form | fused = 32 | 30/30 | 10/30 | 86/120 | 4 / 2 / 4 | 11.2 |
| MLX GPU reference (BF16) | — | 30/30 | 8/30 | 73/120 | 4 / 0 / 4 | — |

| Pair | Identical sequences | Top-1 agreement on identical prefix | Mean / max centered top-16 logit delta |
| --- | ---: | ---: | ---: |
| control vs experimental | 29/30 | 0.990 | 0.26 / 1.91 |
| upstream map vs control | 30/30 | 1.000 | 0.08 / 2.87 |
| control vs MLX | 20/30 | 0.949 | 1.44 / 4.91 |
| experimental vs MLX | 19/30 | 0.940 | 1.43 / 4.64 |

**Reading.**

1. Same conclusion as on 0.6B, now on the model that matters: the exp-form FFN runs in all 32 layers with no regression in schema validity, fields or speed, and one changed token sequence in 30 (net one field). It does not move the model measurably closer to or further from the GPU reference (0.940 vs 0.949 top-1 agreement, within run noise). The correction is safe at model level and does not buy model-level accuracy; its value stays the 8× kernel-level error reduction.
2. Residual-add fused FFN and plain fused FFN produce identical tokens on all 30 cases (30/30), so upstream's default kernel choice is not a numerical liability on this model.
3. The ANE hybrid path diverges from the BF16 GPU reference more on the deeper model (mean centered logit delta 1.4 vs 0.4 on 0.6B; identical sequences 20/30), yet its task scores are equal or higher (10 vs 8 strict; 15–17 vs 17 under a lenient containment rule). No accuracy penalty from ANE placement is visible on this task.
4. The R3 gate (≥ 27/30) is still unmet by every device, including the reference, and the remaining failures are shared: label-boundary mismatches ("the Clinic on Harbour Street" vs "Clinic on Harbour Street"; "team lunch is confirmed" vs "team lunch"), the model's tendency to answer null for fields it should fill on "missing" cases, and a few genuine errors identical across devices. Under the lenient rule the ceiling is 17/30. The limit is the fixture's exact-span labels and the prompt, not the kernels.

**Decision.** R3's question is answered: the kernel correction survives integration on both a dense Qwen3 and a hybrid Qwen3.5 model. Kernel work is not the next lever for this task. Before R4 (concurrent placement, tasks per joule): revise the frozen fixture so labels tolerate determiners and verb framing (or score with a canonical normalization defined per field), re-freeze it, and require the MLX reference to reach the gate; keep one ANE session until the Qwen3.5 batched-lane defect is verified; and pick the next kernel question from the fp16 divergence that grows with depth (per-layer logit drift against the GPU reference), not from the FFN activation.

## R3 (part 3): fixture v2 and the R4 entry decision (2026-09-19)

**Fixture revision.** `prototype/ane/r3/extraction_eval.json` is now version 2 (v1 kept as `extraction_eval_v1.json`). Changes, all frozen before the rerun: the prompt defines "who" as the person or organisation the note comes from; the scorer strips leading determiners and prepositions and a trailing comma-qualifier and accepts the label contained in the prediction with at most 3 extra words (4 for action); six fields carry listed alternates found in the v1 failure review ("assemble" for the fire drill, "return your books" for "books are due back"); null for a stated field and predictions that drop label content stay wrong. A fixture defect was also fixed: the missing-field and misleading cases shared ids `mis-0X` and overwrote each other's per-case files, which had corrupted the MLX reference's score in the first v2 run (its five missing-field cases were scored against the wrong outputs); they are now `mss-0X`.

**Clean rerun on Qwen3.5-4B** (`results/ane-r3-model-4b-v2/`; prompts verified token-identical across all conditions, 107 tokens, same empty think block):

| Condition | FFN kernels | Schema valid | All fields | Fields | Clear / missing / misleading |
| --- | --- | ---: | ---: | ---: | --- |
| upstream kernel map | resadd = 32 | 30/30 | 24/30 | 113/120 | 17 / 3 / 4 |
| control: plain fused, sigmoid | fused = 32 | 30/30 | 24/30 | 113/120 | 17 / 3 / 4 |
| experimental: plain fused, exp-form | fused = 32 | 30/30 | 24/30 | 113/120 | 17 / 3 / 4 |
| MLX GPU reference (BF16) | — | 30/30 | 23/30 | 111/120 | 17 / 2 / 4 |

Token-level: control vs experimental 29/30 identical sequences; ANE vs MLX 19/30 identical, top-1 agreement on identical prefixes 0.978, mean centered logit delta 1.34.

**Reading.** The ANE hybrid path and the BF16 GPU reference are equivalent on this task (24 vs 23, 113 vs 111 fields; the same 5 cases fail on both plus one each). The earlier apparent ANE advantage was the id collision. The kernel-level result is unchanged: the exp-form FFN is neutral at model level. The remaining six ANE failures are two label judgment calls ("Maya from the design team" exceeds the 3-extra-word tolerance; "somewhere quiet" where the label says null) and four model errors shared with the reference ("Facilities:" not recognised as a sender, a null action, "dentist appointment" as a place). Accepting the two judgment calls would give 26/30, still under the gate.

**Decision.** The R3 correctness gate (≥ 27/30 on the MLX reference) is not met: 23/30. R4 is not entered. Per the guideline, the next lever is the task definition, not the kernels or the placement: a v3 with one worked example in the prompt (few-shot) is the cheapest next experiment, to be frozen before running and judged first on the MLX reference. Fixture tuning after seeing outputs stops here, to avoid fitting the fixture to the model. Standing facts for R4 when it opens: ANE placement shows no accuracy penalty on this model and task; the ANE worker stays at one session until the Qwen3.5 batched-lane defect is verified.

## R4: concurrent placement (2026-09-19)

Entered on the outcome owner's decision despite the unmet R3 task gate, recorded as such: the task accuracy limit (24/30 vs 23/30 on the GPU reference) is a model-and-fixture property shared by both devices, not a hardware one. Code `prototype/r4/run_r4.py`; records `results/ane-r4-concurrent/` (run 1) and `results/ane-r4-gpu-rerun/` (GPU-side conditions rerun after a harness fix, see below); readable report `docs/R4-what-changed.md` (generated by `prototype/r4/what_changed.py`).

**Design.** Foreground = `Qwen2.5-3B-Instruct-4bit` on the GPU (MLX), one request every 8 s from a fixed 20-prompt list, 80 tokens max, 38 requests per window. Background = Qwen3.5-4B extraction over the frozen v2 fixture, cycled continuously: on the GPU in BF16 in the same process with token-boundary preemption by the foreground, or on the ANE via ane.cpp serve with one session. Conditions: fg alone, fg+bg GPU, fg+bg ANE, each 3 × 300 s with the start condition rotated per repeat; bg-alone baselines 120 s. Memory sampled every second. Residency rule: one background placement resident at a time. Energy: not measured (no root `powermetrics`; the runner integrates a log if supplied).

**Why the 3B foreground.** The 27B assistant (18.1 GB) plus the 4B ANE worker does not fit under the memory guardrail: in the 27B trial (`results/ane-r4-27b-fg-smoke/`) ANE windows raised swap by 6.7–13.4 GB and foreground TTFT p95 reached 6.9 s (86 s in the first, cold window); GPU windows stayed flat once residency was managed. The ANE worker costs 4.8 GB steady and up to 9.2 GB while loading, in addition to whatever the GPU holds.

**Harness defect found and fixed.** Run 1 retained every finished job's generator, and with it its KV cache, until the end of each window: MLX memory grew 9.5 → 22.9 GB in every GPU-placement window (≈58 MB per background task) and free memory fell to 0.03 GB, inflating the GPU-placement foreground p95 to 454 ms. ANE-placement windows were unaffected (MLX 1.6 → 2.0 GB). Generators are now released on completion; the GPU-side conditions were rerun in one session with a fresh foreground-alone control, and MLX memory then stayed at 9.5–9.7 GB throughout. The table uses the rerun for the GPU side and run 1 for the ANE side.

| Condition | fg TTFT p50 / p95 ms (worst window) | fg gap p95 ms | fg decode tok/s | bg correct / done / offered | bg correct per min | bg latency p50 / p95 ms | swap Δ MB (3 windows) | MLX active peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fg alone (rerun) | 159 / 168 (175) | 6 | 162.5 | — | — | — | −88 | 1.8 |
| fg + bg on GPU (rerun) | 126 / 281 (327) | 7 | 161.8 | 559 / 733 / 736 | 37.3 | 1159 / 1738 | −168 | 9.7 |
| fg + bg on ANE (run 1; fg alone in that run 168 / 187, 158.7 tok/s) | 154 / 168 (175) | 8 | 132.6 | 102 / 128 / 131 | 6.8 | 6937 / 7795 | −432 | 2.0 |
| bg on GPU alone (rerun, 120 s) | — | — | — | 77 / 100 / 101 | 38.5 | 1178 / 1327 | −8 | 9.7 |
| bg on ANE alone (run 1, 120 s) | — | — | — | 13 / 16 / 17 | 6.5 | 7123 / 8073 | −16 | 1.6 |

All foreground requests completed in every window; none exceeded 2 s to first token. Background correctness on the same fixture: GPU 76 %, ANE 80 % (same model; within fixture noise).

**Result theory.**

1. *Throughput.* The ANE path is 5.9× slower than the GPU for this 4B job (6.5 vs 38.5 correct tasks per minute alone; 6.8 vs 37.3 when the GPU also serves the foreground). Its ceiling is set by prompt processing (≈25 tok/s) and decode (11.2 vs 40 tok/s); the R1 cost model attributes ≈70 % of each ANE dispatch to streaming fp16 weights at ≈90 GB/s.
2. *Sharing the GPU is cheap.* With the foreground active the GPU still delivers 97 % of its solo background rate and the foreground loses no decode speed (161.8 vs 162.5 tok/s): foreground requests are short bursts and the scheduler hands every gap to the background at a token boundary.
3. *Foreground protection.* Background on the GPU leaves the median first token faster (126 vs 159 ms, consistent with the GPU staying in a high-clock state under continuous work; a hypothesis, not a measured clock) but raises the p95 from 168 to 281 ms (worst window 327): the background's prompt-processing step is not preemptible, and a foreground arrival during it waits. Background on the ANE makes foreground TTFT indistinguishable from an idle GPU (p95 168). The ANE's measured contribution to the outcome the project started for is that 95th-percentile difference, ≈110 ms; the typical request is unaffected either way.
4. *The ANE is isolated from the GPU, not from the SoC.* Foreground decode fell 16 % with the ANE worker active (158.7 → 132.6 tok/s within run 1) and 0 % with the background on the GPU. The ANE worker's CPU side (prompt processing, norms, attention and sampling on Accelerate) competes with the foreground's CPU-side token loop and with memory bandwidth. Under ANE placement the foreground starts faster at the tail and streams slower.
5. *Memory.* With residency managed and the harness fixed, swap stayed flat or fell in every window of both records. GPU placement holds 9.7 GB in MLX; ANE placement holds 2.0 GB in MLX plus a 4.8 GB second process. Free memory still bottoms near zero in GPU-placement windows on this 36 GB machine, so a larger foreground tips either placement into swap.
6. *Energy.* Unmeasured. Tasks per joule, the plan's adoption metric, remains open; it is the one quantity that could still make the ANE the right home for low-rate background work despite the throughput deficit.

**Decision against the plan's adoption gate** (≥ 15 % better correct tasks per joule than GPU placement, foreground p95 within 5 %, background completion ≥ 90 %): not decidable, energy unavailable. On the measured axes: foreground p95 is 40 % better under ANE placement than under GPU placement (168 vs 281 ms), foreground decode is 16 % worse, background throughput is 82 % worse, and memory is a second model copy. Interpretation for the project: the GPU with the slice-1 scheduler is the bulk background engine; the ANE is a candidate low-rate, foreground-isolating lane whose adoption hinges on energy. Two follow-ups are cheaper than any new mechanism: chunk the background's prompt processing on the GPU (a scheduler change that would likely remove most of the p95 penalty without the ANE), and rerun these windows once with a root `powermetrics` log to compute tasks per joule.

### Change note (instructions.md §6)

1. **Outcome:** the revision 4 outcome (foreground responsiveness under background work) measured under two placements; revision 5's R4 question answered on every axis except energy.
2. **Without it / deferred:** without R4 the ANE's system value was inferred from kernel numbers only. Deferred: energy measurement, chunked background prefill, batched ANE sessions, multi-machine or other-OS results.
3. **Mechanisms:** none new; the ANE worker is the already-justified separate process; the residency rule is a scheduling constraint, not a component.
4. **Verified:** 18 windows over two records (11 + 7), 570 foreground requests, ≈1,900 background tasks scored on the frozen fixture, per-second memory traces, one harness defect found by those traces and fixed, 29 tests passing.
5. **Baseline and result:** fg-alone and bg-alone baselines in the same session; the table above.
6. **Release / recovery:** nothing deployed; recovery is deleting `results/ane-r4-*`; models and caches unchanged.

## R4 addendum: energy measured (2026-09-19, evening)

A third full run, `results/ane-r4-energy/`, repeated all five conditions in one session with the leak-fixed runner while the outcome owner ran `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000` (3,246 one-second samples; every window fully covered). It is now the primary R4 record; the earlier two agree with it within run noise. Energy integration: `prototype/r4/energy.py` → `results/ane-r4-energy/energy.{json,md}`.

| Condition | fg TTFT p50 / p95 ms | fg decode tok/s | bg correct per min | avg W (CPU / GPU / ANE / all) | Energy J (3 × 300 s) | J per correct bg task | Marginal J per task above the fg-alone floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fg alone | 180 / 194 | 163.0 | — | 0.3 / 1.2 / 0.00 / 1.5 | 1317 | — | — |
| fg + bg on GPU | 126 / 296 | 162.2 | 37.7 | 1.5 / 17.0 / 0.00 / 18.5 | 16681 | 29.5 | 27.0 |
| fg + bg on ANE | 152 / 223 (161–162 in the two clean windows; 347 in the first, which absorbed a 4.5 GB swap-in as the ANE server loaded) | 135.6 | 6.8 | 1.0 / 1.4 / 1.42 / 3.8 | 3470 | 34.0 (≈31 in the clean windows) | 20.3 (≈18.5 clean) |
| bg on GPU alone (120 s) | — | — | 39.5 | 2.8 / 16.6 / 0.00 / 19.4 | 2325 | 29.4 | — |
| bg on ANE alone (120 s) | — | — | 6.5 | 0.9 / 0.0 / 1.43 / 2.4 | 292 | 22.5 | — |

**What the energy says.** The ANE rail draws 1.4 W while working; the whole system sits at 3.8 W under ANE placement against 18.5 W under GPU placement. Per correct background task the two placements are close on whole-system energy (34 vs 29.5 J, the ANE 15 % worse, or 5 % worse in its clean windows) because the ANE takes 5.5× longer per task; on marginal energy above the foreground-alone floor the ANE is 25–30 % better (20.3 vs 27.0 J). With no foreground at all the ANE alone costs 22.5 J per task and the GPU alone 29.4 J.

**Decision against the adoption gate** (≥ 15 % better correct tasks per joule than GPU placement; foreground p95 within 5 %; background completion ≥ 90 %):

- As written (whole-system energy over the same interval, failed attempts included): **not met**; the ANE placement is 13 % worse on correct tasks per joule (29.4 vs 33.9 per kJ). Foreground p95 is 25 % better than GPU placement (223 vs 296 ms; 45 % better in clean windows), so that clause is met with margin. Background completion is 98 % on both.
- For the outcome owner's stated use, a rate-limited long-running evaluator that adds a few tasks per minute to a machine that is on anyway, the relevant quantity is the marginal energy per task, and there the ANE lane is 25–30 % cheaper while leaving the GPU's tail latency untouched. That case is **met**, with two prices measured: about 16 % slower foreground token streaming while the ANE worker is active, and a second copy of the model in memory (4.8 GB steady, 9.2 GB while loading; the load itself caused the one swap-in of the run).

**Recommendation.** Use the ANE as the home for a bounded background evaluator on this machine: one session, rate ≤ 7 tasks per minute at 4B, loaded once and kept resident so the load peak is paid once, with the GPU foreground's decode slowdown accepted. Do not use it for bulk background work; the GPU with the slice-1 scheduler finishes 5.5× more tasks for the same total energy per task. Both placements share the same task accuracy. R4 is complete; the next kernel question remains the fp16 divergence that grows with depth, and the next scheduler question is chunked background prompt processing on the GPU.

## Development review checklist

Each change answers the six questions from instructions.md in its existing change note: outcome, deletion test, mechanism/constraint, validation, baseline/result, and recovery. Use this plan and run artifacts rather than creating duplicate process documents.

The immediate next work is R0 followed by R1: execute and inspect an actual ANE projection with a visible 1×1 lowering. The current scheduler is retained as a comparison workload. Neither a Python accessor nor a better queueing policy alone completes the requested ANE research.
