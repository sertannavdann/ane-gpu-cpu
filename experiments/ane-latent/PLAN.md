# L-series: isolated activation-aware ANE factorization experiment

Status: L0 framework implemented following the user's build authorization. Small CPU synthetic diagnostics are implemented and tested; no real-model factorization, native build or ANE execution has run. See README.md and FINDINGS.md. R-series entry gates remain unchanged.

This side experiment follows `docs/instructions.md`. It does not amend, renumber, block or automatically feed changes into R1–Rn. Outcome owner: Sertan; implementation owner: the assigned experiment developer. Proposed effort cap: three active engineering days after entry prerequisites are met, excluding time waiting for an idle device. Review at each gate; stop at the cap and record the result rather than expanding scope.

## Question and scope

Can activation-aware rank reduction of one FFN down-projection reduce complete ANE invocation latency while preserving predefined numerical and task-quality gates?

Start with Qwen3-0.6B layer 0, down-projection 3072 -> 1024, ranks 256 and 512, FP16 factors and the retained exp-form SiLU. This model is a kernel research fixture; its recorded 3/30 extraction result does not qualify it for useful placement. A different model requires its own shape, calibration and baseline manifest, not reuse of this fixture's factors.

Included: fixed-prefix replay, offline output-statistics factorization, bounded factor balancing, one native projection probe, and insertion into an experiment-local copy of the full FFN graph. Deferred: all-layer compression, runtime rank switching, block/hierarchical factorization, attention/KV changes, retraining, new serving protocols, gRPC, Qwen3.5 batching repairs, concurrent placement and tasks/J claims.

Removing this experiment must leave every R-series command and artifact unchanged.

## Isolation contract

Proposed structure (create implementation files only when their stage opens):

```text
experiments/ane-latent/
  PLAN.md
  replay.py                 # fixed-token numerical diagnostics
  calibrate.py              # activation capture and streamed second moments
  factorize.py              # factors and balancing diagnostics
  lowerings.cpp             # experiment-local MIL emitters
  probe.cpp                 # native execution and dispatch instrumentation
  evaluate.py               # paired correctness and timing evaluation
  tests/                    # factor orientation, replay and failure behavior
  fixtures/                 # versioned dataset manifests, not model copies
  patches/                  # instrumentation against an immutable source snapshot
  artifacts/<experiment-id>/
  results/<run-id>/
```

- Read R-series sources/checkpoints only as pinned inputs; record their hashes. Do not import mutable R-series build scripts or invoke scripts that reset/rewrite shared source trees.
- Use a private source snapshot, private virtual environment and dedicated Ninja build directory outside colon-containing paths, such as `/private/tmp/ane-latent-<experiment-id>/build`. Reuse installed dependencies read-only where practical. Do not rebuild shared binaries or change `prototype/ane/ane.cpp-bin`.
- Do not edit `prototype/ane/lowerings.cpp`, upstream vendor source, R3 harness, R-series datasets, the R-series plan, model caches, or global shell settings. Export factor files as new artifacts; never replace checkpoint tensors.
- All run outputs are uniquely named; existing runs are immutable. Hash source, serialized MIL, original weights, FP16 factors, datasets, tokenizer, prompt, compiler flags and runtime versions.
- Inspect native cache path handling before any compile. Namespace the marker cache and temporary compile roots in the private runtime copy. Setting TMPDIR alone is insufficient because the runtime also has a home-directory marker cache. If isolation cannot be demonstrated, stop before compiling; never clear a shared cache.
- Hardware/resource isolation is temporal on this shared SoC. Do not run ANE probes, MLX replay, eigendecomposition or large calibration jobs concurrently with R-series measurements. The L runner uses its own exclusive lock, checks known R-series activity, and requires a confirmed idle measurement window; its lock alone cannot control existing R runners.
- If an R-series run starts, stop admission to the L experiment, finish/reap its bounded child process and mark the overlapping measurement invalid. Never terminate another experiment. R-series work takes priority.
- Before execution record available memory/disk and active measurement conditions. Remain within the existing 26 GB worker planning ceiling, while separately monitoring host pressure/swap. Abort on sustained pressure or increasing swap; do not modify Docker or unrelated processes automatically.
- Use bounded child-process startup, request and shutdown deadlines; reap children after timeout/kill. Delete only this experiment's owned artifacts during recovery.

## Entry prerequisites and independence from R3

L0 preparation can freeze manifests, inspect source, and test pure algebra without ANE execution. It can proceed without changing the R-series schedule.

Before activation capture/factorization (L1), satisfy the requested gate using read-only R-series evidence or an experiment-local reference run:

1. A versioned uncompressed model/prompt reaches at least 27/30 on the extraction contract; inspect every failure. Do not relax labels to pass. Prompt development uses a separate set; a repeatedly inspected evaluation set also needs a fresh confirmation set before making broader claims.
2. Fixed-prefix replay works across native FP16, MLX FP16 and the existing MLX BF16 reference; prompts and forced continuation token IDs are byte-for-byte identical. The purpose is to distinguish precision effects from implementation differences, not demand bitwise cross-device identity.

If these prerequisites remain unmet, the side experiment remains prepared but gated. It imposes no new requirement on R-series completion. Any decision to permit a kernel-only compression study before task qualification must be an explicit scope revision, not a silent bypass.

## L0: trustworthy replay and baseline (up to one day)

Use a fixed recorded continuation per prompt, forcing the same next input token regardless of each backend's argmax. Do not terminate comparisons at first generated-token disagreement.

Capture all-vocabulary logits for a bounded diagnostic subset and selected tensors at the input/output of the chosen projection and transformer block. Record mean and worst errors with identical denominators and steps. Use complete distributions for KL/NLL; do not compare differently centered intersections of top-16 lists.

References:

- Original dense equation with FP32/FP64 accumulation for numerical diagnostics.
- Dense native FP16 and dense MLX FP16, with actual weight/activation casts recorded; FP16 tensors do not imply identical accumulation order or precision.
- Existing BF16 MLX as an additional model reference, not an exact precision match.
- Experiment-local uncompressed exp-form FFN as the performance control. Use the same graph boundary and host packing/unpacking for each condition.

Trace runs and timing runs are separate. No per-token file logging in timed regions. Record known residual-add/fallback selection explicitly; an unintended map invalidates the run.

## L1: offline factorization and balancing (up to one day)

Freeze disjoint calibration, development and evaluation manifests before selecting ranks. Proposed starting calibration: 128 representative sequences capped at 256 tokens, with varied lengths and clear/missing/misleading extraction inputs. Record actual non-padding token count, ranges and coverage. The existing 32-token capture is a smoke fixture, not a compression calibration population.

For row-vector notation:

```
H = SiLU(X Wg) * (X Wu)
Y = H Wd                         Wd: [3072, 1024]
C = sum_batches(Y.T @ Y)          C: [1024, 1024]
V = leading r eigenvectors(C)    V: [1024, r]
A = Wd @ V                       A: [3072, r]
B = V.T                          B: [r, 1024]
Y_approx = (H @ A) @ B
```

C is an uncentered second moment. Accumulate and diagonalize in FP64 for the initial offline implementation; C occupies 8 MiB. Record elapsed time, symmetry/eigensolver residuals and numerical rank. Do not promise a subsecond solve or infer task optimality from a calibration least-squares optimum. Eigenvectors are not unique at repeated eigenvalues; hash the actual exported factors.

Report held-out approximation error separately from factor-quantization and native-execution errors. The nonlinear expansion remains present, but compression can still remove task-critical output directions.

Balancing variants: unbalanced and calibration-scaled, at both ranks. For each latent channel calculate m=max(abs(H A)) and b=max(abs(B_row)); propose d=sqrt(b/(m+epsilon)). This equalizes two peak estimates approximately, not all tensor ranges, and does not center activations.

Safeguards:

- Treat zero/near-zero channels explicitly with finite floors; never divide blindly by zero. Log the floor and clamping policy.
- Check both A*d and B/d weights, the latent Z*d, and final output. Increasing d also enlarges first-stage weights. Constrain d to a feasible range with FP16 headroom; reject the candidate if none exists.
- Use observed ranges on calibration and separate stress/held-out inputs. Finite factors do not guarantee finite products or accumulations.
- Record the FP16 maximum 65504, smallest normal 2^-14 (~6.10e-5), and smallest subnormal 2^-24 (~5.96e-8). Values below the normal threshold are not automatically zero; characterize the actual runtime's handling instead of assuming it.
- Optionally round d to powers of two as one explicitly recorded balancing variant if initial tests identify scaling-rounding error; do not start an unbounded scaling search.
- Export immutable FP16 A/B buffers plus scales, orientation, hashes, ranges and calibration provenance. Re-evaluate using the serialized buffers, not only high-precision in-memory factors.

## L2: private MIL lowering and evaluation (up to one day)

First compare a dense down-projection and a two-convolution factorized down-projection on identical captured H. Then compare the complete experiment-local FFN with gate/up/corrected-SiLU unchanged and only the down-projection replaced.

Map A.T to convolution weights [r,3072,1,1] and B.T to [1024,r,1,1]; inputs [1,3072,1,S_alloc], latent [1,r,1,S_alloc], output [1,1024,1,S_alloc]. Test valid S=1,8,32 with the existing 32-lane allocation and explicit padding policy. S=512 is a later shape experiment, not an implicit extension.

Compile both convolutions into one MIL program. Export only the final output in the timed graph. A separate diagnostic graph may expose the intermediate; its timing cannot establish the original graph's behavior.

Manifest and runtime evidence are distinct:

- Before execution: graph ID, factor hashes, rank, shapes/strides, precision, source patch hashes, requested backend, MIL hash and compiled artifact identity if available.
- At runtime: selected graph ID, compile/load/evaluate success, counted host ANE evaluation calls, input/output surfaces, failures and fallback status. Instrument the actual dispatch boundary; a descriptor alone proves only intent.
- Report one host evaluation separately from internal device operations. A single MIL graph does not prove one physical kernel, no intermediate spill, or no compiler reconstruction of a dense operation.
- Latent size at r=256,S=32 is 16384 bytes = 16 KiB; at S=512 it is 262144 bytes = 256 KiB. Size alone establishes neither SRAM residency nor available SRAM capacity: weights, tiling and other buffers also compete.
- Inspect compiler artifacts or supported counters where available. If they do not expose intermediate placement, record `internal_residency: unknown`. Absence of a host-visible IOSurface does not exclude a compiler-managed DRAM spill. Do not make SRAM proof a mandatory gate if it is unobservable.

Warm up each graph, separate compile/load from steady-state latency, and run at least three paired timing windows with randomized condition order and >=1000 invocations per condition/window. Reuse the R1/R2 timing definitions by copying their specification, not modifying their runners. Report packing, evaluation, unpacking and complete invocation p50/p95, all failures, host memory, and swap delta. Fit-based estimates are screening aids, not performance results.

## Frozen decision gates

These are proposed L-series gates, frozen before measurements; they do not replace R-series gates.

- Native implementation fidelity: relative L2 <=0.005 against the same serialized factor computation on every normal held-out case. Report near-zero absolute error and all basis/extreme-input failures separately; never silently exclude new failures.
- Compression fidelity: relative L2 <=0.005 against the uncompressed dense projection on each normal held-out case. Report per-token tails as well as aggregate error. If both ranks fail, reject these candidates; do not loosen the threshold after seeing results.
- Model fidelity: on identical forced prefixes, mean KL(dense || candidate) <=0.001 nats/token and increase in reference-continuation NLL <=0.01 nats/token. Compute both from full logits; report worst-step values and all nonfinite failures. These are local engineering tolerances, not universal quality guarantees.
- Task quality: at least 27/30 correct extraction outcomes and no decline in correct-case count versus the qualified uncompressed reference; inspect every changed outcome. A local kernel pass cannot substitute for this gate.
- Speed: at least 10% lower complete-invocation p50, p95 regression <=5%, improvement consistent across paired windows, and no additional host evaluation calls relative to the equivalent dense graph. Compare full FFN as well as projection timing; do not advertise a projection-only win as model acceleration.
- Stability: 1000 repeated executions with no nonfinite results, crashes or checksum drift for repeated identical inputs. Cold compile/load failures remain in the report.
- Unknown SRAM residency is acceptable with an explicit limitation. Unknown actual graph selection is not.

A kernel candidate may be retained as a local research artifact when only kernel gates pass. No useful-backend or deployment claim follows until model/task gates pass; energy and concurrency remain outside this slice.

## Results, promotion and recovery

Write one report with provenance, baseline/candidate tables, numerical error decomposition, timing windows, balancing ranges, compile failures and observability limits. Outcomes: retain locally; reject the tested ranks/lowering; or inconclusive due to a named missing measurement. Stop when the question is answered or the effort cap expires.

No automatic upstream patch, replacement of R-series binaries, edit to R-series plans, or opening of R4. Any promotion is a separate review against the then-current R-series baseline, including a concrete patch and repeatable reproduction command.

Change note under instructions.md section 6: this plan addresses measured weight traffic and dispatch cost; omitting it leaves the compression hypothesis untested but preserves all existing outcomes. It introduces only a disposable experimental directory/build. Validation will be the gates above; none have run yet. Recovery removes only owned builds, caches and results. The original checkpoint, runtime and R-series plans remain the recovery baseline.
