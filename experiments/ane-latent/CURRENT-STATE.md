# L-series current state and technical report

Snapshot: 19 September 2026. Installed framework tests were rerun during this audit: **21 passed**. All six framework source hashes match the recorded L0 manifest. The immutable registry contains **84 CPU synthetic cases and four analytical estimates**, with **zero native ANE measurements**.

**L1 update, later on 19 September 2026:** the L0 state above was reproduced after the R4 run exited, then the evidence importer was hardened, a private fixed-prefix trace collector and a host-side packer were added, and the suite grew to **34 passing tests**. The new source manifest is `results/l1-smoke-001/manifest.json`. Details, per-population failure mechanisms and the gate reconciliation are in [L1 report](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/L1-REPORT-2026-09-19.md>). The native gate remains unmet; no ANE execution has occurred.

## 1. Objective and boundaries

The original project investigates actual ANE computation in a local CPU/GPU/ANE agent system: visible tensor lowering, correct transformer subgraphs, useful small-model execution, and eventually concurrent placement and gRPC. The L-series is a separate experiment asking whether activation-aware compression of one FFN down-projection can reduce weight traffic without losing correctness or spending the saving on dispatch overhead.

L-series implementation root:

`/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent`

The R-series remains the main research track. L-series results do not alter R gates, substitute for R model qualification, or authorize automatic adoption. No R-series source, checkpoint, runtime symlink or result was changed by this work. No ANE/MLX model was launched for L0.

Authoritative documents:

- [Development guideline](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/docs/instructions.md>)
- [Main R-series plan](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/docs/agentic-offload-prototype-plan.md>)
- [L-series scope and gates](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/PLAN.md>)
- [Runnable framework guide](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/README.md>)
- [Original L0 findings](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/FINDINGS.md>)
- [CLI transfer and execution guide](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/CLI-HANDOFF.md>)

## 2. Relevant R-series evidence, including newer shared-project work

| Stage | Recorded finding | Implication for L |
| --- | --- | --- |
| R1 | 378 projection cases, zero gate failures; real-activation relative L2 approximately 2.8e-4–4.8e-4 | The matrix-to-1x1-convolution mapping has a measured numerical baseline |
| R1 cost model | Fixed dispatch roughly 50–61 us; streaming slope 7.0–11.8 us/decimal MB varies between runs | Reduce bytes without blindly adding host calls; treat the fit as local screening evidence |
| R2 | Intermediate chunking increased dispatches and lost latency | Smaller individual kernels are not automatically faster |
| R2 | Exp-form SiLU reduced FFN error from roughly 2.2e-2 to 2.6e-3 at similar measured kernel time | Preserve the retained activation correction in any complete-FFN control/candidate pair |
| Initial R3, Qwen3-0.6B | All conditions scored 3/30 under the original contract | The original small-model task was unqualified; kernel parity is not task usefulness |
| Newer R3, Qwen3.5-4B, fixture v2 | Three ANE conditions 24/30, MLX 23/30; 30/30 schema validity | Newer evidence supersedes 3/30 as the latest shared-project task result, but still misses 27/30 |

The latest inspected model report is [Qwen3.5-4B v2 report](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/results/ane-r3-model-4b-v2/report.md>). Its fixture/scorer differs from the original 0.6B evaluation, so score improvements cannot be attributed solely to model size. The main plan also records a fixed duplicate-case-ID bug; future experiments must validate unique IDs before writing case results.

**Shared-project state discrepancy:** the written plan still says R4 is not entered, but R4-named concurrency artifacts exist. At inspection, `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/results/r4-run.log` recorded seven window summaries out of eleven, with no final report observed in `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/results/ane-r4-concurrent`. This is partial evidence, not proof of a currently live process, a completed experiment or a satisfied entry gate. The CLI agent must recheck current logs and run metadata, reconcile whether this is an authorized exploratory run, and avoid interfering. Do not restart, stop, repair or reinterpret that work as part of L.

Illustratively, the partial R4 log shows foreground p95 TTFT around 157–175 ms with ANE background work versus 442–567 ms with GPU background work, but background correct throughput around 6.8/min versus 34–35.2/min respectively. Those are unfinished-run observations under recorded memory pressure, not an adoption result or energy comparison.

R3 logit comparisons also stop at prefix divergence and compare intersecting top-16 sets. Their means cannot partition numerical error by kernel. Claims of equal speed or universally safe correction are stronger than the available evidence. L therefore requires identical forced prefixes and explicit same-precision controls.

## 3. Mathematical hypothesis

Use row-vector notation for the post-gating activation H and down-projection Wd:

```
H = SiLU(X Wg) * (X Wu)
Y = H Wd                         Wd: [3072, 1024]
C = sum(Y_batch.T @ Y_batch)      C: [1024, 1024]
V = leading r eigenvectors(C)    V: [1024, r]
A = Wd @ V                       A: [3072, r]
B = V.T                          B: [r, 1024]
Y_approx = (H @ A) @ B
```

This preserves upstream expansion/gating computation and approximates the linear contraction. The calibration objective is output-space least squares. It does not guarantee reasoning, extraction quality or distribution-shift robustness.

C is an uncentered second moment, accumulated offline in FP64. Do not center it without recording a changed objective and appropriate bias handling. A 1024x1024 FP64 matrix occupies 8 MiB; eigensolver time must be measured rather than promised.

Balance factors through A'=AD, B'=D^-1 B. The implemented heuristic starts from d_i=sqrt(b_i/max(m_i,epsilon)), using latent activation peaks m_i and B-row peaks b_i, then clips to a feasible bound that checks A', B' and calibration latent ranges. It handles zero channels. It is not a proof of safety on unseen inputs or a guarantee about device accumulation.

FP16 has maximum finite magnitude 65504, minimum normal 2^-14, and minimum subnormal 2^-24. Below-normal does not mean automatically zero; ANE subnormal behavior remains unmeasured here.

## 4. Implementation inventory and readiness

All paths in this table are beneath `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent`.

| Component | Implemented | Not established |
| --- | --- | --- |
| `numerics.py` | Output-statistics factorization, bounded balancing, approximation/rounding errors and full-logit fixed-prefix metrics | Real-model calibration, native accumulation fidelity, live trace capture |
| `fixtures.py` | Seven deterministic synthetic populations with separate calibration/evaluation RNG streams | Representative model activation coverage |
| `mil.py` | Inspectible dense/two-convolution projection MIL text; shape/orientation tests | Native compilation, executable weight blobs, full-FFN integration, SRAM residency |
| `theories.py` | Byte accounting and R1 cost-envelope estimates | Measured ANE performance |
| `evidence.py` | Atomic immutable JSON entries, artifact hash checks, per-window native-kernel assessment, Markdown view | Independent device attestation, model adoption assessment, verification that raw files imply producer summaries |
| `lab.py` | `smoke`, `theory`, `assess`, `report` commands | Hardware execution, model replay collection, process scheduling or R-series activity monitoring |
| `tests/test_lab.py` | 21 unit/integration tests | On-device integration tests |
| `replay.py` (L1) | Forced full-logit trace collection with an injected backend, write-once hashed traces, KL/NLL gate comparison | Native, MLX and BF16 backend adapters; any live model run |
| `packing.py` (L1) | `[S,K]` to `[1,K,1,S_alloc]` packing, weight blobs, NumPy 1x1-conv emulation | Runtime plane strides, valid weight-blob headers |
| `tests/test_evidence_contract.py`, `tests/test_replay.py`, `tests/test_packing.py` (L1) | 13 additional tests; suite total 34 | Native compile/timeout, dense-graph mis-selection and cache-namespace tests remain proposed |

There is no C++ native producer, compile-cache isolation implementation, live fixed-prefix model runner or private hardware lock in this slice. Those are explicit future requirements, not features to assume exist.

## 5. Test results and their interpretation

The test suite passed all 21 tests when rerun from the installed project. It covers algebra/orientation, full-rank recovery, held-out distribution shift, zero/nonfinite handling, balancing infeasibility, unseen overflow, deterministic fixtures, forced-prefix continuity, graph-selection failures, artifact corruption, missing evidence, p95 regression, one failing timing window, immutable records and invalid-import receipts.

The smoke matrix is 7 populations x S={1,8,32} x rank={4,8} x balancing={off,on} = 84 cases. Its matrices are 24x16, not actual Qwen matrices.

| Population | Numerical passes | Numerical failures | What it demonstrates |
| --- | ---: | ---: | --- |
| Constructed low rank | 12 | 0 | Expected recoverability and factor orientation |
| Zero | 12 | 0 | Exact-zero handling |
| Flat spectrum | 0 | 12 | Information loss is exposed |
| Shifted activation support | 0 | 12 | Good calibration fit does not hide held-out failure |
| Outlier channels | 8 | 4 | Rank/precision choices matter; balancing is not universally successful |
| Subnormal amplitudes | 0 | 12 | Rounding-sensitive outputs remain visible |
| Held-out overflow | 0 | 12 | Unseen range failure is not suppressed |

**32 passes and 52 numerical failures are expected diagnostic outcomes, not 52 broken unit tests.** The 21 unit tests validate that the harness correctly recognizes successes and failures.

Each smoke record includes raw paired CPU duration samples, p50 gains, p95 regressions, input/source hashes, numerical errors and decision reasons. CPU timing uses resident FP64 NumPy equations, 5 warmups, 30 calls per condition/window, and 3 windows. It does not time an ANE implementation or the explicit FP16-rounding diagnostic. Every synthetic record remains inconclusive for ANE benefit.

Current evidence:

- [Registry report](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/l0-smoke-002/report.md>)
- Manifest: `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/l0-smoke-002/manifest.json`
- Immutable records: `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/l0-smoke-002/registry/`
- Audit receipt: `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/handoff-audit-2026-09-19.json`

The six source hashes match the stored manifest. Test code itself is not included in that source manifest, a provenance improvement to include in the next harness revision. Preserve the existing manifest; append a new run after changes.

## 6. Is the theory worth testing?

Original down-projection weights: 6,291,456 FP16 bytes. Estimated savings:

| Rank | Factor bytes | Weight reduction | Predicted time saved: no extra host call | Predicted time saved: one extra host call |
| --- | ---: | ---: | ---: | ---: |
| 256 | 2,097,152 | 66.7% | 29.4–49.5 us | -31.6 to -0.5 us |
| 512 | 4,194,304 | 33.3% | 14.7–24.7 us | -46.3 to -25.3 us |

These use old R1 local parameters, not newly measured bandwidth. The calculation omits extra intermediate traffic, compiler choices and changed execution efficiency. Across all three FFN weight matrices, rank 256 reduces total weight bytes by about 22.2%, not 66.7%.

**Decision: worth one bounded single-graph native experiment after entry requirements; not yet worth adopting.** Separate dispatches are unlikely to help under this fit. Either rank may fail the stringent 0.005 compression gate. If both fail, publish that negative result rather than widening the gate or escalating to a framework-wide compression search.

A 16 KiB latent at rank 256 and S=32 does not establish SRAM residency. Neither a source descriptor nor one host evaluation proves the physical graph, internal kernel count or absence of spills. Report internal placement as unknown when unobservable.

## 7. Gates: enforced now versus still procedural

Current imported-native kernel assessment requires valid baseline, declared interference-free conditions, matching requested/selected graphs without fallback, backed provenance hashes, one host call per invocation, S=1/8/32 correctness, real held-out evidence, <=0.005 implementation/compression relative L2, and 1000 stable calls. It requires at least 3 paired windows for each S with 1000 durations per condition/window, both condition orders, >=10% p50 gain and <=5% p95 regression in every window. Zero failures are required.

The decision can be `retain_local`, `reject`, `invalid` or `inconclusive`. It always leaves model adoption unevaluated. Both dense and factorized paths must use the same timing boundary and workload.

Procedural/model gates from [L plan](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/PLAN.md>), not implemented in the importer:

- Qualified uncompressed model/prompt reaches >=27/30, with frozen scoring and inspected failures.
- Live fixed-prefix replay succeeds with native FP16, MLX FP16 and the BF16 reference.
- Mean full-distribution KL <=0.001 nats/token and NLL increase <=0.01 nats/token.
- Candidate task outcome count does not regress and remains >=27/30.
- Full-FFN and later model/placement measurements before broader performance claims.

Validation gaps as of the L0 audit: importer trusts producer statements about idle conditions and dispatch; it only requires some real-held-out evidence, not real coverage at every S; it does not establish baseline/candidate input equality from raw artifact content or independently recompute summaries.

L1 status of those gaps: real held-out coverage at every S, unique case/window ids, artifact-backed input hashes with per-window baseline/candidate equality, and raw-vs-summary recomputation are now enforced and tested. Still trusted from the producer: idle conditions, selected graph and the host dispatch count. Hash identity alone does not establish measurement truth.

## 8. Mercor influence

Source directory, read-only:

`/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/Mercor/cuda-golden-kernels-l8hie`

Inspected files: `scripts/seed_benchmark.py`, `scripts/benchmark_framework.py`, `scripts/input_variants/README.md`, and `scripts/io/test_comparison.py`. Exact hashes are in `/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/mercor-influence.json`.

Adopted: deterministic variants, independent reference/candidate paths, explicit timing boundaries, preserved failures and provenance. Not adopted: remote agents, CUDA execution infrastructure, a different speedup gate, or matching nonfinite outputs as success. This is attributed methodological influence, not Mercor accreditation or endorsement.

## 9. Recommended continuation

First reproduce the installed tests and a fresh uniquely named smoke run. Then harden the evidence contract and implement live fixed-prefix collection privately. Reconcile the newer R-series status read-only. Do not run real calibration/factorization or native benchmarks while prerequisites remain unresolved or R hardware measurements are active.

When gates permit: fit one layer only, export immutable FP16 factors, create a private namespaced native runtime, compile dense and two-stage graphs, validate all shapes and numerical cases, then run paired timings and the full-FFN comparison. No automatic promotion. The exact transfer prompt, commands and stop conditions are in [CLI handoff](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/CLI-HANDOFF.md>).
