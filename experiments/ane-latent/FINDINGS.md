# L0 findings and next decision

Status: CPU synthetic harness validated; ANE benefit unmeasured. Evidence: `results/l0-smoke-002/` (84 synthetic records + 4 analytical records). Validation: 21 unit/integration tests passed. No R-series files or checkpoints were modified; no ANE/MLX model was launched.

## Synthetic correctness results

Each population has 12 variants: S=1,8,32 x ranks 4,8 x balancing off/on. Counts below use the combined dense-reference-to-rounded-factor relative L2 <=0.005 gate, with any nonfinite output failing. Matrices are 24x16, not the real model shape.

| Population | Pass | Fail | Interpretation |
| --- | ---: | ---: | --- |
| Low rank | 12 | 0 | Algebra/orientation and explicit FP16 rounding preserve this constructed map |
| Zero | 12 | 0 | Exact-zero handling works |
| Flat spectrum | 0 | 12 | Truncation cannot preserve all equally important directions |
| Distribution shift | 0 | 12 | Calibration-only success is misleading on unseen activation directions |
| Outlier channels | 8 | 4 | Scaling/rank choices still need measurement; no universal balancing guarantee |
| Subnormal amplitudes | 0 | 12 | FP16 rounding can dominate small outputs; failures remain visible |
| Held-out overflow | 0 | 12 | Calibration balancing cannot guarantee unseen-input safety |

These are intentionally constructed populations testing whether the framework detects expected failures. They do not estimate real-model compressibility. Global L2 can also underweight task-relevant low-energy directions in an outlier-dominated tensor; the records include maximum per-token relative L2, but task and fixed-prefix model metrics remain necessary.

Raw CPU timing arrays, per-window median gains and p95 regressions are stored in every completed synthetic record. They time FP64 NumPy equations with arrays resident, not the explicit rounding diagnostic. They are useful for harness plumbing only and do not determine ANE retain/reject decisions.

## Cost-model screening at the actual projection shape

FP16, 3072x1024 down-projection; dense storage 6,291,456 bytes. Estimates use the recorded R1 ranges 7.0–11.8 us/decimal MB and 50–61 us per extra host dispatch. The endpoints are an assumption envelope, not confidence intervals or hardware measurements.

| Rank | Factor bytes | Weight reduction | Estimated saving, no extra dispatch | Estimated saving, one extra dispatch |
| --- | ---: | ---: | ---: | ---: |
| 256 | 2,097,152 | 66.7% | 29.4–49.5 us | -31.6 to -0.5 us |
| 512 | 4,194,304 | 33.3% | 14.7–24.7 us | -46.3 to -25.3 us |

No additional intermediate traffic, compiler folding, layout overhead or graph execution penalty is included. At rank 256 the weight reduction across the whole three-matrix FFN is only about 22.2%, not 66.7%. A host evaluation count of one alone does not prove on-chip fusion or absence of DRAM spill.

Decision: **worth a bounded native single-graph experiment after the existing entry prerequisites**. Two independently dispatched factors are unlikely to help under this fit. No rank is accepted for a real model, and no energy or task-throughput claim is supported.

## Remaining implementation

The next native slice needs a private runtime snapshot, namespaced compile caches, actual dispatch instrumentation, valid weight-blob serialization, paired host-invocation timing and a full-FFN comparison. The MIL drafts have only structural tests, not compilation or device validation. The fixed-prefix utility compares supplied complete traces; native/MLX trace collection is still future work.

Native evidence import checks metadata and hashed files but relies on the producer for honest execution observations; inspect the raw logs before acting on `retain_local`. R-series model/task qualification is unchanged. Promotion is separate from a local kernel result.

## Change note (instructions.md section 6)

Outcome: establish a reproducible way to test the latent-factorization hypothesis without touching the R-series. Deletion test: without this slice there is no isolated evidence registry or adversarial synthetic validation. Mechanisms: local Python/NumPy modules, immutable JSON files and generated reports; no service, remote agent or native daemon. Validation: 21 tests, 84 synthetic cases and four analytical estimates. Baseline: dense equation with identical inputs; CPU and theoretical timings are labeled separately. Recovery: remove only this experiment directory and its staging files; production/R-series runtime remains unchanged.

## L1 addendum (19 September 2026)

L0 was reproduced after the R4 run exited: 21 tests, six matching hashes, and a fresh smoke run (`results/cli-smoke-20260919T175845Z-92262/`) whose 84 records are numerically identical to `l0-smoke-002`. The failure decomposition adds one nuance to the table above: the subnormal population's failures are dominated by approximation error (0.34–0.94) rather than FP16 rounding (0.02–0.08), and the four outlier-channel failures are exactly the S=1 cases. The importer was then hardened (per-S real coverage, unique ids, artifact-backed input identity, raw-vs-summary recomputation), a private forced-replay collector and a packer were added, and the suite grew to 34 tests with a new manifest in `results/l1-smoke-001/`. The native gate is still unmet (MLX reference 23/30). Full detail: `L1-REPORT-2026-09-19.md`.
