# Isolated L-series kernel laboratory

This is the first runnable **L0 framework slice**, inspired by the local Mercor kernel benchmark project. It provides deterministic CPU smoke experiments, activation-aware algebra and balancing, full-logit fixed-prefix comparison utilities, inspectible MIL drafts, and an immutable evidence registry. It does not execute ANE, modify R-series source, or claim model acceleration.

The R-series task-quality prerequisite remains in PLAN.md. The small synthetic matrices exercise infrastructure only; they are not real-model calibration or a bypass of that prerequisite.

## Run

From this directory, use Python 3.12+ with NumPy 2.4.1 (the version used for validation). No PyTorch, MLX, network, account, device permissions or native build is needed for this slice. The existing interpreter may be used read-only; no shared environment is modified. A private virtual environment can install `requirements.txt` when needed.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 lab.py smoke --out results/my-unique-run
python3 lab.py theory --rank 256 --extra-dispatches 0
python3 lab.py theory --rank 256 --extra-dispatches 1
python3 lab.py report results/my-unique-run/registry --out results/my-unique-run/registry-view.md
```

The smoke directory must not already exist. Numerical inputs are deterministic; timestamps and measured CPU timings naturally vary. Tiny CPU timing windows are diagnostic, not reliable device comparisons. Each registry JSON is written atomically with a unique ID and never overwritten by the writer. Markdown reports are derived views. Source and tensor hashes identify each run. Do not edit an existing evidence record: append a new run.

## What is implemented

- `numerics.py`: streamed output second moment, eigenvector factorization, finite bounded balancing, separate approximation/rounding metrics, and full-vocabulary KL/NLL on identical forced prefixes. NumPy FP16 rounding is explicitly not an ANE accumulation emulator.
- `fixtures.py`: seven deterministic input populations; separate calibration and evaluation RNG streams.
- `mil.py`: dense and two-stage 1x1 projection graph drafts with channel-first layout and only the final output exposed. These have not been compiled. A native adapter must serialize valid runtime weight blobs at the indicated offsets; `.npy` or raw FP16 bytes alone are not those blobs.
- `theories.py`: weight-byte accounting and the R1 cost-fit envelope, kept separate from timings.
- `evidence.py`: immutable records, provenance/artifact verification, paired-window assessment, fail-closed correctness and timing gates. A producer's selected graph must match its requested graph; no unreported fallback is accepted.
- `lab.py`: smoke, analytical screening, native evidence import and report commands. Run manifests hash framework and test sources.
- `replay.py` (L1): private forced fixed-prefix trace collector. The backend is an injected `step(prefix_ids) -> full logits` callable; the reference continuation is forced at every step, top-k or partial traces are refused, traces are write-once and hash-checked, and `compare_traces` applies the PLAN.md KL/NLL gates over every step. No model adapter exists yet.
- `packing.py` (L1): host-side `[S,K]` to `[1,K,1,S_alloc]` packing, `[N,K,1,1]` weight blobs and a NumPy 1x1-convolution emulation for layout tests.

## Test populations

Each synthetic population runs at S=1,8,32, ranks 4/8 on a 24x16 matrix, with balancing off/on: 84 cases total. Those scaled-down ranks test algebra and failure behavior; the real-shape analytical/MIL cases use 3072x1024 at ranks 256/512.

| Population | Theory under test | Expected evidence |
| --- | --- | --- |
| Exactly low-rank weights | Factorization preserves a truly low-rank map | Held-out error near rounding floor |
| Flat output spectrum | Low rank is not always useful | Compression-error rejection |
| Shifted activation support | Calibration fit does not imply generalization | Good calibration, poor held-out output |
| Outlier channels | Activation weighting and balancing need empirical checks | Record rank- and precision-dependent failures |
| All zeros | Norm ratios must not hide zero-reference failures | Exact zero passes; nonzero against zero cannot pass |
| Subnormal amplitudes | FP16 normal threshold is not a zero threshold | Explicit rounding error, no blanket exclusion |
| Held-out overflow | Calibration range is not a stability guarantee | Nonfinite output fails even after calibration balancing |

Acceptance tests also cover late divergence in fixed-prefix replay, mislabeled graph selection, missing measurements, corrupted artifacts, tail-latency regression, and a single bad timing window that aggregate speedup would hide.

## Registry and native evidence protocol

The authoritative registry is a directory of individual JSON entries, not an editable leaderboard. It records hypothesis, evidence kind, source/input hashes, conditions, raw paired timings, numerical errors, failures and decision reasons. Evidence kinds are distinct:

- `analytical_estimate`: a cost-model prediction; may be worth testing but never a measured win.
- `cpu_synthetic`: CPU diagnostics; always inconclusive for ANE benefit.
- `native_ane`: imported native measurements. May retain a local kernel, never automatically adopt a model.
- `invalid_import`: a failed import receipt, preserved for audit.

A native importer exists for the next slice; the native producer is not built yet. See `native-evidence.template.json` for the incomplete contract. Empty/missing evidence cannot pass. Supply complete raw durations in microseconds (pack through unpack, excluding warmup/compile), correctness at all three S values, an actual execution map, and hashes backed by local artifact files. Paths are relative to the submitted evidence JSON.

```bash
python3 lab.py assess native-evidence.json --registry results/native-imports/registry
```

Exit 0 means **retain locally**, 2 means rejected, invalid or inconclusive. At least three paired windows per S, 1000 calls per condition/window, varied condition order, 1000 stable repeated calls, <=0.005 numerical errors, >=10% p50 gain and <=5% p95 regression are required. Every window must meet the latency gates. Unknown internal SRAM placement is allowed; unknown graph selection is not.

Since L1 the importer also requires: unique string ids on every case and window; a `real_heldout` case at each of S=1, 8 and 32; `input_sha256` on each case and equal `baseline_input_sha256`/`candidate_input_sha256` on each window, all naming files listed in `artifacts`; and agreement between any producer `summary` block and the raw samples or record counts. See `native-evidence.template.json`.

Import verifies supplied file hashes and required metadata. It cannot independently attest that a producer ran ANE, that its selected graph and dispatch count are true, or that no external workload overlapped; inspect raw artifacts and runtime logs before trusting a retain decision. Synthetic protocol fixtures exist only in unit tests and are never written as hardware results.

## Initial interpretation

`results/l0-smoke-002/` contains the first completed CPU smoke run and four analytical estimates. Its report records all failed numerical cases; none count as native performance improvements. Read `FINDINGS.md` for the initial decision.

The one-dispatch rank-256 hypothesis has a predicted 29–49 microsecond weight-streaming saving under the old local R1 fit. Adding a second host dispatch changes that estimate to a 0.5–32 microsecond loss. These are screening calculations, with compiler traffic, actual layout and new graph performance unknown. The rank-512 saving is smaller. This favors testing a single compiled graph, not claiming it is faster.

## Mercor influence and limits

The read-only influence manifest `mercor-influence.json` hashes the inspected source files. We adopted deterministic workload variants, separate reference/candidate correctness, explicit timing boundaries, complete evidence and failure receipts. We did not copy the remote-agent pipeline, CUDA infrastructure, its 1.2x gate, or its allowance for matching NaN/Infinity outputs. Here nonfinite results fail the numerical gate. Neither this project nor the method is claimed to be accredited or endorsed by Mercor.

## Next bounded slice

After the PLAN.md entry gate and an idle hardware window: implement a private native runtime snapshot with namespaced caches, compile these graphs, verify host dispatch at the actual evaluation boundary, and produce the import contract from raw measurements. Add the full-FFN graph comparison before claiming an FFN benefit. Fixed-prefix metrics are implemented; live native/MLX replay collection and model/task qualification are not yet implemented here.

No hardware runner is exposed in this slice, so CPU smoke commands cannot accidentally start an ANE compile or touch R-series caches. No R-series lock or source modification is needed. Remove this directory to remove the experiment; existing R-series behavior remains unchanged.
