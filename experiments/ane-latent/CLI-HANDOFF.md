# CLI-agent handoff: isolated L-series theories and kernel tests

This document is the task transfer. Read the current-state report first. This is a bounded research experiment within the user's existing authorization, not permission to alter the main R-series or run concurrently with its measurements.

## 1. Copy this task into the CLI agent

```text
Continue the isolated L-series ANE factorization experiment.

Workspace:
/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU

Your writable implementation scope:
/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent

Read, in order:
1. /Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/docs/instructions.md
2. /Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/CURRENT-STATE.md
3. /Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/PLAN.md
4. /Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/README.md
5. /Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/FINDINGS.md
6. /Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/docs/agentic-offload-prototype-plan.md (read-only; reconcile newer run status)

The installed L0 framework has 21 passing tests, 84 CPU synthetic records and
four analytical estimates. No L-series graph has been compiled/executed on ANE.
MIL is draft text. Fixed-prefix metrics exist, but live trace collection does not.
There is no native runner or implemented hardware isolation mechanism yet.

Goal: test whether one activation-aware FFN down-projection factorization is
numerically acceptable and faster at the complete invocation boundary. Preserve
the distinction between analytical predictions, synthetic CPU diagnostics,
native kernel measurements, model quality and concurrent-system outcomes.

Start by reproducing unit tests, verifying the current source hashes against the
stored L0 manifest and running a new uniquely named synthetic smoke experiment.
Inspect failure records and summarize why each theory passed/failed or remains
untested. Preserve all existing registry entries.

Next fix the concrete evidence-validation gaps in CURRENT-STATE.md, adding tests
for real held-out coverage at each S, duplicate case IDs, cross-condition input
identity and raw-record-to-summary consistency. Keep this local and small.
Implement a private fixed-prefix trace collector; do not mistake top-16 shared-
prefix comparisons for the required full-logit replay. Preparation can proceed
while hardware runs are gated; do not start model execution without an idle window.

Before real calibration/factorization, apply the entry gates in PLAN.md. The
newest inspected R3 reference scores 23/30, not the earlier 3/30, and is still below
27/30. R4-named partial logs also exist despite the plan's unresolved gate. Recheck
that evolving state and distinguish exploratory runs from gate completion.
Do not modify, restart or stop the R-series to resolve this discrepancy.

If the gates remain unmet, complete independent L0 work, report the exact missing
evidence and leave native execution pending. Do not silently waive the gate. If
the owner explicitly changes scope to a kernel-only study, record that narrow
change in the L plan without amending R-series acceptance criteria.

Once prerequisites and resource isolation are satisfied, use one immutable
model/layer snapshot, separate calibration and held-out activations, ranks 256
and 512, and balancing off/on. The initial 3072x1024 shape belongs to Qwen3-0.6B
layer 0; changing model requires inspecting new shapes and freezing new inputs.
Build a private native runtime with separate caches and no shared source resets.
Validate actual graph selection and dense/factorized numerical parity before
measuring latency. Evaluate projection and full FFN separately. No silent fallback.

Use the existing local registry for all outcomes, including compile failures,
timeouts, invalid evidence and rejected numerical candidates. Never count CPU
speedup or a theory estimate as ANE improvement. Do not claim SRAM residency
without observable evidence. Do not promote a kernel into an R-series binary.

Use Mercor only as read-only methodological reference:
/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/Mercor/cuda-golden-kernels-l8hie
Its source influence is already hashed in mercor-influence.json. Do not launch
its remote agent/CUDA pipeline or claim accreditation.

Finish each slice with commands, versions/hashes, raw artifact paths, correctness
results, paired p50/p95 comparisons, failures, a retain/reject/inconclusive
judgment, and the next smallest unresolved question. Use full absolute Markdown
document paths. Keep R-series files and existing results unchanged.
```

## 2. Known-good local commands

These commands use the existing interpreter read-only; they do not install into or modify it. It provided Python with NumPy 2.4.1 during the handoff audit. Revalidate availability on another machine/session. The project path contains spaces, a Unicode apostrophe and colons: quote it exactly. Do not use the old pre-move workspace path.

```bash
cd '/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent'
L_PY='/Users/sertanavdan/miniconda3/bin/python3'
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

"$L_PY" -c 'import sys, numpy; print(sys.version); print(numpy.__version__)'
"$L_PY" -m unittest discover -s tests -v
```

Expected: 21 tests pass at this snapshot. A larger count is appropriate after adding meaningful tests. Investigate a failure rather than changing tolerances to match output.

Source-integrity verification, with explicit failure if implementation and original evidence no longer match:

```bash
"$L_PY" - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent')
manifest = json.loads((root / 'results/l0-smoke-002/manifest.json').read_text())
for name, expected in manifest['sources'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Source changed since recorded L0 run: {name}')
print('All recorded framework source hashes match')
PY
```

An intentional new implementation should receive a new source manifest and results; do not edit the old manifest to make this verification pass.

Fresh smoke run (the output directory must not exist):

```bash
L_RUN="/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/cli-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$L_PY" lab.py smoke --out "$L_RUN" --seed 1729
"$L_PY" lab.py report "$L_RUN/registry" --out "$L_RUN/registry-view.md"
```

Expected: 84 synthetic records plus four estimates. Expected diagnostic numerical outcomes: 32 pass, 52 fail. Every CPU record remains inconclusive for ANE benefit. Numerical failures are intentional stress/approximation outcomes; they are not automatically harness failures.

Theory-only screening:

```bash
"$L_PY" lab.py theory --rank 256 --extra-dispatches 0
"$L_PY" lab.py theory --rank 256 --extra-dispatches 1
"$L_PY" lab.py theory --rank 512 --extra-dispatches 0
"$L_PY" lab.py theory --rank 512 --extra-dispatches 1
```

These print predictions, not benchmarks. Their real projection shapes differ from smoke-fixture shapes.

## 3. Current case matrix and extensions

Keep the existing cases: low-rank, flat-spectrum, distribution-shift, outlier-channel, zero, subnormal and held-out-overflow. Preserve their seeds and scale as a regression suite.

Add tests only for concrete next-stage risks:

| Next test | Defect it should catch |
| --- | --- |
| Duplicate case IDs | One result overwriting another, as previously found in R3 |
| Same-shaped but different reference/candidate inputs | Spurious speed or correctness comparisons |
| Real-held-out cases missing at one S | Importer accepting incomplete coverage |
| Sample summaries disagree with raw files | Producer bookkeeping errors |
| Transposed A/B packing with asymmetric values | Correct algebra but incorrect convolution layout |
| Zero-padded vs sentinel-padded unused lanes | Leakage from padding into valid token outputs |
| Forced replay after first argmax disagreement | Silent return to shared-prefix comparison |
| Native compile/evaluate failure and timeout | Lost failure receipts or orphan workers |
| Candidate accidentally selecting dense graph | Apparent success without executing the hypothesis |
| Private cache namespace | Writing to another experiment's marker/compile cache |

The native tests are proposed, not currently implemented or passed. Do not fabricate native evidence to demonstrate the importer; fabricated records belong only in clearly marked unit tests.

## 4. Native measurement protocol when the gate opens

Before execution, record the model/tokenizer/input/source/compiler/runtime hashes, actual shapes, valid/padded S, serialized factors, balancing policy, requested graph and timing boundary. Freeze calibration/development/evaluation separation. Ensure the same model and task referenced by the gate are used for any model-level claim; another model's score does not qualify this fixture.

Use an isolated source snapshot and native build path without colons. Namespace both temporary compiler outputs and the home-directory marker cache in the private runtime; TMPDIR alone is insufficient. Do not mutate HOME or reuse the R build script, which may reset its own source checkout.

Native runner requirements precede performance measurement: verified idle window, bounded memory/disk use, startup/request/shutdown deadlines, process reaping, actual dispatch counter, and explicit failure recording. A local lock does not stop legacy R scripts that do not observe it. Recheck external workload activity; invalidate overlap rather than claiming isolation.

Compare dense vs two-factor projection, then the complete exp-form FFN with only its down-projection changed. Keep diagnostic intermediate-output graphs separate from timed graphs. Define:

- complete invocation = pack + host/surface handoff + synchronous evaluation + read/unpack;
- compile/load = separately reported, excluded from steady-state timings;
- timing population = each S independently, identical input, baseline/candidate pair;
- warmup = preregister a fixed count, exclude from timings;
- windows = >=3 paired windows per S, with both condition orders represented;
- samples = >=1000 invocations per condition per window;
- stability = >=1000 identical-input repetitions, no failures or checksum drift.

Retain locally only with <=0.005 implementation and compression relative L2 on required cases, >=10% p50 gain and <=5% p95 regression in every paired window, and no extra host calls. Do not omit failed shapes or seeds from a winner's denominator. Report error tails, all nonfinite outputs, resource pressure and unknown internals. Projection retention is not full-FFN or model retention.

## 5. Import contract and reporting

Template:

`/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/native-evidence.template.json`

It is intentionally incomplete and cannot pass. A future producer must fill raw samples, execution observations, cases and backed artifact hashes. Do not set `artifacts_verified` manually; the importer recomputes it. Treat producer statements as evidence to audit, not independent attestation.

Example command after producing actual evidence (the source file below does not exist yet):

```bash
"$L_PY" lab.py assess \
  '/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/native-run-001/evidence.json' \
  --registry '/Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/native-imports/registry'
```

Exit 0: retained locally. Exit 2: rejected, invalid or inconclusive; inspect the reason and preserve the receipt. Never translate exit 0 into automatic model adoption. KL/NLL and task gates are separate and not enforced by this importer yet.

For each theory report: hypothesis and predicted effect; unchanged baseline; variable changed; data split and hashes; actual graph executed; numerical outcome; measured paired latency; failures and omissions; limitations; decision; next question. Use new run IDs after every source/data/threshold change.

## 6. Stop and recovery conditions

Stop native admission when an R-series measurement overlaps, isolation cannot be established, host memory pressure/swap rises, graph selection is unknown, or failures make the comparison invalid. Do not stop unrelated processes. If both tested ranks fail correctness, record rejection and stop that search rather than expanding it silently.

No automatic upstream patch, R-plan edit, runtime symlink replacement, model checkpoint rewrite, model download or new service. Keep cache deletion limited to this experiment's owned namespace. The user's build authorization permits necessary isolated work; the CLI sandbox may still require approval to write the relocated directory. Request the precise filesystem permission needed rather than moving work into shared R paths.

References, all absolute:

- [Current verified state](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/CURRENT-STATE.md>)
- [Experiment plan and model gates](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/PLAN.md>)
- [Framework commands and importer limitations](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/README.md>)
- [L0 findings](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/FINDINGS.md>)
- [Registry report](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/experiments/ane-latent/results/l0-smoke-002/report.md>)
- [Main project plan, read-only](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/docs/agentic-offload-prototype-plan.md>)
- [Development guideline](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/ANE:GPU:CPU/docs/instructions.md>)
- [Mercor variant methodology, read-only](</Users/sertanavdan/Documents/Documents - Sertan’s MacBook Pro/Software/Mercor/cuda-golden-kernels-l8hie/scripts/input_variants/README.md>)
