#!/usr/bin/env python3
"""Local L0 lab: theory, CPU diagnostics, native-evidence import, registry report."""
from __future__ import annotations
# Limit tiny smoke tests to one CPU math thread before NumPy imports.
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import uuid
import numpy as np
from evidence import GATES, assess, canonical, digest, import_evidence, record, report
from fixtures import CASES, fixture
from mil import projection
from numerics import balance, errors, factorize, rounded_factors
from theories import cost_theory

ROOT = Path(__file__).resolve().parent


def array_hash(a):
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes(order='C')).hexdigest()


def source_manifest():
    # Framework and test sources are both provenance for a run.
    files = sorted(ROOT.glob('*.py')) + sorted(ROOT.glob('tests/*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def smoke(out, seed):
    out.mkdir(parents=True, exist_ok=False)
    sources = source_manifest()
    # Timing is intentionally small and diagnostic; never used for native gates.
    count = 0
    for kind in CASES:
        for s in (1, 8, 32):
            h, held, w = fixture(kind, seed, s)
            for rank in (4, 8):
                a, b, fit = factorize(h, w, rank)
                for balanced in (False, True):
                    hypothesis = f'{kind}/S{s}/r{rank}/balanced={balanced}'
                    payload = {'hypothesis': hypothesis, 'evidence_kind': 'cpu_synthetic',
                               'seed': seed, 'case': kind, 's': s, 'rank': rank,
                               'shape': list(w.shape), 'sources': sources,
                               'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                               'platform': platform.platform()},
                               'inputs': {'calibration_sha256': array_hash(h), 'heldout_sha256': array_hash(held),
                                          'weights_sha256': array_hash(w)},
                               'timing_policy': 'CPU perf_counter_ns; arrays resident; 5 warmup, 30 calls/window, 3 paired windows; matmul only',
                               'assessment': {'decision': 'inconclusive', 'reasons': ['CPU synthetic diagnostic; ANE benefit unmeasured']}}
                    try:
                        aa, bb, balancing = balance(h, a, b) if balanced else (a, b, {'scale': None})
                        exact = held @ w
                        ideal = (held @ aa) @ bb
                        rounded = rounded_factors(held, aa, bb)
                        payload.update(fit=fit, balancing=balancing,
                                       approximation=errors(exact, ideal),
                                       rounded_implementation=errors(ideal, rounded),
                                       combined_error=errors(exact, rounded),
                                       factor_sha256=[array_hash(aa.astype(np.float16)), array_hash(bb.astype(np.float16))])
                        windows = []
                        rng = np.random.default_rng(seed + s + rank)
                        # These timings are CPU FP64 equations, separate from explicit rounding diagnostic.
                        operations = {'baseline': lambda: held @ w, 'candidate': lambda: (held @ aa) @ bb}
                        for window in range(3):
                            order = list(operations)
                            rng.shuffle(order)
                            row = {'window': window, 'order': order}
                            for side in order:
                                fn = operations[side]
                                for _ in range(5): fn()
                                samples = []
                                for _ in range(30):
                                    start = time.perf_counter_ns()
                                    fn()
                                    samples.append((time.perf_counter_ns() - start) / 1000)
                                row[side + '_us'] = samples
                            row['cpu_p50_gain'] = 1 - float(np.median(row['candidate_us']) / np.median(row['baseline_us']))
                            row['cpu_p95_regression'] = float(np.percentile(row['candidate_us'], 95) / np.percentile(row['baseline_us'], 95) - 1)
                            windows.append(row)
                        payload['windows'] = windows
                        metrics = payload['combined_error']
                        payload['synthetic_correctness'] = 'pass' if (metrics['finite'] and metrics['relative_l2'] is not None and metrics['relative_l2'] <= GATES['relative_l2']) else 'fail'
                        payload['assessment']['reasons'].append('synthetic numerical gate ' + payload['synthetic_correctness'])
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                        payload['failure'] = {'phase': 'offline_numerics', 'type': type(exc).__name__, 'message': str(exc)}
                        payload['assessment']['reasons'].append(str(exc))
                    record(out / 'registry', payload)
                    count += 1
    for rank in (256, 512):
        for extra in (0, 1):
            record(out / 'registry', cost_theory(rank=rank, extra_dispatches=extra))
    (out / 'dense.mil').write_text(projection(3072, 1024))
    for rank in (256, 512):
        (out / f'factorized-r{rank}.mil').write_text(projection(3072, 1024, rank))
    manifest = {'schema_version': 1, 'scope': 'L0 CPU synthetic only', 'seed': seed, 'sources': sources,
                'synthetic_records': count, 'theory_records': 4, 'ane_executed': False,
                'mil_status': 'draft_uncompiled', 'gates': GATES}
    (out / 'manifest.json').write_text(canonical(manifest))
    report(out / 'registry', out / 'report.md')
    print(f'{count} synthetic cases and 4 analytical estimates recorded in {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('smoke', help='small CPU synthetic diagnostics; never launches ANE/MLX')
    p.add_argument('--out', type=Path, default=ROOT / 'results' / ('smoke-' + uuid.uuid4().hex[:10]))
    p.add_argument('--seed', type=int, default=1729)
    p = sub.add_parser('theory')
    p.add_argument('--rank', type=int, choices=(256, 512), default=256)
    p.add_argument('--extra-dispatches', type=int, choices=(0, 1), default=0)
    p = sub.add_parser('assess', help='import producer-supplied native evidence with hashed artifacts')
    p.add_argument('evidence', type=Path)
    p.add_argument('--registry', type=Path, required=True)
    p = sub.add_parser('report')
    p.add_argument('registry', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(args.out, args.seed)
    elif args.command == 'theory':
        print(canonical(cost_theory(rank=args.rank, extra_dispatches=args.extra_dispatches)))
    elif args.command == 'report':
        print(f'{report(args.registry, args.out)} records rendered')
    elif args.command == 'assess':
        try:
            e = import_evidence(args.evidence)
            assessment = assess(e)
        except (ValueError, OSError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            e = {'evidence_kind': 'invalid_import', 'failure': str(exc)}
            assessment = {'decision': 'invalid', 'reasons': [str(exc)], 'adoption': 'not_evaluated'}
        path = record(args.registry, {'hypothesis': 'imported-native-comparison',
                      'evidence_kind': e.get('evidence_kind'), 'evidence': e, 'assessment': assessment})
        print(canonical({'record': str(path), 'assessment': assessment}))
        return 0 if assessment['decision'] == 'retain_local' else 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
