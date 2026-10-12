"""Fail-closed native evidence assessment and immutable local registry."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
from datetime import datetime, timezone
import numpy as np

GATES = {'relative_l2': 0.005, 'p50_gain': 0.10, 'p95_regression': 0.05,
         'windows': 3, 'calls_per_window': 1000, 'stability_calls': 1000}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(root, payload):
    """Each atomic JSON file is a registry entry. No shared append file or overwrites."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    entry = {'schema_version': 1, 'record_id': uuid.uuid4().hex,
             'created_utc': datetime.now(timezone.utc).isoformat(), **payload}
    data = canonical(entry)
    fd, temp = tempfile.mkstemp(prefix='.pending-', dir=root)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        target = root / (entry['record_id'] + '.json')
        # Hard link is atomic and refuses replacement, including accidental UUID collisions.
        os.link(temp, target)
        return target
    finally:
        os.unlink(temp)


def report(root, output):
    rows = [json.loads(p.read_text()) for p in sorted(Path(root).glob('*.json'))]
    lines = ['# L-series evidence registry', '',
             'Theory and CPU diagnostics do not establish ANE performance. Imported native evidence is attributed to its producer, not independently attested.', '',
             '| Hypothesis | Evidence | Decision | Reason |', '| --- | --- | --- | --- |']
    for r in rows:
        decision = r.get('assessment', {})
        cells = [r.get('hypothesis', ''), r.get('evidence_kind', ''), decision.get('decision', 'unmeasured'),
                 '; '.join(decision.get('reasons', []))]
        lines.append('| ' + ' | '.join(str(c).replace('|', '/').replace('\n', ' ') for c in cells) + ' |')
    Path(output).write_text('\n'.join(lines) + '\n')
    return len(rows)


def hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def duplicate_ids(items, label):
    """Every case/window needs a unique non-empty string id; a repeated id could overwrite a result."""
    seen = set()
    for item in items:
        i = item.get('id')
        if not isinstance(i, str) or not i:
            return f'{label} without an id'
        if i in seen:
            return f'duplicate {label} id {i}'
        seen.add(i)
    return None


def summary_mismatch(window, baseline, candidate, tolerance=1e-9):
    """A producer-supplied per-window summary must agree with its own raw samples."""
    declared = window.get('summary')
    if declared is None:
        return None
    actual = {'baseline_p50_us': float(np.median(baseline)), 'candidate_p50_us': float(np.median(candidate)),
              'baseline_p95_us': float(np.percentile(baseline, 95)), 'candidate_p95_us': float(np.percentile(candidate, 95)),
              'baseline_count': len(baseline), 'candidate_count': len(candidate)}
    for key, value in actual.items():
        if key in declared:
            reported = declared[key]
            if type(reported) not in (float, int) or not math.isfinite(reported) or abs(reported - value) > tolerance * max(1.0, abs(value)):
                return f'producer summary {key} disagrees with raw samples in window {window.get("id")}'
    return None


def assess(e):
    """Assess native kernel evidence only; never authorize model adoption."""
    def result(decision, *reasons, **extra):
        return {'decision': decision, 'reasons': list(reasons), 'adoption': 'not_evaluated', **extra}
    if e.get('evidence_kind') != 'native_ane':
        return result('inconclusive', 'ANE timing has not been measured')
    if e.get('baseline_valid') is not True:
        return result('invalid', 'baseline correctness or execution unverified')
    if e.get('interference_free') is not True:
        return result('invalid', 'idle measurement window unverified')
    if e.get('artifacts_verified') is not True:
        return result('invalid', 'raw artifacts not verified by importer')
    if e.get('boundary') not in ('down_projection', 'full_ffn'):
        return result('invalid', 'unknown timing boundary')
    provenance = e.get('provenance', {})
    required = ('source_sha256', 'runtime_sha256', 'weights_sha256', 'inputs_sha256',
                'baseline_mil_sha256', 'candidate_mil_sha256')
    if any(not isinstance(provenance.get(k), str) or len(provenance[k]) != 64 or
           any(c not in '0123456789abcdef' for c in provenance[k]) for k in required):
        return result('invalid', 'incomplete source/input/graph provenance')
    for side in ('baseline', 'candidate'):
        actual = e.get('execution', {}).get(side, {})
        if not actual.get('selected_graph') or actual.get('selected_graph') != actual.get('requested_graph') or actual.get('fallback') is not False:
            return result('invalid', f'{side} graph selection or fallback unverified')
        if actual.get('host_calls_per_invocation') != 1:
            return result('reject', f'{side} host dispatch count is not one')
    cases = e.get('cases', [])
    if not cases or {c.get('s') for c in cases} != {1, 8, 32}:
        return result('inconclusive', 'missing S=1,8,32 correctness evidence')
    problem = duplicate_ids(cases, 'case')
    if problem:
        return result('invalid', problem)
    for c in cases:
        if not hex64(c.get('input_sha256')):
            return result('invalid', f'case {c["id"]} has no input hash')
    for s in (1, 8, 32):
        # Real coverage is required at every S, not merely somewhere in the list.
        if not any(c.get('kind') == 'real_heldout' and c.get('s') == s for c in cases):
            return result('inconclusive', f'no real held-out activation evidence for S={s}')
    for c in cases:
        if c.get('finite') is not True:
            return result('reject', 'nonfinite or missing output evidence')
        for key in ('implementation_relative_l2', 'compression_relative_l2'):
            value = c.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                return result('invalid', f'missing or invalid {key}')
            if value > GATES['relative_l2']:
                return result('reject', f'{key} exceeds frozen gate')
    stability = e.get('stability', {})
    if stability.get('calls', 0) < GATES['stability_calls']:
        return result('inconclusive', 'insufficient stability repetitions')
    if stability.get('failures') != 0 or stability.get('checksum_drift') != 0:
        return result('reject', 'stability failed or missing failure counts')
    windows = e.get('windows', [])
    problem = duplicate_ids(windows, 'window')
    if problem:
        return result('invalid', problem)
    declared = e.get('summary')
    if declared is not None:
        counts = {'cases': len(cases), 'windows': len(windows), 'stability_calls': stability.get('calls')}
        for key, value in counts.items():
            if key in declared and declared[key] != value:
                return result('invalid', f'producer summary {key} disagrees with raw records')
    # At least 3 paired windows for each sequence shape; never pool incomparable shapes.
    statistics = []
    for s in (1, 8, 32):
        group = [w for w in windows if w.get('s') == s]
        if len(group) < GATES['windows']:
            return result('inconclusive', f'insufficient paired timing windows for S={s}')
        if {tuple(w.get('order', [])) for w in group} != {('baseline', 'candidate'), ('candidate', 'baseline')}:
            return result('invalid', f'condition order not varied for S={s}')
        for w in group:
            if w.get('failures') != 0:
                return result('reject', 'timing failures or missing failure count')
            # Both conditions must have timed the same input bytes.
            if not hex64(w.get('baseline_input_sha256')) or w.get('baseline_input_sha256') != w.get('candidate_input_sha256'):
                return result('invalid', f'baseline/candidate input identity unverified for window {w["id"]}')
            arrays = []
            for side in ('baseline_us', 'candidate_us'):
                a = np.asarray(w.get(side, []), dtype=float)
                if a.ndim != 1 or len(a) < GATES['calls_per_window']:
                    return result('inconclusive', 'insufficient invocation samples')
                if not np.isfinite(a).all() or (a <= 0).any():
                    return result('invalid', 'invalid invocation durations')
                arrays.append(a)
            b, c = arrays
            mismatch = summary_mismatch(w, b, c)
            if mismatch:
                return result('invalid', mismatch)
            gain = 1 - float(np.median(c) / np.median(b))
            regression = float(np.percentile(c, 95) / np.percentile(b, 95) - 1)
            statistics.append({'s': s, 'p50_gain': gain, 'p95_regression': regression})
    if any(t['p50_gain'] + 1e-12 < GATES['p50_gain'] or
           t['p95_regression'] > GATES['p95_regression'] + 1e-12 for t in statistics):
        return result('reject', 'latency gate failed in at least one paired window', timing=statistics)
    reason = 'kernel gates passed; full-FFN comparison still required' if e['boundary'] == 'down_projection' else 'full-FFN kernel gates passed; model/task gates still required'
    return result('retain_local', reason, timing=statistics)


def import_evidence(path):
    path = Path(path).resolve()
    def reject_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    e = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(e, dict):
        raise ValueError('evidence must be an object')
    # Ignore producer-supplied verification assertion; inspect actual file bytes.
    e['artifacts_verified'] = False
    artifacts = e.get('artifacts', [])
    if not artifacts:
        return e
    for item in artifacts:
        p = (path.parent / item['path']).resolve()
        if digest(p) != item['sha256']:
            raise ValueError(f'artifact checksum mismatch: {p.name}')
    # Required provenance values must correspond to an actual hashed artifact.
    hashes = {a['sha256'] for a in artifacts}
    if not set(e.get('provenance', {}).values()).issubset(hashes):
        raise ValueError('provenance is not backed by supplied artifacts')
    # Case and window input hashes must name actual hashed input files, so that
    # baseline/candidate identity rests on verified bytes rather than a producer label.
    referenced = {c.get('input_sha256') for c in e.get('cases', [])}
    referenced |= {w.get(k) for w in e.get('windows', []) for k in ('baseline_input_sha256', 'candidate_input_sha256')}
    if not referenced.issubset(hashes):
        raise ValueError('case/window inputs are not backed by supplied artifacts')
    e['artifacts_verified'] = True
    return e
