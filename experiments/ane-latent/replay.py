"""Private fixed-prefix, full-logit trace collection and comparison.

No model is launched here. A backend is injected as a callable
``step(prefix_ids) -> full logits`` so the same collector can later wrap the
native runtime, MLX and a BF16 reference. The collector forces the reference
continuation at every step regardless of the backend's argmax, so a comparison
always covers the whole forced sequence rather than stopping at the first
divergence and never degrades to a shared-prefix or top-k comparison.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from numerics import replay_metrics

# Frozen L-series model-fidelity gates from PLAN.md (local engineering tolerances).
MODEL_GATES = {'mean_kl_nats': 0.001, 'mean_nll_increase_nats': 0.01}


class TraceError(ValueError):
    pass


def _ids(values, name, vocab_size):
    out = []
    for v in values:
        if type(v) is not int or v < 0 or v >= vocab_size:
            raise TraceError(f'{name} contains a token outside the vocabulary')
        out.append(v)
    return out


def collect_trace(step, prompt_ids, forced_ids, vocab_size, label):
    """Run one forced replay. Returns a trace dict; never samples or follows argmax."""
    if type(vocab_size) is not int or vocab_size < 2:
        raise TraceError('invalid vocabulary size')
    prompt = _ids(prompt_ids, 'prompt', vocab_size)
    forced = _ids(forced_ids, 'forced continuation', vocab_size)
    if not prompt or not forced:
        raise TraceError('prompt and forced continuation must be nonempty')
    prefixes, logits, disagreements, nonfinite = [], [], [], []
    prefix = list(prompt)
    for i, token in enumerate(forced):
        out = np.asarray(step(list(prefix)), dtype=np.float64)
        if out.shape != (vocab_size,):
            raise TraceError(f'backend returned {out.shape} at step {i}; full logits of shape ({vocab_size},) are required')
        if not np.isfinite(out).all():
            nonfinite.append(i)
        elif int(out.argmax()) != token:
            disagreements.append(i)
        prefixes.append(list(prefix))
        logits.append(out)
        prefix.append(token)  # forced, independent of the backend's own preference
    return {'schema_version': 1, 'label': str(label), 'full_logits': True, 'vocab_size': vocab_size,
            'prompt_ids': prompt, 'forced_ids': forced, 'steps': len(forced), 'prefixes': prefixes,
            'first_disagreement': disagreements[0] if disagreements else None,
            'argmax_disagreements': disagreements, 'nonfinite_steps': nonfinite,
            'logits': np.stack(logits)}


def write_trace(trace, directory):
    """Immutable on-disk trace: logits.npy plus a JSON manifest carrying its hash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    logits = np.ascontiguousarray(trace['logits'], dtype=np.float32)
    np.save(directory / 'logits.npy', logits)
    meta = {k: v for k, v in trace.items() if k != 'logits'}
    meta['logits_sha256'] = hashlib.sha256((directory / 'logits.npy').read_bytes()).hexdigest()
    meta['logits_dtype'] = 'float32'
    (directory / 'trace.json').write_text(json.dumps(meta, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return directory


def read_trace(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'trace.json').read_text())
    data = (directory / 'logits.npy').read_bytes()
    if hashlib.sha256(data).hexdigest() != meta.get('logits_sha256'):
        raise TraceError('logits file does not match its recorded hash')
    logits = np.load(directory / 'logits.npy')
    if meta.get('full_logits') is not True or logits.shape != (meta['steps'], meta['vocab_size']):
        raise TraceError('trace is not a complete full-logit replay')
    meta['logits'] = logits.astype(np.float64)
    return meta


def compare_traces(reference, candidate):
    """Full-vocabulary KL/NLL over every forced step; refuses partial or mismatched traces."""
    for name, t in (('reference', reference), ('candidate', candidate)):
        if t.get('full_logits') is not True or t['logits'].shape != (t['steps'], t['vocab_size']):
            raise TraceError(f'{name} trace is not a complete full-logit replay')
        if t['steps'] != len(t['forced_ids']) or len(t['prefixes']) != t['steps']:
            raise TraceError(f'{name} trace stopped before the end of the forced continuation')
    for key in ('vocab_size', 'prompt_ids', 'forced_ids', 'prefixes'):
        if reference[key] != candidate[key]:
            raise TraceError(f'traces differ in {key}; forced prefixes must be identical')
    nonfinite = sorted(set(reference['nonfinite_steps']) | set(candidate['nonfinite_steps']))
    out = {'reference': reference['label'], 'candidate': candidate['label'], 'steps': reference['steps'],
           'compared_steps': 0, 'nonfinite_steps': nonfinite,
           'first_disagreement_reference': reference['first_disagreement'],
           'first_disagreement_candidate': candidate['first_disagreement']}
    if nonfinite:
        out.update(decision='invalid', reasons=[f'nonfinite logits at steps {nonfinite}'])
        return out
    metrics = replay_metrics(reference['logits'], candidate['logits'], reference['forced_ids'],
                             reference['prefixes'], candidate['prefixes'])
    out.update(metrics, compared_steps=metrics['steps'])
    reasons = [f'{k} {metrics[k]:.6f} exceeds {v}' for k, v in MODEL_GATES.items() if metrics[k] > v]
    out.update(decision='fail' if reasons else 'pass', reasons=reasons or ['model fidelity gates met on this prefix set only'])
    return out
