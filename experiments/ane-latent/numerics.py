"""Offline algebra only. NumPy FP16 rounding is not an ANE emulator."""
from __future__ import annotations
import numpy as np


def matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return value


def factorize(h, w, rank):
    h, w = matrix(h, 'H'), matrix(w, 'W')
    if h.shape[1] != w.shape[0] or not 1 <= rank <= min(w.shape):
        raise ValueError('incompatible shape or rank')
    # Stream the uncentered output second moment. Never use evaluation inputs here.
    c = np.zeros((w.shape[1], w.shape[1]), dtype=np.float64)
    for start in range(0, len(h), 128):
        y = h[start:start + 128] @ w
        c += y.T @ y
    values, vectors = np.linalg.eigh(c)
    v = vectors[:, -rank:][:, ::-1].copy()
    # Canonical sign for reproducibility (degenerate eigenspaces remain non-unique).
    for i in range(rank):
        if v[np.argmax(np.abs(v[:, i])), i] < 0:
            v[:, i] *= -1
    a, b = w @ v, v.T
    return a, b, {'eigenvalues': values[::-1].tolist(),
                  'calibration_tokens': len(h), 'second_moment_bytes': c.nbytes}


def balance(h, a, b, limit=32752.0):
    h, a, b = matrix(h, 'H'), matrix(a, 'A'), matrix(b, 'B')
    if h.shape[1] != a.shape[0] or a.shape[1] != b.shape[0]:
        raise ValueError('incompatible factors')
    if not np.isfinite(limit) or not 0 < limit <= 65504:
        raise ValueError('invalid FP16 headroom')
    m = np.max(np.abs(h @ a), axis=0)
    ap = np.max(np.abs(a), axis=0)
    bp = np.max(np.abs(b), axis=1)
    eps = 1e-30
    lower = np.maximum(bp / limit, eps)
    upper = limit / np.maximum(np.maximum(m, ap), eps)
    if np.any(lower > upper):
        raise ValueError('no feasible per-channel FP16 balance')
    d = np.clip(np.sqrt(np.maximum(bp, eps) / np.maximum(m, eps)), lower, upper)
    # Entirely zero channels require no scaling.
    d[(m == 0) & (ap == 0) & (bp == 0)] = 1.0
    aa, bb = a * d, b / d[:, None]
    return aa, bb, {'scale': d.tolist(), 'latent_peak_before': float(m.max()),
                   'latent_peak_after': float(np.max(np.abs(h @ aa))),
                   'a_peak_after': float(np.max(np.abs(aa))),
                   'b_peak_after': float(np.max(np.abs(bb))),
                   'headroom_limit': limit}


def errors(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError('output shape mismatch or empty output')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        return {'finite': False, 'relative_l2': None, 'max_abs': None}
    diff = actual - reference
    norm = float(np.linalg.norm(reference))
    delta = float(np.linalg.norm(diff))
    # A zero reference is exact-only, rather than hidden by an arbitrary denominator.
    rel = delta / norm if norm else (0.0 if delta == 0 else None)
    result = {'finite': True, 'relative_l2': rel, 'max_abs': float(np.max(np.abs(diff)))}
    if reference.ndim == 2:
        row_norm = np.linalg.norm(reference, axis=1)
        row_delta = np.linalg.norm(diff, axis=1)
        invalid_zero = bool(np.any((row_norm == 0) & (row_delta != 0)))
        ratios = np.divide(row_delta, row_norm, out=np.zeros_like(row_delta), where=row_norm != 0)
        result['max_token_relative_l2'] = None if invalid_zero else float(ratios.max())
        result['nonzero_against_zero_tokens'] = int(np.sum((row_norm == 0) & (row_delta != 0)))
    return result


def rounded_factors(h, a, b):
    """Explicit rounding points with FP64 matmul; NOT device accumulation semantics."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        h16, a16, b16 = (np.asarray(x, dtype=np.float16) for x in (h, a, b))
        z16 = (h16.astype(np.float64) @ a16.astype(np.float64)).astype(np.float16)
        y16 = (z16.astype(np.float64) @ b16.astype(np.float64)).astype(np.float16)
    return y16.astype(np.float64)


def replay_metrics(ref, candidate, tokens, ref_prefix_ids, candidate_prefix_ids):
    """Full-vocabulary metrics; caller supplies the exact forced input IDs per step."""
    ref, candidate = matrix(ref, 'reference logits'), matrix(candidate, 'candidate logits')
    ids = np.asarray(tokens)
    if ref.shape != candidate.shape or ids.shape != (len(ref),) or ids.dtype.kind not in 'iu':
        raise ValueError('logit/token shape mismatch')
    if len(ref_prefix_ids) != len(ref) or ref_prefix_ids != candidate_prefix_ids:
        raise ValueError('fixed prefixes must match at every step')
    for i, prefix in enumerate(ref_prefix_ids):
        if not prefix or any(type(t) is not int or t < 0 or t >= ref.shape[1] for t in prefix):
            raise ValueError('invalid prefix IDs')
        if i and prefix != ref_prefix_ids[i-1] + [int(ids[i-1])]:
            raise ValueError('prefix does not follow forced continuation')
    if (ids < 0).any() or (ids >= ref.shape[1]).any():
        raise ValueError('token outside vocabulary')
    def log_softmax(x):
        z = x - x.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    p, q = log_softmax(ref), log_softmax(candidate)
    kl = np.maximum((np.exp(p) * (p - q)).sum(axis=1), 0)
    nll_delta = p[np.arange(len(ids)), ids] - q[np.arange(len(ids)), ids]
    return {'steps': len(ids), 'mean_kl_nats': float(kl.mean()),
            'max_kl_nats': float(kl.max()), 'mean_nll_increase_nats': float(nll_delta.mean()),
            'top1_agreement': float(np.mean(ref.argmax(axis=1) == candidate.argmax(axis=1)))}
