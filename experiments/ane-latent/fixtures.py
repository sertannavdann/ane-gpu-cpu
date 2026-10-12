"""Small deterministic synthetic populations. These are not model calibration."""
import numpy as np

CASES = ('low_rank', 'flat_spectrum', 'distribution_shift', 'outlier_channels',
         'zero', 'subnormal', 'overflow')


def fixture(kind, seed=1729, s=32):
    if kind not in CASES or s not in (1, 8, 32):
        raise ValueError('unknown fixture or sequence size')
    # Separate RNG streams for calibration and held-out inputs.
    rng = np.random.default_rng(seed)
    evaluation = np.random.default_rng(seed + 1000003 + s)
    h, held = rng.normal(size=(128, 24)), evaluation.normal(size=(s, 24))
    w = rng.normal(size=(24, 16)) / np.sqrt(24)
    if kind == 'low_rank':
        w = rng.normal(size=(24, 4)) @ rng.normal(size=(4, 16)) / 24
    elif kind == 'flat_spectrum':
        # Equal output singular values: small-rank truncation should lose information.
        w = np.eye(24, 16)
    elif kind == 'distribution_shift':
        h[:, 8:] = 0
        held[:, :8] = 0
        w = np.eye(24, 16)
    elif kind == 'outlier_channels':
        h[:, 0] *= 1000
        held[:, 0] *= 1000
    elif kind == 'zero':
        h[:] = 0
        held[:] = 0
    elif kind == 'subnormal':
        h *= 1e-6
        held *= 1e-6
    elif kind == 'overflow':
        held *= 1e8
    return h, held, w
