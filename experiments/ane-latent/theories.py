"""Screening hypotheses. Every microsecond here is an estimate, never a timing."""
from itertools import product


def cost_theory(k=3072, n=1024, rank=256, extra_dispatches=0):
    if any(type(v) is not int or v < 1 for v in (k, n, rank)) or rank > min(k, n):
        raise ValueError('invalid matrix dimensions or rank')
    if type(extra_dispatches) is not int or extra_dispatches < 0:
        raise ValueError('invalid dispatch count')
    dense = 2 * k * n
    factored = 2 * rank * (k + n)
    saved_mb = (dense - factored) / 1e6
    # Envelope from the recorded R1 runs; combinations are not confidence bounds.
    gains = [saved_mb * slope - extra_dispatches * fixed
             for slope, fixed in product((7.0, 11.8), (50.0, 61.0))]
    return {'hypothesis': f'output-factorization-r{rank}-extra-dispatches-{extra_dispatches}',
            'evidence_kind': 'analytical_estimate', 'shape': [k, n], 'rank': rank,
            'dense_weight_bytes': dense, 'factor_weight_bytes': factored,
            'weight_saving_fraction': 1 - factored / dense,
            'estimated_saved_us_envelope': [min(gains), max(gains)],
            'extra_dispatches': extra_dispatches,
            'assumptions': ['FP16 weights', 'R1 local slope 7.0–11.8 us/decimal MB',
                            'R1 fixed cost 50–61 us', 'no intermediate spill, packing or compiler penalty',
                            'not an independent estimate of effective device bandwidth'],
            'assessment': {'decision': 'worth_testing' if min(gains) > 0 else 'unlikely_under_this_fit',
                           'reasons': ['cost-model screening only; accuracy, ANE latency and energy unmeasured']}}
