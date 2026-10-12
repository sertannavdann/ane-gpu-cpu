import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evidence import assess, canonical, digest, import_evidence, record, report
from fixtures import fixture
from mil import projection
from numerics import balance, errors, factorize, replay_metrics, rounded_factors
from theories import cost_theory


def native_fixture():
    # Fabricated protocol fixture ONLY; never recorded as a hardware run.
    return {'evidence_kind': 'native_ane', 'baseline_valid': True,
            'interference_free': True, 'artifacts_verified': True, 'boundary': 'full_ffn',
            'provenance': {k: 'a' * 64 for k in ('source_sha256', 'runtime_sha256', 'weights_sha256',
                       'inputs_sha256', 'baseline_mil_sha256', 'candidate_mil_sha256')},
            'execution': {side: {'selected_graph': side, 'requested_graph': side,
                          'fallback': False, 'host_calls_per_invocation': 1}
                          for side in ('baseline', 'candidate')},
            'cases': [{'id': f'real-s{s}', 's': s, 'kind': 'real_heldout', 'finite': True,
                       'input_sha256': 'b' * 64,
                       'implementation_relative_l2': .0005, 'compression_relative_l2': .001}
                      for s in (1, 8, 32)],
            'stability': {'calls': 1000, 'failures': 0, 'checksum_drift': 0},
            'windows': [{'id': f'w-s{s}-{i}', 's': s, 'failures': 0, 'baseline_us': [100.] * 1000,
                         'candidate_us': [80.] * 1000,
                         'baseline_input_sha256': 'b' * 64, 'candidate_input_sha256': 'b' * 64,
                         'order': ['baseline', 'candidate'] if i % 2 else ['candidate', 'baseline']}
                        for s in (1, 8, 32) for i in range(3)]}


class NumericsTests(unittest.TestCase):
    def test_low_rank_preserves_heldout_and_balance(self):
        h, x, w = fixture('low_rank')
        a, b, _ = factorize(h, w, 4)
        aa, bb, _ = balance(h, a, b)
        np.testing.assert_allclose(a @ b, aa @ bb, atol=1e-14)
        np.testing.assert_allclose(x @ w, (x @ aa) @ bb, atol=1e-12)
        self.assertLess(errors(x @ w, rounded_factors(x, aa, bb))['relative_l2'], .005)

    def test_full_rank_is_reference(self):
        h, x, w = fixture('flat_spectrum')
        a, b, _ = factorize(h, w, 16)
        np.testing.assert_allclose(x @ w, (x @ a) @ b, atol=1e-12)

    def test_shift_not_hidden_by_calibration_accuracy(self):
        h, x, w = fixture('distribution_shift')
        a, b, _ = factorize(h, w, 8)
        self.assertLess(errors(h @ w, (h @ a) @ b)['relative_l2'], 1e-12)
        self.assertGreater(errors(x @ w, (x @ a) @ b)['relative_l2'], .9)

    def test_zero_nonfinite_and_subnormal(self):
        self.assertEqual(errors([0], [0])['relative_l2'], 0)
        self.assertIsNone(errors([0], [1])['relative_l2'])
        self.assertFalse(errors([float('inf')], [float('inf')])['finite'])
        self.assertFalse(errors([0], [float('nan')])['finite'])
        self.assertGreater(float(np.float16(2**-24)), 0)
        with self.assertRaises(ValueError): errors([1, 2], [1])

    def test_balancing_rejects_impossible_range(self):
        with self.assertRaisesRegex(ValueError, 'no feasible'):
            balance(np.ones((1, 1)), np.array([[1e20]]), np.array([[1e20]]))

    def test_balance_zero_channel_finite(self):
        a, b, info = balance(np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((1, 2)))
        self.assertEqual(info['scale'], [1.0])
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_heldout_overflow_rejected(self):
        h, x, w = fixture('overflow')
        a, b, _ = factorize(h, w, 8)
        aa, bb, _ = balance(h, a, b)
        self.assertFalse(errors(x @ w, rounded_factors(x, aa, bb))['finite'])

    def test_deterministic_fixture_and_separate_inputs(self):
        for a, b in zip(fixture('outlier_channels'), fixture('outlier_channels')):
            np.testing.assert_array_equal(a, b)
        h, x, _ = fixture('low_rank')
        self.assertFalse(np.array_equal(h[:len(x)], x))

    def test_replay_continues_after_argmax_disagreement(self):
        a = np.array([[3., 0, 0], [0, 3., 0]])
        b = np.array([[0., 3, 0], [0, 3., 0]])
        prefixes = [[0], [0, 1]]
        m = replay_metrics(a, b, [1, 1], prefixes, prefixes)
        self.assertEqual(m['steps'], 2)
        self.assertEqual(m['top1_agreement'], .5)
        self.assertGreater(m['mean_kl_nats'], 0)
        same = replay_metrics(a, a + 100, [1, 1], prefixes, prefixes)
        self.assertAlmostEqual(same['mean_kl_nats'], 0)
        self.assertAlmostEqual(same['mean_nll_increase_nats'], 0)
        with self.assertRaises(ValueError):
            replay_metrics(a, b, [1, 1], prefixes, [[0], [0, 0]])
        with self.assertRaises(ValueError):
            replay_metrics(a, b, [1, 1], [[0], [0, 0]], [[0], [0, 0]])


class EvidenceTests(unittest.TestCase):
    def test_protocol_pass_never_adopts_model(self):
        out = assess(native_fixture())
        self.assertEqual(out['decision'], 'retain_local')
        self.assertEqual(out['adoption'], 'not_evaluated')

    def test_no_native_claim_from_theory(self):
        self.assertEqual(assess(cost_theory())['decision'], 'inconclusive')
        self.assertEqual(assess({'evidence_kind': 'cpu_synthetic'})['decision'], 'inconclusive')

    def test_silent_fallback_invalidates_run(self):
        e = native_fixture()
        e['execution']['candidate']['selected_graph'] = 'dense'
        self.assertEqual(assess(e)['decision'], 'invalid')

    def test_correctness_failure_dominates_speed(self):
        e = native_fixture()
        e['cases'][0]['compression_relative_l2'] = .1
        self.assertEqual(assess(e)['decision'], 'reject')

    def test_missing_and_invalid_evidence(self):
        for key, value, expected in [('baseline_valid', False, 'invalid'),
                ('interference_free', False, 'invalid'), ('cases', [], 'inconclusive'),
                ('windows', [], 'inconclusive'), ('artifacts_verified', False, 'invalid')]:
            e = native_fixture(); e[key] = value
            self.assertEqual(assess(e)['decision'], expected)
        e = native_fixture(); e['cases'][0]['compression_relative_l2'] = float('nan')
        self.assertEqual(assess(e)['decision'], 'invalid')

    def test_one_bad_window_cannot_be_averaged_away(self):
        e = native_fixture(); e['windows'][0]['candidate_us'] = [101.] * 1000
        self.assertEqual(assess(e)['decision'], 'reject')

    def test_tail_regression_fails(self):
        e = native_fixture(); e['windows'][0]['candidate_us'] = [80.] * 900 + [110.] * 100
        self.assertEqual(assess(e)['decision'], 'reject')

    def test_registry_no_overwrites_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a = record(td, {'hypothesis': 'x', 'evidence_kind': 'unit_test'})
            b = record(td, {'hypothesis': 'x', 'evidence_kind': 'unit_test'})
            self.assertNotEqual(a, b)
            self.assertEqual(report(td, Path(td) / 'report.md'), 2)
            with self.assertRaises(ValueError): record(td, {'bad': float('nan')})

    def test_nonfinite_import_persists_failure_receipt(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / 'bad.json'
            bad.write_text('{"evidence_kind":"native_ane","bad":NaN}')
            run = subprocess.run([sys.executable, 'lab.py', 'assess', str(bad),
                                  '--registry', str(root / 'registry')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            receipt = json.loads(next((root / 'registry').glob('*.json')).read_text())
            self.assertEqual(receipt['assessment']['decision'], 'invalid')
            self.assertEqual(receipt['evidence_kind'], 'invalid_import')

    def test_importer_verifies_bytes_not_producer_assertion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); raw = root / 'raw.txt'; raw.write_text('evidence')
            e = native_fixture(); e['artifacts'] = [{'path': 'raw.txt', 'sha256': digest(raw)}]
            e['provenance'] = {k: digest(raw) for k in e['provenance']}
            for c in e['cases']: c['input_sha256'] = digest(raw)
            for w in e['windows']: w['baseline_input_sha256'] = w['candidate_input_sha256'] = digest(raw)
            f = root / 'evidence.json'; f.write_text(canonical(e))
            self.assertTrue(import_evidence(f)['artifacts_verified'])
            raw.write_text('changed')
            with self.assertRaises(ValueError): import_evidence(f)
            e['artifacts'] = []; f.write_text(canonical(e))
            self.assertFalse(import_evidence(f)['artifacts_verified'])


class TheoryTests(unittest.TestCase):
    def test_weight_accounting_and_extra_dispatch_penalty(self):
        t = cost_theory(rank=256)
        self.assertEqual(t['dense_weight_bytes'], 6291456)
        self.assertEqual(t['factor_weight_bytes'], 2097152)
        self.assertGreater(t['estimated_saved_us_envelope'][0], 0)
        self.assertLess(cost_theory(rank=256, extra_dispatches=1)['estimated_saved_us_envelope'][1], 0)
        self.assertEqual(cost_theory(rank=768)['weight_saving_fraction'], 0)

    def test_draft_mil_shapes_and_no_latent_output(self):
        m = projection(3072, 1024, 256)
        self.assertIn('[256, 3072, 1, 1]', m)
        self.assertIn('[1024, 256, 1, 1]', m)
        self.assertEqual(m.count('= conv('), 2)
        self.assertIn('} -> (y);', m)
        self.assertNotIn('} -> (z', m)
        with self.assertRaises(ValueError): projection(3, 4, 5)

if __name__ == '__main__':
    unittest.main()
