import tempfile
from pathlib import Path
import unittest
from evidence import assess, canonical, digest, import_evidence
from test_lab import native_fixture


class EvidenceContractTests(unittest.TestCase):
    """Importer gaps named in CURRENT-STATE.md. Fixtures are fabricated protocol data only."""

    def test_real_heldout_required_at_every_s(self):
        e = native_fixture()
        for c in e['cases']:
            if c['s'] == 8:
                c['kind'] = 'random'
        out = assess(e)
        self.assertEqual(out['decision'], 'inconclusive')
        self.assertIn('S=8', out['reasons'][0])
        # Extra cases do not substitute for the missing shape.
        e['cases'].append({'id': 'real-s32-b', 's': 32, 'kind': 'real_heldout', 'finite': True,
                           'input_sha256': 'b' * 64, 'implementation_relative_l2': .0001,
                           'compression_relative_l2': .0001})
        self.assertEqual(assess(e)['decision'], 'inconclusive')

    def test_duplicate_case_and_window_ids_are_invalid(self):
        e = native_fixture(); e['cases'][1]['id'] = e['cases'][0]['id']
        out = assess(e)
        self.assertEqual(out['decision'], 'invalid'); self.assertIn('duplicate case id', out['reasons'][0])
        e = native_fixture(); e['windows'][2]['id'] = e['windows'][0]['id']
        out = assess(e)
        self.assertEqual(out['decision'], 'invalid'); self.assertIn('duplicate window id', out['reasons'][0])
        e = native_fixture(); del e['cases'][0]['id']
        self.assertEqual(assess(e)['decision'], 'invalid')

    def test_cross_condition_input_identity(self):
        e = native_fixture(); e['windows'][0]['candidate_input_sha256'] = 'c' * 64
        out = assess(e)
        self.assertEqual(out['decision'], 'invalid'); self.assertIn('input identity', out['reasons'][0])
        e = native_fixture(); del e['windows'][4]['baseline_input_sha256']
        self.assertEqual(assess(e)['decision'], 'invalid')
        e = native_fixture(); del e['cases'][0]['input_sha256']
        self.assertEqual(assess(e)['decision'], 'invalid')

    def test_input_hashes_must_be_backed_by_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); raw = root / 'raw.txt'; raw.write_text('evidence')
            inputs = root / 'inputs.bin'; inputs.write_bytes(b'\x00\x01')
            e = native_fixture()
            e['artifacts'] = [{'path': 'raw.txt', 'sha256': digest(raw)}, {'path': 'inputs.bin', 'sha256': digest(inputs)}]
            e['provenance'] = {k: digest(raw) for k in e['provenance']}
            for c in e['cases']: c['input_sha256'] = digest(inputs)
            for w in e['windows']: w['baseline_input_sha256'] = w['candidate_input_sha256'] = digest(inputs)
            f = root / 'evidence.json'; f.write_text(canonical(e))
            self.assertTrue(import_evidence(f)['artifacts_verified'])
            # A window that names an input file nobody hashed is refused.
            e['windows'][0]['baseline_input_sha256'] = e['windows'][0]['candidate_input_sha256'] = 'd' * 64
            f.write_text(canonical(e))
            with self.assertRaisesRegex(ValueError, 'not backed'): import_evidence(f)

    def test_summary_must_match_raw_samples(self):
        e = native_fixture()
        e['windows'][0]['summary'] = {'baseline_p50_us': 100.0, 'candidate_p50_us': 80.0,
                                     'baseline_p95_us': 100.0, 'candidate_p95_us': 80.0,
                                     'baseline_count': 1000, 'candidate_count': 1000}
        self.assertEqual(assess(e)['decision'], 'retain_local')
        e['windows'][0]['summary']['candidate_p50_us'] = 79.0
        out = assess(e)
        self.assertEqual(out['decision'], 'invalid'); self.assertIn('disagrees with raw samples', out['reasons'][0])
        e = native_fixture(); e['windows'][0]['summary'] = {'candidate_count': 999}
        self.assertEqual(assess(e)['decision'], 'invalid')
        e = native_fixture(); e['summary'] = {'cases': 3, 'windows': 9, 'stability_calls': 1000}
        self.assertEqual(assess(e)['decision'], 'retain_local')
        e['summary']['windows'] = 12
        self.assertEqual(assess(e)['decision'], 'invalid')

    def test_hardened_fixture_still_passes_and_never_adopts(self):
        out = assess(native_fixture())
        self.assertEqual(out['decision'], 'retain_local')
        self.assertEqual(out['adoption'], 'not_evaluated')


if __name__ == '__main__':
    unittest.main()
