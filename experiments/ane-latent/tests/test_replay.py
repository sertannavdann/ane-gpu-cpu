import tempfile
from pathlib import Path
import unittest
import numpy as np
from replay import TraceError, collect_trace, compare_traces, read_trace, write_trace

V = 6


def backend(bias):
    """Fake full-logit backend: prefers the last prefix token shifted by bias."""
    def step(prefix):
        logits = np.full(V, -5.0)
        logits[(prefix[-1] + bias) % V] = 5.0
        logits[(prefix[-1] + bias + 1) % V] = 1.0
        return logits
    return step


class ReplayTests(unittest.TestCase):
    def test_forced_replay_continues_after_first_argmax_disagreement(self):
        prompt, forced = [0], [1, 2, 3, 4, 5]
        ref = collect_trace(backend(1), prompt, forced, V, 'reference')
        cand = collect_trace(backend(2), prompt, forced, V, 'candidate')
        self.assertEqual(ref['argmax_disagreements'], [])
        self.assertEqual(cand['first_disagreement'], 0)
        # Every step was still collected on the forced prefix, not the candidate's own continuation.
        self.assertEqual(cand['steps'], len(forced))
        self.assertEqual(cand['prefixes'], ref['prefixes'])
        self.assertEqual(cand['prefixes'][-1], prompt + forced[:-1])
        out = compare_traces(ref, cand)
        self.assertEqual(out['compared_steps'], len(forced))
        self.assertEqual(out['decision'], 'fail')
        self.assertLess(out['top1_agreement'], 1.0)
        same = compare_traces(ref, collect_trace(backend(1), prompt, forced, V, 'twin'))
        self.assertEqual(same['decision'], 'pass')
        self.assertAlmostEqual(same['mean_kl_nats'], 0.0)

    def test_top_k_or_partial_logits_are_refused(self):
        with self.assertRaisesRegex(TraceError, 'full logits'):
            collect_trace(lambda p: np.zeros(16), [0], [1], V, 'topk')
        ref = collect_trace(backend(1), [0], [1, 2], V, 'r')
        partial = collect_trace(backend(1), [0], [1, 2], V, 'p')
        partial['logits'] = partial['logits'][:1]
        with self.assertRaisesRegex(TraceError, 'complete full-logit'):
            compare_traces(ref, partial)
        other = collect_trace(backend(1), [0], [1, 3], V, 'other-prefix')
        with self.assertRaisesRegex(TraceError, 'identical'):
            compare_traces(ref, other)
        with self.assertRaises(TraceError):
            collect_trace(backend(1), [0], [V], V, 'oov')

    def test_nonfinite_step_is_reported_not_hidden(self):
        def bad(prefix):
            out = backend(1)(prefix)
            if len(prefix) == 2:
                out[0] = np.nan
            return out
        ref = collect_trace(backend(1), [0], [1, 2, 3], V, 'r')
        cand = collect_trace(bad, [0], [1, 2, 3], V, 'nan')
        self.assertEqual(cand['nonfinite_steps'], [1])
        self.assertEqual(cand['steps'], 3)
        out = compare_traces(ref, cand)
        self.assertEqual(out['decision'], 'invalid')
        self.assertIn('nonfinite', out['reasons'][0])

    def test_trace_files_are_immutable_and_hash_checked(self):
        trace = collect_trace(backend(1), [0], [1, 2, 3], V, 'disk')
        with tempfile.TemporaryDirectory() as td:
            d = write_trace(trace, Path(td) / 'trace-a')
            back = read_trace(d)
            np.testing.assert_allclose(back['logits'], trace['logits'])
            self.assertEqual(back['prefixes'], trace['prefixes'])
            self.assertEqual(compare_traces(trace, back)['decision'], 'pass')
            with self.assertRaises(FileExistsError):
                write_trace(trace, d)
            np.save(d / 'logits.npy', trace['logits'] + 1)
            with self.assertRaisesRegex(TraceError, 'hash'):
                read_trace(d)


if __name__ == '__main__':
    unittest.main()
