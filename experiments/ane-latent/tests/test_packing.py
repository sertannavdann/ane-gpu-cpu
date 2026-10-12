import unittest
import numpy as np
from packing import conv1x1_reference, pack_activations, unpack_outputs, weight_blob


class PackingTests(unittest.TestCase):
    def setUp(self):
        # Asymmetric, non-square values: every (s, k) and (k, n) pair is distinct.
        self.s, self.k, self.n, self.alloc = 5, 6, 3, 8
        self.x = np.arange(self.s * self.k, dtype=np.float64).reshape(self.s, self.k) / 8
        self.w = (np.arange(self.k * self.n, dtype=np.float64).reshape(self.k, self.n) - 7) / 16

    def test_transposed_packing_with_asymmetric_values(self):
        packed = pack_activations(self.x, self.alloc)
        self.assertEqual(packed.shape, (1, self.k, 1, self.alloc))
        for s in range(self.s):
            for k in range(self.k):
                self.assertEqual(float(packed[0, k, 0, s]), float(np.float16(self.x[s, k])))
        blob = weight_blob(self.w)
        self.assertEqual(blob.shape, (self.n, self.k, 1, 1))
        self.assertEqual(float(blob[2, 5, 0, 0]), float(np.float16(self.w[5, 2])))
        y = unpack_outputs(conv1x1_reference(packed, blob), self.s)
        np.testing.assert_allclose(y, self.x @ self.w, rtol=2e-3, atol=2e-3)
        # A transposed weight blob is wrong for a non-square shape and must not be silently accepted.
        with self.assertRaises(ValueError):
            conv1x1_reference(packed, weight_blob(self.w.T))

    def test_padding_lanes_do_not_leak_into_valid_rows(self):
        blob = weight_blob(self.w)
        zero = unpack_outputs(conv1x1_reference(pack_activations(self.x, self.alloc, 0.0), blob), self.s)
        sentinel_surface = conv1x1_reference(pack_activations(self.x, self.alloc, 1e4), blob)
        sentinel = unpack_outputs(sentinel_surface, self.s)
        np.testing.assert_array_equal(zero, sentinel)
        # Sentinel lanes visibly compute something; trimming is what keeps them out.
        self.assertTrue(np.any(sentinel_surface[0, :, 0, self.s:] != 0))
        self.assertTrue(np.isfinite(zero).all())

    def test_two_stage_factor_layout_matches_row_vector_algebra(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(self.k, 2)), rng.normal(size=(2, self.n))
        packed = pack_activations(self.x, self.alloc, 1e4)
        z = conv1x1_reference(packed, weight_blob(a))
        self.assertEqual(z.shape, (1, 2, 1, self.alloc))
        y = unpack_outputs(conv1x1_reference(z, weight_blob(b)), self.s)
        np.testing.assert_allclose(y, (self.x @ a) @ b, rtol=5e-3, atol=5e-3)
        with self.assertRaises(ValueError):
            pack_activations(np.zeros((self.alloc + 1, self.k)), self.alloc)
        with self.assertRaises(ValueError):
            unpack_outputs(z, self.alloc + 1)


if __name__ == '__main__':
    unittest.main()
