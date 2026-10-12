"""Host-side layout for the channel-first [1, C, 1, S] convolution surfaces.

CPU-only helpers with a NumPy 1x1-convolution equivalent, so transposition and
padding-lane mistakes are caught before any native run. They do not describe
the runtime's physical plane strides; a native adapter must still verify those.
"""
from __future__ import annotations
import numpy as np


def pack_activations(x, allocated_s, pad_value=0.0):
    """[S, K] row-vector activations -> fp16 [1, K, 1, allocated_s]; unused lanes hold pad_value."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[0] > allocated_s:
        raise ValueError('activations must be [S, K] with 1 <= S <= allocated_s')
    packed = np.full((1, x.shape[1], 1, allocated_s), pad_value, dtype=np.float16)
    packed[0, :, 0, :x.shape[0]] = x.T.astype(np.float16)
    return packed


def unpack_outputs(y, s):
    """fp16 [1, N, 1, allocated_s] -> [s, N], trimming padded lanes."""
    y = np.asarray(y)
    if y.ndim != 4 or y.shape[0] != 1 or y.shape[2] != 1 or not 1 <= s <= y.shape[3]:
        raise ValueError('output surface must be [1, N, 1, allocated_s]')
    return y[0, :, 0, :s].T.astype(np.float64)


def weight_blob(w):
    """Row-vector weight [K_in, N_out] -> convolution weight [N_out, K_in, 1, 1] in fp16."""
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError('weight must be [K, N]')
    return np.ascontiguousarray(w.T.astype(np.float16)[:, :, None, None])


def conv1x1_reference(packed, blob):
    """out[0, n, 0, s] = sum_k blob[n, k] * packed[0, k, 0, s], FP64 accumulate, fp16 result.

    This is the algebra of the drafted MIL conv; it is not the ANE's accumulation order.
    """
    if packed.ndim != 4 or blob.ndim != 4 or packed.shape[1] != blob.shape[1]:
        raise ValueError('channel mismatch between surface and weight')
    out = blob[:, :, 0, 0].astype(np.float64) @ packed[0, :, 0, :].astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        return out.astype(np.float16)[None, :, None, :]
