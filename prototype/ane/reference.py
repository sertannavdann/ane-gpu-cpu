"""Fixtures, references and error metrics for the projection experiments. No ANE calls here."""
from __future__ import annotations

import numpy as np

S_ALLOC = 32


def bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


def f32_to_bf16_bytes(a: np.ndarray) -> bytes:
    """Round-to-nearest-even bf16 truncation of fp32."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
    return rounded.tobytes()


def fixtures(S: int, K: int, seed: int, act_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Deterministic input cases X[S,K] fp32. Basis vectors check channel order; outliers probe fp16
    saturation; zeros give an absolute-tolerance case."""
    rng = np.random.default_rng(seed)
    cases = {
        "random": (rng.standard_normal((S, K)) * act_scale).astype(np.float32),
        "zeros": np.zeros((S, K), np.float32),
        "basis": np.zeros((S, K), np.float32),
        "outlier": (rng.standard_normal((S, K)) * act_scale).astype(np.float32),
    }
    for s in range(S):
        cases["basis"][s, (s * 7) % K] = 1.0
    idx = rng.choice(K, size=min(4, K), replace=False)
    cases["outlier"][:, idx] = np.where(rng.random((S, len(idx))) < 0.5, -20000.0, 20000.0).astype(np.float32)
    return cases


def ref_fp32(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Accelerate's float64 BLAS raises spurious FP flags on some inputs; results are checked for finiteness downstream.
    with np.errstate(all="ignore"):
        return (X.astype(np.float64) @ W.astype(np.float64).T).astype(np.float32)


def ref_fp16_inputs(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """fp16-rounded inputs and weights, fp32 accumulation: separates input rounding from ANE arithmetic."""
    with np.errstate(all="ignore"):
        return (X.astype(np.float16).astype(np.float64) @ W.astype(np.float16).astype(np.float64).T).astype(np.float32)


def ref_mlx_fp16(X: np.ndarray, W: np.ndarray, repeats: int = 0) -> tuple[np.ndarray, dict]:
    """Matched fp16 matmul on the GPU via MLX. mx.eval is called before the timer stops."""
    import time
    import mlx.core as mx
    x = mx.array(X.astype(np.float16)); w = mx.array(W.astype(np.float16))
    y = mx.matmul(x, w.T); mx.eval(y)
    times = []
    for _ in range(repeats):
        t = time.perf_counter(); y = mx.matmul(x, w.T); mx.eval(y); times.append((time.perf_counter() - t) * 1e6)
    return np.array(y.astype(mx.float32)), {"call_us": times}


def errors(Y: np.ndarray, ref: np.ndarray) -> dict:
    Y = Y.astype(np.float64); ref = ref.astype(np.float64)
    d = Y - ref
    ref_rms = float(np.sqrt((ref ** 2).mean())) if ref.size else 0.0
    ref_norm = float(np.sqrt((ref ** 2).sum()))
    finite = np.isfinite(Y)
    d_f = np.where(finite, d, 0.0)
    out = {"rel_l2": float(np.sqrt((d_f ** 2).sum()) / ref_norm) if ref_norm > 0 else None,
           "rms_err": float(np.sqrt((d_f ** 2).mean())), "max_abs_err": float(np.abs(d_f).max()),
           "norm_max_err": float(np.abs(d_f).max() / max(1.0, ref_rms)), "ref_rms": ref_rms,
           "nonfinite": int((~finite).sum()),
           "cosine": float((Y * ref).sum() / (np.linalg.norm(Y) * np.linalg.norm(ref))) if ref_norm > 0 and np.linalg.norm(Y) > 0 else None}
    return out


GATES = {"projection": {"rel_l2": 0.002, "norm_max_err": 0.02, "zero_abs": 0.001}}


def gate(err: dict, kind: str = "projection", zero_reference: bool = False) -> tuple[bool, list[str]]:
    g = GATES[kind]; reasons = []
    if err["nonfinite"]:
        reasons.append(f"nonfinite={err['nonfinite']}")
    if zero_reference:
        if err["max_abs_err"] > g["zero_abs"]:
            reasons.append(f"zero-ref abs {err['max_abs_err']:.3g} > {g['zero_abs']}")
    else:
        if err["rel_l2"] is None or err["rel_l2"] > g["rel_l2"]:
            reasons.append(f"rel_l2 {err['rel_l2']} > {g['rel_l2']}")
        if err["norm_max_err"] > g["norm_max_err"]:
            reasons.append(f"norm_max {err['norm_max_err']:.3g} > {g['norm_max_err']}")
    return (not reasons), reasons
