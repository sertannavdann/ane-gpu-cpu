"""Hardware-marked: runs only when the native build exists and the ANE runtime is available."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "prototype" / "ane" / "build"

pytestmark = pytest.mark.ane


def _ane_ready() -> bool:
    if not (BUILD / "bench_main").exists() or not (BUILD / "probe").exists():
        return False
    return subprocess.run([str(BUILD / "probe"), "--check"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _ane_ready(), reason="ANE native build or runtime unavailable")
def test_visible_projection_small_case(tmp_path):
    from prototype.ane import reference as R
    K, N, S = 8, 6, 3
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    wb = tmp_path / "W.bf16.bin"; wb.write_bytes(R.f32_to_bf16_bytes(W))
    W = R.bf16_bytes_to_f32(wb.read_bytes(), (N, K))          # the exact bf16 values the ANE receives
    X = np.zeros((S, K), np.float32)
    for s in range(S): X[s, s] = 1.0                            # basis vectors -> rows of W^T
    xp = tmp_path / "x.bin"; X.tofile(xp)
    spec = {"path": "visible", "K": K, "N": N, "S": S, "weight_bf16": str(wb), "sentinel": 5.0,
            "cases": [{"name": "basis", "x": str(xp), "y_out": str(tmp_path / "y.bin")}], "warmup": 1, "calls": 10,
            "measurements_out": "", "mil_out": str(tmp_path / "g.mil"), "summary_out": str(tmp_path / "s.json")}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    p = subprocess.run([str(BUILD / "bench_main"), str(tmp_path / "spec.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    summ = json.loads((tmp_path / "s.json").read_text())
    assert summ["failures"] == 0 and summ["checksum_drift"] == 0
    Yfull = np.fromfile(tmp_path / "y.bin", dtype=np.float32).reshape(32, N)
    ok, why = R.gate(R.errors(Yfull[:S], R.ref_fp32(X, W)))
    assert ok, why
    assert np.allclose(Yfull[:S], W[:, :S].T, atol=2e-3)         # channel order: e_s picks column s of W
    pad_ref = 5.0 * W.sum(axis=1)                                 # padded lanes saw the sentinel vector, independently
    assert np.allclose(Yfull[S:], pad_ref, rtol=2e-2, atol=2e-2)
    assert "conv(" in (tmp_path / "g.mil").read_text() and f"[{N}, {K}, 1, 1]" in (tmp_path / "g.mil").read_text()
