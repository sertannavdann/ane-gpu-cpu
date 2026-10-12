"""Model-derived projection shapes and weights for the ANE experiments.
Reads Qwen3-0.6B config.json and the safetensors header; exports one projection as BF16 bytes and
fp32 numpy, with the FULL sha256 of the tensor bytes and of the shard (no partial fingerprints)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache.nosync" / "hf"))
MODEL_ID = "Qwen/Qwen3-0.6B"


def snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, allow_patterns=["config.json", "*.safetensors"], local_files_only=True))


def projection_shapes(cfg: dict) -> dict:
    h, heads, kv, hd, inter = (cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"],
                               cfg["head_dim"], cfg["intermediate_size"])
    return {"q_proj": (heads * hd, h), "k_proj": (kv * hd, h), "v_proj": (kv * hd, h), "o_proj": (h, heads * hd),
            "gate_proj": (inter, h), "up_proj": (inter, h), "down_proj": (h, inter)}   # (N=out, K=in)


def read_tensor_bf16(shard: Path, name: str) -> tuple[np.ndarray, bytes, dict]:
    with shard.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        meta = header[name]
        assert meta["dtype"] == "BF16", meta["dtype"]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype="<u2").reshape(meta["shape"])
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32, raw, meta


def export(name: str, out_dir: Path, layer: int = 0) -> dict:
    snap = snapshot()
    cfg = json.loads((snap / "config.json").read_text())
    shard = snap / "model.safetensors"
    prefix = "self_attn" if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
    tensor = f"model.layers.{layer}.{prefix}.{name}.weight"
    f32, raw, meta = read_tensor_bf16(shard, tensor)
    n_out, k_in = projection_shapes(cfg)[name]
    assert tuple(meta["shape"]) == (n_out, k_in), (meta["shape"], (n_out, k_in))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.bf16.bin").write_bytes(raw)
    np.save(out_dir / f"{name}.f32.npy", f32)
    rec = {"model_id": MODEL_ID, "snapshot": snap.name, "tensor": tensor, "shape_NK": [n_out, k_in],
           "dtype": meta["dtype"], "tensor_sha256": hashlib.sha256(raw).hexdigest(),
           "shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(), "shard_bytes": shard.stat().st_size,
           "config": {k: cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                                         "intermediate_size", "rms_norm_eps", "torch_dtype")},
           "weight_stats": {"abs_max": float(np.abs(f32).max()), "rms": float(np.sqrt((f32.astype(np.float64) ** 2).mean()))}}
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "q_proj"
    rec = export(name, ROOT / "prototype" / "ane" / "fixtures")
    print(json.dumps(rec, indent=2))
