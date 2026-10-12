"""Capture real layer-0 activations from Qwen3-0.6B with MLX for the R1/R2 fixtures.
x_attn_in  = input_layernorm(embed(tokens))            -> input of q/k/v_proj      [S, hidden]
x_mlp_in   = post_attention_layernorm(h + attn(x_attn_in)) -> input of gate/up_proj [S, hidden]
Saved as fp32 .npy under prototype/ane/fixtures/ with the prompt and token ids recorded."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache.nosync" / "hf"))
FIX = ROOT / "prototype" / "ane" / "fixtures"
PROMPT = ("Explain this routing trace: classification returned capabilities [coding, verification], complexity 0.4, "
          "and the request went direct to the heavy tier. Why direct and why heavy? Answer in three sentences.")


def capture(S: int = 32) -> dict:
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load("Qwen/Qwen3-0.6B")
    ids = tok.encode(PROMPT)[:S]
    assert len(ids) == S, f"prompt has {len(ids)} tokens, need {S}"
    x = mx.array([ids])
    inner = model.model
    h = inner.embed_tokens(x)
    layer = inner.layers[0]
    x_attn_in = layer.input_layernorm(h)
    mask = "causal"
    attn = layer.self_attn(x_attn_in, mask=mask, cache=None)
    h2 = h + attn
    x_mlp_in = layer.post_attention_layernorm(h2)
    mx.eval(x_attn_in, x_mlp_in)
    a = np.array(x_attn_in.astype(mx.float32))[0]; m = np.array(x_mlp_in.astype(mx.float32))[0]
    FIX.mkdir(exist_ok=True)
    np.save(FIX / "real_x_attn_in.npy", a); np.save(FIX / "real_x_mlp_in.npy", m)
    rec = {"model": "Qwen/Qwen3-0.6B", "layer": 0, "S": S, "prompt": PROMPT, "token_ids": ids,
           "x_attn_in": {"shape": list(a.shape), "abs_max": float(np.abs(a).max()), "rms": float(np.sqrt((a.astype(np.float64)**2).mean()))},
           "x_mlp_in": {"shape": list(m.shape), "abs_max": float(np.abs(m).max()), "rms": float(np.sqrt((m.astype(np.float64)**2).mean()))}}
    (FIX / "real_activations.json").write_text(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2)[:900])
