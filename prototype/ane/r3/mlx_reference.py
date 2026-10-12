"""MLX reference for the frozen extraction evaluation: same BF16 Qwen3-0.6B weights, greedy, thinking off.
Writes results/<run>/mlx/<case>.json with text, token ids and per-step top-16 logits."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(ROOT / "cache.nosync" / "hf"))
EVAL = json.loads((ROOT / "prototype/ane/r3/extraction_eval.json").read_text())


def main(out_dir: Path, model_id: str = "Qwen/Qwen3-0.6B", topk: int = 16):
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in EVAL["cases"]:
        user = EVAL["prompt"].format(note=case["note"])
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok.encode(prompt)
        x = mx.array([ids]); cache = None
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
        steps, gen = [], []
        t0 = time.perf_counter()
        logits = model(x, cache=cache)[:, -1, :]
        for step in range(EVAL["max_tokens"]):
            l = np.array(logits.astype(mx.float32))[0]
            top = np.argpartition(-l, topk)[:topk]; top = top[np.argsort(-l[top])]
            tid = int(top[0])
            steps.append({"step": step, "token": tid, "top": [[int(t), float(l[t])] for t in top]})
            gen.append(tid)
            if tid in (tok.eos_token_id, 151645, 151643, 248044, 248046):   # Qwen3 and Qwen3.5 end tokens
                break
            logits = model(mx.array([[tid]]), cache=cache)[:, -1, :]
        text = tok.decode(gen)
        (out_dir / f"{case['id']}.json").write_text(json.dumps({"id": case["id"], "prompt_tokens": len(ids), "prompt_ids": ids, "generated_ids": gen,
                                                                 "text": text, "steps": steps, "wall_s": time.perf_counter() - t0}, ensure_ascii=False))
        print(case["id"], len(gen), "tokens:", text[:90].replace("\n", " "))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/ane-r3-model/mlx")
