"""MLX-LM generation in-process. Imports MLX lazily so the dry workflow never needs it.
Priority switching happens in the scheduler at each yielded token; this module only streams."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

# Prototype-owned model cache; set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "cache.nosync" / "hf"))

from .contracts import ContractError, ErrorKind, GenerateEvent, Request


class MlxGenerator:
    name = "mlx"

    def __init__(self, model_id: str, max_context: int = 4096, expected_hash: str | None = None):
        try:
            import mlx.core as mx
            import mlx_lm
        except ImportError as e:
            raise ImportError("generator_mlx needs mlx and mlx-lm; install requirements-mlx.txt") from e
        self.mx, self.mlx_lm = mx, mlx_lm
        self.model_id, self.max_context = model_id, max_context
        self.model, self.tokenizer = mlx_lm.load(model_id)
        self.weights_hash = self._hash_weights(model_id)
        if expected_hash and expected_hash != self.weights_hash:
            raise RuntimeError(f"model hash {self.weights_hash} != pinned {expected_hash}")
        self.open: set[str] = set()

    @staticmethod
    def _hash_weights(model_id: str) -> str:
        """Hash of the safetensors index/weights on disk so runs pin exactly what was loaded."""
        try:
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(model_id, local_files_only=True))
        except Exception:
            root = Path(model_id)
        h = hashlib.sha256()
        for p in sorted(root.glob("*.safetensors")):
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
            with p.open("rb") as f:
                h.update(f.read(1 << 20))  # first MiB of each shard; full hash is slow and optional
        return h.hexdigest()[:16]

    def probe(self) -> dict:
        """Readiness: one short generation must succeed before the worker reports serving."""
        out = "".join(ev.text for ev in self.stream(
            Request("probe", "probe", "Reply with the single word OK.", request_class=None, max_output_tokens=4),  # type: ignore[arg-type]
            "Reply with the single word OK."))
        self.release("probe")
        return {"ok": bool(out.strip()), "text": out, "memory": self.memory()}

    def memory(self) -> dict:
        mx = self.mx
        return {"active_bytes": int(mx.get_active_memory()), "peak_bytes": int(mx.get_peak_memory()),
                "cache_bytes": int(mx.get_cache_memory())}

    def stream(self, request: Request, prompt: str) -> Iterator[GenerateEvent]:
        if request.request_id in self.open:
            raise RuntimeError(f"generator already open for {request.request_id}")
        self.open.add(request.request_id)
        messages = [{"role": "user", "content": prompt}]
        # Chat template applied exactly once, here.
        rendered = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        count = 0
        try:
            for r in self.mlx_lm.stream_generate(self.model, self.tokenizer, rendered,
                                                 max_tokens=request.max_output_tokens):
                count = getattr(r, "generation_tokens", count + 1)
                if r.text:
                    yield GenerateEvent(text=r.text)
                if getattr(r, "finish_reason", None):
                    yield GenerateEvent(token_count=count, finish_reason=r.finish_reason)
                    return
            yield GenerateEvent(token_count=count, finish_reason="stop")
        except Exception as e:
            yield GenerateEvent(error=ContractError(ErrorKind.GENERATION_ERROR, repr(e)), finish_reason="error")

    def release(self, request_id: str) -> None:
        self.open.discard(request_id)
