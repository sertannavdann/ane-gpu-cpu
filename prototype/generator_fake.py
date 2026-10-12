"""Scripted token stream with per-token delay and failure injection. Used for the dry workflow;
never for a speed or quality claim."""
from __future__ import annotations

import time
from typing import Iterator

from .contracts import ContractError, ErrorKind, GenerateEvent, Request


class ScriptedGenerator:
    name = "fake"

    def __init__(self, token_delay_s: float = 0.002, failures: dict[str, int] | None = None):
        """failures: request_id -> fail after this many tokens with a typed generation error."""
        self.token_delay_s = token_delay_s
        self.failures = failures or {}
        self.open: set[str] = set()
        self.released: list[str] = []
        self.calls: list[str] = []

    def stream(self, request: Request, prompt: str) -> Iterator[GenerateEvent]:
        if request.request_id in self.open:
            raise RuntimeError(f"generator already open for {request.request_id}")
        self.open.add(request.request_id)
        self.calls.append(request.request_id)
        words = ("[fake answer for " + request.request_id + "] " + prompt).split(" ")
        limit = min(len(words), request.max_output_tokens)
        fail_after = self.failures.get(request.request_id)
        for i in range(limit):
            if fail_after is not None and i >= fail_after:
                yield GenerateEvent(error=ContractError(ErrorKind.GENERATION_ERROR, "injected"), finish_reason="error")
                return
            time.sleep(self.token_delay_s)
            yield GenerateEvent(text=words[i] + (" " if i < limit - 1 else ""))
        yield GenerateEvent(token_count=limit, finish_reason="length" if limit < len(words) else "stop")

    def release(self, request_id: str) -> None:
        self.open.discard(request_id)
        self.released.append(request_id)
