"""Admission, lifecycle and scheduling. Owns all mutable request state.

One cooperative loop in one process. Background generation is advanced one token at a time;
between tokens the loop admits arrivals and, when priority is enabled, serves a waiting
foreground request to completion before resuming the background generator. That token yield is
the only preemption boundary and it is the one the generator backend supports.
"""
from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, Iterator, Optional, Protocol

from . import fixtures
from .contracts import (
    Action, Availability, ClassifyResult, ContractError, Decision, ErrorKind, GenerateEvent,
    Lifecycle, Outcome, Request, RequestClass,
)
from .policy import decide


class Router(Protocol):
    name: str
    def classify(self, text: str, availability: Availability) -> ClassifyResult: ...


class Generator(Protocol):
    name: str
    def stream(self, request: Request, prompt: str) -> Iterator[GenerateEvent]: ...
    def release(self, request_id: str) -> None: ...


TERMINAL_TEXT = {
    Action.CLARIFY: "I need one more detail before I can help: {reason}",
    Action.UNAVAILABLE: "That is not available in this configuration: {reason}",
    Action.NEEDS_DECOMPOSITION: "This request needs to be split into steps, which is not supported yet.",
}


class _Active:
    """A background generation parked between tokens."""
    def __init__(self, request: Request, decision: Decision, result: ClassifyResult,
                 tokens: Iterator[GenerateEvent]):
        self.request, self.decision, self.result, self.tokens = request, decision, result, tokens
        self.parts: list[str] = []
        self.count = 0


class Scheduler:
    def __init__(self, router: Router, fallback_router: Router, generator: Generator,
                 availability: Availability, *, fg_capacity: int = 4, bg_capacity: int = 4,
                 priority: bool = True, router_timeout_ms: int = 2_000,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.router, self.fallback_router, self.generator = router, fallback_router, generator
        self.availability = availability
        self.fg_capacity, self.bg_capacity = fg_capacity, bg_capacity
        self.priority = priority
        self.router_timeout_ms = router_timeout_ms
        self.clock, self.sleep = clock, sleep
        self.fg: deque[Request] = deque()
        self.bg: deque[Request] = deque()
        self.active_bg: Optional[_Active] = None
        self.outcomes: list[Outcome] = []
        self._cancel_requested: set[str] = set()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="router")
        self.trace: list[tuple[float, str, str]] = []   # (t, request_id, event) for overlap checks

    # ---- admission -------------------------------------------------------------------------
    def submit(self, req: Request) -> bool:
        req.timestamps["arrival"] = self.clock()
        queue, cap = ((self.fg, self.fg_capacity) if req.request_class is RequestClass.FOREGROUND
                      else (self.bg, self.bg_capacity))
        if len(queue) >= cap:
            self._finish(req, Lifecycle.REJECTED, ContractError(ErrorKind.QUEUE_FULL, req.request_class.value))
            return False
        queue.append(req)
        self._event(req, "queued")
        return True

    def cancel(self, request_id: str) -> None:
        for queue in (self.fg, self.bg):
            for req in list(queue):
                if req.request_id == request_id:
                    queue.remove(req)
                    self._finish(req, Lifecycle.CANCELLED, ContractError(ErrorKind.CANCELLED, "while queued"))
                    return
        self._cancel_requested.add(request_id)

    # ---- main loop -------------------------------------------------------------------------
    def run(self, arrivals: list[tuple[float, Request]]) -> list[Outcome]:
        pending = sorted(arrivals, key=lambda a: a[0])
        t0 = self.clock()
        while pending or self.fg or self.bg or self.active_bg:
            now = self.clock() - t0
            while pending and pending[0][0] <= now:
                self.submit(pending.pop(0)[1])
            if self.fg and (self.priority or self.active_bg is None):
                self._serve_foreground(self.fg.popleft())
            elif self.active_bg is not None:
                self._step_background()
            elif self.bg:
                self._start_background(self.bg.popleft())
            elif pending:
                self.sleep(max(0.0, min(pending[0][0] - (self.clock() - t0), 0.005)))
        return self.outcomes

    # ---- per-request stages ----------------------------------------------------------------
    def _prepare(self, req: Request) -> Optional[tuple[ClassifyResult, Decision, str]]:
        """Admit, route, decide, run tools. Returns None when the request reached a terminal state."""
        req.state = Lifecycle.ADMITTED
        req.timestamps["admitted"] = self.clock()
        self._event(req, "admitted")
        if self._expired(req):
            return None
        req.state = Lifecycle.ROUTING
        req.timestamps["router_start"] = self.clock()
        try:
            result = self._route(req)
        except ContractError as e:
            req.timestamps["router_end"] = self.clock()
            self._finish(req, Lifecycle.FAILED, e)
            return None
        req.timestamps["router_end"] = self.clock()
        decision = decide(result, self.availability, req.context)
        if decision.action in TERMINAL_TEXT:
            text = TERMINAL_TEXT[decision.action].format(reason=decision.reason)
            req.timestamps["first_token"] = self.clock()   # the user sees this text now
            self._event(req, "first_token")
            self._record(req, Lifecycle.DONE, decision, result, text, 0)
            return None
        prompt = req.text
        if decision.action is Action.FIXTURE_TOOLS:
            tool_results = fixtures.run_tools(decision.required_tools, req.context)
            prompt = f"{req.text}\n\nTool results:\n{json.dumps(tool_results, sort_keys=True)}"
        if self._expired(req):
            return None
        req.state = Lifecycle.GENERATING
        req.timestamps["generation_start"] = self.clock()
        self._event(req, "generation_start")
        return result, decision, prompt

    def _route(self, req: Request) -> ClassifyResult:
        """Primary router with timeout; on any typed failure, exactly one fallback attempt."""
        remaining = self._remaining_s(req)
        timeout = min(self.router_timeout_ms / 1000.0, remaining)
        try:
            result = self._call_router(self.router, req.text, timeout).validate()
            req.router_backend = self.router.name
            return result
        except ContractError as first:
            req.fallback_attempts = 1
            self._event(req, f"router_fallback:{first.kind.value}")
            remaining = self._remaining_s(req)
            if remaining <= 0:
                raise ContractError(ErrorKind.DEADLINE, "no time left for fallback") from first
            result = self._call_router(self.fallback_router, req.text,
                                       min(self.router_timeout_ms / 1000.0, remaining)).validate()
            req.router_backend = f"fallback:{self.fallback_router.name}"
            return result

    def _call_router(self, router: Router, text: str, timeout_s: float) -> ClassifyResult:
        future = self._pool.submit(router.classify, text, self.availability)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout:
            # The stuck worker thread cannot be killed; abandon the pool so the fallback attempt
            # is not queued behind it. The abandoned thread ends when the backend call returns.
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="router")
            raise ContractError(ErrorKind.ROUTER_TIMEOUT, f"{router.name} > {timeout_s*1000:.0f} ms")
        except ContractError:
            raise
        except Exception as e:  # any backend exception is a typed router error, never a label
            raise ContractError(ErrorKind.ROUTER_ERROR, f"{router.name}: {e!r}")

    def _serve_foreground(self, req: Request) -> None:
        prepared = self._prepare(req)
        if prepared is None:
            return
        result, decision, prompt = prepared
        parts, count = [], 0
        tokens = self.generator.stream(req, prompt)
        try:
            for ev in tokens:
                stop = self._consume(req, ev, parts)
                count = ev.token_count if ev.token_count is not None else count + (1 if ev.text else 0)
                if stop or self._expired(req) or self._cancel_check(req):
                    break
        finally:
            self._close(tokens, req)
        if req.state is Lifecycle.GENERATING:   # stream ended without a typed failure
            self._record(req, Lifecycle.DONE, decision, result, "".join(parts), count)

    def _start_background(self, req: Request) -> None:
        prepared = self._prepare(req)
        if prepared is None:
            return
        result, decision, prompt = prepared
        self.active_bg = _Active(req, decision, result, self.generator.stream(req, prompt))

    def _step_background(self) -> None:
        a = self.active_bg
        assert a is not None
        req = a.request
        if self._expired(req) or self._cancel_check(req):
            self._close(a.tokens, req)
            self.active_bg = None
            return
        try:
            ev = next(a.tokens)
        except StopIteration:
            self._close(a.tokens, req)
            self.active_bg = None
            self._record(req, Lifecycle.DONE, a.decision, a.result, "".join(a.parts), a.count)
            return
        stop = self._consume(req, ev, a.parts)
        a.count = ev.token_count if ev.token_count is not None else a.count + (1 if ev.text else 0)
        if stop:
            self._close(a.tokens, req)
            self.active_bg = None
            if req.state is Lifecycle.GENERATING:
                self._record(req, Lifecycle.DONE, a.decision, a.result, "".join(a.parts), a.count)

    # ---- helpers ---------------------------------------------------------------------------
    def _consume(self, req: Request, ev: GenerateEvent, parts: list[str]) -> bool:
        """Apply one event. Returns True when the stream is finished."""
        if ev.error is not None:
            self._finish(req, Lifecycle.FAILED, ev.error)
            return True
        if ev.text:
            if "first_token" not in req.timestamps:
                req.timestamps["first_token"] = self.clock()
                self._event(req, "first_token")
            parts.append(ev.text)
        return ev.finish_reason is not None

    def _close(self, tokens: Iterator[GenerateEvent], req: Request) -> None:
        close = getattr(tokens, "close", None)
        if close:
            close()
        self.generator.release(req.request_id)

    def _remaining_s(self, req: Request) -> float:
        return req.deadline_ms / 1000.0 - (self.clock() - req.timestamps["arrival"])

    def _expired(self, req: Request) -> bool:
        if self._remaining_s(req) <= 0:
            self._finish(req, Lifecycle.FAILED, ContractError(ErrorKind.DEADLINE, f"{req.deadline_ms} ms"))
            return True
        return False

    def _cancel_check(self, req: Request) -> bool:
        if req.request_id in self._cancel_requested:
            self._cancel_requested.discard(req.request_id)
            self._finish(req, Lifecycle.CANCELLED, ContractError(ErrorKind.CANCELLED, "at token boundary"))
            return True
        return False

    def _finish(self, req: Request, state: Lifecycle, err: Optional[ContractError]) -> None:
        req.error = err
        self._record(req, state, None, None, "", 0)

    def _record(self, req: Request, state: Lifecycle, decision: Optional[Decision],
                result: Optional[ClassifyResult], output: str, tokens: int) -> None:
        req.state = state
        req.timestamps["finish"] = self.clock()
        self._event(req, state.value)
        self.outcomes.append(Outcome(
            request_id=req.request_id, request_class=req.request_class.value, state=state.value,
            action=decision.action.value if decision else None,
            task_type=result.task_type if result else None,
            capabilities=list(result.capabilities) if result else [],
            tool_intent=result.tool_intent if result else None,
            logical_tier=decision.logical_tier if decision else None,
            required_tools=list(decision.required_tools) if decision else [],
            output=output, output_tokens=tokens, fallback_attempts=req.fallback_attempts,
            router_backend=req.router_backend,
            error=f"{req.error.kind.value}: {req.error.detail}" if req.error else None,
            timestamps=dict(req.timestamps),
        ))

    def _event(self, req: Request, name: str) -> None:
        self.trace.append((self.clock(), req.request_id, name))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
