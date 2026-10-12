"""Deterministic keyword router. Two jobs: the fallback router in production configs, and the
router for harness tests. It is excluded from every model-quality claim. Intent detection copies
the legacy first-substring behaviour from NEXUS intent_patterns.py so known collisions are
reproduced, not hidden."""
from __future__ import annotations

import time

from .contracts import Availability, ClassifyResult, ContractError, ErrorKind, RouterStatus

INTENT_KEYWORDS = {   # order matters: legacy scans dict order, first substring wins
    "leave_time": ["leave", "departure", "when should i go", "what time should i leave", "when to leave"],
    "daily_briefing": ["briefing", "daily summary", "what's my day", "today's schedule", "morning update"],
    "commute_with_calendar": ["get to my meeting", "travel to appointment", "commute to work", "drive to office"],
    "meeting_prep": ["prepare for meeting", "meeting prep", "get ready for my meeting", "before my meeting"],
    "weather_check": ["weather", "temperature", "forecast", "rain", "snow", "cold outside", "hot outside", "umbrella"],
    "weather_commute": ["should i drive", "dress for commute", "weather for my drive", "road conditions"],
    "full_status": ["full status", "everything", "complete overview", "what's happening", "status update"],
    "build_module": ["build a module", "create a module", "build module", "new module", "create a function"],
    "manage_modules": ["list modules", "which modules", "installed modules", "manage modules", "remove module"],
}

TASK_KEYWORDS = [
    ("coding", ["grpc", "python", "function", "bug", "traceback", "exception", "code", "adapter", "refactor", "test fails", "stack trace"]),
    ("finance", ["invoice", "statement", "total", "expense", "budget", "interest", "vat", "tax"]),
    ("math", ["calculate", "compute", "solve", "equation", "percent", "sum of", "probability", "how many"]),
    ("multilingual", ["translate", "in french", "in turkish", "in german", "in spanish", "übersetze", "çevir"]),
    ("search", ["latest news", "search the web", "look up online", "current price", "who won", "stock price today"]),
    ("reasoning", ["explain", "why", "compare", "trace", "analy", "runbook", "according to", "which setting", "policy"]),
]
CAP_FOR_TASK = {"coding": "coding", "finance": "finance", "math": "math", "multilingual": "multilingual",
                "search": "search", "reasoning": "reasoning", "general": "fast_response"}
SEQUENCE_WORDS = ["then", "after that", "and also", "followed by", "next,"]


def detect_intent(text: str):
    lower = text.lower()
    for name, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return name
    return None


class KeywordRouter:
    name = "keyword"

    def classify(self, text: str, availability: Availability) -> ClassifyResult:
        t0 = time.perf_counter()
        words = text.strip().split()
        if len(words) < 3 and "?" not in text:
            return ClassifyResult(RouterStatus.ABSTAIN, reason="too short to classify", backend=self.name,
                                  router_ms=(time.perf_counter() - t0) * 1000)
        lower = text.lower()
        intent = detect_intent(text)
        task_type = "general"
        for name, keywords in TASK_KEYWORDS:
            if any(k in lower for k in keywords):
                task_type = name
                break
        caps = [CAP_FOR_TASK[task_type]]
        if any(k in lower for k in ("as json", "extract", "as a table", "fields")):
            caps.append("extraction")
        if "verify" in lower or "double-check" in lower:
            caps.append("verification")
        complexity = 0.3
        if any(s in lower for s in SEQUENCE_WORDS):
            complexity = 0.7
            if len(caps) == 1:
                caps.append("reasoning" if caps[0] != "reasoning" else "analysis")
        return ClassifyResult(RouterStatus.CLASSIFIED, task_type=task_type, capabilities=caps,
                              complexity=complexity, tool_intent=intent, backend=self.name,
                              router_ms=(time.perf_counter() - t0) * 1000)


class FailureInjectingRouter:
    """Wraps a router; per-request failure modes for replay fault injection.
    modes: {request_text_prefix_or_id: "error" | "hang" | "invalid"}; keyed by text here because
    routers never see request ids (labels and ids are not worker inputs)."""
    name = "injected"

    def __init__(self, inner, modes: dict[str, str], hang_s: float = 0.2):
        self.inner, self.modes, self.hang_s = inner, modes, hang_s
        self.name = inner.name

    def classify(self, text: str, availability: Availability) -> ClassifyResult:
        for key, mode in self.modes.items():
            if text.startswith(key):
                if mode == "error":
                    raise RuntimeError("injected backend failure")
                if mode == "hang":
                    time.sleep(self.hang_s)
                if mode == "invalid":
                    return ClassifyResult(RouterStatus.CLASSIFIED, task_type="poetry", backend=self.name)
        return self.inner.classify(text, availability)
