"""Typed contracts for the prototype. Field semantics follow NEXUS classification
(task_type, capabilities, complexity) so a future adapter is a pure translation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

TASK_TYPES = ("coding", "reasoning", "math", "finance", "multilingual", "search", "general")

# Snapshot of the 13 categories in NEXUS config/routing_config.json (inspected 2026-09-18).
CAPABILITIES = (
    "coding", "reasoning", "analysis", "verification", "deep_research", "finance",
    "multilingual", "math", "fast_response", "routing", "classification", "extraction", "search",
)

# Logical tier per capability. Mirrors NEXUS capability_map intent: heavy for demanding work,
# standard otherwise, search needs an external provider. Prototype policy, not production truth.
CAPABILITY_TIER = {
    "coding": "heavy", "reasoning": "heavy", "analysis": "heavy", "verification": "heavy",
    "deep_research": "heavy", "finance": "heavy", "math": "heavy",
    "multilingual": "standard", "fast_response": "standard", "routing": "standard",
    "classification": "standard", "extraction": "standard", "search": "external",
}

# The nine tool intents from NEXUS intent_patterns.py with their required tools.
TOOL_INTENTS = {
    "leave_time": ["get_user_context", "get_commute_time"],
    "daily_briefing": ["get_user_context"],
    "commute_with_calendar": ["get_user_context", "get_commute_time"],
    "meeting_prep": ["get_user_context"],
    "weather_check": ["get_user_context"],
    "weather_commute": ["get_user_context", "get_commute_time"],
    "full_status": ["get_user_context"],
    "build_module": ["build_module", "validate_module"],
    "manage_modules": ["list_modules"],
}
# Intents whose fixture tool needs a destination in context; otherwise the correct answer is clarify.
INTENTS_NEEDING_DESTINATION = {"leave_time", "commute_with_calendar", "weather_commute"}


class RequestClass(str, Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class Lifecycle(str, Enum):
    QUEUED = "queued"
    ADMITTED = "admitted"
    ROUTING = "routing"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RouterStatus(str, Enum):
    CLASSIFIED = "classified"
    ABSTAIN = "abstain"
    ERROR = "error"


class Action(str, Enum):
    ANSWER = "answer"
    FIXTURE_TOOLS = "fixture_tools"
    CLARIFY = "clarify"
    NEEDS_DECOMPOSITION = "needs_decomposition"
    UNAVAILABLE = "unavailable"


class ErrorKind(str, Enum):
    ROUTER_TIMEOUT = "router_timeout"
    ROUTER_ERROR = "router_error"
    ROUTER_INVALID = "router_invalid"
    GENERATION_ERROR = "generation_error"
    DEADLINE = "deadline_exceeded"
    QUEUE_FULL = "queue_full"
    CANCELLED = "cancelled"


class ContractError(Exception):
    def __init__(self, kind: ErrorKind, detail: str = ""):
        super().__init__(f"{kind.value}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class Availability:
    """What this deployment can actually do. Never inferred from the request."""
    tools: frozenset[str]
    tiers: frozenset[str]          # e.g. {"heavy", "standard"}; "external" absent means no provider
    fixtures: frozenset[str] = frozenset()


@dataclass
class Request:
    request_id: str
    session_id: str
    text: str
    request_class: RequestClass
    context: dict = field(default_factory=dict)      # fixture context supplied with the request
    max_output_tokens: int = 128
    deadline_ms: int = 30_000
    # lifecycle bookkeeping, owned by the scheduler
    state: Lifecycle = Lifecycle.QUEUED
    timestamps: dict = field(default_factory=dict)   # name -> monotonic seconds
    error: Optional[ContractError] = None
    fallback_attempts: int = 0
    router_backend: str = ""


@dataclass
class ClassifyResult:
    status: RouterStatus
    task_type: str = "general"
    capabilities: list[str] = field(default_factory=list)
    complexity: float = 0.0
    tool_intent: Optional[str] = None
    reason: str = ""
    backend: str = ""
    router_ms: float = 0.0

    def validate(self) -> "ClassifyResult":
        """Reject anything outside the declared vocabulary. Invalid output is a typed failure."""
        if self.status is RouterStatus.ERROR:
            raise ContractError(ErrorKind.ROUTER_ERROR, self.reason)
        if self.task_type not in TASK_TYPES:
            raise ContractError(ErrorKind.ROUTER_INVALID, f"task_type={self.task_type!r}")
        bad = [c for c in self.capabilities if c not in CAPABILITIES]
        if bad:
            raise ContractError(ErrorKind.ROUTER_INVALID, f"capabilities={bad}")
        if not (isinstance(self.complexity, (int, float)) and 0.0 <= self.complexity <= 1.0):
            raise ContractError(ErrorKind.ROUTER_INVALID, f"complexity={self.complexity!r}")
        if self.tool_intent is not None and self.tool_intent not in TOOL_INTENTS:
            raise ContractError(ErrorKind.ROUTER_INVALID, f"tool_intent={self.tool_intent!r}")
        return self


@dataclass
class Decision:
    action: Action
    logical_tier: str = "standard"
    required_tools: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class GenerateEvent:
    """One streamed token, or the terminal event. Text is not a token count."""
    text: str = ""
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None     # "stop" | "length" | "error" | "cancelled"
    error: Optional[ContractError] = None


@dataclass
class Outcome:
    request_id: str
    request_class: str
    state: str
    action: Optional[str]
    task_type: Optional[str]
    capabilities: list[str]
    tool_intent: Optional[str]
    logical_tier: Optional[str]
    required_tools: list[str]
    output: str
    output_tokens: int
    fallback_attempts: int
    router_backend: str
    error: Optional[str]
    timestamps: dict
