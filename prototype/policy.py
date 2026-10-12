"""Capability -> action. Pure functions; no I/O, no model calls.

Order of checks is the product policy: a required-tool decision can never be bypassed by the
router's choice of tier, and unavailable capabilities are reported, not silently substituted."""
from __future__ import annotations

from .contracts import (
    Action, Availability, CAPABILITY_TIER, ClassifyResult, Decision, INTENTS_NEEDING_DESTINATION,
    RouterStatus, TOOL_INTENTS,
)

COMPLEXITY_THRESHOLD_DIRECT = 0.5   # legacy analyze_and_route rule, kept as a reference input


def required_tier(capabilities: list[str]) -> str:
    """Highest logical tier among capabilities. Unlike legacy get_required_tier, 'external' is
    reported separately and never collapsed to 'standard'; see plan source-audit row 5."""
    tiers = {CAPABILITY_TIER.get(c, "standard") for c in capabilities}
    if "external" in tiers:
        return "external"
    if "heavy" in tiers:
        return "heavy"
    return "standard"


def decide(result: ClassifyResult, availability: Availability, context: dict) -> Decision:
    if result.status is RouterStatus.ABSTAIN:
        return Decision(Action.CLARIFY, reason=f"router abstained: {result.reason}")

    if result.tool_intent:
        tools = TOOL_INTENTS[result.tool_intent]
        missing = [t for t in tools if t not in availability.tools]
        if missing:
            return Decision(Action.UNAVAILABLE, required_tools=tools,
                            reason=f"tools unavailable: {missing}")
        if result.tool_intent in INTENTS_NEEDING_DESTINATION and not context.get("destination"):
            return Decision(Action.CLARIFY, required_tools=tools, reason="destination missing")
        return Decision(Action.FIXTURE_TOOLS, logical_tier="standard", required_tools=tools,
                        reason=f"intent {result.tool_intent}")

    tier = required_tier(result.capabilities)
    if tier == "external":
        if "external" not in availability.tiers:
            return Decision(Action.UNAVAILABLE, logical_tier="external",
                            reason="search requested; no external provider configured")
        tier = "standard"
    if tier not in availability.tiers:
        return Decision(Action.UNAVAILABLE, logical_tier=tier, reason=f"tier {tier} not configured")

    if result.complexity >= COMPLEXITY_THRESHOLD_DIRECT and len(result.capabilities) > 1:
        return Decision(Action.NEEDS_DECOMPOSITION, logical_tier=tier,
                        reason="multi-capability, high complexity; decomposition is out of slice")

    return Decision(Action.ANSWER, logical_tier=tier, reason="direct answer")
