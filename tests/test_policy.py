from prototype.contracts import Action, Availability, ClassifyResult, RouterStatus
from prototype.policy import decide, required_tier

AV = Availability(frozenset({"get_user_context", "get_commute_time", "list_modules"}), frozenset({"heavy", "standard"}))


def ok(**kw):
    return ClassifyResult(RouterStatus.CLASSIFIED, **kw)


def test_abstain_becomes_clarify():
    d = decide(ClassifyResult(RouterStatus.ABSTAIN, reason="short"), AV, {})
    assert d.action is Action.CLARIFY


def test_intent_needing_destination_without_context_clarifies():
    d = decide(ok(task_type="general", capabilities=["fast_response"], tool_intent="leave_time"), AV, {})
    assert d.action is Action.CLARIFY and d.required_tools == ["get_user_context", "get_commute_time"]


def test_intent_with_destination_runs_tools():
    d = decide(ok(task_type="general", capabilities=["fast_response"], tool_intent="leave_time"), AV, {"destination": "Office"})
    assert d.action is Action.FIXTURE_TOOLS


def test_missing_tool_is_unavailable_not_answer():
    d = decide(ok(task_type="coding", capabilities=["coding"], tool_intent="build_module"), AV, {})
    assert d.action is Action.UNAVAILABLE and "build_module" in d.reason


def test_search_without_provider_is_unavailable():
    d = decide(ok(task_type="search", capabilities=["search"]), AV, {})
    assert d.action is Action.UNAVAILABLE and d.logical_tier == "external"


def test_external_not_collapsed_to_standard():
    assert required_tier(["search"]) == "external"
    assert required_tier(["search", "coding"]) == "external"
    assert required_tier(["fast_response"]) == "standard"
    assert required_tier(["coding", "fast_response"]) == "heavy"


def test_missing_tier_is_unavailable():
    only_std = Availability(frozenset(), frozenset({"standard"}))
    d = decide(ok(task_type="coding", capabilities=["coding"]), only_std, {})
    assert d.action is Action.UNAVAILABLE


def test_multi_capability_high_complexity_needs_decomposition():
    d = decide(ok(task_type="math", capabilities=["math", "reasoning"], complexity=0.7), AV, {})
    assert d.action is Action.NEEDS_DECOMPOSITION


def test_single_capability_answers_on_tier():
    d = decide(ok(task_type="math", capabilities=["math"], complexity=0.7), AV, {})
    assert d.action is Action.ANSWER and d.logical_tier == "heavy"
