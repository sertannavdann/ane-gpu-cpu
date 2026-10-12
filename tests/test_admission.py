import time

from prototype.admission import Scheduler
from prototype.contracts import Lifecycle
from tests.conftest import bg, fg


def outcomes_by_id(outs):
    return {o.request_id: o for o in outs}


def test_bounded_queue_rejects_with_typed_error(router, generator, availability):
    s = Scheduler(router, router, generator, availability, fg_capacity=1, bg_capacity=1)
    assert s.submit(fg("a", "What is the weather tomorrow?"))
    assert not s.submit(fg("b", "What is the weather tomorrow?"))
    o = outcomes_by_id(s.outcomes)["b"]
    assert o.state == "rejected" and o.error.startswith("queue_full")
    s.close()


def test_lifecycle_timestamps_are_ordered(router, generator, availability):
    s = Scheduler(router, router, generator, availability)
    outs = s.run([(0.0, fg("a", "What is the weather in Fixture City tomorrow?"))])
    ts = outs[0].timestamps
    order = ["arrival", "admitted", "router_start", "router_end", "generation_start", "first_token", "finish"]
    assert [k for k in order if k in ts] == order
    assert all(ts[a] <= ts[b] for a, b in zip(order, order[1:]))
    assert outs[0].state == "done" and outs[0].action == "fixture_tools"
    s.close()


def test_priority_serves_foreground_before_background_finishes(router, availability):
    from prototype.generator_fake import ScriptedGenerator
    gen = ScriptedGenerator(token_delay_s=0.005)
    s = Scheduler(router, router, gen, availability, priority=True)
    long_bg = bg("bg", "Prepare tomorrow's briefing from my calendar and notes. " * 8, max_output_tokens=120)
    outs = outcomes_by_id(s.run([(0.0, long_bg), (0.05, fg("fg", "What is the weather tomorrow?"))]))
    assert outs["fg"].state == "done" and outs["bg"].state == "done"
    assert outs["fg"].timestamps["finish"] < outs["bg"].timestamps["finish"]
    assert outs["fg"].timestamps["first_token"] > outs["bg"].timestamps["generation_start"]
    assert "[fake answer for bg]" in outs["bg"].output and "[fake answer for fg]" in outs["fg"].output
    s.close()


def test_without_priority_foreground_waits_for_background(router, availability):
    from prototype.generator_fake import ScriptedGenerator
    gen = ScriptedGenerator(token_delay_s=0.005)
    s = Scheduler(router, router, gen, availability, priority=False)
    long_bg = bg("bg", "Prepare tomorrow's briefing from my calendar and notes. " * 8, max_output_tokens=120)
    outs = outcomes_by_id(s.run([(0.0, long_bg), (0.05, fg("fg", "What is the weather tomorrow?"))]))
    assert outs["bg"].timestamps["finish"] <= outs["fg"].timestamps["admitted"]
    s.close()


def test_deadline_expiry_is_typed_failure_and_releases(router, availability):
    from prototype.generator_fake import ScriptedGenerator
    gen = ScriptedGenerator(token_delay_s=0.01)
    s = Scheduler(router, router, gen, availability)
    outs = s.run([(0.0, fg("a", "Explain why the request went to the heavy tier in detail. " * 10,
                            max_output_tokens=100, deadline_ms=60))])
    assert outs[0].state == "failed" and outs[0].error.startswith("deadline_exceeded")
    assert gen.released == ["a"] and not gen.open
    s.close()


def test_cancel_queued_and_cancel_active_background(router, availability):
    from prototype.generator_fake import ScriptedGenerator
    gen = ScriptedGenerator(token_delay_s=0.005)
    s = Scheduler(router, router, gen, availability)
    a = bg("a", "Prepare tomorrow's briefing from my calendar and notes. " * 8, max_output_tokens=200)
    b = bg("b", "Prepare tomorrow's briefing again please.")
    s.submit(a); s.submit(b)
    s.cancel("b")                       # queued: removed immediately
    assert outcomes_by_id(s.outcomes)["b"].state == "cancelled" and len(s.bg) == 1
    s._start_background(s.bg.popleft())
    for _ in range(3):
        s._step_background()
    s.cancel("a")                       # active: cancelled at the next token boundary
    s._step_background()
    o = outcomes_by_id(s.outcomes)["a"]
    assert o.state == "cancelled" and s.active_bg is None and gen.released == ["a"] and not gen.open
    s.close()


def test_terminal_actions_do_not_generate(router, generator, availability):
    s = Scheduler(router, router, generator, availability)
    outs = outcomes_by_id(s.run([(0.0, fg("c", "When should I leave?")),
                                 (0.0, fg("u", "Search the web for the latest news on MLX releases.")),
                                 (0.0, fg("d", "Calculate 15 percent of 84 and then explain why percentages are multiplicative."))]))
    assert outs["c"].action == "clarify" and outs["u"].action == "unavailable" and outs["d"].action == "needs_decomposition"
    assert generator.calls == [] and all(o.state == "done" for o in outs.values())
    s.close()
