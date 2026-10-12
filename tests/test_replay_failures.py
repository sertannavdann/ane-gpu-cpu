from prototype.admission import Scheduler
from prototype.generator_fake import ScriptedGenerator
from prototype.router_fake import FailureInjectingRouter, KeywordRouter
from tests.conftest import fg


def run_one(router, availability, req, generator=None, **kw):
    gen = generator or ScriptedGenerator(token_delay_s=0.001)
    s = Scheduler(router, KeywordRouter(), gen, availability, **kw)
    out = s.run([(0.0, req)])[0]
    s.close()
    return out


def test_router_exception_falls_back_once(availability):
    r = FailureInjectingRouter(KeywordRouter(), {"What": "error"})
    o = run_one(r, availability, fg("a", "What is the weather tomorrow?"))
    assert o.state == "done" and o.fallback_attempts == 1 and o.router_backend == "fallback:keyword"


def test_router_timeout_falls_back_within_deadline(availability):
    r = FailureInjectingRouter(KeywordRouter(), {"What": "hang"}, hang_s=0.3)
    o = run_one(r, availability, fg("a", "What is the weather tomorrow?"), router_timeout_ms=50)
    assert o.state == "done" and o.fallback_attempts == 1
    assert o.timestamps["router_end"] - o.timestamps["router_start"] < 0.25


def test_invalid_schema_is_never_a_label(availability):
    r = FailureInjectingRouter(KeywordRouter(), {"What": "invalid"})
    o = run_one(r, availability, fg("a", "What is the weather tomorrow?"))
    assert o.fallback_attempts == 1 and o.task_type == "general"


def test_fallback_failure_is_typed_not_silent(availability):
    class Broken:
        name = "broken"
        def classify(self, text, availability):
            raise RuntimeError("down")
    s = Scheduler(Broken(), Broken(), ScriptedGenerator(0.001), availability)
    o = s.run([(0.0, fg("a", "What is the weather tomorrow?"))])[0]
    s.close()
    assert o.state == "failed" and o.error.startswith("router_error") and o.action is None


def test_generation_error_is_typed_and_not_replayed(availability):
    gen = ScriptedGenerator(0.001, failures={"a": 3})
    o = run_one(KeywordRouter(), availability, fg("a", "Explain why the request went to the heavy tier."), generator=gen)
    assert o.state == "failed" and o.error.startswith("generation_error")
    assert gen.calls == ["a"] and gen.released == ["a"]


def test_clarify_is_not_converted_to_generation(availability):
    gen = ScriptedGenerator(0.001)
    o = run_one(KeywordRouter(), availability, fg("a", "Fix that."), generator=gen)
    assert o.action == "clarify" and gen.calls == [] and o.output_tokens == 0
