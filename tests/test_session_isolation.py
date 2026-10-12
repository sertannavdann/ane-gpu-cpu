from prototype.admission import Scheduler
from prototype.generator_fake import ScriptedGenerator
from prototype.router_fake import KeywordRouter
from tests.conftest import bg, fg


def test_same_session_requests_get_separate_generation_state(availability):
    gen = ScriptedGenerator(0.001)
    s = Scheduler(KeywordRouter(), KeywordRouter(), gen, availability)
    outs = {o.request_id: o for o in s.run([
        (0.0, bg("a", "Prepare tomorrow's briefing from my calendar and notes.", session="s1")),
        (0.0, fg("b", "What is the weather tomorrow?", session="s1")),
    ])}
    s.close()
    assert sorted(gen.released) == ["a", "b"] and not gen.open
    assert "[fake answer for a]" in outs["a"].output and "[fake answer for a]" not in outs["b"].output
    assert "[fake answer for b]" in outs["b"].output and "[fake answer for b]" not in outs["a"].output
