"""The outcome brief's acceptance example, run on fakes. Proves the workflow shape, not speed."""
from prototype.admission import Scheduler
from prototype.bench import metrics
from prototype.generator_fake import ScriptedGenerator
from prototype.router_fake import KeywordRouter
from tests.conftest import bg, fg


def test_acceptance_example(availability):
    gen = ScriptedGenerator(token_delay_s=0.01)
    s = Scheduler(KeywordRouter(), KeywordRouter(), gen, availability, priority=True)
    background = bg("ast-02", "Prepare tomorrow's briefing from my calendar and notes.", max_output_tokens=100)
    foreground = fg("ast-01", "What is the weather in Fixture City tomorrow?")
    outs = {o.request_id: o for o in s.run([(0.0, background), (0.2, foreground)])}
    s.close()
    f, b = outs["ast-01"], outs["ast-02"]
    assert f.tool_intent == "weather_check" and f.action == "fixture_tools" and f.required_tools == ["get_user_context"]
    assert f.task_type == "general"
    assert "light rain" in f.output                      # cites the fixture forecast
    assert b.state == "done" and b.error is None and b.output_tokens > 0 and "Design review" in b.output
    assert b.timestamps["generation_start"] < f.timestamps["arrival"] < b.timestamps["finish"]
    assert metrics.overlap_count(s.trace, list(outs.values())) == 1
    ttft_ms = (f.timestamps["first_token"] - f.timestamps["arrival"]) * 1000
    assert ttft_ms < 500
