import json
from collections import Counter
from pathlib import Path

from prototype.contracts import CAPABILITIES, TASK_TYPES, TOOL_INTENTS

W = Path(__file__).resolve().parents[1] / "prototype" / "bench" / "workload"


def rows(name):
    return [json.loads(l) for l in (W / name).read_text().splitlines() if l.strip()]


def test_replay_has_60_rows_with_valid_vocabulary():
    r = rows("replay.jsonl")
    assert len(r) == 60 and len({x["id"] for x in r}) == 60
    for x in r:
        e = x["expected"]
        assert e["task_type"] in TASK_TYPES
        assert set(e["capabilities"]) <= set(CAPABILITIES)
        assert e["tool_intent"] is None or e["tool_intent"] in TOOL_INTENTS
        if e["tool_intent"]:
            assert e["required_tools"] == TOOL_INTENTS[e["tool_intent"]]
        assert x["class"] in ("foreground", "background")


def test_bucket_counts_match_plan():
    c = Counter(x["bucket"] for x in rows("replay.jsonl"))
    assert c == {"coding": 9, "reasoning": 8, "finance": 9, "multilingual": 6, "assistant": 10,
                 "ambiguous": 8, "mixed": 6, "unavailable": 4}


def test_no_family_crosses_train_and_replay():
    assert not {x["family_id"] for x in rows("train.jsonl")} & {x["family_id"] for x in rows("replay.jsonl")}
    assert not {x["input"] for x in rows("train.jsonl")} & {x["input"] for x in rows("replay.jsonl")}


def test_all_capabilities_and_intents_covered():
    r = rows("replay.jsonl")
    caps = {c for x in r for c in x["expected"]["capabilities"]}
    intents = {x["expected"]["tool_intent"] for x in r} - {None}
    assert caps == set(CAPABILITIES), set(CAPABILITIES) - caps
    assert intents == set(TOOL_INTENTS), set(TOOL_INTENTS) - intents
