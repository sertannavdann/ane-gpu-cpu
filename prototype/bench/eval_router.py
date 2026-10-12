"""Offline router evaluation on the frozen replay set: no generation, no timing claims.
Usage: python -m prototype.bench.eval_router [keyword|tfidf]"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from ..contracts import Availability, RouterStatus
from ..policy import decide
from ..router_fake import KeywordRouter

ROOT = Path(__file__).resolve().parents[2]
AV = Availability(frozenset({"get_user_context", "get_commute_time", "build_module", "validate_module", "list_modules"}),
                  frozenset({"heavy", "standard"}))


def evaluate(router, rows):
    task_ok = disp_ok = intent_ok = abstain = 0
    confusion = Counter()
    for r in rows:
        e = r["expected"]
        res = router.classify(r["input"], AV)
        if res.status is RouterStatus.ABSTAIN:
            abstain += 1
        d = decide(res, AV, r.get("context", {}))
        task_ok += res.status is RouterStatus.CLASSIFIED and res.task_type == e["task_type"]
        intent_ok += res.tool_intent == e["tool_intent"]
        ok = d.action.value == e["action"]
        disp_ok += ok
        if not ok:
            confusion[(e["action"], d.action.value)] += 1
    n = len(rows)
    return {"n": n, "task_type_acc": task_ok / n, "tool_intent_acc": intent_ok / n,
            "disposition_acc": disp_ok / n, "abstain_rate": abstain / n,
            "disposition_confusion(expected->got)": {f"{a}->{b}": c for (a, b), c in confusion.most_common()}}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
    rows = [json.loads(l) for l in (ROOT / "prototype/bench/workload/replay.jsonl").read_text().splitlines() if l.strip()]
    if which == "tfidf":
        from ..router_tfidf import TfidfRouter
        router = TfidfRouter.from_training_file(ROOT / "prototype/bench/workload/train.jsonl")
        print("calibration:", json.dumps(router.calibration))
    else:
        router = KeywordRouter()
    print(json.dumps(evaluate(router, rows), indent=2))


if __name__ == "__main__":
    main()
