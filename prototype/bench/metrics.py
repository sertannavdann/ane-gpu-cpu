"""Outcome accounting. Failures stay in every denominator."""
from __future__ import annotations

from collections import Counter


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _ms(ts: dict, a: str, b: str) -> float | None:
    return (ts[b] - ts[a]) * 1000 if a in ts and b in ts else None


def summarize(outcomes, rows_by_id: dict) -> dict:
    """outcomes: list[Outcome]; rows_by_id: workload rows with expected labels."""
    summary: dict = {"n": len(outcomes), "by_class": {}, "states": dict(Counter(o.state for o in outcomes)),
                     "errors": dict(Counter(o.error.split(":")[0] for o in outcomes if o.error)),
                     "fallbacks": sum(o.fallback_attempts for o in outcomes)}
    for cls in ("foreground", "background"):
        group = [o for o in outcomes if o.request_class == cls]
        if not group:
            continue
        ttft = [_ms(o.timestamps, "arrival", "first_token") for o in group]
        ttft_ok = [t for t in ttft if t is not None]
        queue = [_ms(o.timestamps, "arrival", "admitted") for o in group if "admitted" in o.timestamps]
        router = [_ms(o.timestamps, "router_start", "router_end") for o in group if "router_end" in o.timestamps]
        total = [_ms(o.timestamps, "arrival", "finish") for o in group]
        done = [o for o in group if o.state == "done"]
        disp_ok = rubric_ok = eligible = 0
        for o in group:
            row = rows_by_id.get(o.request_id)
            if not row:
                continue
            eligible += 1
            if o.action == row["expected"]["action"]:
                disp_ok += 1
            if row["expected"].get("rubric") and o.output and row["expected"]["rubric"].lower() in o.output.lower():
                rubric_ok += 1
        summary["by_class"][cls] = {
            "offered": len(group), "completed": len(done),
            "completion_rate": len(done) / len(group),
            "ttft_population": len(group), "ttft_measured": len(ttft_ok),
            "ttft_misses": len(group) - len(ttft_ok),
            "ttft_p50_ms": percentile(ttft_ok, 0.5), "ttft_p95_ms": percentile(ttft_ok, 0.95),
            "queue_p50_ms": percentile(queue, 0.5), "queue_p95_ms": percentile(queue, 0.95),
            "router_p50_ms": percentile(router, 0.5), "router_p95_ms": percentile(router, 0.95),
            "total_p50_ms": percentile(total, 0.5), "total_p95_ms": percentile(total, 0.95),
            "disposition_accuracy": (disp_ok / eligible) if eligible else None,
            "rubric_pass": rubric_ok, "eligible": eligible,
            "output_tokens": sum(o.output_tokens for o in group),
        }
    return summary


def overlap_count(trace: list[tuple[float, str, str]], outcomes) -> int:
    """Number of foreground requests whose admitted..finish interval overlaps a background
    generation interval. Proves the concurrent condition actually occurred."""
    bg = [(o.timestamps.get("generation_start"), o.timestamps.get("finish"))
          for o in outcomes if o.request_class == "background" and "generation_start" in o.timestamps]
    n = 0
    for o in outcomes:
        if o.request_class != "foreground" or "admitted" not in o.timestamps:
            continue
        a, b = o.timestamps["arrival"], o.timestamps["finish"]
        if any(s is not None and e is not None and s < b and a < e for s, e in bg):
            n += 1
    return n
