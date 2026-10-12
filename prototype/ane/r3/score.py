"""Scorer for the frozen extraction evaluation (v2 rule; see extraction_eval.json 'scoring')."""
from __future__ import annotations
import json, re

KEYS = ("who", "when", "where", "action")
LEAD = {"the", "a", "an", "at", "in", "on", "from", "your", "my"}
EXTRA = {"who": 3, "when": 3, "where": 3, "action": 4}

def norm(v):
    if v is None: return None
    s = str(v).split(",")[0] if "," in str(v) else str(v)          # drop a trailing comma-qualifier ("IT counter, second floor")
    w = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip().split()
    while w and w[0] in LEAD: w = w[1:]
    return " ".join(w)

def parse(text: str):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None

def field_ok(k: str, pred, label, accept: list | None) -> bool:
    if label is None or pred is None:
        return pred is None and label is None
    p = norm(pred)
    if not p: return False
    for lab in [label] + list(accept or []):
        l = norm(lab)
        if p == l: return True
        if l and re.search(r"(^| )" + re.escape(l) + r"( |$)", p) and len(p.split()) - len(l.split()) <= EXTRA[k]:
            return True
    return False

def score(text: str, label: dict, accept: dict | None = None) -> dict:
    obj = parse(text)
    valid = obj is not None and set(obj.keys()) == set(KEYS)
    fields = {k: field_ok(k, obj.get(k) if obj else "__missing__", label[k], (accept or {}).get(k)) for k in KEYS}
    return {"schema_valid": valid, "fields": fields, "fields_correct": sum(fields.values()), "all_correct": valid and all(fields.values())}
