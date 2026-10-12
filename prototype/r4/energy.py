"""Energy from a `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -o LOG` text log.
Integrates CPU, GPU and ANE mW over each window's [unix_t0, unix_t1] and writes tasks-per-joule into a run's summary.
Usage: python -m prototype.r4.energy --log results/power-r4.log --run results/ane-r4-energy"""
from __future__ import annotations
import argparse, datetime, json, re
from pathlib import Path
import numpy as np

HEADER = re.compile(r"\*\*\* Sampled system activity \(([A-Za-z]{3} [A-Za-z]{3}\s+\d+ \d\d:\d\d:\d\d \d{4})[^)]*\)\s*\(([\d.]+)ms elapsed\)")
POWER = [("CPU Power", "cpu"), ("GPU Power", "gpu"), ("ANE Power", "ane"), ("Combined Power (CPU + GPU + ANE)", "combined")]


def parse(path: Path):
    """-> list of (unix_ts_of_sample_start, elapsed_s, {cpu,gpu,ane,combined} in W). Sample header time is the sample END on macOS."""
    rows, cur, ts, el = [], {}, None, None
    for line in path.read_text(errors="ignore").splitlines():
        m = HEADER.search(line)
        if m:
            if ts is not None and cur: rows.append((ts, el, cur))
            ts = datetime.datetime.strptime(m.group(1).replace("  ", " "), "%a %b %d %H:%M:%S %Y").timestamp(); el = float(m.group(2)) / 1000; cur = {}
            continue
        for key, name in POWER:
            if line.startswith(key):
                mm = re.search(r":\s*([\d.]+)\s*mW", line)
                if mm: cur[name] = float(mm.group(1)) / 1000
    if ts is not None and cur: rows.append((ts, el, cur))
    return rows


def window_energy(rows, t0, t1):
    """Energy in J per rail within [t0, t1]; each sample covers (ts - elapsed, ts]. Partial overlap is prorated."""
    J = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0, "combined": 0.0}; n = 0; cover = 0.0
    for ts, el, p in rows:
        a, b = ts - el, ts
        ov = max(0.0, min(b, t1) - max(a, t0))
        if ov <= 0: continue
        n += 1; cover += ov
        for k in J: J[k] += p.get(k, 0.0) * ov
    if n == 0: return None
    J["samples"] = n; J["coverage_s"] = cover; J["window_s"] = t1 - t0
    J["avg_W"] = {k: J[k] / cover for k in ("cpu", "gpu", "ane", "combined")}
    return J


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--log", required=True); ap.add_argument("--run", required=True); a = ap.parse_args()
    run = Path(a.run); rows = parse(Path(a.log)); print(f"{len(rows)} power samples; log span {rows[0][0]:.0f}..{rows[-1][0]:.0f}" if rows else "no samples parsed")
    windows = [json.loads(l) for l in (run / "windows.jsonl").read_text().splitlines()]
    per_cond = {}
    for w in windows:
        e = window_energy(rows, w["unix_t0"], w["unix_t1"]); w["energy_J"] = e
        per_cond.setdefault(w["condition"], []).append(w)
    out = {"log": a.log, "conditions": {}}
    for c, ws in per_cond.items():
        es = [w["energy_J"] for w in ws if w.get("energy_J")]
        if not es: out["conditions"][c] = {"energy": "unavailable"}; continue
        J = float(np.sum([e["combined"] or (e["cpu"] + e["gpu"] + e["ane"]) for e in es])); span = float(np.sum([e["window_s"] for e in es]))
        bg_correct = sum(w["bg"]["correct"] for w in ws); bg_done = sum(w["bg"]["completed"] for w in ws); fg_done = sum(w["fg"]["completed"] for w in ws)
        out["conditions"][c] = {"windows": len(ws), "energy_J_total": J, "avg_W": {k: float(np.mean([e["avg_W"][k] for e in es])) for k in ("cpu", "gpu", "ane", "combined")},
                                "bg_correct": bg_correct, "bg_completed": bg_done, "fg_completed": fg_done,
                                "correct_bg_tasks_per_kJ": 1000 * bg_correct / J if J else None, "J_per_correct_bg_task": J / bg_correct if bg_correct else None,
                                "J_per_fg_request": J / fg_done if fg_done else None, "coverage_fraction": float(np.mean([e["coverage_s"] / e["window_s"] for e in es]))}
    (run / "energy.json").write_text(json.dumps(out, indent=2))
    L = ["| condition | windows | avg W (CPU / GPU / ANE / combined) | energy J | correct bg tasks | J per correct bg task | correct bg tasks per kJ | J per fg request | coverage |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c, v in out["conditions"].items():
        if "avg_W" not in v: L.append(f"| {c} | — | unavailable | | | | | | |"); continue
        w = v["avg_W"]; f = lambda x, d=1: "n/a" if x is None else f"{x:.{d}f}"
        L.append(f"| {c} | {v['windows']} | {f(w['cpu'])} / {f(w['gpu'])} / {f(w['ane'],2)} / {f(w['combined'])} | {f(v['energy_J_total'],0)} | {v['bg_correct']} | {f(v['J_per_correct_bg_task'])} | {f(v['correct_bg_tasks_per_kJ'],2)} | {f(v['J_per_fg_request'])} | {v['coverage_fraction']:.2f} |")
    (run / "energy.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
