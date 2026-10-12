"""Entry point: python -m prototype.run_replay --config configs/baseline.json"""
from __future__ import annotations

import argparse
from pathlib import Path

from .bench.replay import ROOT, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--workload", default="prototype/bench/workload/replay.jsonl")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    out = run(ROOT / a.config, ROOT / a.workload, a.run_id)
    print((out / "report.md").read_text())
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
