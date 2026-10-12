"""Load a workload and config, run the scheduler, write results/<run-id>/."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ..admission import Scheduler
from ..contracts import Availability, Request, RequestClass
from ..generator_fake import ScriptedGenerator
from ..router_fake import FailureInjectingRouter, KeywordRouter
from . import metrics

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_arrivals(rows: list[dict], cfg: dict) -> list[tuple[float, Request]]:
    """Background requests start at t=0 spaced by bg_interval_s; foreground requests arrive from
    fg_start_s spaced by fg_interval_s, so they land while background work is in flight."""
    arrivals, fg_i, bg_i = [], 0, 0
    for r in rows:
        cls = RequestClass(r["class"])
        if cls is RequestClass.FOREGROUND:
            t = cfg["fg_start_s"] + fg_i * cfg["fg_interval_s"]; fg_i += 1
        else:
            t = bg_i * cfg["bg_interval_s"]; bg_i += 1
        arrivals.append((t, Request(r["id"], r.get("session_id", r["id"]), r["input"], cls,
                                    context=dict(r.get("context", {})),
                                    max_output_tokens=r.get("max_output_tokens", cfg["max_output_tokens"][r["class"]]),
                                    deadline_ms=cfg["deadline_ms"][r["class"]])))
    return arrivals


def make_components(cfg: dict):
    fallback = KeywordRouter()
    if cfg["router"] == "keyword":
        router = KeywordRouter()
    elif cfg["router"] == "tfidf":
        from ..router_tfidf import TfidfRouter
        router = TfidfRouter.from_training_file(ROOT / cfg["train_path"])
    else:
        raise ValueError(cfg["router"])
    if cfg.get("router_failures"):
        router = FailureInjectingRouter(router, cfg["router_failures"], hang_s=cfg.get("hang_s", 0.5))
    if cfg["generator"] == "fake":
        generator = ScriptedGenerator(cfg.get("fake_token_delay_s", 0.01), cfg.get("generation_failures"))
    elif cfg["generator"] == "mlx":
        from ..generator_mlx import MlxGenerator
        generator = MlxGenerator(cfg["model_id"], cfg.get("max_context", 4096), cfg.get("model_hash"))
        probe = generator.probe()
        if not probe["ok"]:
            raise RuntimeError(f"MLX probe failed: {probe}")
    else:
        raise ValueError(cfg["generator"])
    availability = Availability(frozenset(cfg["availability"]["tools"]), frozenset(cfg["availability"]["tiers"]))
    return router, fallback, generator, availability


def host_manifest() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception as e:
            return f"unavailable: {e!r}"
    return {"os": sh(["sw_vers", "-productVersion"]), "build": sh(["sw_vers", "-buildVersion"]),
            "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]), "python": sys.version.split()[0],
            "platform": platform.platform(),
            "docker_vm_running": bool(sh(["pgrep", "-f", "com.docker"])),
            "resident_python_llm_procs": sh(["pgrep", "-fl", "llm_service"])}


def run(config_path: Path, workload_path: Path, run_id: str | None = None) -> Path:
    cfg = json.loads(config_path.read_text())
    rows = load_rows(workload_path)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + config_path.stem
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    router, fallback, generator, availability = make_components(cfg)
    sched = Scheduler(router, fallback, generator, availability, fg_capacity=cfg["fg_capacity"],
                      bg_capacity=cfg["bg_capacity"], priority=cfg["priority"],
                      router_timeout_ms=cfg["router_timeout_ms"])
    mem_before = generator.memory() if hasattr(generator, "memory") else None
    t0 = time.monotonic()
    outcomes = sched.run(build_arrivals(rows, cfg))
    wall = time.monotonic() - t0
    sched.close()
    mem_after = generator.memory() if hasattr(generator, "memory") else None
    summary = metrics.summarize(outcomes, {r["id"]: r for r in rows})
    summary["wall_s"] = wall
    summary["fg_overlapping_bg"] = metrics.overlap_count(sched.trace, outcomes)
    summary["router_backend"] = router.name
    summary["router_calibration"] = getattr(router, "calibration", None)
    manifest = {"run_id": run_id, "config": cfg, "config_path": str(config_path),
                "workload_path": str(workload_path),
                "workload_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
                "host": host_manifest(), "generator": generator.name,
                "model_id": getattr(generator, "model_id", None), "model_hash": getattr(generator, "weights_hash", None),
                "memory_before": mem_before, "memory_after": mem_after,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out / "requests.jsonl").open("w") as f:
        for o in outcomes:
            f.write(json.dumps(asdict(o)) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(report(summary, manifest))
    return out


def report(s: dict, m: dict) -> str:
    lines = [f"# Replay {m['run_id']}", "",
             f"Config `{Path(m['config_path']).name}`, router `{s['router_backend']}`, generator `{m['generator']}`, "
             f"priority {m['config']['priority']}, wall {s['wall_s']:.1f} s, workload sha `{m['workload_sha256'][:12]}`.",
             "", "Foreground requests overlapping an active background generation: "
             f"{s['fg_overlapping_bg']}.", "",
             "| class | offered | completed | completion | TTFT misses | TTFT p50 ms | TTFT p95 ms | queue p95 ms | router p95 ms | total p95 ms | disposition acc | rubric |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cls, c in s["by_class"].items():
        f = lambda v: "n/a" if v is None else (f"{v:.0f}" if isinstance(v, float) and v > 2 else f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {cls} | {c['offered']} | {c['completed']} | {c['completion_rate']:.2f} | {c['ttft_misses']} | "
                     f"{f(c['ttft_p50_ms'])} | {f(c['ttft_p95_ms'])} | {f(c['queue_p95_ms'])} | {f(c['router_p95_ms'])} | "
                     f"{f(c['total_p95_ms'])} | {f(c['disposition_accuracy'])} | {c['rubric_pass']}/{c['eligible']} |")
    lines += ["", f"States: {s['states']}. Errors: {s['errors']}. Fallbacks: {s['fallbacks']}.", "",
              "Fake router/generator results are plumbing evidence only; they support no speed or quality claim."]
    return "\n".join(lines) + "\n"
