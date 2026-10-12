"""R4: concurrent placement. Foreground = interactive requests on the GPU (MLX, fixed model, fixed arrival schedule).
Background = extraction tasks from the frozen R3 fixture, cycled continuously, placed either on the GPU (same process,
token-boundary preemption by the foreground) or on the ANE (ane.cpp serve, one session). Conditions:
  fg_alone     foreground only                      bg_gpu_alone  background on GPU only (rate baseline)
  fg_bg_gpu    foreground + background on GPU       bg_ane_alone  background on ANE only (rate baseline)
  fg_bg_ane    foreground + background on ANE
Each condition runs `--repeats` windows of `--window-s` seconds in alternating order. Records per-request timestamps,
per-second memory/swap, and (if --power-log points at a `sudo powermetrics -i 1000` text log) integrated energy.
Usage: python -m prototype.r4.run_r4 [--window-s 300 --repeats 3] [--smoke]"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache.nosync" / "hf"))
sys.path.insert(0, str(ROOT / "prototype/ane/r3")); sys.path.insert(0, str(ROOT))
from score import score  # noqa: E402
from prototype.scenario_parallel import AneServer, vm  # noqa: E402

EVAL = json.loads((ROOT / "prototype/ane/r3/extraction_eval.json").read_text())
FG_PROMPTS = json.loads((ROOT / "prototype/r4/fg_prompts.json").read_text())
END_TOKENS = {151645, 151643, 248044, 248046}


def pct(v, q): return float(np.percentile(v, q)) if len(v) else None


# ----------------------------------------------------------------------------- GPU side: step-wise generators
class GpuModel:
    def __init__(self, model_id: str):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx; self.id = model_id
        t = time.perf_counter(); self.model, self.tok = load(model_id); mx.eval(self.model.parameters()); self.load_s = time.perf_counter() - t
        self.active_gb = mx.get_active_memory() / 2**30

    def unload(self):
        import gc
        self.model = None; self.tok = None; gc.collect(); self.mx.clear_cache()
        return self.mx.get_active_memory() / 2**30

    def start(self, user: str, max_tokens: int):
        from mlx_lm import stream_generate
        prompt = self.tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens)   # Python generator; each next() = one token


class Job:
    """One request, foreground or background, driven one token at a time."""
    def __init__(self, kind, idx, text, max_tokens, arrival):
        self.kind, self.idx, self.text, self.max_tokens, self.arrival = kind, idx, text, max_tokens, arrival
        self.gen = None; self.start_t = None; self.first_t = None; self.token_t = []; self.parts = []; self.done_t = None; self.tokens = 0; self.error = None
    def step(self, model: GpuModel) -> bool:
        """Advance one token. Returns True when finished."""
        if self.gen is None:
            self.start_t = time.perf_counter(); self.gen = model.start(self.text, self.max_tokens)
        try:
            r = next(self.gen)
        except StopIteration:
            self.done_t = time.perf_counter(); self.release(); return True
        except Exception as e:  # typed failure; never silent
            self.error = repr(e)[:200]; self.done_t = time.perf_counter(); self.release(); return True
        now = time.perf_counter()
        if self.first_t is None: self.first_t = now
        self.token_t.append(now); self.parts.append(r.text); self.tokens = r.generation_tokens
        if r.finish_reason or (r.token in END_TOKENS):
            self.done_t = now; self.release(); return True
        return False
    def release(self):
        """Drop the generator so its KV cache is freed now, not when the window's job list dies (R4 run 1 leaked ~58 MB per task)."""
        g = self.gen; self.gen = None
        if g is not None:
            try: g.close()
            except Exception: pass
    def record(self):
        gaps = np.diff(self.token_t) * 1000 if len(self.token_t) > 1 else np.array([])
        return {"kind": self.kind, "idx": self.idx, "arrival": self.arrival, "start": self.start_t, "first": self.first_t, "done": self.done_t,
                "ttft_ms": (self.first_t - self.arrival) * 1000 if self.first_t else None, "queue_ms": (self.start_t - self.arrival) * 1000 if self.start_t else None,
                "wall_ms": (self.done_t - self.arrival) * 1000 if self.done_t else None, "tokens": self.tokens,
                "gap_p50_ms": pct(gaps, 50), "gap_p95_ms": pct(gaps, 95), "gap_max_ms": float(gaps.max()) if len(gaps) else None,
                "decode_tps": (len(self.token_t) - 1) / (self.token_t[-1] - self.token_t[0]) if len(self.token_t) > 2 else None,
                "text": "".join(self.parts)[:400], "error": self.error}


# ----------------------------------------------------------------------------- memory sampler
class MemSampler(threading.Thread):
    def __init__(self, mx=None):
        super().__init__(daemon=True); self.mx = mx; self.rows = []; self.stop = threading.Event()
    def run(self):
        while not self.stop.is_set():
            v = vm(); self.rows.append({"t": time.perf_counter(), **v, "mlx_active_gb": (self.mx.get_active_memory() / 2**30) if self.mx else None})
            self.stop.wait(1.0)


# ----------------------------------------------------------------------------- ANE background worker
class AneBgWorker(threading.Thread):
    def __init__(self, server: AneServer, cases, max_tokens, end_time):
        super().__init__(daemon=True); self.srv, self.cases, self.max_tokens, self.end = server, cases, max_tokens, end_time
        self.records = []; self.offered = 0; self.busy_intervals = []
    def run(self):
        i = 0
        while time.perf_counter() < self.end:
            c = self.cases[i % len(self.cases)]; i += 1; self.offered += 1
            t0 = time.perf_counter()
            r = self.srv.request(i, EVAL["prompt"].format(note=c["note"]), self.max_tokens)
            t1 = time.perf_counter(); self.busy_intervals.append((t0, t1))
            sc = score(r.get("text", ""), c["label"], c.get("accept"))
            self.records.append({"kind": "bg_ane", "case": c["id"], "arrival": t0, "done": t1, "wall_ms": (t1 - t0) * 1000, "ok": bool(r.get("ok")),
                                 "gen_tokens": r.get("generation_tokens"), "gen_tps": r.get("generation_tps"), "prompt_tps": r.get("prompt_tps"),
                                 "all_correct": sc["all_correct"], "fields_correct": sc["fields_correct"], "completed_in_window": t1 <= self.end})


# ----------------------------------------------------------------------------- one window
def run_window(cond: str, fg: GpuModel, bg_gpu: GpuModel | None, ane: AneServer | None, a, cases) -> dict:
    W = a.window_s; t0 = time.perf_counter(); end = t0 + W
    rng = random.Random(a.seed)
    arrivals = [] if cond.startswith("bg_") else [t0 + k * a.fg_interval_s + rng.uniform(0, 1.0) for k in range(int(W / a.fg_interval_s) + 1) if k * a.fg_interval_s < W]
    fg_jobs, bg_jobs, fg_queue = [], [], []
    fg_active = None; bg_active = None; bg_i = 0; next_arr = 0
    bg_gpu_on = cond in ("fg_bg_gpu", "bg_gpu_alone"); fg_on = not cond.startswith("bg_")
    ane_worker = AneBgWorker(ane, cases, a.bg_max_tokens, end) if cond in ("fg_bg_ane", "bg_ane_alone") else None
    mem = MemSampler(fg.mx); mem.start(); mem0 = vm()
    if ane_worker: ane_worker.start()
    bg_parked_while_fg = 0.0; last = time.perf_counter()
    while True:
        now = time.perf_counter()
        while next_arr < len(arrivals) and arrivals[next_arr] <= now:
            fg_queue.append(Job("fg", next_arr, FG_PROMPTS[next_arr % len(FG_PROMPTS)], a.fg_max_tokens, arrivals[next_arr])); next_arr += 1
        window_over = now >= end
        if fg_active is None and fg_queue and not window_over:
            fg_active = fg_queue.pop(0)
        if fg_active is not None:                      # foreground has absolute priority; background is parked between tokens
            if bg_active is not None: bg_parked_while_fg += time.perf_counter() - last
            if fg_active.step(fg): fg_jobs.append(fg_active); fg_active = None
        elif bg_gpu_on and not window_over:
            if bg_active is None:
                c = cases[bg_i % len(cases)]; bg_active = Job("bg_gpu", bg_i, EVAL["prompt"].format(note=c["note"]), a.bg_max_tokens, time.perf_counter()); bg_active.case = c; bg_i += 1
            if bg_active.step(bg_gpu):
                bg_jobs.append(bg_active); bg_active = None
        else:
            if window_over and fg_active is None and not fg_queue: break
            if fg_on and next_arr < len(arrivals): time.sleep(min(0.005, max(0.0, arrivals[next_arr] - time.perf_counter())))
            elif not fg_on: time.sleep(0.05)
            else: break
        last = time.perf_counter()
        if not fg_on and time.perf_counter() >= end: break
    if bg_active is not None: bg_active.done_t = None; bg_active.release(); bg_jobs.append(bg_active)   # in flight at window end: offered, not completed
    if ane_worker: ane_worker.join(timeout=120)
    mem.stop.set(); mem.join(timeout=2); mem1 = vm()
    fg_rec = [j.record() for j in fg_jobs]
    bg_rec = []
    for j in bg_jobs:
        r = j.record(); r["case"] = j.case["id"]; r["completed_in_window"] = j.done_t is not None and j.done_t <= end
        sc = score("".join(j.parts), j.case["label"], j.case.get("accept")); r["all_correct"] = sc["all_correct"] and r["completed_in_window"]; r["fields_correct"] = sc["fields_correct"]; bg_rec.append(r)
    if ane_worker: bg_rec = ane_worker.records
    fg_t = [r["ttft_ms"] for r in fg_rec if r["ttft_ms"] is not None]; gaps95 = [r["gap_p95_ms"] for r in fg_rec if r["gap_p95_ms"] is not None]
    bg_done = [r for r in bg_rec if r.get("completed_in_window")]
    span = time.perf_counter() - t0
    out = {"condition": cond, "window_s": W, "actual_span_s": span, "fg": {
               "offered": len(arrivals), "completed": len(fg_rec), "failed": sum(1 for r in fg_rec if r["error"]),
               "ttft_p50_ms": pct(fg_t, 50), "ttft_p95_ms": pct(fg_t, 95), "ttft_max_ms": max(fg_t) if fg_t else None,
               "gap_p50_ms": pct([r["gap_p50_ms"] for r in fg_rec if r["gap_p50_ms"]], 50), "gap_p95_ms": pct(gaps95, 95),
               "wall_p95_ms": pct([r["wall_ms"] for r in fg_rec if r["wall_ms"]], 95),
               "decode_tps_median": pct([r["decode_tps"] for r in fg_rec if r["decode_tps"]], 50),
               "deadline_misses_ttft_gt_2s": sum(1 for t in fg_t if t > 2000)},
           "bg": {"device": "gpu" if bg_gpu_on else ("ane" if ane_worker else None), "offered": len(bg_rec) if not ane_worker else ane_worker.offered,
                  "completed": len(bg_done), "correct": sum(1 for r in bg_done if r.get("all_correct")),
                  "tasks_per_min": len(bg_done) / (W / 60), "correct_per_min": sum(1 for r in bg_done if r.get("all_correct")) / (W / 60),
                  "latency_p50_ms": pct([r["wall_ms"] for r in bg_done if r.get("wall_ms")], 50), "latency_p95_ms": pct([r["wall_ms"] for r in bg_done if r.get("wall_ms")], 95),
                  "decode_tps_median": pct([r.get("decode_tps") or r.get("gen_tps") for r in bg_done if (r.get("decode_tps") or r.get("gen_tps"))], 50),
                  "bg_parked_while_fg_s": bg_parked_while_fg},
           "memory": {"swap_used_mb_start": mem0["swap_used_mb"], "swap_used_mb_end": mem1["swap_used_mb"], "swap_delta_mb": (mem1["swap_used_mb"] or 0) - (mem0["swap_used_mb"] or 0),
                      "free_gb_min": min(r["free_gb"] for r in mem.rows) if mem.rows else None, "mlx_active_gb_max": max((r["mlx_active_gb"] or 0) for r in mem.rows) if mem.rows else None,
                      "ane_rss_gb": ane.rss_gb() if ane else None},
           "t0": t0, "t1": time.perf_counter(), "fg_records": fg_rec, "bg_records": bg_rec, "mem_rows": mem.rows}
    return out


# ----------------------------------------------------------------------------- energy (optional)
def parse_powermetrics(path: Path):
    """Parses `powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000` text output into (unix_time, {cpu,gpu,ane} mW)."""
    import datetime
    rows, cur, ts = [], {}, None
    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(r"\*\*\* Sampled system activity \((.*?)\)", line)
        if m:
            if ts and cur: rows.append((ts, cur))
            try: ts = datetime.datetime.strptime(m.group(1).split(" (")[0].strip(), "%a %b %d %H:%M:%S %Y").timestamp()
            except Exception: ts = None
            cur = {}; continue
        for key, name in (("CPU Power", "cpu"), ("GPU Power", "gpu"), ("ANE Power", "ane")):
            if line.startswith(key):
                mm = re.search(r"([\d.]+)\s*mW", line)
                if mm: cur[name] = float(mm.group(1))
    if ts and cur: rows.append((ts, cur))
    return rows


def energy_for(rows, t_start_unix, t_end_unix):
    sel = [(t, p) for t, p in rows if t_start_unix <= t <= t_end_unix]
    if len(sel) < 2: return None
    J = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0}
    for (t0, p0), (t1, _) in zip(sel, sel[1:]):
        dt = t1 - t0
        for k in J: J[k] += p0.get(k, 0.0) / 1000 * dt
    J["total_cpu_gpu_ane"] = sum(J.values()); J["samples"] = len(sel); return J


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg-model", default="mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit"); ap.add_argument("--bg-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--window-s", type=float, default=300); ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--fg-interval-s", type=float, default=8.0)
    ap.add_argument("--fg-max-tokens", type=int, default=80); ap.add_argument("--bg-max-tokens", type=int, default=64); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conditions", default="fg_alone,fg_bg_gpu,fg_bg_ane,bg_gpu_alone,bg_ane_alone"); ap.add_argument("--baseline-window-s", type=float, default=120)
    ap.add_argument("--run-id", default="ane-r4-concurrent"); ap.add_argument("--power-log", default=None); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--port", type=int, default=55171); ap.add_argument("--settle-s", type=float, default=10)
    a = ap.parse_args()
    if a.smoke: a.window_s, a.repeats, a.baseline_window_s, a.settle_s = 40, 1, 30, 3
    out = ROOT / "results" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    conds = a.conditions.split(","); cases = EVAL["cases"]
    log = {"args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "mem_start": vm(), "docker_vm": bool(subprocess.run(["pgrep", "-f", "com.docker.virtualization|Virtualization.VirtualMachine"], capture_output=True).stdout)}
    fg = GpuModel(a.fg_model); log["fg_load_s"] = fg.load_s; log["fg_active_gb"] = fg.active_gb
    snap = str(next((ROOT / "cache.nosync/hf/hub" / ("models--" + a.bg_model.replace("/", "--")) / "snapshots").iterdir()))
    main_conds = [c for c in conds if c.startswith("fg_")]; base_conds = [c for c in conds if c.startswith("bg_")]
    plan = []
    for r in range(a.repeats):                    # rotate the starting condition per repeat instead of random order
        order = main_conds[r % len(main_conds):] + main_conds[:r % len(main_conds)]; plan += [(c, a.window_s) for c in order]
    plan += [(c, a.baseline_window_s) for c in base_conds]
    # Residency rule (memory guardrail): exactly one background placement resident at a time. The 4B on MLX is loaded only for
    # GPU windows and unloaded before ANE windows; the ANE server runs only during ANE windows. The foreground model stays resident.
    bg_gpu = None; ane = None; windows = []; residency_log = []
    def need(placement):
        nonlocal bg_gpu, ane
        if placement != "gpu" and bg_gpu is not None:
            left = bg_gpu.unload(); bg_gpu = None; residency_log.append({"t": time.time(), "event": "unload_bg_gpu", "mlx_active_gb_after": left})
        if placement != "ane" and ane is not None:
            ane.close(); ane = None; residency_log.append({"t": time.time(), "event": "stop_ane"}); time.sleep(3)
        if placement == "gpu" and bg_gpu is None:
            bg_gpu = GpuModel(a.bg_model); residency_log.append({"t": time.time(), "event": "load_bg_gpu", "load_s": bg_gpu.load_s, "mlx_active_gb": fg.mx.get_active_memory() / 2**30})
            list(bg_gpu.start(EVAL["prompt"].format(note=cases[0]["note"]), 8))   # warm
        if placement == "ane" and ane is None:
            ane = AneServer(snap, a.port, 1, out / "ane-serve.log"); residency_log.append({"t": time.time(), "event": "start_ane", "startup_s": ane.startup_s, "rss_gb": ane.rss_gb()})
            ane.request(0, EVAL["prompt"].format(note=cases[0]["note"]), a.bg_max_tokens)   # warm
        list(fg.start(FG_PROMPTS[0], 4))          # warm the foreground model too (pages back in after any pressure)
        time.sleep(a.settle_s); return vm()
    for i, (cond, ws) in enumerate(plan):
        a.window_s = ws
        placement = "gpu" if cond in ("fg_bg_gpu", "bg_gpu_alone") else ("ane" if cond in ("fg_bg_ane", "bg_ane_alone") else "none")
        mem_pre = need(placement)
        print(f"== window {i+1}/{len(plan)}: {cond} ({ws:.0f} s) resident: fg{'+bg_gpu' if bg_gpu else ''}{'+ane' if ane else ''} | {mem_pre}", flush=True)
        w = run_window(cond, fg, bg_gpu, ane, a, cases); w["index"] = i; w["resident"] = {"bg_gpu": bg_gpu is not None, "ane": ane is not None, "mem_pre_window": mem_pre}
        w["unix_t0"] = time.time() - (time.perf_counter() - w["t0"]); w["unix_t1"] = time.time() - (time.perf_counter() - w["t1"])
        windows.append(w)
        f, b, m = w["fg"], w["bg"], w["memory"]
        print(f"   fg {f['completed']}/{f['offered']} ttft p50/p95 {f['ttft_p50_ms'] or 0:.0f}/{f['ttft_p95_ms'] or 0:.0f} ms gap p95 {f['gap_p95_ms'] or 0:.0f} ms | "
              f"bg {b['completed']}/{b['offered']} correct {b['correct']} ({b['correct_per_min']:.1f}/min) lat p50 {b['latency_p50_ms'] or 0:.0f} ms | swap Δ {m['swap_delta_mb']:.0f} MB", flush=True)
        with (out / "windows.jsonl").open("a") as fh: fh.write(json.dumps({k: v for k, v in w.items() if k != "mem_rows"}, default=str) + "\n")
        with (out / "memory.jsonl").open("a") as fh: fh.write(json.dumps({"window": i, "condition": cond, "rows": w["mem_rows"]}) + "\n")
    log["residency"] = residency_log
    if ane: ane.close()
    # energy
    power_rows = parse_powermetrics(Path(a.power_log)) if a.power_log and Path(a.power_log).exists() else None
    for w in windows:
        w["energy_J"] = energy_for(power_rows, w["unix_t0"], w["unix_t1"]) if power_rows else None
    summary = aggregate(windows, log)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "manifest.json").write_text(json.dumps({"log": log, "host": host()}, indent=2, default=str))
    (out / "report.md").write_text(report(summary, log)); print((out / "report.md").read_text())


def host():
    sh = lambda c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
    return {"chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]), "os": sh(["sw_vers", "-productVersion"]), "build": sh(["sw_vers", "-buildVersion"]),
            "pin": json.loads((ROOT / "prototype/ane/vendor/PIN.json").read_text())["commit"], "power_mode": sh(["pmset", "-g", "batt"]).split("\n")[0]}


def aggregate(windows, log):
    conds = {}
    for w in windows: conds.setdefault(w["condition"], []).append(w)
    def agg(ws, path, how="mean"):
        vals = []
        for w in ws:
            v = w
            for k in path.split("."): v = v.get(k) if isinstance(v, dict) else None
            if v is not None: vals.append(v)
        if not vals: return None
        return float(np.mean(vals)) if how == "mean" else (float(np.max(vals)) if how == "max" else float(np.sum(vals)))
    s = {"conditions": {}, "n_windows": len(windows), "energy_available": any(w.get("energy_J") for w in windows)}
    for c, ws in conds.items():
        e = [w["energy_J"] for w in ws if w.get("energy_J")]
        s["conditions"][c] = {"windows": len(ws), "window_s": ws[0]["window_s"],
            "fg_offered": agg(ws, "fg.offered", "sum"), "fg_completed": agg(ws, "fg.completed", "sum"), "fg_failed": agg(ws, "fg.failed", "sum"),
            "fg_ttft_p50_ms": agg(ws, "fg.ttft_p50_ms"), "fg_ttft_p95_ms": agg(ws, "fg.ttft_p95_ms"), "fg_ttft_p95_ms_worst_window": agg(ws, "fg.ttft_p95_ms", "max"),
            "fg_gap_p50_ms": agg(ws, "fg.gap_p50_ms"), "fg_gap_p95_ms": agg(ws, "fg.gap_p95_ms"), "fg_decode_tps": agg(ws, "fg.decode_tps_median"), "fg_deadline_misses": agg(ws, "fg.deadline_misses_ttft_gt_2s", "sum"),
            "bg_device": ws[0]["bg"]["device"], "bg_offered": agg(ws, "bg.offered", "sum"), "bg_completed": agg(ws, "bg.completed", "sum"), "bg_correct": agg(ws, "bg.correct", "sum"),
            "bg_correct_per_min": agg(ws, "bg.correct_per_min"), "bg_tasks_per_min": agg(ws, "bg.tasks_per_min"), "bg_latency_p50_ms": agg(ws, "bg.latency_p50_ms"), "bg_latency_p95_ms": agg(ws, "bg.latency_p95_ms"),
            "bg_decode_tps": agg(ws, "bg.decode_tps_median"), "swap_delta_mb_total": agg(ws, "memory.swap_delta_mb", "sum"), "free_gb_min": min((w["memory"]["free_gb_min"] or 99) for w in ws),
            "mlx_active_gb_max": agg(ws, "memory.mlx_active_gb_max", "max"),
            "energy_J_total": float(np.sum([x["total_cpu_gpu_ane"] for x in e])) if e else None,
            "correct_tasks_per_J": (agg(ws, "bg.correct", "sum") / float(np.sum([x["total_cpu_gpu_ane"] for x in e]))) if e and agg(ws, "bg.correct", "sum") else None}
    return s


def report(s, log):
    c = s["conditions"]; f = lambda v, d=0: "n/a" if v is None else f"{v:.{d}f}"
    L = [f"# R4 concurrent placement ({log['started']})", "",
         f"Foreground `{log['args']['fg_model']}` on the GPU ({log['fg_active_gb']:.1f} GB active), one request every {log['args']['fg_interval_s']} s, {log['args']['fg_max_tokens']} tokens max. "
         f"Background `{log['args']['bg_model']}`: BF16 on the GPU in the same process with token-boundary preemption by the foreground, or fp16 hybrid on the ANE via ane.cpp serve (one session; RSS {log.get('ane_rss_gb','n/a')} GB). "
         f"Residency rule: one background placement resident at a time (4B on MLX loaded only for GPU windows, ANE server only for ANE windows); foreground model always resident. Docker VM present: {log['docker_vm']}. Energy: {'integrated from powermetrics log' if s['energy_available'] else 'unavailable (no root powermetrics log supplied)'}.", "",
         "| condition | windows | fg done/offered | fg TTFT p50 / p95 ms (worst-window p95) | fg gap p50 / p95 ms | fg tok/s | fg TTFT > 2 s | bg device | bg correct/done/offered | bg correct/min | bg latency p50 / p95 ms | bg tok/s | swap Δ MB | MLX peak GB |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k, v in c.items():
        L.append(f"| {k} | {v['windows']} × {v['window_s']:.0f} s | {f(v['fg_completed'])}/{f(v['fg_offered'])} | {f(v['fg_ttft_p50_ms'])} / {f(v['fg_ttft_p95_ms'])} ({f(v['fg_ttft_p95_ms_worst_window'])}) | "
                 f"{f(v['fg_gap_p50_ms'])} / {f(v['fg_gap_p95_ms'])} | {f(v['fg_decode_tps'],1)} | {f(v['fg_deadline_misses'])} | {v['bg_device'] or '—'} | {f(v['bg_correct'])}/{f(v['bg_completed'])}/{f(v['bg_offered'])} | "
                 f"{f(v['bg_correct_per_min'],2)} | {f(v['bg_latency_p50_ms'])} / {f(v['bg_latency_p95_ms'])} | {f(v['bg_decode_tps'],1)} | {f(v['swap_delta_mb_total'])} | {f(v['mlx_active_gb_max'],1)} |")
    if s["energy_available"]:
        L += ["", "| condition | energy J (CPU+GPU+ANE) | correct bg tasks / J |", "| --- | ---: | ---: |"]
        for k, v in c.items(): L.append(f"| {k} | {f(v['energy_J_total'])} | {f(v['correct_tasks_per_J'],4)} |")
    L += ["", "Foreground TTFT is arrival-to-first-token including queueing; gaps are visible token gaps; background correctness uses the frozen v2 extraction fixture. Windows alternate per repeat; baselines run once at the end."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
