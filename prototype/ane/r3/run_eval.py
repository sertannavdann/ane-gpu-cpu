"""R3 model-level comparison on the frozen extraction set.
Conditions (same BF16 Qwen3-0.6B weights, same prompt tokens, greedy, thinking off):
  upstream_map : trace build, no env            -> upstream kernel map (residual-add FFN)
  control      : trace build, ANE_FORCE_PLAIN_FFN -> plain fused FFN, sigmoid form
  exp          : exp build,   ANE_FORCE_PLAIN_FFN -> plain fused FFN, g/(1+exp(-g)) form
  mlx          : MLX GPU reference (prototype/ane/r3/mlx_reference.py)
Usage: python -m prototype.ane.r3.run_eval [--conditions a,b,c] [--limit N]"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
R3 = ROOT / "prototype/ane/r3"; sys.path.insert(0, str(R3))
from score import score  # noqa: E402
EVAL = json.loads((R3 / "extraction_eval.json").read_text())
CONDS = {"upstream_map": ("ane-trace", {}), "control": ("ane-trace", {"ANE_FORCE_PLAIN_FFN": "1"}), "exp": ("ane-exp", {"ANE_FORCE_PLAIN_FFN": "1"})}


MODEL = {"id": "Qwen/Qwen3-0.6B"}


def model_dir():
    repo = "models--" + MODEL["id"].replace("/", "--")
    return str(next((ROOT / "cache.nosync/hf/hub" / repo / "snapshots").iterdir()))


def run_batch(binary: Path, env: dict, cases: list, trace: Path, prompts_path: Path) -> list[dict]:
    """All cases in one process (--prompts-file); model state is reset between cases."""
    if trace.exists(): trace.unlink()
    prompts_path.write_text(json.dumps([EVAL["prompt"].format(note=c["note"]) for c in cases], ensure_ascii=False))
    e = dict(os.environ, ANE_TRACE_PATH=str(trace), **env); e.setdefault("HF_HOME", str(ROOT / "cache.nosync/hf"))
    t0 = time.perf_counter()
    p = subprocess.run([str(binary), "generate", "--model", model_dir(), "--prompts-file", str(prompts_path), "--max-tokens", str(EVAL["max_tokens"]), "--temp", "0"],
                       capture_output=True, text=True, env=e, timeout=7200)
    wall = time.perf_counter() - t0
    err = p.stderr
    km = re.search(r"R3 FFN kernel map: (.*)", err)
    blocks = re.split(r"### CASE (\d+)\n", err)
    per_case = {int(blocks[i]): blocks[i + 1] for i in range(1, len(blocks) - 1, 2)}
    lines = [json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
    traces, cur = {}, None
    for l in lines:
        if "case" in l: cur = l["case"]; traces[cur] = {"prompt_ids": None, "steps": []}
        elif cur is not None and "prompt_ids" in l: traces[cur]["prompt_ids"] = l["prompt_ids"]
        elif cur is not None and "step" in l: traces[cur]["steps"].append(l)
    out = []
    for i, c in enumerate(cases):
        blk = per_case.get(i, "")
        m = re.search(r"==========\n(.*)\n==========", blk, re.S); text = m.group(1) if m else ""
        tps = re.search(r"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec", blk); ptps = re.search(r"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec", blk)
        tr = traces.get(i, {"prompt_ids": None, "steps": []})
        out.append({"id": c["id"], "text": text, "exit": p.returncode, "wall_s": wall / len(cases), "kernel_map": km.group(1) if km else None,
                    "prompt_tokens": int(ptps.group(1)) if ptps else None, "prompt_tps": float(ptps.group(2)) if ptps else None,
                    "gen_tokens": int(tps.group(1)) if tps else None, "gen_tps": float(tps.group(2)) if tps else None,
                    "prompt_ids": tr["prompt_ids"], "generated_ids": [s_["token"] for s_ in tr["steps"]], "steps": tr["steps"], "score": score(text, c["label"], c.get("accept")),
                    "stderr_tail": err[-400:] if p.returncode else ""})
    return out


def run_case(binary: Path, env: dict, case: dict, trace: Path) -> dict:
    if trace.exists(): trace.unlink()
    e = dict(os.environ, ANE_TRACE_PATH=str(trace), **env); e.setdefault("HF_HOME", str(ROOT / "cache.nosync/hf"))
    user = EVAL["prompt"].format(note=case["note"])
    t0 = time.perf_counter()
    p = subprocess.run([str(binary), "generate", "--model", model_dir(), "--prompt", user, "--max-tokens", str(EVAL["max_tokens"]), "--temp", "0"],
                       capture_output=True, text=True, env=e, timeout=600)
    wall = time.perf_counter() - t0
    err = p.stderr
    m = re.search(r"==========\n(.*)\n==========", err, re.S); text = m.group(1) if m else ""
    km = re.search(r"R3 FFN kernel map: (.*)", err)
    tps = re.search(r"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec", err); ptps = re.search(r"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec", err)
    lines = [json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
    prompt_ids = next((l["prompt_ids"] for l in lines if "prompt_ids" in l), None)
    steps = [l for l in lines if "step" in l]
    return {"id": case["id"], "text": text, "exit": p.returncode, "wall_s": wall, "kernel_map": km.group(1) if km else None,
            "prompt_tokens": int(ptps.group(1)) if ptps else None, "prompt_tps": float(ptps.group(2)) if ptps else None,
            "gen_tokens": int(tps.group(1)) if tps else None, "gen_tps": float(tps.group(2)) if tps else None,
            "prompt_ids": prompt_ids, "generated_ids": [s["token"] for s in steps], "steps": steps, "score": score(text, case["label"], case.get("accept")),
            "stderr_tail": err[-300:] if p.returncode else ""}


def compare(a: dict, b: dict) -> dict:
    """Token agreement and logit deltas while prefixes are identical (same prompt ids, same chosen tokens so far)."""
    if not a.get("steps") or not b.get("steps"): return {"comparable": False}
    same_prompt = a.get("prompt_ids") == b.get("prompt_ids")
    n, top1_agree, max_delta, deltas = 0, 0, 0.0, []
    for sa, sb in zip(a["steps"], b["steps"]):
        ta = {t: v for t, v in sa["top"]}; tb = {t: v for t, v in sb["top"]}
        common = set(ta) & set(tb)
        if common:
            ca = np.array([ta[t] for t in common]); cb = np.array([tb[t] for t in common])
            d = float(np.abs((ca - ca.mean()) - (cb - cb.mean())).max()); deltas.append(d); max_delta = max(max_delta, d)
        top1_agree += sa["top"][0][0] == sb["top"][0][0]; n += 1
        if sa["token"] != sb["token"]: break     # prefixes diverge after this step
    return {"comparable": True, "same_prompt_ids": same_prompt, "steps_on_identical_prefix": n, "top1_agreement": top1_agree / n if n else None,
            "max_centered_logit_delta": max_delta, "mean_centered_logit_delta": float(np.mean(deltas)) if deltas else None,
            "diverged": n < min(len(a["steps"]), len(b["steps"])) or a["generated_ids"] != b["generated_ids"], "identical_sequence": a["generated_ids"] == b["generated_ids"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conditions", default="upstream_map,control,exp,mlx"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default="ane-r3-model"); ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--per-process", action="store_true", help="one process per case (old mode)")
    a = ap.parse_args(); MODEL["id"] = a.model
    out = ROOT / "results" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    cases = EVAL["cases"][: a.limit] if a.limit else EVAL["cases"]
    results = {}
    for cond in a.conditions.split(","):
        cdir = out / cond; cdir.mkdir(exist_ok=True); recs = []
        if cond == "mlx":
            from .mlx_reference import main as mlx_main
            mlx_main(cdir, model_id=a.model)
            for c in cases:
                r = json.loads((cdir / f"{c['id']}.json").read_text()); r["score"] = score(r["text"], c["label"], c.get("accept")); recs.append(r)
        else:
            binary, env = CONDS[cond]
            if a.per_process:
                for c in cases:
                    recs.append(run_case(R3 / "bin" / binary, env, c, cdir / "trace.jsonl"))
            else:
                recs = run_batch(R3 / "bin" / binary, env, cases, cdir / "trace.jsonl", cdir / "prompts.json")
            for r in recs:
                (cdir / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False))
                print(cond, r["id"], "ok" if r["score"]["all_correct"] else f"{r['score']['fields_correct']}/4", f"{r['gen_tps'] or 0:.1f} tok/s", r["text"][:70].replace("\n", " "), flush=True)
        results[cond] = recs
    summary = {"n_cases": len(cases), "model": a.model, "conditions": {}}
    for cond, recs in results.items():
        s = {"schema_valid": sum(r["score"]["schema_valid"] for r in recs), "all_correct": sum(r["score"]["all_correct"] for r in recs),
             "fields_correct": sum(r["score"]["fields_correct"] for r in recs), "fields_total": 4 * len(recs),
             "by_kind": {k: sum(r["score"]["all_correct"] for r, c in zip(recs, cases) if c["kind"] == k) for k in ("clear", "missing", "misleading")},
             "kernel_map": next((r.get("kernel_map") for r in recs if r.get("kernel_map")), None),
             "gen_tps_median": float(np.median([r["gen_tps"] for r in recs if r.get("gen_tps")])) if any(r.get("gen_tps") for r in recs) else None,
             "prompt_tps_median": float(np.median([r["prompt_tps"] for r in recs if r.get("prompt_tps")])) if any(r.get("prompt_tps") for r in recs) else None,
             "failures": sum(1 for r in recs if r.get("exit"))}
        summary["conditions"][cond] = s
    pairs = [("control", "exp"), ("upstream_map", "control"), ("control", "mlx"), ("exp", "mlx"), ("upstream_map", "mlx")]
    summary["pairs"] = {}
    for x, y in pairs:
        if x in results and y in results:
            cs = [compare(a_, b_) for a_, b_ in zip(results[x], results[y])]
            cc = [c for c in cs if c.get("comparable")]
            summary["pairs"][f"{x}_vs_{y}"] = {"identical_sequences": sum(c["identical_sequence"] for c in cc), "comparable": len(cc),
                                              "mean_top1_agreement_on_prefix": float(np.mean([c["top1_agreement"] for c in cc if c["top1_agreement"] is not None])) if cc else None,
                                              "mean_steps_on_identical_prefix": float(np.mean([c["steps_on_identical_prefix"] for c in cc])) if cc else None,
                                              "max_centered_logit_delta": max((c["max_centered_logit_delta"] for c in cc), default=None),
                                              "mean_centered_logit_delta": float(np.mean([c["mean_centered_logit_delta"] for c in cc if c["mean_centered_logit_delta"] is not None])) if cc else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    L = [f"# R3 model-level comparison ({out.name}, {len(cases)} cases, model {a.model})", "", "| condition | kernel map | schema valid | all fields correct | fields | clear/missing/misleading | prompt tok/s | gen tok/s | failures |", "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for cond, s in summary["conditions"].items():
        bk = s["by_kind"]; L.append(f"| {cond} | {s['kernel_map'] or '—'} | {s['schema_valid']}/{len(cases)} | {s['all_correct']}/{len(cases)} | {s['fields_correct']}/{s['fields_total']} | {bk['clear']}/{bk['missing']}/{bk['misleading']} | {s['prompt_tps_median'] or 0:.1f} | {s['gen_tps_median'] or 0:.1f} | {s['failures']} |")
    L += ["", "| pair | identical sequences | mean steps on identical prefix | top-1 agreement on prefix | mean / max centered top-16 logit delta |", "| --- | ---: | ---: | ---: | ---: |"]
    for k, p in summary["pairs"].items():
        L.append(f"| {k} | {p['identical_sequences']}/{p['comparable']} | {p['mean_steps_on_identical_prefix'] or 0:.1f} | {p['mean_top1_agreement_on_prefix'] or 0:.3f} | {p['mean_centered_logit_delta'] or 0:.3f} / {p['max_centered_logit_delta'] or 0:.3f} |")
    L += ["", "Logit deltas are computed on the tokens common to both top-16 lists, after subtracting each list's mean (softmax is shift-invariant), only while the two runs have chosen identical tokens so far."]
    (out / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
