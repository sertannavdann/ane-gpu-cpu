"""Paper predictions for every lab: write the number down BEFORE measuring.

Each prediction is (label, predicted value, unit, how it was computed, how to
read the measured value from this run's records). The inputs are public facts
and stated assumptions, never this run's measurements, except where a lab is
explicitly predicted from an earlier lab's measured roof (marked "from run").
"""
from __future__ import annotations

import json
import math
import pathlib

# Inputs. Change these to predict a different chip.
SPEC = {
    "clock_ghz": 4.5,      # M4 P-core max clock, widely reported (Apple does not publish it)
    "fma_pipes": 4,        # 128-bit FP/SIMD pipes per P-core (public microbenchmark reports)
    "fma_latency": 4,      # cycles
    "loads_per_cycle": 3,  # 128-bit loads per cycle per P-core (public microbenchmark reports)
    "l1_cycles": 3,        # load-to-use latency
    "dram_ns": 100,        # typical LPDDR SoC DRAM latency (assumption)
    "dram_gbs": 410,       # Apple spec sheet, 14-core M4 Max
    "p_cores": 10,
}


def _recs(path: pathlib.Path, lab: str, case: str | None = None) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("lab") == lab and (case is None or r.get("case") == case):
                out.append(r)
    return out


def _pick(path, lab, case, key="median", where=None, agg=max):
    rs = [r for r in _recs(path, lab, case) if key in r and r[key] is not None and (where is None or where(r))]
    return agg(r[key] for r in rs) if rs else None


def predictions(lab: str, ceil: dict) -> list[dict]:
    S = SPEC
    core_peak = S["fma_pipes"] * 8 * S["clock_ghz"]  # FMAs/cycle x 8 FLOP (4 lanes x 2) x GHz
    P = []

    def add(label, pred, unit, how, measure):
        P.append({"label": label, "pred": pred, "unit": unit, "how": how, "measure": measure})

    if lab == "peak":
        add("fp32, 1 dependency chain", 8 * S["clock_ghz"] / S["fma_latency"], "GFLOP/s",
            "one FMA (8 FLOP) every 4 cycles x 4.5 GHz",
            lambda j: _pick(j, "p1.peak", "chains_f32", where=lambda r: r["chains"] == 1))
        add("fp32, 1 core, best", core_peak, "GFLOP/s", "4 pipes x 8 FLOP x 4.5 GHz",
            lambda j: _pick(j, "p1.peak", "chains_f32"))
        add("chains needed to saturate", S["fma_pipes"] * S["fma_latency"], "chains", "pipes x latency = 4 x 4",
            lambda j: _sat_chains(j))
        add("fp16 / fp32 throughput", 2.0, "x", "8 vs 4 lanes per 128-bit register",
            lambda j: _ratio(_pick(j, "p1.peak", "f16_1core"), _pick(j, "p1.peak", "chains_f32")))
        add("fp32, 8 P-cores", 8 * core_peak, "GFLOP/s", "8 x one-core peak (assumes no clock drop)",
            lambda j: _pick(j, "p1.peak", "threads_f32", where=lambda r: r["threads"] == 8))
    elif lab == "bandwidth":
        add("L1 read, 1 core", S["loads_per_cycle"] * 16 * S["clock_ghz"], "GB/s", "3 loads x 16 B x 4.5 GHz",
            lambda j: _pick(j, "p1.bandwidth", "sweep", where=lambda r: r.get("level") == "L1"))
        add("DRAM read, best CPU", S["dram_gbs"], "GB/s", "Apple spec sheet (whole SoC)",
            lambda j: _pick(j, "p1.bandwidth", "dram_read"))
        add("DRAM read, 1 core", None, "GB/s", "unknown without the core's miss-queue depth (Little's law)",
            lambda j: _pick(j, "p1.bandwidth", "dram_read", where=lambda r: r["threads"] == 1))
    elif lab == "latency":
        add("L1 hit", S["l1_cycles"] / S["clock_ghz"], "ns", "3 cycles / 4.5 GHz",
            lambda j: _pick(j, "p1.latency", "random", "ns", agg=min))
        add("DRAM miss", S["dram_ns"], "ns", "typical LPDDR SoC (assumption)",
            lambda j: _pick(j, "p1.latency", "random", "ns", agg=max))
    elif lab == "gemm":
        add("packed NEON, 1 core", core_peak, "GFLOP/s", "one-core peak (upper bound)",
            lambda j: _pick(j, "p1.gemm", "steps_packed+NEON 1T"))
        bw = ceil.get("bw", S["dram_gbs"])
        add("M=1 (decode) from DRAM", 0.5 * bw, "GFLOP/s", f"0.5 FLOP/B x {bw:.0f} GB/s (from run)",
            lambda j: _pick(j, "p1.gemm", "shape_accel", where=lambda r: r["m"] == 1))
        if "peak_all" in ceil and "bw" in ceil:
            add("tokens where GEMM turns compute-bound", 2 * ceil["peak_all"] / ceil["bw"], "M",
                "intensity ~ M/2 FLOP/B = peak / bandwidth (from run)", lambda j: None)
    elif lab == "gemv":
        add("fp32 GEMV, 1 core", 4 * 4 / S["fma_latency"] * S["clock_ghz"] * 4, "GB/s",
            "4 chains x 4 lanes / 4 cycles x 4.5 GHz x 4 B",
            lambda j: _pick(j, "p1.gemv", "NEON fp32", where=lambda r: r["threads"] == 1))
        bw = ceil.get("bw", S["dram_gbs"])
        add("fp32 GEMV, best", bw, "GB/s", "DRAM read roof (from run)" if "bw" in ceil else "spec",
            lambda j: _pick(j, "p1.gemv", "NEON fp32"))
        add("fp16 tok/s vs fp32, best", 2.0, "x", "half the bytes per weight",
            lambda j: _gemv_tok_ratio(j))
    elif lab == "quant":
        emax = 2.07  # expected max |x| of 32 standard normal samples
        for bits in (8, 4):
            qmax = 2 ** (bits - 1) - 1
            add(f"{bits}-bit block-32 error, Gaussian", emax / qmax / math.sqrt(12), "rel",
                f"step = max/{qmax}; rounding noise = step/sqrt(12); max of 32 normals ~ 2.07",
                lambda j, b=bits: _pick(j, "p2.quant", "block_sweep", "err_gauss",
                                        where=lambda r: r["bits"] == b and r["block"] == 32))
        for fmt, bpw in (("Q8_0", 34 / 32), ("Q4_0", 18 / 32)):
            add(f"{fmt} decode speedup vs fp16", 2 / bpw, "x", f"bytes ratio 2 / {bpw:.3f}",
                lambda j, f=fmt: _qspeed(j, f))
    return P


def _ratio(a, b):
    return a / b if a and b else None


def _gemv_tok_ratio(j):
    # tokens/s is proportional to weights/s: GB/s divided by bytes per weight.
    f16 = _pick(j, "p1.gemv", None, where=lambda r: r["case"].startswith("NEON fp16"))
    f32 = _pick(j, "p1.gemv", "NEON fp32")
    return _ratio(f16 / 2 if f16 else None, f32 / 4 if f32 else None)


def _sat_chains(j):
    rs = _recs(j, "p1.peak", "chains_f32")
    if not rs:
        return None
    best = max(r["median"] for r in rs)
    return min(r["chains"] for r in rs if r["median"] >= 0.9 * best)


def _qspeed(j, fmt):
    rs = [r for r in _recs(j, "p2.quant") if r["case"].startswith("gemv_")]
    if not rs:
        return None
    t = max(r["threads"] for r in rs)
    at = {r["case"][5:]: r["median"] for r in rs if r["threads"] == t}
    return _ratio(at.get(fmt), at.get("fp16"))


def p3_predictions(path: pathlib.Path) -> list[dict]:
    """The service lab predicts and measures in one go; just collect its rows."""
    P = []
    for r in _recs(path, "p3.service"):
        if "decode_pred" not in r:
            continue
        P.append({"label": f"{r['model']} ({r['engine']}) decode", "pred": r["decode_pred"], "unit": "tok/s",
                  "how": "bandwidth / bytes per token", "value": r["decode_meas"]})
        P.append({"label": f"{r['model']} ({r['engine']}) prefill", "pred": r["prefill_pred"], "unit": "tok/s",
                  "how": "TFLOP/s / (2 x params)", "value": r["prefill_meas"]})
    return P


def fmt(v, unit):
    """Number only; the unit is printed in its own column (except 'x', which is part of the number)."""
    if v is None:
        return "?"
    if unit == "rel":
        return f"{v:.4f}"
    if unit == "x":
        return f"{v:.2f}x"
    if unit == "ns":
        return f"{v:.2f}"
    return f"{v:,.1f}" if v < 100 else f"{v:,.0f}"


def unit_label(unit):
    return "" if unit == "x" else unit
