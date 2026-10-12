"""R2 numerical probe: how the ANE lowers the SwiGLU activation.
Runs sigmoid/silu formulations on the real layer-0 gate values and inside the full fused FFN,
records error vs float64 and evaluate time. Output: results/ane-r2-ffn/silu-variants.json"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np
from . import reference as R

ROOT = Path(__file__).resolve().parents[2]; FIX = ROOT / "prototype/ane/fixtures"; OUT = ROOT / "results/ane-r2-ffn"; work = OUT / "work"
D, I = 1024, 3072
HDR = 'program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{"coremlc-version", "3505.4.1"}})]\n{\n'
CONSTS = ('        tensor<string, []> pt = const()[name = tensor<string, []>("pt"), val = tensor<string, []>("valid")];\n'
          '        tensor<int32, [2]> st = const()[name = tensor<string, []>("st"), val = tensor<int32, [2]>([1, 1])];\n'
          '        tensor<int32, [4]> pd = const()[name = tensor<string, []>("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
          '        tensor<int32, [2]> dl = const()[name = tensor<string, []>("dl"), val = tensor<int32, [2]>([1, 1])];\n'
          '        tensor<int32, []> gr = const()[name = tensor<string, []>("gr"), val = tensor<int32, []>(1)];\n'
          '        tensor<fp16, []> half = const()[name = tensor<string, []>("half"), val = tensor<fp16, []>(0.5)];\n'
          '        tensor<fp16, []> one = const()[name = tensor<string, []>("one"), val = tensor<fp16, []>(1.0)];\n'
          '        tensor<fp16, []> mone = const()[name = tensor<string, []>("mone"), val = tensor<fp16, []>(-1.0)];\n')
T = lambda C: f"tensor<fp16, [1, {C}, 1, 32]>"
def wconst(nm, N, K): return f'        tensor<fp16, [{N}, {K}, 1, 1]> {nm.upper()} = const()[name = tensor<string, []>("{nm.upper()}"), val = tensor<fp16, [{N}, {K}, 1, 1]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/{nm}.bin"), offset = tensor<uint64, []>(64)))];\n'
def conv(out, N, w, x): return f'        {T(N)} {out} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = {w}, x = {x})[name = tensor<string, []>("{out}")];\n'
# activation formulations: each maps input tensor `g` to output `y` (silu) ; C channels
ACT = {
    "sigmoid_mul": lambda C, g, y: f'        {T(C)} sg = sigmoid(x = {g})[name = tensor<string, []>("sg")];\n        {T(C)} {y} = mul(x = {g}, y = sg)[name = tensor<string, []>("{y}")];\n',
    "silu_op":     lambda C, g, y: f'        {T(C)} {y} = silu(x = {g})[name = tensor<string, []>("{y}")];\n',
    "tanh_form":   lambda C, g, y: (f'        {T(C)} gh = mul(x = {g}, y = half)[name = tensor<string, []>("gh")];\n        {T(C)} th = tanh(x = gh)[name = tensor<string, []>("th")];\n'
                                    f'        {T(C)} t1 = add(x = th, y = one)[name = tensor<string, []>("t1")];\n        {T(C)} sg = mul(x = t1, y = half)[name = tensor<string, []>("sg")];\n        {T(C)} {y} = mul(x = {g}, y = sg)[name = tensor<string, []>("{y}")];\n'),
    "exp_form":    lambda C, g, y: (f'        {T(C)} ng = mul(x = {g}, y = mone)[name = tensor<string, []>("ng")];\n        {T(C)} ex = exp(x = ng)[name = tensor<string, []>("ex")];\n'
                                    f'        {T(C)} den = add(x = ex, y = one)[name = tensor<string, []>("den")];\n        {T(C)} sg = real_div(x = {g}, y = den)[name = tensor<string, []>("sg")];\n        {T(C)} {y} = {"sg" if False else "sg"}\n'),
}
ACT["exp_form"] = lambda C, g, y: (f'        {T(C)} ng = mul(x = {g}, y = mone)[name = tensor<string, []>("ng")];\n        {T(C)} ex = exp(x = ng)[name = tensor<string, []>("ex")];\n'
                                    f'        {T(C)} den = add(x = ex, y = one)[name = tensor<string, []>("den")];\n        {T(C)} {y} = real_div(x = {g}, y = den)[name = tensor<string, []>("{y}")];\n')

def pack(A): S, C = A.shape; P = np.zeros((C, 32), np.float16); P[:, :S] = A.T.astype(np.float16); return P.tobytes()
def unpack(p, C, S): return np.fromfile(p, dtype=np.float32).reshape(C, 32)[:, :S].T
def run(tag, body, cin, cout, inp, weights, calls=200):
    mil = HDR + f"    func main<ios16>({T(cin)} x) {{\n" + CONSTS + "".join(wconst(n, N, K) for n, N, K in weights) + body + "    } -> (y);\n}\n"
    (work / f"{tag}.mil").write_text(mil); (work / f"{tag}.in").write_bytes(pack(inp))
    spec = {"mil": str(work / f"{tag}.mil"), "weights": [{"name": n, "bf16": str(FIX / f"{n}_proj.bf16.bin"), "numel": N * K} for n, N, K in weights],
            "input_bytes": cin * 64, "output_bytes": cout * 64, "input": str(work / f"{tag}.in"), "output": str(work / f"{tag}.out"), "calls": calls}
    (work / f"{tag}.spec").write_text(json.dumps(spec, ensure_ascii=False))
    p = subprocess.run([str(ROOT / "prototype/ane/build/mil_main"), str(work / f"{tag}.spec")], capture_output=True, text=True)
    if p.returncode: return None, {"error": p.stderr[-300:], "exit": p.returncode}
    return unpack(work / f"{tag}.out", cout, 32), json.loads(p.stdout.strip().splitlines()[-1])

def main():
    Wg = np.load(FIX / "gate_proj.f32.npy"); Wu = np.load(FIX / "up_proj.f32.npy"); Wd = np.load(FIX / "down_proj.f32.npy")
    X = np.fromfile(work / "S32_s1_real.x.bin", dtype=np.float32).reshape(32, D)
    x = X.astype(np.float64); g = x @ Wg.T; u = x @ Wu.T; sig = 1 / (1 + np.exp(-g)); silu = g * sig; y = (silu * u) @ Wd.T
    res = {"input": "real layer-0 activations, S=32", "gate_value_range": [float(g.min()), float(g.max())], "variants": {}}
    # error shape of the plain sigmoid op
    Y, _ = run("sig_only", f'        {T(I)} y = sigmoid(x = x)[name = tensor<string, []>("y")];\n', I, I, g.astype(np.float32), [], calls=3)
    if Y is not None:
        err = Y.astype(np.float64) - sig; bins = [-np.inf, -6, -4, -2, 0, 2, 4, np.inf]
        res["sigmoid_op_abs_error_by_gate_bin"] = {f"g in [{bins[i]},{bins[i+1]})": {"max_abs": float(np.abs(err[m]).max()) if m.any() else None, "n": int(m.sum())} for i in range(len(bins) - 1) for m in [(g >= bins[i]) & (g < bins[i+1])]}
    for name, act in ACT.items():
        v = {}
        Y, m = run(f"act_{name}", act(I, "x", "y"), I, I, g.astype(np.float32), [], calls=3)          # activation alone on true g
        v["silu_alone"] = R.errors(Y, silu) if Y is not None else m
        Y, m = run(f"full_{name}", conv("gate", I, "GATE", "x") + conv("up", I, "UP", "x") + act(I, "gate", "sl") + f'        {T(I)} fu = mul(x = sl, y = up)[name = tensor<string, []>("fu")];\n' + conv("y", D, "DOWN", "fu"),
                   D, D, X, [("gate", I, D), ("up", I, D), ("down", D, I)], calls=300)
        v["full_ffn"] = R.errors(Y, y) if Y is not None else m; v["full_ffn_timing"] = m
        res["variants"][name] = v
        e = v["full_ffn"]; print(f"{name:12s} silu-alone relL2 {v['silu_alone'].get('rel_l2', 'fail')!s:>10}  full-FFN relL2 {e.get('rel_l2', 'fail')!s:>10}  normMax {e.get('norm_max_err','-')!s:>10}  eval {m.get('eval_us_mean','-')} us  compile {m.get('compile_us','-')} us")
    (OUT / "silu-variants.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res.get("sigmoid_op_abs_error_by_gate_bin"), indent=1))

if __name__ == "__main__":
    main()
