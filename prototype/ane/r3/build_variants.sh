#!/usr/bin/env bash
# Build two instrumented ane.cpp variants from the pinned vendor tree (outside the workspace: CMake's fetcher
# cannot handle colons in paths):
#   trace : upstream + passive trace hook (ANE_TRACE_PATH) + kernel-map log + ANE_FORCE_PLAIN_FFN runtime switch
#   exp   : trace + patches-proposed/0001 (SiLU as g/(1+exp(-g)) in mil_gen_fused_ffn / chunk_accum)
# Patch records are written to r3/patches/. Binaries linked as r3/bin/ane-<variant>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ANE="$(dirname "$HERE")"; VENDOR="$ANE/vendor/ane.cpp"
mkdir -p "$HERE/patches" "$HERE/bin"
for v in trace exp; do
  SRC="$HOME/.cache/ane-cpp-src-$v"; BLD="$HOME/.cache/ane-cpp-build-$v"
  rm -rf "$SRC"; cp -R "$VENDOR" "$SRC"; git -C "$SRC" checkout -q -- .
  python3 - "$SRC" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1])
g = src / "generate.cpp"; s = g.read_text()
hook = r'''
// --- R3 trace hook (passive): ANE_TRACE_PATH=<file> appends one JSON line per step with the top-16 logits.
#include <algorithm>
static FILE* r3_trace_file() { static FILE* f = nullptr; static bool init = false; if (!init) { init = true; const char* p = getenv("ANE_TRACE_PATH"); if (p) f = fopen(p, "a"); } return f; }
static void r3_trace_prompt(const std::vector<int>& ids) { FILE* f = r3_trace_file(); if (!f) return; fprintf(f, "{\"prompt_ids\":["); for (size_t i = 0; i < ids.size(); i++) fprintf(f, "%s%d", i ? "," : "", ids[i]); fprintf(f, "]}\n"); }
static void r3_trace_step(int step, const float* logits, int vocab, int token) {
    FILE* f = r3_trace_file(); if (!f) return;
    std::vector<int> idx(vocab); for (int i = 0; i < vocab; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 16, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    fprintf(f, "{\"step\":%d,\"token\":%d,\"top\":[", step, token);
    for (int i = 0; i < 16; i++) fprintf(f, "%s[%d,%.6g]", i ? "," : "", idx[i], logits[idx[i]]);
    fprintf(f, "]}\n"); fflush(f);
}
'''
assert "void stream_generate(" in s
s = s.replace("void stream_generate(", hook + "\nvoid stream_generate(", 1)
s = s.replace("    std::vector<int> prompt_tokens = tokenizer.encode(formatted);\n", "    std::vector<int> prompt_tokens = tokenizer.encode(formatted);\n    r3_trace_prompt(prompt_tokens);\n", 1)
old = "        int next_token = sample_token(logits, sampler_vocab, sampling, generated_tokens);\n        if (is_stop_token(next_token, tokenizer)) {\n            break;\n        }\n\n        emit_token(next_token);"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "        int next_token = sample_token(logits, sampler_vocab, sampling, generated_tokens);\n        r3_trace_step(n_generated, logits, sampler_vocab, next_token);\n        if (is_stop_token(next_token, tokenizer)) {\n            break;\n        }\n\n        emit_token(next_token);")
g.write_text(s)
q = src / "models/llm/qwen3.cpp"; s = q.read_text()
old = "            // Try fused FFN + residual add first\n            if (use_blobs) {"
assert s.count(old) == 1
s = s.replace(old, "            // Try fused FFN + residual add first (R3: ANE_FORCE_PLAIN_FFN=1 skips it so the plain fused FFN is exercised)\n            static const bool r3_force_plain = getenv(\"ANE_FORCE_PLAIN_FFN\") != nullptr;\n            if (r3_force_plain) {\n                ane_layers_[L].ffn_resadd = nullptr;\n            } else if (use_blobs) {")
old2 = "        if (!ane_layers_[L].ffn_resadd && !ane_layers_[L].fused_ffn) {\n            if (ffn_chunks <= 1) ffn_chunks = 2;"
assert s.count(old2) == 1
s = s.replace(old2, "        if (ane_layers_[L].ffn_resadd) r3_ffn_resadd++; else if (ane_layers_[L].fused_ffn) r3_ffn_fused++; else r3_ffn_chunked++;\n" + old2)
# counters declared before the layer loop: find the LOG("Compiling Qwen3 ANE kernels
old3 = '    LOG("Compiling Qwen3 ANE kernels%s...\\n", use_blobs ? " (from blobs)" : "");'
assert s.count(old3) == 1
s = s.replace(old3, old3 + "\n    int r3_ffn_resadd = 0, r3_ffn_fused = 0, r3_ffn_chunked = 0;")
old4 = '    LOG("  %d ANE layer kernels ready'
assert s.count(old4) == 1
s = s.replace(old4, '    fprintf(stderr, "R3 FFN kernel map: resadd=%d fused=%d chunked=%d force_plain=%d\\n", r3_ffn_resadd, r3_ffn_fused, r3_ffn_chunked, getenv("ANE_FORCE_PLAIN_FFN") != nullptr);\n' + old4)
q.write_text(s)
# ---- same instrumentation for the Qwen3.5 loader
q = src / "models/llm/qwen3_5.cpp"; s = q.read_text()
old = "            // Try fused FFN + residual add first\n            if (use_blobs) {"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "            // Try fused FFN + residual add first (R3: ANE_FORCE_PLAIN_FFN=1 skips it so the plain fused FFN is exercised)\n            static const bool r3_force_plain = getenv(\"ANE_FORCE_PLAIN_FFN\") != nullptr;\n            if (r3_force_plain) {\n                ane_layers_[L].ffn_resadd = nullptr;\n            } else if (use_blobs) {")
old2 = "        if (!ane_layers_[L].ffn_resadd && !ane_layers_[L].fused_ffn) {\n            // Fall back to chunked FFN"
assert s.count(old2) == 1
s = s.replace(old2, "        if (ane_layers_[L].ffn_resadd) r3_ffn_resadd++; else if (ane_layers_[L].fused_ffn) r3_ffn_fused++; else r3_ffn_chunked++;\n" + old2)
old3 = '    LOG("Compiling ANE kernels%s...\\n", use_blobs ? " (from blobs)" : "");'
assert s.count(old3) == 1
s = s.replace(old3, old3 + "\n    int r3_ffn_resadd = 0, r3_ffn_fused = 0, r3_ffn_chunked = 0;")
old4 = '    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d)\\n",'
assert s.count(old4) == 1
s = s.replace(old4, '    fprintf(stderr, "R3 FFN kernel map: resadd=%d fused=%d chunked=%d force_plain=%d\\n", r3_ffn_resadd, r3_ffn_fused, r3_ffn_chunked, getenv("ANE_FORCE_PLAIN_FFN") != nullptr);\n' + old4)
q.write_text(s)
# ---- main.cpp: --prompts-file batch mode (one process, model reset between cases, case markers on stderr and in the trace)
m = src / "main.cpp"; s = m.read_text()
s = s.replace('    const char* prompt = "Hello";\n', '    const char* prompt = "Hello";\n    const char* prompts_file = nullptr;   // R3: JSON array of prompts, run sequentially in one process\n', 1)
old = '        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {\n            args.prompt = argv[++i];\n'
assert s.count(old) == 1
s = s.replace(old, old + '        } else if (strcmp(argv[i], "--prompts-file") == 0 && i + 1 < argc) {\n            args.prompts_file = argv[++i];\n')
old = 'static int cmd_generate(LLMModel& model, Tokenizer& tokenizer,\n                        DraftModelContext* draft, const Args& args) {\n'
assert s.count(old) == 1
batch = (
    '    if (args.prompts_file) {\n'
    '        std::ifstream pf(args.prompts_file);\n'
    '        nlohmann::json arr = nlohmann::json::parse(pf);\n'
    '        int i = 0;\n'
    '        for (auto& p : arr) {\n'
    '            std::string ps = p.get<std::string>();\n'
    '            fprintf(stderr, "### CASE %d\\n", i);\n'
    '            if (const char* tp = getenv("ANE_TRACE_PATH")) { FILE* tf = fopen(tp, "a"); if (tf) { fprintf(tf, "{\\"case\\":%d}\\n", i); fclose(tf); } }\n'
    '            model.reset();\n'
    '            if (draft && draft->model) draft->model->reset();\n'
    '            Args one = args; one.prompt = ps.c_str(); one.prompts_file = nullptr;\n'
    '            cmd_generate(model, tokenizer, draft, one);\n'
    '            i++;\n'
    '        }\n'
    '        return 0;\n'
    '    }\n')
s = s.replace(old, old + batch)
if "#include <fstream>" not in s: s = s.replace("#include <nlohmann/json.hpp>", "#include <nlohmann/json.hpp>\n#include <fstream>", 1)
m.write_text(s)
PY
  if [ "$v" = "exp" ]; then git -C "$SRC" apply "$ANE/vendor/patches-proposed/0001-ffn-silu-exp-form.patch"; fi
  git -C "$SRC" diff > "$HERE/patches/$v.patch"
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$(xcrun -f clang)" -DCMAKE_CXX_COMPILER="$(xcrun -f clang++)" -Wno-dev > "$BLD.configure.log" 2>&1
  cmake --build "$BLD" > "$BLD.build.log" 2>&1 || { echo "build $v FAILED, see $BLD.build.log"; tail -20 "$BLD.build.log"; exit 1; }
  ln -sfn "$BLD/ane.cpp" "$HERE/bin/ane-$v"
  echo "built $v: $(wc -l < "$HERE/patches/$v.patch") patch lines -> $HERE/bin/ane-$v"
done
ln -sfn "$HOME/.cache/ane-cpp-build-full/ane.cpp" "$HERE/bin/ane-upstream"
