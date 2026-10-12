// R0 feasibility probe: init the private ANE runtime, compile one upstream matmul kernel,
// evaluate it repeatedly, compare with a CPU fp32 reference, and print one JSON line.
// Exit codes: 0 ok, 2 ANE unavailable, 3 compile failed, 4 eval failed, 5 numerical mismatch.
#include "ane_runtime.h"
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ane_lm;

static double now_ms() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}
static uint16_t f32_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }  // truncation
static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
struct Lcg { uint64_t s; explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double uniform() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53); } };
static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); double k = (v.size() - 1) * p; size_t lo = (size_t)k, hi = std::min(lo + 1, v.size() - 1); return v[lo] + (v[hi] - v[lo]) * (k - lo); }
static long rss_bytes() { mach_task_basic_info info; mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT; return task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &c) == KERN_SUCCESS ? (long)info.resident_size : -1; }
static void write_bin(const std::string& p, const void* d, size_t n) { std::ofstream f(p, std::ios::binary); f.write((const char*)d, n); }
static std::string json_escape(const std::string& s) { std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') { o += "\\n"; continue; } o += c; } return o; }

// Capture stdout+stderr of a call into a string (ane_print_model_info prints to stdout).
template <class F> static std::string capture_stdout(F fn) {
    fflush(stdout); fflush(stderr); int saved = dup(1), saved2 = dup(2); char path[] = "/tmp/ane_probe_capXXXXXX"; int fd = mkstemp(path);
    dup2(fd, 1); dup2(fd, 2); fn(); fflush(stdout); fflush(stderr); dup2(saved, 1); dup2(saved2, 2); close(saved); close(saved2);
    lseek(fd, 0, SEEK_SET); std::string out; char buf[4096]; ssize_t n; while ((n = read(fd, buf, sizeof buf)) > 0) out.append(buf, n);
    close(fd); unlink(path); return out;
}

// Find the compile directory ($TMPDIR/<modelId>/model.mil) created after t0.
static std::string find_new_mil(time_t t0, std::string* model_id) {
    const char* tmp = getenv("TMPDIR"); std::string root = tmp ? tmp : "/tmp";
    DIR* d = opendir(root.c_str()); if (!d) return "";
    std::string best; time_t best_t = 0; dirent* e;
    while ((e = readdir(d))) {
        std::string p = root + "/" + e->d_name + "/model.mil"; struct stat st;
        if (stat(p.c_str(), &st) == 0 && st.st_mtime >= t0 && st.st_mtime >= best_t) { best = p; best_t = st.st_mtime; *model_id = e->d_name; }
    }
    closedir(d); return best;
}

int main(int argc, char** argv) {
    int K = 256, N = 512, calls = 100; uint64_t seed = 1; std::string out_dir; bool check_only = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--check") check_only = true;
        else if (a == "--k" && i + 1 < argc) K = atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc) N = atoi(argv[++i]);
        else if (a == "--calls" && i + 1 < argc) calls = atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
    }
    std::ostringstream js; js << "{";
    double t0 = now_ms(); ane_init(); bool avail = ane_available(); double init_ms = now_ms() - t0;
    js << "\"ane_available\":" << (avail ? "true" : "false") << ",\"init_ms\":" << init_ms;
    js << ",\"skip_locks_env\":\"" << (getenv("ANE_USE_LOCKS") ? getenv("ANE_USE_LOCKS") : "") << "\"";
    if (!avail) { js << ",\"error\":\"ANE private runtime unavailable (see stderr)\"}"; puts(js.str().c_str()); return 2; }
    if (check_only) { js << "}"; puts(js.str().c_str()); return 0; }

    ane_set_persist_cache(true);   // keep $TMPDIR/<modelId>/model.mil so we can record the emitted MIL
    // Deterministic weights and input; the same bytes are written to out_dir for the Python reference.
    std::vector<uint16_t> W((size_t)N * K); std::vector<float> Wf((size_t)N * K), x(K), y(N), yref(N, 0.f);
    Lcg rng(seed);
    for (size_t i = 0; i < W.size(); i++) { float v = (float)((rng.uniform() * 2 - 1) * 0.05); W[i] = f32_to_bf16(v); Wf[i] = bf16_to_f32(W[i]); }
    for (int k = 0; k < K; k++) x[k] = (float)(rng.uniform() * 2 - 1);
    for (int n = 0; n < N; n++) { double acc = 0; for (int k = 0; k < K; k++) acc += (double)Wf[(size_t)n * K + k] * x[k]; yref[n] = (float)acc; }

    time_t wall0 = time(nullptr) - 1;
    int compiles_before = ane_compile_count(), cache_before = ane_cache_loads();
    double tc = now_ms(); ANEKernel* k = ane_compile_matmul(W.data(), N, K); double compile_ms = now_ms() - tc;
    js << ",\"shape\":{\"K\":" << K << ",\"N\":" << N << ",\"S_alloc\":" << ANE_SPATIAL << "}";
    js << ",\"compile_ms\":" << compile_ms << ",\"compile_count_delta\":" << (ane_compile_count() - compiles_before)
       << ",\"cache_loads_delta\":" << (ane_cache_loads() - cache_before);
    if (!k) { js << ",\"error\":\"compile failed (see stderr)\"}"; puts(js.str().c_str()); return 3; }
    std::string model_id, mil = find_new_mil(wall0, &model_id);
    js << ",\"model_id\":\"" << model_id << "\",\"mil_path\":\"" << mil << "\"";
    js << ",\"input_surface_bytes\":" << ane_get_input_size(k, 0) << ",\"output_surface_bytes\":" << ane_get_output_size(k, 0)
       << ",\"n_inputs\":" << ane_get_n_inputs(k) << ",\"n_outputs\":" << ane_get_n_outputs(k);
    std::string info = capture_stdout([&] { ane_print_model_info(k); });
    js << ",\"model_info\":\"" << json_escape(info) << "\"";

    // First call separately (any lazy first-dispatch cost), then the timed loop.
    double tf = now_ms(); bool ok = ane_matvec(k, y.data(), x.data(), K, N); double first_ms = now_ms() - tf;
    js << ",\"first_call_ms\":" << first_ms;
    if (!ok) { js << ",\"error\":\"eval failed (see stderr)\"}"; puts(js.str().c_str()); ane_free(k); return 4; }
    std::vector<double> us; us.reserve(calls); int failures = 0;
    for (int i = 0; i < calls; i++) { double t = now_ms(); if (!ane_matvec(k, y.data(), x.data(), K, N)) failures++; us.push_back((now_ms() - t) * 1000); }
    double num = 0, den = 0, maxabs = 0; int nonfinite = 0;
    for (int n = 0; n < N; n++) { if (!std::isfinite(y[n])) nonfinite++; double d = y[n] - yref[n]; num += d * d; den += (double)yref[n] * yref[n]; maxabs = std::max(maxabs, std::fabs(d)); }
    double rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    js << ",\"calls\":" << calls << ",\"eval_failures\":" << failures
       << ",\"call_us\":{\"min\":" << pct(us, 0) << ",\"p50\":" << pct(us, .5) << ",\"p95\":" << pct(us, .95) << ",\"max\":" << pct(us, 1) << "}"
       << ",\"rel_l2\":" << rel_l2 << ",\"max_abs_err\":" << maxabs << ",\"nonfinite\":" << nonfinite
       << ",\"rss_bytes\":" << rss_bytes();
    if (!out_dir.empty()) {
        write_bin(out_dir + "/W.bf16.bin", W.data(), W.size() * 2); write_bin(out_dir + "/x.f32.bin", x.data(), x.size() * 4);
        write_bin(out_dir + "/y.f32.bin", y.data(), y.size() * 4);
        if (!mil.empty()) { std::ifstream in(mil, std::ios::binary); std::ofstream o(out_dir + "/graph.mil", std::ios::binary); o << in.rdbuf(); }
    }
    ane_free(k);
    bool pass = failures == 0 && nonfinite == 0 && rel_l2 <= 0.01;
    js << ",\"pass\":" << (pass ? "true" : "false") << "}";
    puts(js.str().c_str());
    return pass ? 0 : 5;
}
