// R1 driver. Usage: bench_main spec.json
// spec: {"path":"upstream"|"visible"|"upstream_matvec","K":..,"N":..,"S":..,"weight_bf16":"W.bin",
//        "cases":[{"name":..,"x":"x.f32.bin","y_out":"y.f32.bin"}],"warmup":n,"calls":n,
//        "sentinel":0.0,"measurements_out":"m.jsonl","mil_out":"graph.mil","summary_out":"s.json"}
// Each case: X is fp32 row-major [S,K]; y_out receives fp32 row-major [S_alloc,N] (all lanes).
// Timed loop runs on case 0. One process per experiment; process start is never counted.
#include "lowerings.h"
#include <mach/mach_time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ane_lm;

static double now_us() { static mach_timebase_info_data_t tb = {0, 0}; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e3; }
static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
static void spit(const std::string& p, const void* d, size_t n) { std::ofstream f(p, std::ios::binary); f.write((const char*)d, n); }
// Minimal JSON getters for our own flat spec (no third-party dependency for a 12-key file).
static std::string jstr(const std::string& j, const std::string& key, size_t from = 0) {
    size_t k = j.find("\"" + key + "\"", from); if (k == std::string::npos) return ""; size_t c = j.find(':', k), q1 = j.find('"', c); size_t q2 = j.find('"', q1 + 1); return j.substr(q1 + 1, q2 - q1 - 1); }
static double jnum(const std::string& j, const std::string& key, double dflt) {
    size_t k = j.find("\"" + key + "\""); if (k == std::string::npos) return dflt; return atof(j.c_str() + j.find(':', k) + 1); }
static std::string find_new_mil(time_t t0, std::string* id) {
    const char* tmp = getenv("TMPDIR"); std::string root = tmp ? tmp : "/tmp"; DIR* d = opendir(root.c_str()); if (!d) return "";
    std::string best; time_t bt = 0; dirent* e;
    while ((e = readdir(d))) { std::string p = root + "/" + e->d_name + "/model.mil"; struct stat st;
        if (stat(p.c_str(), &st) == 0 && st.st_mtime >= t0 && st.st_mtime >= bt) { best = p; bt = st.st_mtime; *id = e->d_name; } }
    closedir(d); return best; }
static uint64_t fnv(const std::vector<float>& v) { uint64_t h = 1469598103934665603ULL; for (float f : v) { uint32_t u; memcpy(&u, &f, 4); h ^= u; h *= 1099511628211ULL; } return h; }

struct Call { double pack, handoff, eval, read, transpose, unpack, total; uint64_t checksum; bool ok; };

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_main spec.json\n"); return 1; }
    std::string spec = slurp(argv[1]);
    lowering::Projection p; p.K = (int)jnum(spec, "K", 0); p.N = (int)jnum(spec, "N", 0); p.S = (int)jnum(spec, "S", 1);
    std::string path = jstr(spec, "path"), wpath = jstr(spec, "weight_bf16");
    int warmup = (int)jnum(spec, "warmup", 5), calls = (int)jnum(spec, "calls", 100);
    float sentinel = (float)jnum(spec, "sentinel", 0.0);
    std::string m_out = jstr(spec, "measurements_out"), mil_out = jstr(spec, "mil_out"), s_out = jstr(spec, "summary_out");
    if (p.K <= 0 || p.N <= 0 || p.S < 1 || p.S > p.S_alloc) { fprintf(stderr, "bad shape\n"); return 1; }
    if (path == "upstream_matvec" && p.S != 1) { fprintf(stderr, "upstream_matvec requires S=1\n"); return 1; }

    std::string wbytes = slurp(wpath);
    if (wbytes.size() != (size_t)p.N * p.K * 2) { fprintf(stderr, "weight size %zu != N*K*2\n", wbytes.size()); return 1; }
    const uint16_t* W = (const uint16_t*)wbytes.data();

    // cases
    std::vector<std::pair<std::string, std::string>> cases;   // (x path, y_out path)
    for (size_t pos = spec.find("\"cases\""); (pos = spec.find("\"x\"", pos + 1)) != std::string::npos;) {
        std::string x = jstr(spec, "x", pos), y = jstr(spec, "y_out", pos); if (x.empty()) break; cases.push_back({x, y}); pos = spec.find("}", pos); }
    if (cases.empty()) { fprintf(stderr, "no cases\n"); return 1; }

    ane_init(); if (!ane_available()) { fprintf(stderr, "ANE unavailable\n"); return 2; }
    ane_set_persist_cache(true);
    time_t wall0 = time(nullptr) - 1; int comp0 = ane_compile_count(), cache0 = ane_cache_loads();
    double tc = now_us();
    ANEKernel* k = (path == "visible") ? lowering::compile_visible(p, W) : ane_compile_matmul(W, p.N, p.K);
    double compile_us = now_us() - tc;
    if (!k) { fprintf(stderr, "compile failed\n"); return 3; }
    std::string model_id, mil = find_new_mil(wall0, &model_id);
    if (!mil_out.empty() && !mil.empty()) spit(mil_out, slurp(mil).data(), slurp(mil).size());
    if (ane_get_input_size(k, 0) != p.input_bytes() || ane_get_output_size(k, 0) != p.output_bytes()) {
        fprintf(stderr, "surface size mismatch: in %zu/%zu out %zu/%zu\n", ane_get_input_size(k, 0), p.input_bytes(), ane_get_output_size(k, 0), p.output_bytes()); return 3; }

    std::vector<uint16_t> staging; std::vector<float> dense(p.output_elems()), Yfull, X((size_t)p.S * p.K);
    auto one_call = [&](const float* Xp, std::vector<float>& Yout) -> Call {
        Call c{}; double t0 = now_us();
        if (path == "upstream_matvec") {
            std::vector<float> y(p.N); double te = now_us(); c.ok = ane_matvec(k, y.data(), Xp, p.K, p.N); c.eval = now_us() - te;   // pack+handoff+eval+unpack inside upstream
            Yout.assign((size_t)p.S_alloc * p.N, 0.f); for (int n = 0; n < p.N; n++) Yout[n] = y[n];
        } else {
            double t = now_us(); lowering::pack(p, Xp, sentinel, staging); c.pack = now_us() - t;
            t = now_us(); bool w = ane_write_surface_raw(k, 0, staging.data(), staging.size() * 2); c.handoff = now_us() - t;
            t = now_us(); bool e = w && ane_eval_qos(k, 21); c.eval = now_us() - t;   // synchronous evaluate: sync is inside this number
            t = now_us(); bool r = e && ane_read_output_raw(k, 0, dense.data(), (int)dense.size()); c.read = now_us() - t;   // lock + fp16->fp32 of the whole surface
            t = now_us(); lowering::unpack_full(p, dense.data(), Yout); c.transpose = now_us() - t;                                  // [N,S_alloc] -> [S_alloc,N]
            c.unpack = c.read + c.transpose;
            c.ok = w && e && r;
        }
        c.total = now_us() - t0; c.checksum = fnv(Yout); return c;
    };

    // numerics: every case once, write full-lane Y
    for (auto& cs : cases) {
        std::string xb = slurp(cs.first); if (xb.size() != X.size() * 4) { fprintf(stderr, "x size mismatch %s\n", cs.first.c_str()); return 1; }
        memcpy(X.data(), xb.data(), xb.size());
        Call c = one_call(X.data(), Yfull); if (!c.ok) { fprintf(stderr, "eval failed on %s\n", cs.first.c_str()); return 4; }
        if (!cs.second.empty()) spit(cs.second, Yfull.data(), Yfull.size() * 4);
    }
    // timing: case 0
    { std::string xb = slurp(cases[0].first); memcpy(X.data(), xb.data(), xb.size()); }
    double tf = now_us(); Call first = one_call(X.data(), Yfull); double first_us = now_us() - tf; uint64_t ref_ck = first.checksum;
    for (int i = 0; i < warmup; i++) one_call(X.data(), Yfull);
    std::ofstream mf(m_out.empty() ? "/dev/null" : m_out); int failures = 0, drift = 0;
    for (int i = 0; i < calls; i++) {
        Call c = one_call(X.data(), Yfull); if (!c.ok) failures++; if (c.checksum != ref_ck) drift++;
        mf << "{\"i\":" << i << ",\"pack_us\":" << c.pack << ",\"handoff_us\":" << c.handoff << ",\"eval_us\":" << c.eval
           << ",\"read_us\":" << c.read << ",\"transpose_us\":" << c.transpose << ",\"unpack_us\":" << c.unpack << ",\"call_us\":" << c.total << ",\"ok\":" << (c.ok ? "true" : "false") << ",\"checksum\":\"" << std::hex << c.checksum << std::dec << "\"}\n";
    }
    std::ostringstream js;
    js << "{\"path\":\"" << path << "\",\"index_map\":" << lowering::json_index_map(p) << ",\"compile_us\":" << compile_us
       << ",\"compiled\":" << (ane_compile_count() - comp0) << ",\"cache_loaded\":" << (ane_cache_loads() - cache0)
       << ",\"model_id\":\"" << model_id << "\",\"first_call_us\":" << first_us << ",\"warmup\":" << warmup << ",\"calls\":" << calls
       << ",\"failures\":" << failures << ",\"checksum_drift\":" << drift << ",\"sentinel\":" << sentinel << ",\"cases\":" << cases.size() << "}";
    if (!s_out.empty()) spit(s_out, js.str().data(), js.str().size());
    puts(js.str().c_str());
    ane_free(k);
    return (failures || drift) ? 5 : 0;
}
