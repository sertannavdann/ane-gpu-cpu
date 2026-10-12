// R2 driver: SwiGLU FFN on the ANE. Baseline = upstream fused kernel (one dispatch).
// Change under test = upstream chunked-accumulate kernels (N dispatches over intermediate channels).
// spec: {"mode":"fused"|"chunked"|"fused_matvec"|"chunked_matvec","chunks":N,"dim":D,"inter":I,"S":s,
//        "wg":"gate.bf16.bin","wu":"up.bf16.bin","wd":"down.bf16.bin",
//        "cases":[{"name":..,"x":"x.f32.bin","y_out":"y.f32.bin"}],"warmup":n,"calls":n,
//        "measurements_out":..,"summary_out":..,"mil_out":..}
// x is fp32 [S,D]; y_out is fp32 [32,D] (all lanes). *_matvec modes are upstream's own S=1 call path.
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
static std::string jstr(const std::string& j, const std::string& key, size_t from = 0) {
    size_t k = j.find("\"" + key + "\"", from); if (k == std::string::npos) return ""; size_t c = j.find(':', k), q1 = j.find('"', c); size_t q2 = j.find('"', q1 + 1); return j.substr(q1 + 1, q2 - q1 - 1); }
static double jnum(const std::string& j, const std::string& key, double dflt) {
    size_t k = j.find("\"" + key + "\""); if (k == std::string::npos) return dflt; return atof(j.c_str() + j.find(':', k) + 1); }
static uint64_t fnv(const std::vector<float>& v) { uint64_t h = 1469598103934665603ULL; for (float f : v) { uint32_t u; memcpy(&u, &f, 4); h ^= u; h *= 1099511628211ULL; } return h; }
static std::vector<std::string> new_mils(time_t t0) {
    const char* tmp = getenv("TMPDIR"); std::string root = tmp ? tmp : "/tmp"; DIR* d = opendir(root.c_str()); std::vector<std::string> out; if (!d) return out; dirent* e;
    while ((e = readdir(d))) { std::string p = root + "/" + e->d_name + "/model.mil"; struct stat st; if (stat(p.c_str(), &st) == 0 && st.st_mtime >= t0) out.push_back(p); }
    closedir(d); return out; }

struct Call { double pack = 0, handoff = 0, eval = 0, read = 0, transpose = 0, total = 0; int dispatches = 0; uint64_t checksum = 0; bool ok = false; };

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ffn_main spec.json\n"); return 1; }
    std::string spec = slurp(argv[1]);
    int D = (int)jnum(spec, "dim", 0), I = (int)jnum(spec, "inter", 0), S = (int)jnum(spec, "S", 1), chunks = (int)jnum(spec, "chunks", 1);
    int warmup = (int)jnum(spec, "warmup", 5), calls = (int)jnum(spec, "calls", 100);
    std::string mode = jstr(spec, "mode"), m_out = jstr(spec, "measurements_out"), s_out = jstr(spec, "summary_out"), mil_out = jstr(spec, "mil_out");
    std::string wg = slurp(jstr(spec, "wg")), wu = slurp(jstr(spec, "wu")), wd = slurp(jstr(spec, "wd"));
    if (D <= 0 || I <= 0 || S < 1 || S > ANE_SPATIAL) { fprintf(stderr, "bad shape\n"); return 1; }
    if (wg.size() != (size_t)I * D * 2 || wu.size() != (size_t)I * D * 2 || wd.size() != (size_t)D * I * 2) { fprintf(stderr, "weight sizes wrong\n"); return 1; }
    bool matvec = mode.find("matvec") != std::string::npos, chunked = mode.rfind("chunked", 0) == 0;
    std::string variant = mode == "fused_exp" ? "exp" : mode == "fused_vis" ? "sigmoid" : "";   // our emitter vs upstream compile
    if (matvec && S != 1) { fprintf(stderr, "matvec modes need S=1\n"); return 1; }
    if (!chunked) chunks = 1;
    std::vector<std::pair<std::string, std::string>> cases;
    for (size_t pos = spec.find("\"cases\""); (pos = spec.find("\"x\"", pos + 1)) != std::string::npos;) {
        std::string x = jstr(spec, "x", pos), y = jstr(spec, "y_out", pos); if (x.empty()) break; cases.push_back({x, y}); pos = spec.find("}", pos); }

    ane_init(); if (!ane_available()) return 2;
    ane_set_persist_cache(true);
    time_t wall0 = time(nullptr) - 1; int comp0 = ane_compile_count(), cache0 = ane_cache_loads();
    const uint16_t *G = (const uint16_t*)wg.data(), *U = (const uint16_t*)wu.data(), *Wd = (const uint16_t*)wd.data();
    double tc = now_us();
    ANEKernel* fused = nullptr; ChunkedFFN cffn = {};
    if (chunked) { if (!ane_compile_chunked_ffn(&cffn, G, U, Wd, D, I, chunks)) { fprintf(stderr, "chunked compile failed\n"); return 3; } }
    else if (!variant.empty()) { fused = lowering::compile_ffn(D, I, variant, G, U, Wd); if (!fused) { fprintf(stderr, "variant compile failed\n"); return 3; } }
    else { fused = ane_compile_fused_ffn(G, U, Wd, D, I); if (!fused) { fprintf(stderr, "fused compile failed\n"); return 3; } }
    double compile_us = now_us() - tc;
    auto mils = new_mils(wall0);
    if (!mil_out.empty() && !mils.empty()) { std::string t = slurp(mils[0]); spit(mil_out, t.data(), t.size()); }

    lowering::Projection px; px.S = S; px.K = D; px.N = D;      // pack/unpack helpers for [S,D] lanes
    std::vector<uint16_t> stage, packed((size_t)2 * D * ANE_SPATIAL); std::vector<float> dense((size_t)D * ANE_SPATIAL), acc((size_t)D * ANE_SPATIAL), Y, X((size_t)S * D);
    size_t weight_bytes = ((size_t)2 * I * D + (size_t)D * I) * 2;

    auto one_call = [&](const float* Xp, std::vector<float>& Yout) -> Call {
        Call c; double t0 = now_us();
        if (mode == "fused_matvec") {
            std::vector<float> y(D); double t = now_us(); c.ok = ane_matvec(fused, y.data(), Xp, D, D); c.eval = now_us() - t; c.dispatches = 1;
            Yout.assign((size_t)ANE_SPATIAL * D, 0.f); memcpy(Yout.data(), y.data(), D * 4);
        } else if (mode == "chunked_matvec") {
            std::vector<float> y(D); double t = now_us(); c.ok = ane_eval_chunked_ffn(&cffn, y.data(), Xp); c.eval = now_us() - t; c.dispatches = chunks;
            Yout.assign((size_t)ANE_SPATIAL * D, 0.f); memcpy(Yout.data(), y.data(), D * 4);
        } else if (mode == "fused" || !variant.empty()) {
            double t = now_us(); lowering::pack(px, Xp, 0.f, stage); c.pack = now_us() - t;
            t = now_us(); bool w = ane_write_surface_raw(fused, 0, stage.data(), stage.size() * 2); c.handoff = now_us() - t;
            t = now_us(); bool e = w && ane_eval_qos(fused, 21); c.eval = now_us() - t; c.dispatches = 1;
            t = now_us(); bool r = e && ane_read_output_raw(fused, 0, dense.data(), (int)dense.size()); c.read = now_us() - t;
            t = now_us(); lowering::unpack_full(px, dense.data(), Yout); c.transpose = now_us() - t; c.ok = w && e && r;
        } else {   // chunked, S lanes: packed input = [x planes (D) | acc planes (D)], acc chained through chunks on the host
            double t = now_us(); lowering::pack(px, Xp, 0.f, stage); std::fill(acc.begin(), acc.end(), 0.f); c.pack = now_us() - t; c.ok = true;
            for (int ci = 0; ci < chunks && c.ok; ci++) {
                ANEKernel* k = cffn.chunks[ci];
                t = now_us();
                memcpy(packed.data(), stage.data(), stage.size() * 2);
                for (size_t i = 0; i < acc.size(); i++) packed[stage.size() + i] = lowering::f32_to_f16_bits(acc[i]);
                c.pack += now_us() - t;
                t = now_us(); bool w = ane_write_surface_raw(k, 0, packed.data(), packed.size() * 2); c.handoff += now_us() - t;
                t = now_us(); bool e = w && ane_eval_qos(k, 21); c.eval += now_us() - t; c.dispatches++;
                t = now_us(); bool r = e && ane_read_output_raw(k, 0, acc.data(), (int)acc.size()); c.read += now_us() - t;
                c.ok = w && e && r;
            }
            t = now_us(); lowering::unpack_full(px, acc.data(), Yout); c.transpose = now_us() - t;
        }
        c.total = now_us() - t0; c.checksum = fnv(Yout); return c;
    };

    for (auto& cs : cases) {
        std::string xb = slurp(cs.first); if (xb.size() != X.size() * 4) { fprintf(stderr, "x size mismatch %s\n", cs.first.c_str()); return 1; }
        memcpy(X.data(), xb.data(), xb.size());
        Call c = one_call(X.data(), Y); if (!c.ok) { fprintf(stderr, "eval failed on %s\n", cs.first.c_str()); return 4; }
        if (!cs.second.empty()) spit(cs.second, Y.data(), Y.size() * 4);
    }
    { std::string xb = slurp(cases[0].first); memcpy(X.data(), xb.data(), xb.size()); }
    double tf = now_us(); Call first = one_call(X.data(), Y); double first_us = now_us() - tf; uint64_t ref = first.checksum;
    for (int i = 0; i < warmup; i++) one_call(X.data(), Y);
    std::ofstream mf(m_out.empty() ? "/dev/null" : m_out); int failures = 0, drift = 0;
    for (int i = 0; i < calls; i++) {
        Call c = one_call(X.data(), Y); if (!c.ok) failures++; if (c.checksum != ref) drift++;
        mf << "{\"i\":" << i << ",\"pack_us\":" << c.pack << ",\"handoff_us\":" << c.handoff << ",\"eval_us\":" << c.eval << ",\"read_us\":" << c.read
           << ",\"transpose_us\":" << c.transpose << ",\"call_us\":" << c.total << ",\"dispatches\":" << c.dispatches << ",\"ok\":" << (c.ok ? "true" : "false") << "}\n";
    }
    std::ostringstream js;
    js << "{\"mode\":\"" << mode << "\",\"chunks\":" << chunks << ",\"dim\":" << D << ",\"inter\":" << I << ",\"S\":" << S << ",\"weight_bytes\":" << weight_bytes
       << ",\"compile_us\":" << compile_us << ",\"compiled\":" << (ane_compile_count() - comp0) << ",\"cache_loaded\":" << (ane_cache_loads() - cache0)
       << ",\"kernels\":" << (chunked ? chunks : 1) << ",\"first_call_us\":" << first_us << ",\"warmup\":" << warmup << ",\"calls\":" << calls
       << ",\"failures\":" << failures << ",\"checksum_drift\":" << drift << ",\"cases\":" << cases.size() << "}";
    if (!s_out.empty()) spit(s_out, js.str().data(), js.str().size());
    puts(js.str().c_str());
    if (fused) ane_free(fused); if (chunked) ane_free_chunked_ffn(&cffn);
    return (failures || drift) ? 5 : 0;
}
