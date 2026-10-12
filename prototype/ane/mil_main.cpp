// Generic single-program probe: compile a MIL text with named bf16 weights, write one raw fp16 input
// surface, evaluate once (plus optional timed repeats), dump the raw fp16 output surface.
// spec: {"mil":"prog.mil","weights":[{"name":"wg","bf16":"file","numel":N},...],"input_bytes":B,"output_bytes":B,
//        "input":"in.f16.bin","output":"out.f16.bin","calls":n}
#include "ane_runtime.h"
#include <mach/mach_time.h>
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
static std::string jstr(const std::string& j, const std::string& key, size_t from = 0) {
    size_t k = j.find("\"" + key + "\"", from); if (k == std::string::npos) return ""; size_t c = j.find(':', k), q1 = j.find('"', c); size_t q2 = j.find('"', q1 + 1); return j.substr(q1 + 1, q2 - q1 - 1); }
static double jnum(const std::string& j, const std::string& key, double d, size_t from = 0) {
    size_t k = j.find("\"" + key + "\"", from); if (k == std::string::npos) return d; return atof(j.c_str() + j.find(':', k) + 1); }
int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string spec = slurp(argv[1]);
    std::string mil = slurp(jstr(spec, "mil")), in = slurp(jstr(spec, "input")), outp = jstr(spec, "output");
    size_t in_b = (size_t)jnum(spec, "input_bytes", 0), out_b = (size_t)jnum(spec, "output_bytes", 0); int calls = (int)jnum(spec, "calls", 1);
    std::vector<std::string> blobs; std::vector<MILWeight> ws; std::vector<std::string> names;
    for (size_t pos = spec.find("\"weights\""); (pos = spec.find("\"name\"", pos + 1)) != std::string::npos;) {
        std::string nm = jstr(spec, "name", pos), path = jstr(spec, "bf16", pos); int numel = (int)jnum(spec, "numel", 0, pos);
        if (nm.empty() || spec.find("\"input_bytes\"") < pos) break;
        names.push_back(nm); blobs.push_back(slurp(path));
        if (blobs.back().size() != (size_t)numel * 2) { fprintf(stderr, "weight %s size %zu != %d*2\n", nm.c_str(), blobs.back().size(), numel); return 1; }
        pos = spec.find("}", pos);
    }
    for (size_t i = 0; i < names.size(); i++) ws.push_back({names[i].c_str(), (const uint16_t*)blobs[i].data(), (int)(blobs[i].size() / 2)});
    if (in.size() != in_b) { fprintf(stderr, "input size %zu != %zu\n", in.size(), in_b); return 1; }
    ane_init(); if (!ane_available()) return 2;
    double tc = now_us();
    ANEKernel* k = ane_compile_mil_weighted(mil.c_str(), 1, &in_b, 1, &out_b, ws.data(), (int)ws.size());
    if (!k) { fprintf(stderr, "compile failed\n"); return 3; }
    double compile_us = now_us() - tc;
    if (!ane_write_surface_raw(k, 0, (const uint16_t*)in.data(), in.size())) return 4;
    std::vector<double> ev;
    for (int i = 0; i < calls; i++) { double t = now_us(); if (!ane_eval_qos(k, 21)) return 4; ev.push_back(now_us() - t); }
    std::vector<float> out(out_b / 2); if (!ane_read_output_raw(k, 0, out.data(), (int)out.size())) return 4;
    std::ofstream(outp, std::ios::binary).write((const char*)out.data(), out.size() * 4);   // fp32 of the fp16 surface
    double s = 0; for (double v : ev) s += v;
    printf("{\"compile_us\":%.0f,\"eval_us_mean\":%.1f,\"calls\":%d}\n", compile_us, s / ev.size(), calls);
    ane_free(k); return 0;
}
