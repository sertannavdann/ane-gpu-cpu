#include "lowerings.h"
#include <cstdio>
#include <cstring>

namespace lowering {

uint16_t f32_to_f16_bits(float f) { __fp16 h = (__fp16)f; uint16_t b; memcpy(&b, &h, 2); return b; }

std::string emit_mil(const Projection& p, const char* wname) {
    char buf[4096];
    snprintf(buf, sizeof(buf),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/%s.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = W, x = x)"
        "[name = tensor<string, []>(\"cv\")];\n"
        "    } -> (y);\n"
        "}\n",
        p.K, p.S_alloc, p.N, p.K, p.N, p.K, wname, p.N, p.S_alloc);
    return buf;
}

void pack(const Projection& p, const float* X, float sentinel, std::vector<uint16_t>& st) {
    st.resize(p.input_elems());
    const uint16_t sent = f32_to_f16_bits(sentinel);
    for (int k = 0; k < p.K; k++) {
        uint16_t* plane = st.data() + (size_t)k * p.S_alloc;
        for (int s = 0; s < p.S; s++) plane[s] = f32_to_f16_bits(X[(size_t)s * p.K + k]);
        for (int s = p.S; s < p.S_alloc; s++) plane[s] = sent;
    }
}

void unpack_full(const Projection& p, const float* out, std::vector<float>& Y) {
    // [N, S_alloc] plane layout -> [S_alloc, N] rows, transposed in 32x32 tiles so both the reads
    // (one 64-byte plane per n) and the writes (128-byte row segments) stay cache-resident.
    // The first version of this loop (n outer, s inner) cost ~176 us for N=2048 and was mistaken
    // for an ANE cost until the timer was split; see results/ane-r1-projection/report.md.
    Y.resize((size_t)p.S_alloc * p.N);
    const int T = 32;
    for (int n0 = 0; n0 < p.N; n0 += T) {
        int n1 = n0 + T < p.N ? n0 + T : p.N;
        for (int s = 0; s < p.S_alloc; s++) {
            float* row = Y.data() + (size_t)s * p.N;
            for (int n = n0; n < n1; n++) row[n] = out[(size_t)n * p.S_alloc + s];
        }
    }
}

ane_lm::ANEKernel* compile_visible(const Projection& p, const uint16_t* W_bf16) {
    std::string mil = emit_mil(p);
    size_t in_sz = p.input_bytes(), out_sz = p.output_bytes();
    ane_lm::MILWeight w{"weight", W_bf16, p.N * p.K};
    return ane_lm::ane_compile_mil_weighted(mil.c_str(), 1, &in_sz, 1, &out_sz, &w, 1);
}

std::string json_index_map(const Projection& p) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"activation\":\"A[0,k,0,s]=X[s,k] at element k*%d+s\",\"weight\":\"F[n,k,0,0]=W[n,k] fp16 row-major\","
        "\"output\":\"Y[s,n]=Z[0,n,0,s] at element n*%d+s\",\"S\":%d,\"S_alloc\":%d,\"K\":%d,\"N\":%d,"
        "\"padded_lanes\":%d,\"input_bytes\":%zu,\"output_bytes\":%zu,\"weight_bytes\":%zu}",
        p.S_alloc, p.S_alloc, p.S, p.S_alloc, p.K, p.N, p.padded_lanes(), p.input_bytes(), p.output_bytes(), p.weight_bytes());
    return buf;
}

std::string emit_ffn_mil(int D, int I, const std::string& variant) {
    auto T = [](int C) { return "tensor<fp16, [1, " + std::to_string(C) + ", 1, 32]>"; };
    auto W = [](const char* nm, const char* up, int N, int K) {
        return "        tensor<fp16, [" + std::to_string(N) + ", " + std::to_string(K) + ", 1, 1]> " + up + " = const()[name = tensor<string, []>(\"" + up +
               "\"), val = tensor<fp16, [" + std::to_string(N) + ", " + std::to_string(K) + ", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/" + nm +
               ".bin\"), offset = tensor<uint64, []>(64)))];\n"; };
    auto conv = [&](const char* out, int N, const char* w, const char* x) {
        return "        " + T(N) + " " + out + " = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = " + w + ", x = " + x +
               ")[name = tensor<string, []>(\"" + out + "\")];\n"; };
    std::string m = "program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n";
    m += "    func main<ios16>(" + T(D) + " x) {\n";
    m += "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n";
    m += W("wg", "Wg", I, D) + W("wu", "Wu", I, D) + W("wd", "Wd", D, I);
    m += conv("gate", I, "Wg", "x") + conv("up", I, "Wu", "x");
    if (variant == "exp") {
        m += "        tensor<fp16, []> one = const()[name = tensor<string, []>(\"one\"), val = tensor<fp16, []>(1.0)];\n"
             "        tensor<fp16, []> mone = const()[name = tensor<string, []>(\"mone\"), val = tensor<fp16, []>(-1.0)];\n";
        m += "        " + T(I) + " ng = mul(x = gate, y = mone)[name = tensor<string, []>(\"ng\")];\n";
        m += "        " + T(I) + " ex = exp(x = ng)[name = tensor<string, []>(\"ex\")];\n";
        m += "        " + T(I) + " den = add(x = ex, y = one)[name = tensor<string, []>(\"den\")];\n";
        m += "        " + T(I) + " silu = real_div(x = gate, y = den)[name = tensor<string, []>(\"sl\")];\n";
    } else {
        m += "        " + T(I) + " sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n";
        m += "        " + T(I) + " silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n";
    }
    m += "        " + T(I) + " fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n";
    m += conv("out", D, "Wd", "fused");
    m += "    } -> (out);\n}\n";
    return m;
}

ane_lm::ANEKernel* compile_ffn(int D, int I, const std::string& variant, const uint16_t* wg, const uint16_t* wu, const uint16_t* wd) {
    std::string mil = emit_ffn_mil(D, I, variant);
    size_t in_sz = (size_t)D * ane_lm::ANE_SPATIAL * 2, out_sz = in_sz;
    ane_lm::MILWeight w[3] = {{"wg", wg, I * D}, {"wu", wu, I * D}, {"wd", wd, D * I}};
    return ane_lm::ane_compile_mil_weighted(mil.c_str(), 1, &in_sz, 1, &out_sz, w, 3);
}

}  // namespace lowering
