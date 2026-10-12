// Visible projection lowering: Y[S,N] = X[S,K] W[N,K]^T as a 1x1 convolution on the ANE.
//   A[0,k,0,s] = X[s,k]   activation surface [1,K,1,S_alloc], fp16, element index k*S_alloc + s
//   F[n,k,0,0] = W[n,k]   weight [N,K,1,1], fp16 BLOBFILE
//   Y[s,n]     = Z[0,n,0,s]                 output  [1,N,1,S_alloc], element index n*S_alloc + s
// Lanes s >= S are padding: filled with `sentinel` on input and discarded on output.
#pragma once
#include "ane_runtime.h"
#include <cstdint>
#include <string>
#include <vector>

namespace lowering {

struct Projection {
    int S = 1, K = 0, N = 0;
    int S_alloc = ane_lm::ANE_SPATIAL;            // upstream runtime floor: IOSurface width >= 32
    size_t input_elems() const { return (size_t)K * S_alloc; }
    size_t output_elems() const { return (size_t)N * S_alloc; }
    size_t input_bytes() const { return input_elems() * 2; }
    size_t output_bytes() const { return output_elems() * 2; }
    size_t weight_bytes() const { return (size_t)N * K * 2 + 128; }   // fp16 payload + upstream blob header
    int padded_lanes() const { return S_alloc - S; }
    // Index maps, kept as functions so tests can read them.
    size_t in_index(int s, int k) const { return (size_t)k * S_alloc + s; }
    size_t out_index(int s, int n) const { return (size_t)n * S_alloc + s; }
};

// Our MIL text for the conv form (same op sequence as upstream mil_gen_matmul; emitted by us).
std::string emit_mil(const Projection& p, const char* weight_name = "weight");

// Pack X[S,K] fp32 row-major into an fp16 staging buffer of input_elems(); padding lanes get sentinel.
void pack(const Projection& p, const float* X, float sentinel, std::vector<uint16_t>& staging);
// Unpack the full output surface (already fp32, N*S_alloc) into Y[S_alloc,N] row-major (all lanes kept).
void unpack_full(const Projection& p, const float* out_dense, std::vector<float>& Y_full);

// Compile our MIL with the given BF16 weights [N,K].
ane_lm::ANEKernel* compile_visible(const Projection& p, const uint16_t* W_bf16);

uint16_t f32_to_f16_bits(float f);

// SwiGLU FFN as one program: conv(Wg), conv(Wu), activation, mul, conv(Wd). Weights wg/wu [I,D], wd [D,I].
// variant "sigmoid": upstream form  silu = g * sigmoid(g)     (ANE sigmoid ~3e-3 abs error; FFN rel L2 ~2e-2)
// variant "exp":     silu = g / (1 + exp(-g))                  (measured FFN rel L2 ~2.6e-3, same dispatch, same time)
std::string emit_ffn_mil(int D, int I, const std::string& variant);
ane_lm::ANEKernel* compile_ffn(int D, int I, const std::string& variant, const uint16_t* wg, const uint16_t* wu, const uint16_t* wd);
std::string json_index_map(const Projection& p);

}  // namespace lowering
