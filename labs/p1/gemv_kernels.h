// GEMV: y[R] = W[R,K] · x[K], W row-major. This is what one decode step does
// for every linear layer: each weight is read once and used for one
// multiply-add, i.e. 2 FLOPs per 4 bytes (fp32) or per 2 bytes (fp16).
#pragma once

#include <arm_neon.h>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "../common/pool.h"

namespace gemv {

// Plain loop. Without -ffast-math the compiler may not reorder float adds, so
// this stays one serial dependency chain of scalar FMAs: latency-bound.
inline void scalar(long r0, long r1, int K, const float* W, const float* x, float* y) {
  for (long r = r0; r < r1; ++r) {
    const float* w = W + r * K;
    float s = 0;
    for (int k = 0; k < K; ++k) s += w[k] * x[k];
    y[r] = s;
  }
}

// NEON, 4 independent accumulators (16 floats per step).
inline void f32(long r0, long r1, int K, const float* W, const float* x, float* y) {
  for (long r = r0; r < r1; ++r) {
    const float* w = W + r * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < K; k += 16) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(x + k));
      a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
      a2 = vfmaq_f32(a2, vld1q_f32(w + k + 8), vld1q_f32(x + k + 8));
      a3 = vfmaq_f32(a3, vld1q_f32(w + k + 12), vld1q_f32(x + k + 12));
    }
    y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  }
}

// fp16 weights, fp32 activations and accumulation: half the bytes per weight.
inline void f16(long r0, long r1, int K, const float16_t* W, const float* x, float* y) {
  for (long r = r0; r < r1; ++r) {
    const float16_t* w = W + r * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < K; k += 16) {
      float16x8_t h0 = vld1q_f16(w + k), h1 = vld1q_f16(w + k + 8);
      a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(x + k));
      a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h0), vld1q_f32(x + k + 4));
      a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(x + k + 8));
      a3 = vfmaq_f32(a3, vcvt_high_f32_f16(h1), vld1q_f32(x + k + 12));
    }
    y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  }
}

// fp16 weights AND fp16 activations, fp32 accumulation via FMLAL/FMLAL2
// (FEAT_FHM): one instruction widens 4 fp16 products into 4 fp32 sums, so the
// separate fp16->fp32 convert disappears from the inner loop.
inline void f16_fmlal(long r0, long r1, int K, const float16_t* W, const float16_t* xh, float* y) {
  for (long r = r0; r < r1; ++r) {
    const float16_t* w = W + r * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < K; k += 16) {
      float16x8_t w0 = vld1q_f16(w + k), w1 = vld1q_f16(w + k + 8);
      float16x8_t x0 = vld1q_f16(xh + k), x1 = vld1q_f16(xh + k + 8);
      a0 = vfmlalq_low_f16(a0, w0, x0);
      a1 = vfmlalq_high_f16(a1, w0, x0);
      a2 = vfmlalq_low_f16(a2, w1, x1);
      a3 = vfmlalq_high_f16(a3, w1, x1);
    }
    y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  }
}

inline void accel(int R, int K, const float* W, const float* x, float* y) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, R, K, 1.0f, W, K, x, 1, 0.0f, y, 1);
}

}  // namespace gemv
