// Block quantization in the style of llama.cpp (same Q8_0 / Q4_0 layouts,
// written from the format description).
//
// A block holds 32 consecutive weights of one row and one fp16 scale d:
//   Q8_0: q in [-127, 127], w ≈ d·q.        34 bytes / 32 weights = 8.5 bits/weight
//   Q4_0: q in [0, 15],     w ≈ d·(q - 8).  18 bytes / 32 weights = 4.5 bits/weight
// Q4_0 nibble order: byte j holds weight j in its low nibble and weight j+16
// in its high nibble, so one 16-byte load unpacks into two contiguous halves.
//
// Dot products never dequantize to float. The activation vector is quantized
// to Q8_0 once, then each block is an integer dot product (NEON sdot: 16
// int8 x int8 products summed into 4 int32 lanes in one instruction) times the
// two scales. That is why low-bit decode is both fewer bytes and cheap math.
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace quant {

constexpr int QK = 32;

struct block_q8_0 {
  float16_t d;
  int8_t qs[QK];
};
struct block_q4_0 {
  float16_t d;
  uint8_t qs[QK / 2];
};
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block must be 34 bytes");
static_assert(sizeof(block_q4_0) == 18, "Q4_0 block must be 18 bytes");

inline void quantize_q8_0(const float* x, block_q8_0* y, long n) {
  for (long b = 0; b < n / QK; ++b) {
    const float* xb = x + b * QK;
    float amax = 0;
    for (int j = 0; j < QK; ++j) amax = std::max(amax, std::fabs(xb[j]));
    float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    y[b].d = (float16_t)d;
    for (int j = 0; j < QK; ++j) y[b].qs[j] = (int8_t)std::lround(xb[j] * id);
  }
}

// The scale maps the largest-magnitude weight (with its sign) to -8, so that
// value is represented exactly and the asymmetric range [-8, 7] is used fully.
inline void quantize_q4_0(const float* x, block_q4_0* y, long n) {
  for (long b = 0; b < n / QK; ++b) {
    const float* xb = x + b * QK;
    float amax = 0, vmax = 0;
    for (int j = 0; j < QK; ++j)
      if (std::fabs(xb[j]) > amax) { amax = std::fabs(xb[j]); vmax = xb[j]; }
    float d = vmax / -8.0f, id = d ? 1.0f / d : 0.0f;
    y[b].d = (float16_t)d;
    for (int j = 0; j < QK / 2; ++j) {
      int q0 = std::min(15, (int)(xb[j] * id + 8.5f));
      int q1 = std::min(15, (int)(xb[j + QK / 2] * id + 8.5f));
      y[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

inline void dequantize_q8_0(const block_q8_0* x, float* y, long n) {
  for (long b = 0; b < n / QK; ++b)
    for (int j = 0; j < QK; ++j) y[b * QK + j] = (float)x[b].d * x[b].qs[j];
}

inline void dequantize_q4_0(const block_q4_0* x, float* y, long n) {
  for (long b = 0; b < n / QK; ++b) {
    float d = (float)x[b].d;
    for (int j = 0; j < QK / 2; ++j) {
      y[b * QK + j] = d * ((x[b].qs[j] & 0x0F) - 8);
      y[b * QK + j + QK / 2] = d * ((x[b].qs[j] >> 4) - 8);
    }
  }
}

// ---- dot products: weight row (n values) · activation row (Q8_0) ----------

inline float dot_q8_0_scalar(const block_q8_0* w, const block_q8_0* a, long n) {
  float s = 0;
  for (long b = 0; b < n / QK; ++b) {
    int32_t acc = 0;
    for (int j = 0; j < QK; ++j) acc += (int32_t)w[b].qs[j] * a[b].qs[j];
    s += (float)acc * (float)w[b].d * (float)a[b].d;
  }
  return s;
}

inline float dot_q4_0_scalar(const block_q4_0* w, const block_q8_0* a, long n) {
  float s = 0;
  for (long b = 0; b < n / QK; ++b) {
    int32_t acc = 0;
    for (int j = 0; j < QK / 2; ++j) {
      acc += ((w[b].qs[j] & 0x0F) - 8) * a[b].qs[j];
      acc += ((w[b].qs[j] >> 4) - 8) * a[b].qs[j + QK / 2];
    }
    s += (float)acc * (float)w[b].d * (float)a[b].d;
  }
  return s;
}

inline float dot_q8_0_neon(const block_q8_0* w, const block_q8_0* a, long n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  long nb = n / QK, b = 0;
  for (; b + 1 < nb; b += 2) {  // two blocks per step for instruction-level parallelism
    int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w[b].qs), vld1q_s8(a[b].qs));
    p0 = vdotq_s32(p0, vld1q_s8(w[b].qs + 16), vld1q_s8(a[b].qs + 16));
    int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w[b + 1].qs), vld1q_s8(a[b + 1].qs));
    p1 = vdotq_s32(p1, vld1q_s8(w[b + 1].qs + 16), vld1q_s8(a[b + 1].qs + 16));
    s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(p0), (float)w[b].d * (float)a[b].d);
    s1 = vmlaq_n_f32(s1, vcvtq_f32_s32(p1), (float)w[b + 1].d * (float)a[b + 1].d);
  }
  float s = vaddvq_f32(vaddq_f32(s0, s1));
  if (b < nb) s += dot_q8_0_scalar(w + b, a + b, QK);
  return s;
}

inline float dot_q4_0_neon(const block_q4_0* w, const block_q8_0* a, long n) {
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  const int8x16_t s8 = vdupq_n_s8(8);
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  long nb = n / QK, b = 0;
  auto block = [&](long i) {
    uint8x16_t v = vld1q_u8(w[i].qs);
    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);  // weights 0..15
    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8);  // weights 16..31
    int32x4_t p = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(a[i].qs));
    return vdotq_s32(p, hi, vld1q_s8(a[i].qs + 16));
  };
  for (; b + 1 < nb; b += 2) {
    int32x4_t p0 = block(b), p1 = block(b + 1);
    s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(p0), (float)w[b].d * (float)a[b].d);
    s1 = vmlaq_n_f32(s1, vcvtq_f32_s32(p1), (float)w[b + 1].d * (float)a[b + 1].d);
  }
  float s = vaddvq_f32(vaddq_f32(s0, s1));
  if (b < nb) s += dot_q4_0_scalar(w + b, a + b, QK);
  return s;
}

// ---- generic symmetric block quantizer, for the accuracy study --------------
// bits b, block size B: q = round(w / d), d = absmax / (2^(b-1) - 1).
// Storage: b bits per weight + one 16-bit scale per block.
inline double bits_per_weight(int bits, int block) { return bits + 16.0 / block; }

inline void fake_quant(const float* x, float* y, long n, int bits, int block) {
  const float qmax = (float)((1 << (bits - 1)) - 1);
  for (long b0 = 0; b0 < n; b0 += block) {
    long b1 = std::min(n, b0 + block);
    float amax = 0;
    for (long j = b0; j < b1; ++j) amax = std::max(amax, std::fabs(x[j]));
    float d = (float)(float16_t)(amax / qmax);  // scale stored as fp16
    for (long j = b0; j < b1; ++j) y[j] = d ? d * std::nearbyint(std::clamp(x[j] / d, -qmax, qmax)) : 0;
  }
}

}  // namespace quant
