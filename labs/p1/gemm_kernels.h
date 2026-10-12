// GEMM, one optimisation at a time.  C[M,N] = A[M,K] · B[K,N], row-major fp32.
//
//   naive    i-j-k loops. The inner loop walks B down a column: a stride of N
//            floats, so each multiply touches a new cache line.
//   ikj      Swap loops so the inner loop walks B and C along rows. Unit
//            stride: every byte of each fetched line is used, and the compiler
//            can vectorise it.
//   blocked  ikj on tiles small enough to stay in L1/L2, so each tile of B is
//            reused many times before it is evicted.
//   packed   BLIS-style: copy (pack) tiles of A and B into contiguous panels,
//            then an 8x8 NEON micro-kernel keeps a 8x8 block of C in 16
//            registers and does 16 FMAs per 4 loads. Optional threads split M.
//   accel    Apple Accelerate cblas_sgemm (uses the SME matrix unit).
#pragma once

#include <arm_neon.h>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/pool.h"

namespace gemm {

inline void naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[(long)i * K + k] * B[(long)k * N + j];
      C[(long)i * N + j] = s;
    }
}

inline void ikj(int M, int N, int K, const float* A, const float* B, float* C) {
  std::memset(C, 0, sizeof(float) * (size_t)M * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      float a = A[(long)i * K + k];
      const float* b = B + (long)k * N;
      float* c = C + (long)i * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];
    }
}

inline void blocked(int M, int N, int K, const float* A, const float* B, float* C, int T = 64) {
  std::memset(C, 0, sizeof(float) * (size_t)M * N);
  const int TJ = 256;  // a 64 x 256 tile of B is 64 KiB: fits the 128 KiB L1D
  for (int i0 = 0; i0 < M; i0 += T)
    for (int k0 = 0; k0 < K; k0 += T)
      for (int j0 = 0; j0 < N; j0 += TJ) {
        int i1 = std::min(M, i0 + T), k1 = std::min(K, k0 + T), j1 = std::min(N, j0 + TJ);
        for (int i = i0; i < i1; ++i)
          for (int k = k0; k < k1; ++k) {
            float a = A[(long)i * K + k];
            const float* b = B + (long)k * N;
            float* c = C + (long)i * N;
            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
          }
      }
}

// ---- packed + NEON micro-kernel --------------------------------------------
constexpr int MR = 8, NR = 8;          // register tile: 8x8 floats = 16 q-registers
constexpr int KC = 256;                // depth of a packed panel
constexpr int MC = 128;                // rows of A packed at once (MC x KC x 4 B = 128 KiB)
constexpr int NC = 2048;               // cols of B packed at once (KC x NC x 4 B = 2 MiB, in L2)

// acc(8x8) = Ap(MR x kc, k-major) · Bp(kc x NR, k-major)
inline void micro_8x8(int kc, const float* Ap, const float* Bp, float* out, int ldo) {
  float32x4_t c[8][2];
  for (int i = 0; i < 8; ++i) c[i][0] = c[i][1] = vdupq_n_f32(0);
  for (int k = 0; k < kc; ++k) {
    float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4);
    float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4);
    Ap += MR; Bp += NR;
    c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0, 0); c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0, 0);
    c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0, 1); c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0, 1);
    c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0, 2); c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0, 2);
    c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0, 3); c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0, 3);
    c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a1, 0); c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a1, 0);
    c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a1, 1); c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a1, 1);
    c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a1, 2); c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a1, 2);
    c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a1, 3); c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a1, 3);
  }
  for (int i = 0; i < 8; ++i) {
    vst1q_f32(out + i * ldo, c[i][0]);
    vst1q_f32(out + i * ldo + 4, c[i][1]);
  }
}

// Pack rows [i0, i0+mc) x cols [k0, k0+kc) of A into MR-row panels, zero-padded.
inline void pack_A(const float* A, int K, int i0, int mc, int k0, int kc, float* Ap) {
  for (int ir = 0; ir < mc; ir += MR)
    for (int k = 0; k < kc; ++k)
      for (int r = 0; r < MR; ++r)
        *Ap++ = (ir + r < mc) ? A[(long)(i0 + ir + r) * K + k0 + k] : 0.0f;
}

// Pack one NR-wide panel (columns [j, j+NR)) of B rows [k0, k0+kc), zero-padded.
inline void pack_B_panel(const float* B, int N, int k0, int kc, int j, int nmax, float* Bp) {
  for (int k = 0; k < kc; ++k) {
    const float* row = B + (long)(k0 + k) * N;
    for (int c = 0; c < NR; ++c) *Bp++ = (j + c < nmax) ? row[j + c] : 0.0f;
  }
}

inline void packed(int M, int N, int K, const float* A, const float* B, float* C, lab::Pool* pool) {
  lab::Pool single(1);
  lab::Pool& P = pool ? *pool : single;
  int nt = P.size();
  std::memset(C, 0, sizeof(float) * (size_t)M * N);
  std::vector<float> Bp((size_t)KC * (NC + NR));
  std::vector<std::vector<float>> Ap(nt, std::vector<float>((size_t)(MC + MR) * KC));

  for (int jc = 0; jc < N; jc += NC) {
    int nc = std::min(NC, N - jc);
    int npanels = (nc + NR - 1) / NR;
    for (int pc = 0; pc < K; pc += KC) {
      int kc = std::min(KC, K - pc);
      P.run([&](int t) {  // pack B panels in parallel
        for (int p = t; p < npanels; p += nt)
          pack_B_panel(B, N, pc, kc, jc + p * NR, N, Bp.data() + (size_t)p * kc * NR);
      });
      // Work unit = (row block of C, range of B panels). With few row blocks
      // (small M, e.g. prefill of 32 tokens) the panels are split as well, so
      // every thread gets work. Each thread writes a disjoint part of C.
      int nblocks = (M + MC - 1) / MC;
      int nsplit = std::min(npanels, std::max(1, (nt + nblocks - 1) / nblocks));
      int units = nblocks * nsplit;
      P.run([&](int t) {
        float tile[MR * NR];
        for (int u = t; u < units; u += nt) {
          int blk = u / nsplit, s = u % nsplit;
          int p0 = (int)((long)npanels * s / nsplit), p1 = (int)((long)npanels * (s + 1) / nsplit);
          int ic = blk * MC, mc = std::min(MC, M - ic);
          pack_A(A, K, ic, mc, pc, kc, Ap[t].data());
          for (int p = p0; p < p1; ++p) {
            int j = jc + p * NR, nr = std::min(NR, N - j);
            for (int ir = 0; ir < mc; ir += MR) {
              int mr = std::min(MR, mc - ir);
              micro_8x8(kc, Ap[t].data() + (size_t)ir * kc, Bp.data() + (size_t)p * kc * NR, tile, NR);
              for (int r = 0; r < mr; ++r) {
                float* c = C + (long)(ic + ir + r) * N + j;
                const float* tr = tile + r * NR;
                if (nr == NR) {
                  vst1q_f32(c, vaddq_f32(vld1q_f32(c), vld1q_f32(tr)));
                  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vld1q_f32(tr + 4)));
                } else {
                  for (int q = 0; q < nr; ++q) c[q] += tr[q];
                }
              }
            }
          }
        }
      });
    }
  }
}

inline void accel(int M, int N, int K, const float* A, const float* B, float* C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

// ||X - R||_F / ||R||_F
inline double rel_err(const float* X, const float* R, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = (double)X[i] - R[i];
    num += d * d;
    den += (double)R[i] * R[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace gemm
