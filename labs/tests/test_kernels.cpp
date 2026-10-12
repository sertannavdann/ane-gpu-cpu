// Correctness tests for every kernel the labs time. A fast wrong kernel is
// worthless, so `ctest` (or run_labs.py) runs these before any benchmark.
#include <cstdio>
#include <random>
#include <string>

#include "../common/pool.h"
#include "../p1/gemm_kernels.h"
#include "../p1/gemv_kernels.h"
#include "../p2/quant_kernels.h"

static int failures = 0, checks = 0;

static void expect(bool ok, const std::string& what) {
  ++checks;
  if (!ok) { ++failures; std::printf("  FAIL  %s\n", what.c_str()); }
}

static std::vector<float> randn(size_t n, unsigned seed, float scale = 1) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

static void test_gemm() {
  lab::Pool pool(4);
  // Odd shapes exercise every edge path: partial micro-tiles, partial KC/MC blocks.
  int shapes[][3] = {{1, 1, 1}, {7, 9, 13}, {13, 29, 37}, {64, 64, 64}, {129, 65, 300}, {1, 3072, 1024}, {33, 17, 513}};
  for (auto& s : shapes) {
    int M = s[0], N = s[1], K = s[2];
    auto A = randn((size_t)M * K, 1), B = randn((size_t)K * N, 2);
    std::vector<float> R((size_t)M * N), C((size_t)M * N);
    gemm::accel(M, N, K, A.data(), B.data(), R.data());
    std::string tag = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    gemm::naive(M, N, K, A.data(), B.data(), C.data());
    expect(gemm::rel_err(C.data(), R.data(), C.size()) < 1e-5, "gemm naive " + tag);
    gemm::ikj(M, N, K, A.data(), B.data(), C.data());
    expect(gemm::rel_err(C.data(), R.data(), C.size()) < 1e-5, "gemm ikj " + tag);
    gemm::blocked(M, N, K, A.data(), B.data(), C.data());
    expect(gemm::rel_err(C.data(), R.data(), C.size()) < 1e-5, "gemm blocked " + tag);
    gemm::packed(M, N, K, A.data(), B.data(), C.data(), nullptr);
    expect(gemm::rel_err(C.data(), R.data(), C.size()) < 1e-5, "gemm packed 1T " + tag);
    gemm::packed(M, N, K, A.data(), B.data(), C.data(), &pool);
    expect(gemm::rel_err(C.data(), R.data(), C.size()) < 1e-5, "gemm packed 4T " + tag);
  }
}

static void test_gemv() {
  int R = 96, K = 1024;
  auto W = randn((size_t)R * K, 3, 0.05f), x = randn(K, 4);
  std::vector<float16_t> W16(W.size());
  std::vector<float> W16f(W.size());
  for (size_t i = 0; i < W.size(); ++i) { W16[i] = (float16_t)W[i]; W16f[i] = (float)W16[i]; }
  std::vector<float> ref(R), y(R), ref16(R);
  gemv::accel(R, K, W.data(), x.data(), ref.data());
  gemv::accel(R, K, W16f.data(), x.data(), ref16.data());
  gemv::scalar(0, R, K, W.data(), x.data(), y.data());
  expect(gemm::rel_err(y.data(), ref.data(), R) < 1e-5, "gemv scalar");
  gemv::f32(0, R, K, W.data(), x.data(), y.data());
  expect(gemm::rel_err(y.data(), ref.data(), R) < 1e-5, "gemv neon f32");
  gemv::f16(0, R, K, W16.data(), x.data(), y.data());
  // Same fp16-rounded weights on both sides: only summation order differs.
  expect(gemm::rel_err(y.data(), ref16.data(), R) < 1e-5, "gemv neon f16 vs rounded reference");
  // fmlal also rounds x to fp16, so compare against a reference with both rounded.
  std::vector<float16_t> xh(K);
  std::vector<float> xhf(K), ref16x(R);
  for (int k = 0; k < K; ++k) { xh[k] = (float16_t)x[k]; xhf[k] = (float)xh[k]; }
  gemv::accel(R, K, W16f.data(), xhf.data(), ref16x.data());
  gemv::f16_fmlal(0, R, K, W16.data(), xh.data(), y.data());
  expect(gemm::rel_err(y.data(), ref16x.data(), R) < 1e-5, "gemv neon f16 fmlal vs rounded reference");
}

static void test_quant() {
  using namespace quant;
  // Layout: weight j in the low nibble of byte j, weight j+16 in the high nibble.
  float w[32];
  for (int j = 0; j < 32; ++j) w[j] = (float)(j % 16) - 8;  // exactly representable with d = 1
  w[0] = -8;
  block_q4_0 b4;
  quantize_q4_0(w, &b4, 32);
  expect((float)b4.d == 1.0f, "q4_0 scale for range [-8,7] is 1");
  expect((b4.qs[1] & 0x0F) == (int)w[1] + 8, "q4_0 low nibble holds weight j");
  expect((b4.qs[1] >> 4) == (int)w[17] + 8, "q4_0 high nibble holds weight j+16");
  float back[32];
  dequantize_q4_0(&b4, back, 32);
  bool exact = true;
  for (int j = 0; j < 32; ++j) exact &= back[j] == w[j];
  expect(exact, "q4_0 round-trips exactly representable values");

  // Round-trip error bounds. Rounding to the nearest level costs at most d/2.
  // The scale is computed in fp32 but stored as fp16 (relative rounding error
  // up to 2^-11), and dequantization uses the stored value, so a level q also
  // shifts by up to |q|·d·2^-11: 127·2^-11 ≈ 0.06 steps for Q8, 8·2^-11 for Q4.
  long n = 32 * 512;
  auto v = randn(n, 5);
  std::vector<block_q8_0> q8(n / 32);
  std::vector<block_q4_0> q4(n / 32);
  std::vector<float> d8(n), d4(n);
  quantize_q8_0(v.data(), q8.data(), n);
  quantize_q4_0(v.data(), q4.data(), n);
  dequantize_q8_0(q8.data(), d8.data(), n);
  dequantize_q4_0(q4.data(), d4.data(), n);
  bool in8 = true, in4 = true;
  for (long i = 0; i < n; ++i) {
    float s8 = std::fabs((float)q8[i / 32].d), s4 = std::fabs((float)q4[i / 32].d);
    const float h = 1.0f / 2048;
    in8 &= std::fabs(v[i] - d8[i]) <= s8 * (0.5f + 127 * h) + 1e-6f;
    in4 &= std::fabs(v[i] - d4[i]) <= s4 * (1.0f + 8 * h) + 1e-6f;  // q clamped at 15 can cost a full step
  }
  expect(in8, "q8_0 error within half a step + fp16 scale rounding");
  expect(in4, "q4_0 error within one step + fp16 scale rounding");

  // NEON dot products agree with the scalar reference (same integer sums).
  auto a = randn(n, 6);
  std::vector<block_q8_0> aq(n / 32);
  quantize_q8_0(a.data(), aq.data(), n);
  for (long len : {32L, 64L, 96L, 1024L, n}) {
    float s8 = dot_q8_0_scalar(q8.data(), aq.data(), len), n8 = dot_q8_0_neon(q8.data(), aq.data(), len);
    float s4 = dot_q4_0_scalar(q4.data(), aq.data(), len), n4 = dot_q4_0_neon(q4.data(), aq.data(), len);
    expect(std::fabs(s8 - n8) <= 1e-4f * (1 + std::fabs(s8)), "q8_0 neon == scalar, n=" + std::to_string(len));
    expect(std::fabs(s4 - n4) <= 1e-4f * (1 + std::fabs(s4)), "q4_0 neon == scalar, n=" + std::to_string(len));
  }
}

int main() {
  test_gemm();
  test_gemv();
  test_quant();
  std::printf("kernel tests: %d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
