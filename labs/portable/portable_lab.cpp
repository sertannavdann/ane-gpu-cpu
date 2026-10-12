// Portable lab: the same three measurements on any 64-bit machine.
//
//   1. memory bandwidth (STREAM triad), 1..N threads
//   2. fp32 GEMV (decode-style), with the SIMD path this CPU has
//   3. int8 dot product (the core of Q8/Q4 kernels), with this CPU's dot instruction
//
// One file, standard C++20, no Apple APIs. The SIMD paths are chosen at compile
// time, which is exactly how llama.cpp/ggml handles many ISAs:
//
//   macOS (Apple silicon)  clang++ -O3 -std=c++20 -mcpu=native portable_lab.cpp -o portable_lab
//   Linux arm64 / x86-64   g++ -O3 -std=c++20 -march=native -pthread portable_lab.cpp -o portable_lab
//   Windows x64 (MSVC)     cl /O2 /std:c++20 /arch:AVX2 /EHsc portable_lab.cpp
//   Windows on Arm (MSVC)  cl /O2 /std:c++20 /EHsc portable_lab.cpp        (NEON is always on)
//
// Run:  ./portable_lab --seconds 1
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define ISA_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define ISA_AVX2 1
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Run fn(tid) on n threads and wait. Threads are created per call: simple and
// portable, but NOT free. The GEMV below calls this once per layer, and with
// many threads each layer's share of work (~0.1 ms) is close to the cost of
// creating and joining the threads, so GEMV scaling stalls well below the
// bandwidth the triad reaches. labs/p1_gemv uses a persistent thread pool for
// exactly this reason; compare the two.
template <class F>
static void parallel(int n, F&& fn) {
  std::vector<std::thread> ts;
  for (int t = 1; t < n; ++t) ts.emplace_back(fn, t);
  fn(0);
  for (auto& t : ts) t.join();
}

// Repeat body() for `seconds` (after a warm-up) and return the median rate.
template <class F>
static double rate(double work, double seconds, F&& body) {
  body();
  std::vector<double> r;
  double start = now();
  while (now() - start < seconds || r.size() < 3) {
    double a = now();
    body();
    r.push_back(work / (now() - a));
  }
  std::sort(r.begin(), r.end());
  return r[r.size() / 2];
}

static void platform() {
  std::printf("== Platform\n");
#if defined(__APPLE__)
  const char* os = "macOS";
#elif defined(_WIN32)
  const char* os = "Windows";
#elif defined(__linux__)
  const char* os = "Linux";
#else
  const char* os = "other";
#endif
#if defined(ISA_NEON)
  const char* isa = "arm64 NEON (128-bit)";
#elif defined(ISA_AVX2)
  const char* isa = "x86-64 AVX2+FMA (256-bit)";
#else
  const char* isa = "scalar (no SIMD path compiled)";
#endif
  long page = 0;
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  page = si.dwPageSize;
#else
  page = sysconf(_SC_PAGESIZE);
#endif
  std::printf("   OS %s, SIMD path %s, %u hardware threads, page %ld B\n", os, isa,
              std::thread::hardware_concurrency(), page);
#if defined(__ARM_FEATURE_DOTPROD)
  std::printf("   int8 dot instruction: sdot (Armv8.2 dotprod)\n");
#elif defined(__AVXVNNI__) || defined(__AVX512VNNI__)
  std::printf("   int8 dot instruction: vpdpbusd (VNNI)\n");
#elif defined(ISA_AVX2)
  std::printf("   int8 dot instruction: none; using AVX2 vpmaddubsw + vpmaddwd\n");
#else
  std::printf("   int8 dot instruction: none (scalar)\n");
#endif
}

// ---------------------------------------------------------------- 1. bandwidth
static void bandwidth(double seconds, const std::vector<int>& threads) {
  const size_t n = (size_t)64 << 20;  // 64 M floats = 256 MiB per array, 768 MiB total
  std::vector<float> a(n), b(n, 1.0f), c(n, 2.0f);
  std::printf("\n== 1. Memory bandwidth, STREAM triad a = b + s*c (3 x %zu MiB)\n", n * 4 >> 20);
  std::printf("   %-8s %10s\n", "threads", "GB/s");
  for (int t : threads) {
    double gbs = rate(3.0 * n * 4 / 1e9, seconds, [&] {
      parallel(t, [&](int id) {
        size_t lo = n * id / t, hi = n * (id + 1) / t;
        for (size_t i = lo; i < hi; ++i) a[i] = b[i] + 3.0f * c[i];  // compilers vectorize this
      });
    });
    std::printf("   %-8d %10.1f\n", t, gbs);
  }
}

// ---------------------------------------------------------------- 2. GEMV fp32
static float dot_f32(const float* w, const float* x, int k) {
#if defined(ISA_NEON)
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  for (int i = 0; i < k; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(w + i), vld1q_f32(x + i));
    s1 = vfmaq_f32(s1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(w + i + 8), vld1q_f32(x + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#elif defined(ISA_AVX2)
  __m256 s0 = _mm256_setzero_ps(), s1 = s0;
  for (int i = 0; i < k; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), s1);
  }
  __m256 s = _mm256_add_ps(s0, s1);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  return _mm_cvtss_f32(h);
#else
  float s = 0;
  for (int i = 0; i < k; ++i) s += w[i] * x[i];
  return s;
#endif
}

static void gemv(double seconds, const std::vector<int>& threads) {
  const int rows = 3072, k = 1024, layers = 64;  // 768 MiB of fp32 weights: far bigger than any cache
  std::vector<float> W((size_t)layers * rows * k), x(k), y(rows);
  for (size_t i = 0; i < W.size(); ++i) W[i] = (float)((i * 2654435761u) % 1000) * 1e-4f - 0.05f;
  for (int i = 0; i < k; ++i) x[i] = 0.01f * (i % 17);
  std::printf("\n== 2. fp32 GEMV over %d layers of %dx%d (%zu MiB per 'token')\n", layers, rows, k,
              W.size() * 4 >> 20);
  std::printf("   %-8s %10s %14s\n", "threads", "GB/s", "tok/s @ 1 GB");
  for (int t : threads) {
    double gbs = rate(W.size() * 4.0 / 1e9, seconds, [&] {
      for (int l = 0; l < layers; ++l) {
        const float* Wl = W.data() + (size_t)l * rows * k;
        parallel(t, [&](int id) {
          for (int r = rows * id / t; r < rows * (id + 1) / t; ++r) y[r] = dot_f32(Wl + (size_t)r * k, x.data(), k);
        });
      }
    });
    std::printf("   %-8d %10.1f %14.1f\n", t, gbs, gbs);
  }
}

// ---------------------------------------------------------------- 3. int8 dot
static int32_t dot_i8(const int8_t* a, const int8_t* b, int n) {
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t s0 = vdupq_n_s32(0), s1 = s0;
  for (int i = 0; i < n; i += 32) {
    s0 = vdotq_s32(s0, vld1q_s8(a + i), vld1q_s8(b + i));
    s1 = vdotq_s32(s1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  return vaddvq_s32(vaddq_s32(s0, s1));
#elif defined(ISA_AVX2)
  // x86 has no signed x signed byte multiply-add: vpmaddubsw is UNSIGNED x signed.
  // Trick used by llama.cpp: |a| x (b with a's sign) gives the same products.
  __m256i acc = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);
  for (int i = 0; i < n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
    __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(va, va), _mm256_sign_epi8(vb, va));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
  }
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  return _mm_cvtsi128_si32(s);
#else
  int32_t s = 0;
  for (int i = 0; i < n; ++i) s += (int32_t)a[i] * b[i];
  return s;
#endif
}

static void int8dot(double seconds) {
  const int n = 4096, reps = 2048;  // 8 KiB: stays in L1, so this measures math, not memory
  std::vector<int8_t> a(n), b(n);
  int32_t ref = 0;
  for (int i = 0; i < n; ++i) {
    a[i] = (int8_t)((i * 37) % 255 - 127);
    b[i] = (int8_t)((i * 91) % 255 - 127);
    ref += (int32_t)a[i] * b[i];
  }
  std::atomic<int32_t> sink{0};
  int32_t got = dot_i8(a.data(), b.data(), n);
  double gops = rate(2.0 * n * reps / 1e9, seconds, [&] {
    int32_t s = 0;
    for (int r = 0; r < reps; ++r) s += dot_i8(a.data(), b.data(), n);
    sink += s;
  });
  std::printf("\n== 3. int8 dot product, one core, L1-resident: %.1f GOP/s  (check: %s)\n", gops,
              got == ref ? "correct" : "WRONG");
}

int main(int argc, char** argv) {
  double seconds = 1.0;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--seconds") seconds = std::atof(argv[i + 1]);
  int hw = (int)std::max(1u, std::thread::hardware_concurrency());
  std::vector<int> threads;
  for (int t : {1, 2, 4, 8}) if (t < hw) threads.push_back(t);
  threads.push_back(hw);
  platform();
  bandwidth(seconds, threads);
  gemv(seconds, threads);
  int8dot(seconds);
  return 0;
}
