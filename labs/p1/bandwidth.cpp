// P1 lab: the memory ceiling.
//
// Streams arrays of growing size through four STREAM-style kernels and reports
// GB/s. As the working set outgrows each cache level, bandwidth drops to the
// next level's rate. The plateau for arrays much larger than every cache is the
// DRAM bandwidth that bounds LLM decode.
//
// Byte counting follows the STREAM convention (bytes the program asks for).
// A store to a line not in cache may first read that line (write-allocate), so
// the hardware can move more bytes than are counted for `write` and `copy`.
#include <arm_neon.h>

#include <cstring>

#include "../common/harness.h"
#include "../common/hwinfo.h"
#include "../common/pool.h"

using namespace lab;

static float* alloc_floats(size_t n) {
  size_t bytes = (n * sizeof(float) + 16383) / 16384 * 16384;
  return (float*)std::aligned_alloc(16384, bytes);
}

// 8 independent accumulators: enough loads in flight to cover memory latency.
static float read_kernel(const float* a, long b, long e) {
  float32x4_t s[8];
  for (auto& v : s) v = vdupq_n_f32(0);
  for (long i = b; i < e; i += 32) {
#pragma unroll
    for (int k = 0; k < 8; ++k) s[k] = vaddq_f32(s[k], vld1q_f32(a + i + 4 * k));
  }
  for (int k = 1; k < 8; ++k) s[0] = vaddq_f32(s[0], s[k]);
  return vaddvq_f32(s[0]);
}

static void write_kernel(float* a, long b, long e, float v) {
  float32x4_t x = vdupq_n_f32(v);
  for (long i = b; i < e; i += 32) {
#pragma unroll
    for (int k = 0; k < 8; ++k) vst1q_f32(a + i + 4 * k, x);
  }
}

static void copy_kernel(float* dst, const float* src, long b, long e) {
  for (long i = b; i < e; i += 32) {
#pragma unroll
    for (int k = 0; k < 8; ++k) vst1q_f32(dst + i + 4 * k, vld1q_f32(src + i + 4 * k));
  }
}

static void triad_kernel(float* a, const float* b_, const float* c, float s, long b, long e) {
  float32x4_t vs = vdupq_n_f32(s);
  for (long i = b; i < e; i += 32) {
#pragma unroll
    for (int k = 0; k < 8; ++k)
      vst1q_f32(a + i + 4 * k, vfmaq_f32(vld1q_f32(b_ + i + 4 * k), vld1q_f32(c + i + 4 * k), vs));
  }
}

struct Arrays {
  float *a = nullptr, *b = nullptr, *c = nullptr;
  long n = 0;
  Arrays(long n_, Pool& pool) : n(n_) {
    a = alloc_floats(n); b = alloc_floats(n); c = alloc_floats(n);
    // First touch in parallel, by the threads that will use each range.
    pool.run([&](int t) {
      auto [lo, hi] = Pool::range(n, t, pool.size(), 32);
      for (long i = lo; i < hi; ++i) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.5f; }
    });
  }
  ~Arrays() { std::free(a); std::free(b); std::free(c); }
};

static int arrays_for(const std::string& mode) {
  return mode == "triad" ? 3 : mode == "copy" ? 2 : 1;
}

static double bytes_per_call(const std::string& mode, long n) {
  return (double)n * sizeof(float) * arrays_for(mode);
}

static Summary measure(const std::string& mode, Arrays& arr, Pool& pool, const RunCfg& cfg,
                       const std::string& label) {
  alignas(128) static float partial[64 * 32];
  long n = arr.n;
  return run_for(label, bytes_per_call(mode, n), 1e9, "GB/s", cfg, [&] {
    pool.run([&](int t) {
      auto [lo, hi] = Pool::range(n, t, pool.size(), 32);
      if (mode == "read") partial[t * 32] = read_kernel(arr.a, lo, hi);  // 128 B apart: no false sharing
      else if (mode == "write") write_kernel(arr.a, lo, hi, 3.0f);
      else if (mode == "copy") copy_kernel(arr.a, arr.b, lo, hi);
      else triad_kernel(arr.a, arr.b, arr.c, 3.0f, lo, hi);
    });
    keep(partial[0]);
  });
}

static std::string human(double bytes) {
  char buf[32];
  if (bytes >= 1 << 30) std::snprintf(buf, sizeof buf, "%.1f GiB", bytes / (1 << 30));
  else if (bytes >= 1 << 20) std::snprintf(buf, sizeof buf, "%.1f MiB", bytes / (1 << 20));
  else std::snprintf(buf, sizeof buf, "%.0f KiB", bytes / 1024);
  return buf;
}

int main(int argc, char** argv) {
  Args args(argc, argv);
  RunCfg cfg = RunCfg::from(args);
  Recorder rec(args);
  HwInfo hw;
  std::string part = args.str("part", "all");
  std::string qos = args.str("qos", "user-interactive");

  banner("P1.bandwidth  STREAM-style memory bandwidth",
         "How fast can the CPU move bytes from each level of the memory hierarchy?");
  std::printf("  P-core L1D %s, P-cluster L2 %s, line %ld B, page %ld B. Spec DRAM peak: %.0f GB/s\n",
              human(hw.p_l1d).c_str(), human(hw.p_l2).c_str(), hw.line, hw.page, hw.spec_bw_gbs());

  // Part 1: working-set sweep on one core, read kernel.
  if (part == "all" || part == "sweep") {
    int t = (int)args.num("sweep-threads", 1);
    Pool pool(t, parse_qos(qos));
    std::printf("\n  Part 1: read bandwidth vs working-set size, %d thread(s)\n", t);
    std::printf("  %-10s %-7s %10s %10s %10s\n", "set", "level", "GB/s", "p10", "p90");
    std::vector<long> kib = args.list("sizes-kib", {16, 64, 96, 256, 1024, 4096, 12288, 32768,
                                                     65536, 262144, 1048576});
    for (long k : kib) {
      long n = std::max(32L * t, k * 1024 / 4 / 32 * 32);
      Arrays arr(n, pool);
      Summary s = measure("read", arr, pool, cfg, "read " + human(n * 4.0));
      std::printf("  %-10s %-7s %10.1f %10.1f %10.1f\n", human(n * 4.0).c_str(),
                  level_for(hw, n * 4.0, t), s.median, s.p10, s.p90);
      rec.emit_summary("p1.bandwidth", "sweep", s, "GB/s",
                       {{"bytes", n * 4.0}, {"threads", (double)t}}, {{"level", level_for(hw, n * 4.0, t)}});
    }
  }

  // Part 2: DRAM-sized arrays, each kernel, thread scaling.
  if (part == "all" || part == "threads") {
    long mib = (long)args.num("dram-mib", 512);
    long n = mib * 1024 * 1024 / 4 / 32 * 32;
    std::vector<long> threads = args.list("threads", {1, 2, 4, 6, 8, hw.p_cores, (long)hw.cores()});
    std::printf("\n  Part 2: DRAM bandwidth (%ld MiB per array) vs threads and kernel\n", mib);
    std::printf("  %-8s %10s %10s %10s %10s\n", "threads", "read", "write", "copy", "triad");
    double best = 0;
    for (long t : threads) {
      Pool pool((int)t, parse_qos(qos));
      Arrays arr(n, pool);
      std::printf("  %-8ld", t);
      for (const char* mode : {"read", "write", "copy", "triad"}) {
        Summary s = measure(mode, arr, pool, cfg, std::string(mode) + " t=" + std::to_string(t));
        std::printf(" %10.1f", s.median);
        std::fflush(stdout);
        rec.emit_summary("p1.bandwidth", std::string("dram_") + mode, s, "GB/s",
                         {{"threads", (double)t}, {"bytes", bytes_per_call(mode, n)}}, {{"qos", qos}});
        if (std::string(mode) == "read") best = std::max(best, s.median);
      }
      std::printf("\n");
    }
    // E-cores alone.
    {
      Pool pool((int)hw.e_cores, QOS_CLASS_BACKGROUND);
      Arrays arr(n, pool);
      Summary s = measure("read", arr, pool, cfg, "read E-cores");
      std::printf("  %-8s %10.1f   <- E-cores only (QoS background)\n",
                  (std::to_string(hw.e_cores) + "E").c_str(), s.median);
      rec.emit_summary("p1.bandwidth", "dram_read_ecores", s, "GB/s", {{"threads", (double)hw.e_cores}},
                       {{"qos", "background"}});
    }
    rec.emit("p1.bandwidth", "ceiling", {{"dram_read_gbs", best}, {"spec_gbs", hw.spec_bw_gbs()}});
    std::printf("\n  Best CPU DRAM read: %.0f GB/s = %.0f%% of the %.0f GB/s spec figure.\n", best,
                100 * best / hw.spec_bw_gbs(), hw.spec_bw_gbs());
    std::printf("  The CPU clusters alone usually cannot pull the whole SoC bandwidth; the GPU can.\n");
  }
  return 0;
}
