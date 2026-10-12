// P1 lab: GEMM step by step, then the prefill/decode roofline.
//
// Part 1 ("steps"): one square GEMM, each optimisation from gemm_kernels.h.
// Part 2 ("shapes"): a transformer projection (K=1024 -> N=3072, the Qwen3-0.6B
// FFN up-projection) with M = number of tokens processed together. M=1 is
// decode, M>=32 is prefill. Arithmetic intensity (FLOPs per byte of memory
// traffic) grows with M, so the same weights move from memory-bound to
// compute-bound. This is the reason decode and prefill behave so differently.
#include <random>

#include "../common/harness.h"
#include "../common/hwinfo.h"
#include "gemm_kernels.h"

using namespace lab;

static void fill(std::vector<float>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0, 1);
  for (auto& x : v) x = d(rng);
}

// Minimum bytes a GEMM must move if each matrix crosses memory exactly once.
static double gemm_bytes(double M, double N, double K) { return 4 * (M * K + K * N + M * N); }

int main(int argc, char** argv) {
  Args args(argc, argv);
  RunCfg cfg = RunCfg::from(args);
  Recorder rec(args);
  HwInfo hw;
  std::string part = args.str("part", "all");
  int threads = (int)args.num("threads", hw.p_cores);
  double peak1 = args.num("peak-1core", 0), peakN = args.num("peak-all", 0);
  double bw = args.num("bw", 0);
  Pool pool(threads);

  banner("P1.gemm  matrix multiply from scratch",
         "What does each optimisation buy, and where does decode vs prefill sit on the roofline?");

  if (part == "all" || part == "steps") {
    int n = (int)args.num("n", 1024);
    std::printf("  Part 1: C = A·B, %d x %d x %d fp32 (%.2f GFLOP per call)\n", n, n, n, 2.0 * n * n * n / 1e9);
    std::vector<float> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n), R((size_t)n * n);
    fill(A, 1); fill(B, 2);
    gemm::accel(n, n, n, A.data(), B.data(), R.data());
    double flops = 2.0 * n * n * n;
    std::printf("  %-22s %10s %10s %12s %10s\n", "variant", "GFLOP/s", "speedup", "% of NEON pk", "rel err");
    double base = 0;
    auto step = [&](const char* name, bool multi, auto&& fn) {
      Summary s = run_for(name, flops, 1e9, "GFLOP/s", cfg, fn);
      double err = gemm::rel_err(C.data(), R.data(), C.size());
      if (base == 0) base = s.median;
      double pk = multi ? peakN : peak1;
      std::printf("  %-22s %10.2f %9.1fx %12s %10.1e\n", name, s.median, s.median / base,
                  pk > 0 ? (std::to_string((int)(100 * s.median / pk)) + "%").c_str() : "-", err);
      std::fflush(stdout);
      rec.emit_summary("p1.gemm", std::string("steps_") + name, s, "GFLOP/s",
                       {{"n", (double)n}, {"rel_err", err}, {"threads", multi ? (double)threads : 1}});
    };
    if (n <= (int)args.num("naive-max", 1024))
      step("naive (ijk)", false, [&] { gemm::naive(n, n, n, A.data(), B.data(), C.data()); });
    step("ikj (unit stride)", false, [&] { gemm::ikj(n, n, n, A.data(), B.data(), C.data()); });
    step("blocked", false, [&] { gemm::blocked(n, n, n, A.data(), B.data(), C.data()); });
    step("packed+NEON 1T", false, [&] { gemm::packed(n, n, n, A.data(), B.data(), C.data(), nullptr); });
    step(("packed+NEON " + std::to_string(threads) + "T").c_str(), true,
         [&] { gemm::packed(n, n, n, A.data(), B.data(), C.data(), &pool); });
    step("Accelerate (SME)", true, [&] { gemm::accel(n, n, n, A.data(), B.data(), C.data()); });
    std::printf("\n  Accelerate's error is 0 because it is the reference. Errors ~1e-7 are fp32\n");
    std::printf("  rounding from a different summation order, not bugs.\n");
  }

  if (part == "all" || part == "shapes") {
    int K = (int)args.num("k", 1024), N = (int)args.num("nout", 3072);
    std::vector<long> Ms = args.list("m", {1, 4, 16, 32, 64, 128, 256, 512});
    long maxM = *std::max_element(Ms.begin(), Ms.end());
    std::vector<float> A((size_t)maxM * K), B((size_t)K * N), C((size_t)maxM * N), R((size_t)maxM * N);
    fill(A, 3); fill(B, 4);
    std::printf("\n  Part 2: tokens x weights, K=%d -> N=%d (weights %.1f MiB fp32), %d threads\n", K, N,
                4.0 * K * N / (1 << 20), threads);
    if (bw <= 0 || peakN <= 0)
      std::printf("  (no --bw/--peak-all given: roofline columns left blank; run_labs.py fills them)\n");
    std::printf("  %-6s %9s %12s %12s %12s %12s %8s\n", "M", "FLOP/B", "roof GF/s", "packed GF/s",
                "accel GF/s", "accel GB/s", "bound");
    for (long M : Ms) {
      double flops = 2.0 * M * N * K, bytes = gemm_bytes(M, N, K), ai = flops / bytes;
      double roof = (bw > 0 && peakN > 0) ? std::min(peakN, ai * bw) : 0;
      Summary sp = run_for("packed M=" + std::to_string(M), flops, 1e9, "GFLOP/s", cfg,
                           [&] { gemm::packed((int)M, N, K, A.data(), B.data(), C.data(), &pool); });
      Summary sa = run_for("accel M=" + std::to_string(M), flops, 1e9, "GFLOP/s", cfg,
                           [&] { gemm::accel((int)M, N, K, A.data(), B.data(), R.data()); });
      double err = gemm::rel_err(C.data(), R.data(), (size_t)M * N);
      const char* bound = roof == 0 ? "?" : (ai * bw < peakN ? "memory" : "compute");
      std::printf("  %-6ld %9.2f %12s %12.1f %12.1f %12.1f %8s%s\n", M, ai,
                  roof > 0 ? std::to_string((int)roof).c_str() : "-", sp.median, sa.median,
                  sa.median * 1e9 / ai / 1e9, bound, err > 1e-5 ? "  ERR!" : "");
      std::fflush(stdout);
      rec.emit_summary("p1.gemm", "shape_packed", sp, "GFLOP/s",
                       {{"m", (double)M}, {"k", (double)K}, {"n", (double)N}, {"ai", ai}, {"roof", roof},
                        {"rel_err", err}, {"threads", (double)threads}});
      rec.emit_summary("p1.gemm", "shape_accel", sa, "GFLOP/s",
                       {{"m", (double)M}, {"k", (double)K}, {"n", (double)N}, {"ai", ai}, {"roof", roof}});
    }
    std::printf("\n  Note the weights here are %.0f MiB, which fits in the L2/SLC once warm, so even\n",
                4.0 * K * N / (1 << 20));
    std::printf("  M=1 can beat the DRAM roof. The gemv lab streams a model-sized footprint instead.\n");
    std::printf("  'accel GB/s' = GFLOP/s / intensity: the bandwidth the kernel effectively used.\n");
  }
  return 0;
}
