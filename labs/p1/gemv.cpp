// P1 lab: decode is a bandwidth problem.
//
// A model's decode step reads every weight once per token. To reproduce that,
// this lab allocates a stack of "layers" whose total size (--footprint-mib,
// default 1024) is far larger than every cache, and runs one GEMV per layer
// per call, like one token through the model. Throughput is reported as GB/s
// of weights streamed and as the tokens/s this rate implies for a model of the
// same size: tokens/s = bandwidth / bytes of weights.
#include "../common/harness.h"
#include "../common/hwinfo.h"
#include "gemv_kernels.h"

using namespace lab;

int main(int argc, char** argv) {
  Args args(argc, argv);
  RunCfg cfg = RunCfg::from(args);
  Recorder rec(args);
  HwInfo hw;
  const int R = (int)args.num("rows", 3072), K = (int)args.num("k", 1024);
  const double footprint = args.num("footprint-mib", 1024) * (1 << 20);
  const int L = std::max(1, (int)(footprint / (4.0 * R * K)));
  double bw_ref = args.num("bw", 0);

  banner("P1.gemv  decode = matrix-vector, bandwidth-bound",
         "How close does a decode-style GEMV get to the DRAM roof, and what do fewer bytes buy?");
  std::printf("  %d layers of %d x %d weights: %.0f MiB fp32 / %.0f MiB fp16 per 'token'\n\n", L, R, K,
              4.0 * L * R * K / (1 << 20), 2.0 * L * R * K / (1 << 20));

  // Cheap xorshift fill: a gigabyte of std::normal_distribution takes seconds.
  uint64_t st = 0x9E3779B97F4A7C15ull;
  auto urand = [&] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (float)(st >> 40) / (1 << 24) - 0.5f; };
  std::vector<std::vector<float>> W32(L, std::vector<float>((size_t)R * K));
  std::vector<std::vector<float16_t>> W16(L, std::vector<float16_t>((size_t)R * K));
  for (int l = 0; l < L; ++l)
    for (size_t i = 0; i < W32[l].size(); ++i) { W32[l][i] = 0.07f * urand(); W16[l][i] = (float16_t)W32[l][i]; }
  std::vector<float> x(K), y(R), yref(R);
  for (auto& v : x) v = 2.0f * urand();
  std::vector<float16_t> xh(K);
  for (int k = 0; k < K; ++k) xh[k] = (float16_t)x[k];

  // Correctness: every kernel against Accelerate on layer 0.
  gemv::accel(R, K, W32[0].data(), x.data(), yref.data());
  // Each benchmark leaves the LAST layer's output in y; check() takes a function
  // that recomputes layer 0 so it can be compared with the layer-0 reference.
  auto check = [&](auto&& layer0) {
    layer0();
    double num = 0, den = 0;
    for (int i = 0; i < R; ++i) { num += (y[i] - yref[i]) * (y[i] - yref[i]); den += yref[i] * yref[i]; }
    return std::sqrt(num / den);
  };

  std::vector<long> threads = args.list("threads", {1, 2, 4, 8, hw.p_cores, (long)hw.cores()});
  std::printf("  %-26s %8s %10s %10s %14s %10s\n", "kernel", "threads", "GB/s", "GFLOP/s", "tok/s @1GiB", "rel err");
  double best16 = 0, best32 = 0;
  auto report = [&](const std::string& name, long t, const Summary& s, double bytes_per_w, double err) {
    double gflops = s.median / bytes_per_w * 2;
    std::printf("  %-26s %8s %10.1f %10.1f %14.1f %10.1e\n", name.c_str(),
                t == 0 ? "auto" : std::to_string(t).c_str(), s.median, gflops,
                s.median / 1.073741824, err);
    std::fflush(stdout);
    rec.emit_summary("p1.gemv", name, s, "GB/s",
                     {{"threads", (double)t}, {"bytes_per_weight", bytes_per_w}, {"rel_err", err}});
  };

  double bytes32 = 4.0 * L * R * K, bytes16 = 2.0 * L * R * K;
  {
    Pool one(1);
    Summary s = run_for("scalar", bytes32, 1e9, "GB/s", cfg, [&] {
      for (int l = 0; l < L; ++l) gemv::scalar(0, R, K, W32[l].data(), x.data(), y.data());
    });
    report("scalar fp32 (1 chain)", 1, s, 4, check([&] { gemv::scalar(0, R, K, W32[0].data(), x.data(), y.data()); }));
  }
  {
    Summary s = run_for("accel", bytes32, 1e9, "GB/s", cfg, [&] {
      for (int l = 0; l < L; ++l) gemv::accel(R, K, W32[l].data(), x.data(), y.data());
    });
    report("Accelerate sgemv fp32", 0, s, 4, check([&] { gemv::accel(R, K, W32[0].data(), x.data(), y.data()); }));
  }
  for (long t : threads) {
    Pool pool((int)t);
    Summary s32 = run_for("f32 t=" + std::to_string(t), bytes32, 1e9, "GB/s", cfg, [&] {
      for (int l = 0; l < L; ++l)
        pool.run([&](int tid) {
          auto [a, b] = Pool::range(R, tid, (int)t, 8);
          gemv::f32(a, b, K, W32[l].data(), x.data(), y.data());
        });
    });
    report("NEON fp32", t, s32, 4, check([&] { gemv::f32(0, R, K, W32[0].data(), x.data(), y.data()); }));
    Summary s16 = run_for("f16 t=" + std::to_string(t), bytes16, 1e9, "GB/s", cfg, [&] {
      for (int l = 0; l < L; ++l)
        pool.run([&](int tid) {
          auto [a, b] = Pool::range(R, tid, (int)t, 8);
          gemv::f16(a, b, K, W16[l].data(), x.data(), y.data());
        });
    });
    report("NEON fp16 (convert+fma)", t, s16, 2, check([&] { gemv::f16(0, R, K, W16[0].data(), x.data(), y.data()); }));
    Summary sh = run_for("f16 fmlal t=" + std::to_string(t), bytes16, 1e9, "GB/s", cfg, [&] {
      for (int l = 0; l < L; ++l)
        pool.run([&](int tid) {
          auto [a, b] = Pool::range(R, tid, (int)t, 8);
          gemv::f16_fmlal(a, b, K, W16[l].data(), xh.data(), y.data());
        });
    });
    report("NEON fp16 (fmlal)", t, sh, 2, check([&] { gemv::f16_fmlal(0, R, K, W16[0].data(), xh.data(), y.data()); }));
    s16.median = std::max(s16.median, sh.median);
    best32 = std::max(best32, s32.median);
    best16 = std::max(best16, s16.median);
  }
  rec.emit("p1.gemv", "best", {{"f32_gbs", best32}, {"f16_gbs", best16}, {"bw_ref", bw_ref}});
  std::printf("\n  'tok/s @1GiB' = tokens/s for a model with 1 GiB of weights at this rate.\n");
  std::printf("  At equal GB/s, fp16 doubles tokens/s because each weight is half the bytes.\n");
  if (bw_ref > 0)
    std::printf("  Best fp32 GEMV: %.0f%% of the measured DRAM read bandwidth (%.0f GB/s).\n",
                100 * best32 / bw_ref, bw_ref);
  std::printf("  fp16 error ~1e-4 is weight rounding (fmlal also rounds x to fp16), not the kernel.\n");
  std::printf("  If fp16 GB/s is below fp32 GB/s at the same thread count, that kernel is limited by\n");
  std::printf("  instructions, not memory: compare convert+fma (2 extra converts per 8 weights) with fmlal.\n");
  return 0;
}
