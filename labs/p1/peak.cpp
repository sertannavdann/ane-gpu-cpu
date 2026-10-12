// P1 lab: the compute ceiling.
//
// A fused multiply-add (FMA) on a 128-bit NEON register does 4 fp32 (or 8
// fp16) multiply-adds = 8 (16) FLOPs. A core has several FMA pipes, but each
// FMA takes a few cycles to produce its result. If every FMA needs the previous
// one's result (one dependency chain), the core waits out the full latency
// each time. With enough independent chains, all pipes stay busy.
//
//   chains needed to saturate ≈ pipes × latency  (e.g. 4 pipes × 4 cycles = 16)
//
// This lab measures both regimes, then scales across threads and core types.
//
// Write the chain as acc = vfmaq(acc, a, b), i.e. acc += a*b. The first
// version of this lab wrote vfmaq(b, acc, a) = b + acc*a; the compiler then had
// to copy b into a fresh register and move the result back every iteration,
// so each link of the chain was fmla + mov (≈6 cycles), not fmla (4 cycles).
// Check the assembly (objdump -d) before trusting a latency microbenchmark.
#include <arm_neon.h>

#include "../common/harness.h"
#include "../common/hwinfo.h"
#include "../common/pool.h"

using namespace lab;

static constexpr long kIters = 1 << 16;

template <int C>
__attribute__((noinline)) float32x4_t fma_chains_f32(long iters) {
  float32x4_t acc[C];
  for (int c = 0; c < C; ++c) acc[c] = vdupq_n_f32(0.001f * c);
  const float32x4_t a = vdupq_n_f32(0.9999f), b = vdupq_n_f32(0.0001f);
  for (long i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = vfmaq_f32(acc[c], a, b);  // acc += a*b: one fmla per link
  }
  float32x4_t s = acc[0];
  for (int c = 1; c < C; ++c) s = vaddq_f32(s, acc[c]);
  return s;
}

template <int C>
__attribute__((noinline)) float16x8_t fma_chains_f16(long iters) {
  float16x8_t acc[C];
  for (int c = 0; c < C; ++c) acc[c] = vdupq_n_f16((float16_t)(0.001f * c));
  const float16x8_t a = vdupq_n_f16((float16_t)0.999f), b = vdupq_n_f16((float16_t)0.001f);
  for (long i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = vfmaq_f16(acc[c], a, b);
  }
  float16x8_t s = acc[0];
  for (int c = 1; c < C; ++c) s = vaddq_f16(s, acc[c]);
  return s;
}

template <int C>
static Summary measure_f32(Pool& pool, const RunCfg& cfg, const std::string& label) {
  double flops = (double)kIters * C * 4 * 2 * pool.size();
  return run_for(label, flops, 1e9, "GFLOP/s", cfg, [&] {
    pool.run([&](int) { keep(fma_chains_f32<C>(kIters)); });
  });
}

template <int C>
static Summary measure_f16(Pool& pool, const RunCfg& cfg, const std::string& label) {
  double flops = (double)kIters * C * 8 * 2 * pool.size();
  return run_for(label, flops, 1e9, "GFLOP/s", cfg, [&] {
    pool.run([&](int) { keep(fma_chains_f16<C>(kIters)); });
  });
}

int main(int argc, char** argv) {
  Args args(argc, argv);
  RunCfg cfg = RunCfg::from(args);
  Recorder rec(args);
  HwInfo hw;
  const double assumed_latency = args.num("fma-latency", 4);  // cycles, Apple P-core assumption
  const std::string part = args.str("part", "all");  // all | chains | f16 | threads | soak

  if (part == "soak") {
    // Hold every chosen core at full FMA throughput for --seconds and print the
    // rate each --interval. A falling rate means the clock dropped (power or
    // thermal limit); other busy apps show up as dips and noise.
    int t = (int)args.num("threads", hw.p_cores);
    double seconds = args.num("seconds", 30), interval = args.num("interval", 1.0);
    Pool pool(t, QOS_CLASS_USER_INTERACTIVE);
    banner("P1.peak soak  sustained FMA load",
           "Does throughput hold over time, or does the chip throttle under sustained load?");
    std::printf("  %d threads x 16 chains fp32 for %.0f s, one line per %.1f s\n\n", t, seconds, interval);
    std::printf("  %8s %12s %12s %12s\n", "t (s)", "GFLOP/s", "per thread", "% of first");
    double start = now_s(), first = 0, flops_per_call = (double)kIters * 16 * 8 * t;
    std::vector<double> rates;
    while (now_s() - start < seconds) {
      double a = now_s();
      long calls = 0;
      while (now_s() - a < interval) {
        pool.run([&](int) { keep(fma_chains_f32<16>(kIters)); });
        ++calls;
      }
      double g = calls * flops_per_call / (now_s() - a) / 1e9;
      if (first == 0) first = g;
      rates.push_back(g);
      std::printf("  %8.1f %12.1f %12.1f %11.1f%%\n", now_s() - start, g, g / t, 100 * g / first);
      std::fflush(stdout);
      rec.emit("p1.peak", "soak", {{"t", now_s() - start}, {"gflops", g}, {"threads", (double)t}});
    }
    Summary sm = summarize(rates, now_s() - start);
    std::printf("\n  median %.1f, min %.1f, max %.1f GFLOP/s; last/first = %.1f%%\n", sm.median, sm.min, sm.max,
                100 * rates.back() / rates.front());
    return 0;
  }

  banner("P1.peak  FMA latency vs throughput",
         "How many FLOP/s can one core do, and why do independent chains matter?");

  // Part 1: one P-core, vary the number of independent chains.
  if (part == "all" || part == "chains") {
    Pool one(1, QOS_CLASS_USER_INTERACTIVE);
    std::printf("  Part 1: one P-core, fp32, vary independent FMA chains\n");
    std::printf("  %-8s %12s %10s %14s\n", "chains", "GFLOP/s", "spread%", "FMA instr/ns");
    double ghz_est = 0;
    auto row = [&](int c, const Summary& s) {
      double instr_per_ns = s.median / 8.0;  // 8 FLOPs per fp32x4 FMA
      std::printf("  %-8d %12.2f %10.1f %14.2f\n", c, s.median, s.spread_pct(), instr_per_ns);
      rec.emit_summary("p1.peak", "chains_f32", s, "GFLOP/s", {{"chains", (double)c}, {"threads", 1}});
      if (c == 1) ghz_est = instr_per_ns * assumed_latency;
    };
    row(1, measure_f32<1>(one, cfg, "f32 chains=1"));
    row(2, measure_f32<2>(one, cfg, "f32 chains=2"));
    row(4, measure_f32<4>(one, cfg, "f32 chains=4"));
    row(8, measure_f32<8>(one, cfg, "f32 chains=8"));
    row(12, measure_f32<12>(one, cfg, "f32 chains=12"));
    row(16, measure_f32<16>(one, cfg, "f32 chains=16"));
    row(24, measure_f32<24>(one, cfg, "f32 chains=24"));
    std::printf("\n  With 1 chain each FMA waits for the previous one: time per FMA = latency.\n");
    std::printf("  If the FMA latency is %.0f cycles, the core clock is ≈ %.2f GHz.\n", assumed_latency,
                ghz_est);
    std::printf("  (That clock is inferred from an assumed latency, not read from hardware.)\n");
    rec.emit("p1.peak", "clock_estimate", {{"ghz", ghz_est}, {"assumed_latency", assumed_latency}});
  }

  // Part 2: fp16 doubles the lanes per register, so it doubles peak FLOP/s.
  if (part == "all" || part == "f16") {
    Pool one(1, QOS_CLASS_USER_INTERACTIVE);
    std::printf("\n  Part 2: one P-core, fp16 vs fp32 at 16 chains\n");
    Summary s32 = measure_f32<16>(one, cfg, "f32 x16");
    Summary s16 = measure_f16<16>(one, cfg, "f16 x16");
    std::printf("  fp32: %8.2f GFLOP/s   fp16: %8.2f GFLOP/s   ratio %.2fx\n", s32.median, s16.median,
                s16.median / s32.median);
    rec.emit_summary("p1.peak", "f16_1core", s16, "GFLOP/s", {{"threads", 1}});
  }

  // Part 3: scale threads; then E-cores only.
  if (part != "all" && part != "threads") return 0;
  std::vector<long> threads = args.list("threads", {1, 2, 4, 8, hw.p_cores, (long)hw.cores()});
  std::printf("\n  Part 3: fp32 peak vs thread count (QoS user-interactive: P-cores first)\n");
  std::printf("  %-8s %12s %14s %10s\n", "threads", "GFLOP/s", "per thread", "spread%");
  double best_all = 0;
  for (long t : threads) {
    Pool pool((int)t, QOS_CLASS_USER_INTERACTIVE);
    Summary s = measure_f32<16>(pool, cfg, "f32 x16 t=" + std::to_string(t));
    std::printf("  %-8ld %12.1f %14.1f %10.1f\n", t, s.median, s.median / t, s.spread_pct());
    rec.emit_summary("p1.peak", "threads_f32", s, "GFLOP/s", {{"threads", (double)t}},
                     {{"qos", "user-interactive"}});
    best_all = std::max(best_all, s.median);
  }
  {
    Pool pool((int)hw.e_cores, QOS_CLASS_BACKGROUND);
    Summary s = measure_f32<16>(pool, cfg, "E-cores only");
    std::printf("  %-8s %12.1f %14.1f %10.1f   <- QoS background = E-cores only\n",
                (std::to_string(hw.e_cores) + "E").c_str(), s.median, s.median / hw.e_cores,
                s.spread_pct());
    rec.emit_summary("p1.peak", "ecores_f32", s, "GFLOP/s", {{"threads", (double)hw.e_cores}},
                     {{"qos", "background"}});
  }
  rec.emit("p1.peak", "ceiling", {{"gflops_f32_all", best_all}});
  std::printf("\n  NEON fp32 ceiling for this machine (all cores): %.0f GFLOP/s.\n", best_all);
  std::printf("  This is the 'compute roof' used by the GEMM and roofline labs. Accelerate's\n");
  std::printf("  sgemm can beat it because it uses the separate SME matrix unit, not NEON.\n");
  return 0;
}
