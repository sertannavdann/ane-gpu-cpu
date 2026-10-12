// P2 lab: quantized kernels and their accuracy cost.
//
// Part 1  accuracy vs bits and block size (why llama.cpp uses blocks of 32)
// Part 2  activation outliers: why W8A8 breaks, SmoothQuant and AWQ fixes
// Part 3  decode speed: fp32 / fp16 / Q8_0 / Q4_0 GEMV over a model-sized footprint
#include <random>

#include "../common/harness.h"
#include "../common/hwinfo.h"
#include "../p1/gemv_kernels.h"
#include "quant_kernels.h"

using namespace lab;
using namespace quant;

static double rel(const std::vector<float>& x, const std::vector<float>& ref) {
  double n = 0, d = 0;
  for (size_t i = 0; i < x.size(); ++i) { n += (x[i] - ref[i]) * (x[i] - ref[i]); d += ref[i] * ref[i]; }
  return std::sqrt(n / d);
}

// y[t, r] = sum_k X[t, k] W[r, k]
static std::vector<float> matmul_t(const std::vector<float>& X, const std::vector<float>& W, int T, int R, int K) {
  std::vector<float> Y((size_t)T * R);
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < R; ++r) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += (double)X[(size_t)t * K + k] * W[(size_t)r * K + k];
      Y[(size_t)t * R + r] = (float)s;
    }
  return Y;
}

static void part1(Recorder& rec) {
  std::printf("  Part 1: weight error vs bits and block size (symmetric absmax, fp16 scale)\n");
  std::printf("  Two weight populations: plain Gaussian, and Gaussian with 0.2%% outliers at 12x.\n\n");
  const long n = 1 << 20;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 1);
  std::uniform_real_distribution<float> u(0, 1);
  std::vector<float> g(n), o(n), q(n);
  for (long i = 0; i < n; ++i) { g[i] = nd(rng); o[i] = g[i] * (u(rng) < 0.002f ? 12.0f : 1.0f); }
  std::printf("  %-5s %-6s %10s %14s %14s\n", "bits", "block", "bits/wt", "err gaussian", "err outliers");
  for (int bits : {8, 4, 3}) {
    for (int block : {16, 32, 64, 128, 256, 4096}) {
      fake_quant(g.data(), q.data(), n, bits, block);
      double eg = rel(q, g);
      fake_quant(o.data(), q.data(), n, bits, block);
      double eo = rel(q, o);
      std::printf("  %-5d %-6d %10.3f %14.4f %14.4f\n", bits, block, bits_per_weight(bits, block), eg, eo);
      rec.emit("p2.quant", "block_sweep",
               {{"bits", (double)bits}, {"block", (double)block}, {"bits_per_weight", bits_per_weight(bits, block)},
                {"err_gauss", eg}, {"err_outlier", eo}});
    }
    std::printf("\n");
  }
  std::printf("  Smaller blocks track local magnitude better (lower error) but spend more bits on\n");
  std::printf("  scales: 16 bits / 32 weights = 0.5 bits each. One outlier inflates its block's scale\n");
  std::printf("  and crushes its 31 neighbours toward zero, so outlier error grows with block size.\n\n");
}

static void part2(Recorder& rec) {
  const int T = 64, K = 1024, R = 256;
  std::printf("  Part 2: activation outliers. X[%d tokens, %d channels], W[%d, %d].\n", T, K, R, K);
  std::printf("  1%% of input channels carry 30x larger activations, as seen in real LLMs.\n\n");
  std::mt19937 rng(2);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> X((size_t)T * K), W((size_t)R * K), chan_scale(K, 1.0f);
  for (int k = 0; k < K; ++k) if (k % 100 == 7) chan_scale[k] = 30.0f;
  for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) X[(size_t)t * K + k] = nd(rng) * chan_scale[k];
  for (auto& w : W) w = nd(rng) * 0.02f;
  auto Yref = matmul_t(X, W, T, R, K);

  // (a) SmoothQuant: W8A8 with per-tensor int8 activations.
  std::printf("  (a) W8A8 (int8 weights per row, int8 activations per tensor) with SmoothQuant\n");
  std::printf("      s_k = max|X_k|^a / max|W_k|^(1-a); X' = X/s, W' = W*s, then quantize both.\n");
  std::printf("  %-8s %14s\n", "alpha", "output err");
  std::vector<float> xmax(K, 0), wmax(K, 0);
  for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) xmax[k] = std::max(xmax[k], std::fabs(X[(size_t)t * K + k]));
  for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) wmax[k] = std::max(wmax[k], std::fabs(W[(size_t)r * K + k]));
  for (double a : {0.0, 0.25, 0.5, 0.75}) {
    std::vector<float> s(K), Xs(X), Ws(W);
    for (int k = 0; k < K; ++k) s[k] = (float)(std::pow(xmax[k], a) / std::pow(wmax[k], 1 - a));
    if (a == 0.0) std::fill(s.begin(), s.end(), 1.0f);  // baseline: no smoothing
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) Xs[(size_t)t * K + k] /= s[k];
    for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) Ws[(size_t)r * K + k] *= s[k];
    std::vector<float> Xq(Xs.size()), Wq(Ws.size());
    fake_quant(Xs.data(), Xq.data(), (long)Xs.size(), 8, (int)Xs.size());  // one scale for the tensor
    for (int r = 0; r < R; ++r) fake_quant(&Ws[(size_t)r * K], &Wq[(size_t)r * K], K, 8, K);
    double e = rel(matmul_t(Xq, Wq, T, R, K), Yref);
    std::printf("  %-8s %14.4f\n", a == 0 ? "none" : std::to_string(a).substr(0, 4).c_str(), e);
    rec.emit("p2.quant", "smoothquant", {{"alpha", a}, {"err", e}});
  }
  std::printf("  Without smoothing the 30x channels set the activation scale and the other 99%%\n");
  std::printf("  of channels get only a few int8 levels. Smoothing moves that range into the\n");
  std::printf("  weights, which are quantized per row and tolerate it.\n\n");

  // (b) AWQ-style: weight-only Q4 (block 32), activations fp32.
  std::printf("  (b) W4A16 (4-bit weights, block 32; activations stay float) with AWQ-style scaling\n");
  std::printf("      s_k = mean|X_k|^a; quantize W*s, run with X/s. Salient channels gain precision.\n");
  std::printf("  %-8s %14s\n", "alpha", "output err");
  std::vector<float> xmean(K, 0);
  for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) xmean[k] += std::fabs(X[(size_t)t * K + k]) / T;
  double best = 1e9, best_a = 0;
  for (double a : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    std::vector<float> s(K), Ws(W), Wq(W.size()), Xs(X);
    for (int k = 0; k < K; ++k) s[k] = (float)std::pow(xmean[k], a);
    for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) Ws[(size_t)r * K + k] *= s[k];
    for (int r = 0; r < R; ++r) fake_quant(&Ws[(size_t)r * K], &Wq[(size_t)r * K], K, 4, 32);
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) Xs[(size_t)t * K + k] /= s[k];
    double e = rel(matmul_t(Xs, Wq, T, R, K), Yref);
    std::printf("  %-8.2f %14.4f%s\n", a, e, a == 0 ? "   (plain round-to-nearest)" : "");
    rec.emit("p2.quant", "awq", {{"alpha", a}, {"err", e}});
    if (e < best) { best = e; best_a = a; }
  }
  std::printf("  Best alpha %.2f. AWQ searches alpha per layer on calibration data exactly like this;\n",
              best_a);
  std::printf("  too large an alpha over-scales and the block's other weights lose precision.\n");
  std::printf("  GPTQ goes further: it quantizes columns one at a time and updates the remaining\n");
  std::printf("  weights to cancel the error, using second-order (Hessian) information from X.\n\n");
}

static void part3(const Args& args, const RunCfg& cfg, Recorder& rec, const HwInfo& hw) {
  const int R = 3072, K = 1024;
  const double footprint_w = args.num("footprint-mweights", 256) * 1e6;
  const int L = std::max(1, (int)(footprint_w / ((double)R * K)));
  const long nw = (long)R * K, nbk = nw / QK;
  std::printf("  Part 3: decode GEMV over %d layers of %d x %d (%.0f M weights per 'token')\n", L, R, K,
              (double)L * nw / 1e6);
  std::printf("  Bytes per token: fp32 %.0f MiB, fp16 %.0f MiB, Q8_0 %.0f MiB, Q4_0 %.0f MiB\n\n",
              4.0 * L * nw / (1 << 20), 2.0 * L * nw / (1 << 20), 34.0 / 32 * L * nw / (1 << 20),
              18.0 / 32 * L * nw / (1 << 20));

  uint64_t st = 0x2545F4914F6CDD1Dull;
  auto urand = [&] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (float)(st >> 40) / (1 << 24) - 0.5f; };
  std::vector<std::vector<float>> W32(L);
  std::vector<std::vector<float16_t>> W16(L);
  std::vector<std::vector<block_q8_0>> W8(L);
  std::vector<std::vector<block_q4_0>> W4(L);
  for (int l = 0; l < L; ++l) {
    W32[l].resize(nw); W16[l].resize(nw); W8[l].resize(nbk); W4[l].resize(nbk);
    for (long i = 0; i < nw; ++i) { W32[l][i] = 0.07f * urand(); W16[l][i] = (float16_t)W32[l][i]; }
    quantize_q8_0(W32[l].data(), W8[l].data(), nw);
    quantize_q4_0(W32[l].data(), W4[l].data(), nw);
  }
  std::vector<float> x(K), y(R), yref(R);
  for (auto& v : x) v = 2.0f * urand();
  std::vector<block_q8_0> xq(K / QK);
  gemv::accel(R, K, W32[0].data(), x.data(), yref.data());

  struct Fmt { const char* name; double bytes_per_w; };
  const Fmt fmts[] = {{"fp32", 4}, {"fp16", 2}, {"Q8_0", 34.0 / 32}, {"Q4_0", 18.0 / 32}};
  std::vector<long> threads = args.list("threads", {1, 4, hw.p_cores});
  std::printf("  %-6s %8s %10s %12s %10s %12s %10s\n", "format", "threads", "GB/s", "Gweights/s",
              "vs fp16", "byte-ratio", "rel err");
  for (long t : threads) {
    Pool pool((int)t);
    double gw16 = 0;
    for (const Fmt& f : fmts) {
      std::string fname = f.name;
      auto layer = [&](int l) {
        pool.run([&](int tid) {
          auto [a, b] = Pool::range(R, tid, (int)t, 8);
          if (fname == "fp32") gemv::f32(a, b, K, W32[l].data(), x.data(), y.data());
          else if (fname == "fp16") gemv::f16(a, b, K, W16[l].data(), x.data(), y.data());
          else if (fname == "Q8_0") for (long r = a; r < b; ++r) y[r] = dot_q8_0_neon(&W8[l][r * K / QK], xq.data(), K);
          else for (long r = a; r < b; ++r) y[r] = dot_q4_0_neon(&W4[l][r * K / QK], xq.data(), K);
        });
      };
      double weights = (double)L * nw;
      Summary s = run_for(fname + " t=" + std::to_string(t), weights, 1e9, "Gw/s", cfg, [&] {
        quantize_q8_0(x.data(), xq.data(), K);  // activations -> Q8_0 once per token, as llama.cpp does
        for (int l = 0; l < L; ++l) layer(l);
      });
      quantize_q8_0(x.data(), xq.data(), K);
      layer(0);
      double err = rel(y, yref);
      if (fname == "fp16") gw16 = s.median;
      std::printf("  %-6s %8ld %10.1f %12.2f %10s %12s %10.1e\n", f.name, t, s.median * f.bytes_per_w,
                  s.median, gw16 > 0 ? (std::to_string(s.median / gw16).substr(0, 4) + "x").c_str() : "-",
                  (std::to_string(2.0 / f.bytes_per_w).substr(0, 4) + "x").c_str(), err);
      std::fflush(stdout);
      rec.emit_summary("p2.quant", std::string("gemv_") + f.name, s, "Gweights/s",
                       {{"threads", (double)t}, {"gbs", s.median * f.bytes_per_w},
                        {"bytes_per_weight", f.bytes_per_w}, {"rel_err", err}});
    }
    std::printf("\n");
  }
  std::printf("  'byte-ratio' is the speedup over fp16 predicted if decode were purely bandwidth-bound.\n");
  std::printf("  When the measured speedup falls short, the kernel has become compute-bound: unpacking\n");
  std::printf("  nibbles and scaling blocks costs instructions, and a single core cannot issue them\n");
  std::printf("  fast enough. More threads bring Q4 back toward the bandwidth roof.\n");
  std::printf("  Q4_0's output error (a few %%) means nothing on its own: judge 'good enough' with\n");
  std::printf("  perplexity and a task fixture on a real model (see the P2 notes).\n");
}

int main(int argc, char** argv) {
  Args args(argc, argv);
  RunCfg cfg = RunCfg::from(args);
  Recorder rec(args);
  HwInfo hw;
  std::string part = args.str("part", "all");
  banner("P2.quant  4-bit and 8-bit block quantization",
         "What does a block format cost in accuracy, and what does it buy in decode speed?");
  if (part == "all" || part == "accuracy") part1(rec);
  if (part == "all" || part == "outliers") part2(rec);
  if (part == "all" || part == "speed") part3(args, cfg, rec, hw);
  return 0;
}
