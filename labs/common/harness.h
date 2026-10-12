// Timing harness shared by every lab (the P7 v0 core).
//
// Rules it enforces, because each one is a classic way to fool yourself:
//   1. Warm up before measuring (page faults, cold caches, CPU clock ramp-up).
//   2. Make each timed sample long enough (>= 2 ms) that timer cost is noise.
//   3. Run for a fixed wall-clock budget and keep every sample, then report the
//      median and the 10th/90th percentiles instead of a single best run.
//   4. Write machine-readable records so results can be compared later.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lab {

inline double now_s() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// --key=value or --key value; bare --flag means "1".
struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) continue;
      a = a.substr(2);
      auto eq = a.find('=');
      if (eq != std::string::npos) {
        kv[a.substr(0, eq)] = a.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv[a] = argv[++i];
      } else {
        kv[a] = "1";
      }
    }
  }
  bool has(const char* k) const { return kv.count(k) != 0; }
  double num(const char* k, double d) const { return has(k) ? std::atof(kv.at(k).c_str()) : d; }
  std::string str(const char* k, const std::string& d) const { return has(k) ? kv.at(k) : d; }
  std::vector<long> list(const char* k, std::vector<long> d) const {
    if (!has(k)) return d;
    std::vector<long> out;
    std::string s = kv.at(k);
    size_t p = 0;
    while (p < s.size()) {
      size_t q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      out.push_back(std::atol(s.substr(p, q - p).c_str()));
      p = q + 1;
    }
    return out;
  }
};

struct Summary {
  double median = 0, p10 = 0, p90 = 0, min = 0, max = 0;
  int samples = 0;
  double seconds = 0;
  double spread_pct() const { return median > 0 ? 100.0 * (p90 - p10) / median : 0; }
};

inline double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  double pos = q * (v.size() - 1);
  size_t lo = (size_t)pos, hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

inline Summary summarize(const std::vector<double>& v, double seconds) {
  Summary s;
  if (v.empty()) return s;
  s.median = percentile(v, 0.5);
  s.p10 = percentile(v, 0.1);
  s.p90 = percentile(v, 0.9);
  s.min = *std::min_element(v.begin(), v.end());
  s.max = *std::max_element(v.begin(), v.end());
  s.samples = (int)v.size();
  s.seconds = seconds;
  return s;
}

struct RunCfg {
  double seconds = 1.0;    // measurement budget per case
  double interval = 0.5;   // --watch print period
  bool watch = false;      // print a live line every interval
  double min_sample = 2e-3;
  static RunCfg from(const Args& a) {
    RunCfg c;
    c.seconds = a.num("seconds", 1.0);
    c.interval = a.num("interval", 0.5);
    c.watch = a.has("watch");
    return c;
  }
};

// Calls body() repeatedly for cfg.seconds and returns throughput samples in
// (work units per second) / unit_scale, e.g. bytes/s / 1e9 = GB/s.
// `work_per_call` is how much work one body() call does (bytes, FLOPs, loads).
template <class F>
Summary run_for(const std::string& label, double work_per_call, double unit_scale,
                const char* unit, const RunCfg& cfg, F&& body) {
  // Warm-up: at least 2 calls and 50 ms.
  double t0 = now_s();
  int warm = 0;
  while (warm < 2 || now_s() - t0 < 0.05) { body(); ++warm; }

  // Calibrate how many calls make one sample of >= min_sample seconds.
  long reps = 1;
  for (;;) {
    double a = now_s();
    for (long r = 0; r < reps; ++r) body();
    double dt = now_s() - a;
    if (dt >= cfg.min_sample || reps > (1L << 30)) break;
    reps *= 2;
  }

  std::vector<double> rates;
  double start = now_s(), next_print = start + cfg.interval;
  std::vector<double> window;
  for (;;) {
    double a = now_s();
    for (long r = 0; r < reps; ++r) body();
    double b = now_s();
    double rate = reps * work_per_call / (b - a) / unit_scale;
    rates.push_back(rate);
    window.push_back(rate);
    if (cfg.watch && b >= next_print) {
      std::printf("    t=%6.2fs  %-28s %10.2f %s  (window min %.2f, max %.2f)\n", b - start,
                  label.c_str(), percentile(window, 0.5), unit,
                  *std::min_element(window.begin(), window.end()),
                  *std::max_element(window.begin(), window.end()));
      std::fflush(stdout);
      window.clear();
      next_print += cfg.interval;
    }
    if (b - start >= cfg.seconds) break;
  }
  return summarize(rates, now_s() - start);
}

// One JSON object per line, appended to --json <path> when given.
class Recorder {
 public:
  explicit Recorder(const Args& a) {
    if (a.has("json")) f_ = std::fopen(a.str("json", "").c_str(), "a");
  }
  ~Recorder() { if (f_) std::fclose(f_); }
  void emit(const std::string& lab, const std::string& kase,
            std::vector<std::pair<std::string, double>> nums,
            std::vector<std::pair<std::string, std::string>> strs = {}) {
    if (!f_) return;
    std::fprintf(f_, "{\"lab\":\"%s\",\"case\":\"%s\"", lab.c_str(), kase.c_str());
    for (auto& [k, v] : strs) std::fprintf(f_, ",\"%s\":\"%s\"", k.c_str(), v.c_str());
    for (auto& [k, v] : nums) {
      if (std::isfinite(v)) std::fprintf(f_, ",\"%s\":%.6g", k.c_str(), v);
      else std::fprintf(f_, ",\"%s\":null", k.c_str());
    }
    std::fprintf(f_, "}\n");
    std::fflush(f_);
  }
  void emit_summary(const std::string& lab, const std::string& kase, const Summary& s,
                    const char* unit, std::vector<std::pair<std::string, double>> extra = {},
                    std::vector<std::pair<std::string, std::string>> strs = {}) {
    std::vector<std::pair<std::string, double>> nums = {
        {"median", s.median}, {"p10", s.p10}, {"p90", s.p90}, {"samples", (double)s.samples}};
    nums.insert(nums.end(), extra.begin(), extra.end());
    strs.push_back({"unit", unit});
    emit(lab, kase, nums, strs);
  }

 private:
  FILE* f_ = nullptr;
};

inline void banner(const char* title, const char* question) {
  std::printf("\n== %s ==\n   Question: %s\n\n", title, question);
}

// Keeps the optimizer from deleting a computation whose result is unused.
template <class T>
inline void keep(const T& v) {
  asm volatile("" : : "r,m"(v) : "memory");
}

}  // namespace lab
