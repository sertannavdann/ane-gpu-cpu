// What the machine says about itself, via sysctl.
#pragma once

#include <sys/sysctl.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace lab {

inline long sysctl_long(const char* name, long fallback = -1) {
  int64_t v = 0;
  size_t sz = sizeof(v);
  if (sysctlbyname(name, &v, &sz, nullptr, 0) != 0) return fallback;
  return (long)v;
}

inline std::string sysctl_str(const char* name) {
  char buf[256] = {0};
  size_t sz = sizeof(buf);
  if (sysctlbyname(name, buf, &sz, nullptr, 0) != 0) return "?";
  return buf;
}

struct HwInfo {
  std::string cpu = sysctl_str("machdep.cpu.brand_string");
  long p_cores = sysctl_long("hw.perflevel0.physicalcpu");
  long e_cores = sysctl_long("hw.perflevel1.physicalcpu");
  long p_l1d = sysctl_long("hw.perflevel0.l1dcachesize");
  long p_l2 = sysctl_long("hw.perflevel0.l2cachesize");
  long e_l1d = sysctl_long("hw.perflevel1.l1dcachesize");
  long e_l2 = sysctl_long("hw.perflevel1.l2cachesize");
  long line = sysctl_long("hw.cachelinesize");
  long page = sysctl_long("hw.pagesize");
  long mem = sysctl_long("hw.memsize");

  // Apple spec-sheet DRAM bandwidth, keyed on the P-core count of the bin.
  // This is a marketing peak, not a measurement: the bandwidth lab measures it.
  double spec_bw_gbs() const {
    if (cpu.find("M4 Max") != std::string::npos) return p_cores >= 12 ? 546 : 410;
    if (cpu.find("M4 Pro") != std::string::npos) return 273;
    if (cpu.find("M4") != std::string::npos) return 120;
    return 0;
  }
  int cores() const { return (int)(p_cores + e_cores); }
};

// Name the cache level a working set of `bytes` most likely lives in.
inline const char* level_for(const HwInfo& h, double bytes, int threads) {
  if (bytes <= h.p_l1d * 0.75 * threads) return "L1";
  if (bytes <= h.p_l2 * 0.75) return "L2";
  if (bytes <= 64.0 * 1024 * 1024) return "L2/SLC";
  return "DRAM";
}

}  // namespace lab
