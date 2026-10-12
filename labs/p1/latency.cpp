// P1 lab: memory latency and the cache hierarchy.
//
// Pointer chasing: each load's address comes from the previous load, so the
// CPU cannot overlap them. Time per load = latency of wherever the data lives.
// One node per cache line, visited in random order so the hardware prefetcher
// cannot guess the next address. The same chase in sequential order shows what
// the prefetcher hides.
#include <cstdint>
#include <numeric>
#include <random>

#include "../common/harness.h"
#include "../common/hwinfo.h"

using namespace lab;

struct Chain {
  std::vector<char> buf;
  void* head = nullptr;
  Chain(size_t bytes, size_t stride, bool random) : buf(bytes + stride) {
    size_t nodes = std::max<size_t>(2, bytes / stride);
    std::vector<size_t> order(nodes);
    std::iota(order.begin(), order.end(), 0);
    if (random) {
      // Sattolo's algorithm: a random permutation that is one single cycle.
      std::mt19937_64 rng(42);
      for (size_t i = nodes - 1; i > 0; --i) {
        size_t j = std::uniform_int_distribution<size_t>(0, i - 1)(rng);
        std::swap(order[i], order[j]);
      }
    }
    char* base = (char*)(((uintptr_t)buf.data() + stride - 1) / stride * stride);
    for (size_t i = 0; i < nodes; ++i) {
      void** node = (void**)(base + order[i] * stride);
      *node = base + order[(i + 1) % nodes] * stride;
    }
    head = base + order[0] * stride;
  }
};

__attribute__((noinline)) static void* chase(void* p, long steps) {
  for (long i = 0; i < steps; i += 8) {
    p = *(void**)p; p = *(void**)p; p = *(void**)p; p = *(void**)p;
    p = *(void**)p; p = *(void**)p; p = *(void**)p; p = *(void**)p;
  }
  return p;
}

int main(int argc, char** argv) {
  Args args(argc, argv);
  RunCfg cfg = RunCfg::from(args);
  Recorder rec(args);
  HwInfo hw;
  size_t stride = (size_t)args.num("stride", hw.line);
  const long steps = 1 << 16;

  banner("P1.latency  pointer-chase load latency",
         "How long does one dependent load take from L1, L2, the SLC and DRAM?");
  std::printf("  stride %zu B (one node per cache line), %ld dependent loads per call\n\n", stride,
              steps);
  std::printf("  %-10s %-7s %14s %14s\n", "set", "level", "random ns", "sequential ns");

  std::vector<long> kib = args.list(
      "sizes-kib", {8, 32, 64, 112, 192, 512, 2048, 8192, 14336, 24576, 49152, 131072, 524288});
  for (long k : kib) {
    size_t bytes = (size_t)k * 1024;
    double ns[2];
    for (int r = 0; r < 2; ++r) {
      Chain ch(bytes, stride, r == 0);
      void* p = ch.head;
      Summary s = run_for(std::to_string(k) + " KiB " + (r == 0 ? "random" : "seq"), steps, 1, "loads/s",
                          cfg, [&] { p = chase(p, steps); });
      keep(p);
      ns[r] = 1e9 / s.median;
      rec.emit("p1.latency", r == 0 ? "random" : "sequential",
               {{"bytes", (double)bytes}, {"ns", ns[r]}, {"samples", (double)s.samples}},
               {{"level", level_for(hw, bytes, 1)}});
    }
    char label[32];
    std::snprintf(label, sizeof label, k >= 1024 ? "%.0f MiB" : "%.0f KiB", k >= 1024 ? k / 1024.0 : (double)k);
    std::printf("  %-10s %-7s %14.2f %14.2f\n", label, level_for(hw, bytes, 1), ns[0], ns[1]);
    std::fflush(stdout);
  }
  std::printf("\n  Random-order latency steps up at each capacity edge. Sequential order stays low\n");
  std::printf("  far past the caches: the prefetcher fetches the next lines before they are asked for.\n");
  std::printf("  Bandwidth = bytes in flight / latency (Little's law): the bandwidth lab's kernels\n");
  std::printf("  reach high GB/s only because many independent loads are outstanding at once.\n");
  return 0;
}
