// Persistent spinning thread pool.
//
// Why not std::thread per call: creating a thread costs tens of microseconds,
// which would swamp a 20 µs L2-resident kernel. Workers here spin on an atomic
// generation counter, so dispatch costs well under a microsecond.
//
// macOS has no user-visible core pinning. Instead, a thread's QoS class picks
// the cluster: USER_INTERACTIVE lets the scheduler use performance (P) cores,
// BACKGROUND confines the thread to efficiency (E) cores. That is the knob the
// labs use to compare core types.
#pragma once

#include <pthread.h>
#include <pthread/qos.h>

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace lab {

inline qos_class_t parse_qos(const std::string& s) {
  if (s == "background") return QOS_CLASS_BACKGROUND;
  if (s == "utility") return QOS_CLASS_UTILITY;
  if (s == "user-initiated") return QOS_CLASS_USER_INITIATED;
  return QOS_CLASS_USER_INTERACTIVE;
}

class Pool {
 public:
  Pool(int n, qos_class_t qos = QOS_CLASS_USER_INTERACTIVE) : n_(n < 1 ? 1 : n) {
    pthread_set_qos_class_self_np(qos, 0);
    for (int i = 1; i < n_; ++i) {
      workers_.emplace_back([this, i, qos] {
        pthread_set_qos_class_self_np(qos, 0);
        loop(i);
      });
    }
  }
  ~Pool() {
    stop_.store(true, std::memory_order_release);
    for (auto& t : workers_) t.join();
  }
  int size() const { return n_; }

  // Runs f(tid) on every thread, tid in [0, size). Returns when all finish,
  // so a call is also a barrier.
  void run(const std::function<void(int)>& f) {
    if (n_ == 1) { f(0); return; }
    fn_ = &f;
    done_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    f(0);
    while (done_.load(std::memory_order_acquire) < n_ - 1) { spin_hint(); }
    fn_ = nullptr;
  }

  // Splits [0, n) into contiguous ranges, one per thread, aligned to `align`.
  static std::pair<long, long> range(long n, int tid, int nt, long align = 1) {
    long per = (n / nt + align - 1) / align * align;
    long b = std::min(n, per * tid), e = std::min(n, b + per);
    if (tid == nt - 1) e = n;
    return {b, e};
  }

 private:
  static void spin_hint() {
#if defined(__aarch64__)
    asm volatile("yield");
#endif
  }
  void loop(int tid) {
    uint64_t seen = 0;
    for (;;) {
      uint64_t g;
      int spins = 0;
      while ((g = gen_.load(std::memory_order_acquire)) == seen) {
        if (stop_.load(std::memory_order_acquire)) return;
        if (++spins > 4096) { std::this_thread::yield(); spins = 0; } else spin_hint();
      }
      seen = g;
      (*fn_)(tid);
      done_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  int n_;
  std::vector<std::thread> workers_;
  const std::function<void(int)>* fn_ = nullptr;
  alignas(128) std::atomic<uint64_t> gen_{0};
  alignas(128) std::atomic<int> done_{0};
  alignas(128) std::atomic<bool> stop_{false};
};

}  // namespace lab
