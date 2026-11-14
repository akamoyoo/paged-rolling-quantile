// SPDX-License-Identifier: Apache-2.0
//
// Peak heap memory held by each implementation, per value in the window.
//
// The global allocation functions are replaced with versions that keep a running total of the
// bytes handed out. Each implementation is constructed and fed uniformly distributed values, and
// the largest total seen while it runs is reported. On glibc the size of a block is what malloc
// set aside for it (its own per-block header excluded); elsewhere it is the size asked for.
//
//   memory_footprint [window ...]      (default: 1024 65536 1048576)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "reference/multiset.hpp"
#include "reference/nth_element.hpp"
#include "reference/sort_every_step.hpp"
#include "reference/sorted_array.hpp"
#include "reference/two_heaps.hpp"
#include "rolling_quantile/rolling_quantile.hpp"

namespace {

std::size_t live_bytes = 0;
std::size_t peak_bytes = 0;

#if defined(__GLIBC__)
const std::size_t header = 0;
std::size_t size_of_block(void* block, std::size_t) { return malloc_usable_size(block); }
#else
// Without a way to ask the allocator, the requested size is stored in front of each block.
const std::size_t header = alignof(std::max_align_t);
std::size_t size_of_block(void* block, std::size_t) { return *static_cast<std::size_t*>(block); }
#endif

void* counted_allocation(std::size_t size) {
  void* const block = std::malloc(size + header + (size == 0 ? 1 : 0));
  if (block == nullptr) throw std::bad_alloc();
  if (header != 0) *static_cast<std::size_t*>(block) = size;
  live_bytes += size_of_block(block, size);
  if (live_bytes > peak_bytes) peak_bytes = live_bytes;
  return static_cast<char*>(block) + header;
}

void counted_release(void* p) noexcept {
  if (p == nullptr) return;
  void* const block = static_cast<char*>(p) - header;
  live_bytes -= size_of_block(block, 0);
  std::free(block);
}

void* counted_allocation(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return counted_allocation(size);
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

}  // namespace

// Every replaceable form that the standard library uses for ordinary alignment.
void* operator new(std::size_t size) { return counted_allocation(size); }
void* operator new[](std::size_t size) { return counted_allocation(size); }
void* operator new(std::size_t size, const std::nothrow_t& tag) noexcept {
  return counted_allocation(size, tag);
}
void* operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
  return counted_allocation(size, tag);
}
void operator delete(void* p) noexcept { counted_release(p); }
void operator delete[](void* p) noexcept { counted_release(p); }
void operator delete(void* p, std::size_t) noexcept { counted_release(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { counted_release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { counted_release(p); }

namespace {

// `turnovers`: how many windows' worth of values to feed after the window has filled.
template <class Impl>
void report(const char* name, std::size_t window, const std::vector<double>& input,
            std::size_t turnovers) {
  const std::size_t before = live_bytes;
  peak_bytes = live_bytes;
  {
    Impl impl(window, 0.5);
    const std::size_t count = window + turnovers * window + 16;
    double last = 0.0;
    for (std::size_t i = 0; i < count; ++i) last = impl.update(input[i]);
    if (last != last) std::fprintf(stderr, "%s: unexpected NaN result\n", name);
  }
  const double held = static_cast<double>(peak_bytes - before);
  std::printf("%zu,%s,%.1f\n", window, name, held / static_cast<double>(window));
}

// RollingQuantile reserves page buffers for the worst case. This reports what the buffers that
// actually hold values at the end of the same run amount to, together with the ring buffer.
void report_buffers_in_use(std::size_t window, const std::vector<double>& input) {
  rq::RollingQuantile<double> rolling(window, 0.5);
  for (std::size_t i = 0; i < 4 * window + 16; ++i) rolling.push(input[i]);
  const std::size_t values = rolling.sorted().page_count() * rolling.sorted().page_capacity();
  const double held = static_cast<double>((values + window) * sizeof(double));
  std::printf("%zu,paged (buffers in use),%.1f\n", window, held / static_cast<double>(window));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::size_t> windows;
  for (int i = 1; i < argc; ++i) {
    const std::size_t window = static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10));
    if (window != 0) windows.push_back(window);
  }
  if (windows.empty()) windows = {1024, 65536, 1048576};

  std::puts("window,impl,peak_bytes_per_value");
  for (const std::size_t window : windows) {
    std::vector<double> input(4 * window + 16);
    std::mt19937_64 rng(1);
    for (double& v : input) v = static_cast<double>(rng() >> 11) * 0x1.0p-53;
    // The three that rework the whole window at every step hold the same memory from the first
    // full window on, so they are not run for long.
    report<rq::RollingQuantile<double>>("paged", window, input, 3);
    report_buffers_in_use(window, input);
    report<rq::reference::Multiset<double>>("multiset", window, input, 3);
    report<rq::reference::TwoHeaps<double>>("two_heaps", window, input, 3);
    report<rq::reference::SortedArray<double>>("sorted_array", window, input, 0);
    report<rq::reference::NthElement<double>>("nth_element", window, input, 0);
    report<rq::reference::SortEveryStep<double>>("sort", window, input, 0);
  }
  return 0;
}
