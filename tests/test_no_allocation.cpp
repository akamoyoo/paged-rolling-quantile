// SPDX-License-Identifier: Apache-2.0
//
// RollingQuantile allocates in its constructor and never again; PagedMultiset does not allocate
// while its size stays within what reserve() was given. This program replaces the global
// allocation functions with counting ones and checks both claims on streams chosen to make pages
// split, merge and rebalance as often as possible.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include "check.hpp"
#include "rolling_quantile/rolling_quantile.hpp"
#include "streams.hpp"

namespace {

std::size_t allocation_count = 0;

void* counted_allocation(std::size_t size) {
  ++allocation_count;
  if (void* p = std::malloc(size != 0 ? size : 1)) return p;
  throw std::bad_alloc();
}

void* counted_allocation(std::size_t size, const std::nothrow_t&) noexcept {
  ++allocation_count;
  return std::malloc(size != 0 ? size : 1);
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
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

using test::below;
using test::Rng;
using test::Shape;

template <class T>
void rolling(std::size_t window, std::size_t page_capacity, Shape shape, std::uint64_t seed) {
  rq::RollingQuantile<T> r(window, 0.5, page_capacity);
  test::Stream<T> stream(shape, {1.0 / 200, 1.0 / 200, 1.0 / 50}, seed);
  Rng rng(seed + 1);
  std::vector<T> in(32);
  std::vector<T> out(32);
  std::vector<T> state(window);
  const std::size_t bytes = r.memory_bytes();
  std::size_t results = 0;  // non-NaN results seen: the loop does real work

  const std::size_t before = allocation_count;
  const std::size_t steps = 30 * window + 3000;
  for (std::size_t step = 0; step < steps; ++step) {
    switch (below(rng, 40)) {
      case 0:
        for (std::size_t i = below(rng, window + 2); i > 0; --i) r.pop();
        break;
      case 1:
        for (T& x : in) x = stream.next();
        r.update(in.data(), in.size(), out.data());
        break;
      case 2:
        if (below(rng, 50) == 0) r.clear();
        break;
      case 3:
        r.copy_window(state.begin());
        break;
      default: {
        const T y = r.update(stream.next());
        if (y == y) ++results;
        break;
      }
    }
    const T y = r.quantile(test::unit(rng));
    if (y == y) ++results;
  }
  CHECK(allocation_count == before);
  CHECK(r.memory_bytes() == bytes);
  CHECK(results > steps / 2);
}

void reserved_multiset(std::size_t page_capacity, std::uint64_t seed) {
  const std::size_t limit = 5000;
  rq::PagedMultiset<double> set(page_capacity);
  set.reserve(limit);
  Rng rng(seed);
  std::vector<double> held;
  held.reserve(limit);

  const std::size_t before = allocation_count;
  // Phases: fill in increasing order, drain from the front; fill in decreasing order, drain
  // from the back; then churn at random around the limit.
  for (std::size_t i = 0; i < limit; ++i) set.insert(static_cast<double>(i));
  for (std::size_t i = 0; i < limit; ++i) CHECK(set.erase(static_cast<double>(i)));
  for (std::size_t i = limit; i > 0; --i) set.insert(static_cast<double>(i));
  for (std::size_t i = limit; i > 0; --i) CHECK(set.erase(static_cast<double>(i)));
  for (std::size_t step = 0; step < 200000; ++step) {
    const bool insert = held.empty() || (held.size() < limit && below(rng, 100) < 55);
    if (insert) {
      const double v = static_cast<double>(below(rng, 3000));
      set.insert(v);
      held.push_back(v);
    } else {
      const std::size_t at = below(rng, held.size());
      CHECK(set.erase(held[at]));
      held[at] = held.back();
      held.pop_back();
    }
  }
  CHECK(allocation_count == before);
  CHECK(set.size() == held.size());
}

}  // namespace

int main() {
  const Shape shapes[] = {Shape::uniform,    Shape::ticks,    Shape::constant, Shape::ascending,
                          Shape::descending, Shape::sawtooth, Shape::runs,     Shape::few};
  // Among them windows just above one page of the capacities used below, where the number of
  // buffers to reserve changes from one to several.
  const std::size_t windows[] = {1, 2, 3, 7, 12, 20, 64, 300, 1000, 5000};
  const std::size_t page_capacities[] = {2, 3, 4, 8, 9, 64, 128};
  std::uint64_t seed = 1;
  for (const Shape shape : shapes) {
    for (const std::size_t window : windows) {
      rolling<double>(window, 0, shape, seed++);
      rolling<double>(window, 2, shape, seed++);
      rolling<double>(window, 8, shape, seed++);
      rolling<float>(window, 16, shape, seed++);
    }
  }
  for (const std::size_t page_capacity : page_capacities) reserved_multiset(page_capacity, seed++);
  CHECK(allocation_count > 0);  // the counting itself works
  return test::finish("test_no_allocation");
}
