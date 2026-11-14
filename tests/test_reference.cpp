// SPDX-License-Identifier: Apache-2.0
//
// Two things. First, each simple implementation in reference/ is checked against the quantile
// definition. Second, RollingQuantile is compared with reference::Multiset on long streams and
// on windows far too large for the definition (a sort per step) to be affordable.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <vector>

#include "check.hpp"
#include "reference/definition.hpp"
#include "reference/multiset.hpp"
#include "reference/nth_element.hpp"
#include "reference/sort_every_step.hpp"
#include "reference/sorted_array.hpp"
#include "reference/two_heaps.hpp"
#include "rolling_quantile/rolling_quantile.hpp"
#include "streams.hpp"

namespace {

using test::Shape;
using test::Specials;

[[noreturn]] void fail(const char* what, const char* name, std::size_t window, double q,
                       Shape shape, std::uint64_t seed, std::size_t step, double got, double want) {
  std::fprintf(stderr,
               "test_reference: %s: %s window=%zu q=%a shape=%s seed=%llu step=%zu\n"
               "  got %.17g, want %.17g\n",
               what, name, window, q, test::name(shape), static_cast<unsigned long long>(seed),
               step, got, want);
  std::exit(1);
}

// `exact`: whether the implementation ranks equal values in arrival order, as the definition
// does. Those that sort with an unstable algorithm, or do not keep equal values apart, may
// return a zero of the other sign.
template <class Impl>
void against_definition(const char* name, bool exact) {
  const std::size_t windows[] = {1, 2, 3, 4, 5, 8, 13, 32, 100, 257};
  const double qs[] = {0.0, 0.05, 0.25, 1.0 / 3.0, 0.5, 0.75, 0.99, 1.0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t seed = 1;
  for (const std::size_t window : windows) {
    for (const double q : qs) {
      for (int shape_index = 0; shape_index < test::shape_count; ++shape_index, ++seed) {
        const Shape shape = static_cast<Shape>(shape_index);
        Specials specials;
        if (seed % 2 == 0) specials = {1.0 / 50, 1.0 / 25, 1.0 / 10};
        test::Stream<double> stream(shape, specials, seed);
        Impl impl(window, q);
        std::deque<double> model;
        const std::size_t steps = 3 * window + 60;
        for (std::size_t step = 0; step < steps; ++step) {
          const double x = stream.next();
          model.push_back(x);
          if (model.size() > window) model.pop_front();
          const double got = impl.update(x);
          const double want =
              model.size() == window
                  ? rq::reference::quantile(std::vector<double>(model.begin(), model.end()), q)
                  : nan;
          const bool ok = exact ? test::same(got, want) : test::same_value(got, want);
          if (!ok) {
            fail("differs from the definition", name, window, q, shape, seed, step, got, want);
          }
        }
      }
    }
  }
}

// RollingQuantile and the std::multiset implementation must agree bit for bit at every step.
void long_stream(std::size_t window, double q, std::size_t page_capacity, Shape shape,
                 Specials specials, std::uint64_t seed, std::size_t steps) {
  test::Stream<double> stream(shape, specials, seed);
  rq::RollingQuantile<double> paged(window, q, page_capacity);
  rq::reference::Multiset<double> tree(window, q);
  for (std::size_t step = 0; step < steps; ++step) {
    const double x = stream.next();
    const double got = paged.update(x);
    const double want = tree.update(x);
    if (!test::same(got, want)) {
      fail("RollingQuantile differs from Multiset", "long_stream", window, q, shape, seed, step,
           got, want);
    }
  }
  CHECK(paged.sorted().validate());
}

void long_streams() {
  const Shape shapes[] = {Shape::uniform,    Shape::walk,     Shape::ticks, Shape::few,
                          Shape::ascending,  Shape::descending, Shape::sawtooth, Shape::runs,
                          Shape::wide};
  const double qs[] = {0.5, 0.01, 0.999, 0.25, 1.0, 0.0, 0.9, 2.0 / 3.0, 0.75};
  std::uint64_t seed = 77;
  std::size_t i = 0;
  for (const Shape shape : shapes) {
    const double q = qs[i++ % 9];
    // Signed zeros and infinities are frequent; a NaN forces Multiset to start over, so NaNs
    // are kept rare where the window is large.
    long_stream(1000, q, 0, shape, {1.0 / 5000, 1.0 / 500, 1.0 / 50}, seed++, 150000);
    long_stream(1000, q, 16, shape, {1.0 / 5000, 1.0 / 500, 1.0 / 50}, seed++, 100000);
    long_stream(20000, q, 0, shape, {1.0 / 100000, 1.0 / 2000, 1.0 / 100}, seed++, 200000);
    long_stream(20000, q, 64, shape, {0.0, 1.0 / 2000, 1.0 / 100}, seed++, 120000);
  }
  // Windows at the upper end of the benchmarked range.
  long_stream(200000, 0.5, 0, Shape::uniform, {0.0, 1.0 / 5000, 1.0 / 100}, seed++, 500000);
  long_stream(200000, 0.9, 0, Shape::ticks, {0.0, 0.0, 0.0}, seed++, 500000);
  long_stream(200000, 0.1, 0, Shape::ascending, {0.0, 0.0, 0.0}, seed++, 500000);
  long_stream(1000000, 0.5, 0, Shape::walk, {0.0, 0.0, 1.0 / 1000}, seed++, 1600000);
}

}  // namespace

int main() {
  against_definition<rq::reference::SortEveryStep<double>>("SortEveryStep", false);
  against_definition<rq::reference::NthElement<double>>("NthElement", false);
  against_definition<rq::reference::SortedArray<double>>("SortedArray", true);
  against_definition<rq::reference::Multiset<double>>("Multiset", true);
  against_definition<rq::reference::TwoHeaps<double>>("TwoHeaps", false);
  long_streams();
  return test::finish("test_reference");
}
