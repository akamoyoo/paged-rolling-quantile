// SPDX-License-Identifier: Apache-2.0
//
// The quantile definition and the documented behaviour of RollingQuantile, on small cases whose
// answers are written out: worked examples, values produced by pandas and NumPy, the treatment of
// NaN, infinities, overflow and signed zeros, and the interface contract.

#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "check.hpp"
#include "reference/definition.hpp"
#include "rolling_quantile/rolling_quantile.hpp"

namespace {

using test::same;

const double nan = std::numeric_limits<double>::quiet_NaN();
const double inf = std::numeric_limits<double>::infinity();

// Feeds `values` to a fresh RollingQuantile and returns what update() reported after the last.
double last_update(std::size_t window, double q, const std::vector<double>& values) {
  rq::RollingQuantile<double> r(window, q);
  double result = nan;
  for (const double v : values) result = r.update(v);
  return result;
}

// Both the implementation and the literal definition must give `want`.
void expect(const std::vector<double>& window, double q, double want) {
  CHECK(same(last_update(window.size(), q, window), want));
  CHECK(same(rq::reference::quantile(window, q), want));
}

void worked_examples() {
  // h = q (n - 1) = 1.5: halfway between the values of rank 1 and 2.
  expect({4, 1, 3, 2}, 0.5, 2.5);
  expect({4, 1, 3, 2}, 0.25, 1.75);
  expect({4, 1, 3, 2}, 0.75, 3.25);
  expect({4, 1, 3, 2}, 0.0, 1.0);
  expect({4, 1, 3, 2}, 1.0, 4.0);
  // An odd count: the median is an element, as is every quantile with integral h.
  expect({30, 10, 20}, 0.5, 20.0);
  expect({30, 10, 20}, 0.25, 15.0);
  expect({5, 1, 4, 2, 3}, 0.25, 2.0);
  expect({5, 1, 4, 2, 3}, 0.75, 4.0);
  // Duplicates: interpolating between equal neighbours returns that value.
  expect({7, 7, 7, 7}, 0.3, 7.0);
  expect({1, 2, 2, 3}, 0.5, 2.0);
  // A single value is every quantile of itself.
  expect({42}, 0.0, 42.0);
  expect({42}, 0.37, 42.0);
  expect({42}, 1.0, 42.0);
}

struct RollingCase {
  std::size_t window;
  double q;
  std::vector<double> want;  // update() results from the first full window on
};

// Series.rolling(window).quantile(q, interpolation="linear") as computed by pandas 2.3.2.
void pandas_rolling_values() {
  const std::vector<double> series = {0.5, -1.25, 3.1,  2.75, -0.125, 10.0,       0.001,
                                      7.3, -2.5,  4.0,  4.0,  1e-07,  123456.789, -0.7};
  const std::vector<RollingCase> cases = {
      {1,
       0.5,
       {0x1.0000000000000p-1, -0x1.4000000000000p+0, 0x1.8cccccccccccdp+1, 0x1.6000000000000p+1,
        -0x1.0000000000000p-3, 0x1.4000000000000p+3, 0x1.0624dd2f1a9fcp-10, 0x1.d333333333333p+2,
        -0x1.4000000000000p+1, 0x1.0000000000000p+2, 0x1.0000000000000p+2, 0x1.ad7f29abcaf48p-24,
        0x1.e240c9fbe76c9p+16, -0x1.6666666666666p-1}},
      {2,
       0.5,
       {-0x1.8000000000000p-2, 0x1.d999999999998p-1, 0x1.7666666666666p+1, 0x1.5000000000000p+0,
        0x1.3c00000000000p+2, 0x1.40083126e978ep+2, 0x1.d34395810624dp+1, 0x1.3333333333334p+1,
        0x1.8000000000000p-1, 0x1.0000000000000p+2, 0x1.0000006b5fca7p+1, 0x1.e240c9fbe91a1p+15,
        0x1.e24016c8b4396p+15}},
      {3,
       0.25,
       {-0x1.8000000000000p-2, 0x1.8000000000000p-1, 0x1.5000000000000p+0, 0x1.5000000000000p+0,
        -0x1.fbe76c8b43958p-5, 0x1.d34395810624dp+1, -0x1.3fdf3b645a1cbp+0, 0x1.8000000000000p-1,
        0x1.8000000000000p-1, 0x1.0000006b5fca7p+1, 0x1.0000006b5fca7p+1, -0x1.6666630b68131p-2}},
      {4,
       0.9,
       {0x1.7f5c28f5c28f6p+1, 0x1.7f5c28f5c28f6p+1, 0x1.fb851eb851ebap+2, 0x1.f4ccccccccccep+2,
        0x1.26147ae147ae2p+3, 0x1.26147ae147ae2p+3, 0x1.93d70a3d70a3ep+2, 0x1.93d70a3d70a3ep+2,
        0x1.0000000000000p+2, 0x1.5194f3c9eecc1p+16, 0x1.5194f3c9eecc1p+16}},
      {5,
       1.0 / 3.0,
       {0x1.5555555555552p-4, 0x1.aaaaaaaaaaaa9p-1, 0x1.d5acb6f46508dp-1, 0x1.d5acb6f46508dp-1,
        -0x1.53f7ced916874p-4, 0x1.55810624dd2f0p+0, 0x1.55810624dd2f0p+0, 0x1.55555673aa1bcp+0,
        0x1.55555673aa1bcp+0, 0x1.55555673aa1bcp+0}},
      {7,
       0.99,
       {0x1.32c083126e977p+3, 0x1.3ad0e56041893p+3, 0x1.3ad0e56041893p+3, 0x1.3ad0e56041893p+3,
        0x1.3ad0e56041893p+3, 0x1.3ad0e56041893p+3, 0x1.c551d1d53cddap+16, 0x1.c551d1d53cddap+16}},
      {8,
       0.01,
       {-0x1.2bd70a3d70a3dp+0, -0x1.34ccccccccccdp+1, -0x1.2ab851eb851ecp+1, -0x1.2ab851eb851ecp+1,
        -0x1.2ab851eb851ecp+1, -0x1.2999998a914f3p+1, -0x1.2fdf3b645a1cbp+1}},
      {14, 0.75, {0x1.0000000000000p+2}},
  };
  for (const RollingCase& c : cases) {
    rq::RollingQuantile<double> r(c.window, c.q);
    CHECK(c.want.size() == series.size() - c.window + 1);
    for (std::size_t i = 0; i < series.size(); ++i) {
      const double got = r.update(series[i]);
      if (i + 1 < c.window) {
        CHECK(std::isnan(got));
      } else {
        CHECK(same(got, c.want[i + 1 - c.window]));
      }
    }
  }
}

// numpy.quantile(series[:n], q, method="linear") as computed by NumPy 2.2.6. These cases have
// gamma < 0.5, where NumPy evaluates the interpolation with the same expression as this library;
// for gamma >= 0.5 it uses hi - (hi - lo) * (1 - gamma), which can differ in the last place.
void numpy_values() {
  const std::vector<double> series = {0.5, -1.25, 3.1,  2.75, -0.125, 10.0,       0.001,
                                      7.3, -2.5,  4.0,  4.0,  1e-07,  123456.789, -0.7};
  struct Case {
    std::size_t n;
    double q;
    double want;
  };
  const Case cases[] = {
      {5, 0.3, -0x1.0000000000000p-55},      {10, 0.9, 0x1.e47ae147ae146p+2},
      {14, 0.1, -0x1.15c28f5c28f5cp+0},      {7, 0.4, 0x1.9ad42c3c9eed2p-3},
      {14, 1.0 / 3.0, 0x1.5d98618b35527p-12}, {3, 0.2, -0x1.1999999999999p-1},
  };
  for (const Case& c : cases) {
    const auto end = series.begin() + static_cast<std::ptrdiff_t>(c.n);
    const std::vector<double> window(series.begin(), end);
    expect(window, c.q, c.want);
  }
}

void nan_policy() {
  // A NaN occupies a slot like any other value; while it is in the window the result is NaN.
  rq::RollingQuantile<double> r(3, 0.5);
  CHECK(std::isnan(r.update(1.0)));  // one value: window not full
  CHECK(std::isnan(r.update(nan)));  // two values
  CHECK(r.size() == 2 && r.nan_count() == 1 && r.sorted().size() == 1);
  CHECK(std::isnan(r.update(3.0)));  // {1, NaN, 3}
  CHECK(std::isnan(r.update(4.0)));  // {NaN, 3, 4}
  CHECK(std::isnan(r.quantile(0.5)));
  CHECK(same(r.update(5.0), 4.0));  // {3, 4, 5}: the NaN has left
  CHECK(r.nan_count() == 0);
  CHECK(same(r.update(6.0), 5.0));

  // A window of NaNs, then recovery.
  rq::RollingQuantile<double> all(2, 0.0);
  CHECK(std::isnan(all.update(nan)));
  CHECK(std::isnan(all.update(nan)));
  CHECK(all.nan_count() == 2 && all.sorted().empty());
  CHECK(std::isnan(all.update(7.0)));
  CHECK(same(all.update(8.0), 7.0));

  // The definition agrees: any NaN makes the quantile NaN, and so does an empty window.
  CHECK(std::isnan(rq::reference::quantile(std::vector<double>{1.0, nan, 3.0}, 0.5)));
  CHECK(std::isnan(rq::reference::quantile(std::vector<double>{}, 0.5)));
}

void infinities_and_overflow() {
  // Infinities are ordinary values at the ends of the order.
  expect({-inf, 5.0, inf}, 0.5, 5.0);
  expect({inf, 1.0, -inf, 2.0, 3.0}, 0.5, 2.0);
  expect({inf, 1.0, -inf}, 0.0, -inf);
  expect({inf, 1.0, -inf}, 1.0, inf);
  // Interpolating towards an infinite neighbour gives that infinity ...
  expect({1.0, inf}, 0.5, inf);
  expect({-inf, 1.0}, 0.5, -inf);
  expect({-inf, 1.0}, 0.999, -inf);
  expect({inf, inf}, 0.5, inf);
  expect({-inf, -inf}, 0.25, -inf);
  // ... and between opposite infinities there is no answer.
  expect({-inf, inf}, 0.5, nan);

  // Finite neighbours further apart than the largest double: the difference overflows, the
  // quantile does not.
  const double big = std::numeric_limits<double>::max();
  expect({-big, big}, 0.5, 0.0);
  expect({-big, big}, 0.25, 0.75 * -big + 0.25 * big);
  expect({-big, big}, 0.75, 0.25 * -big + 0.75 * big);
  expect({-big, big / 2}, 0.5, -big / 4);
}

void signed_zeros() {
  // +0.0 and -0.0 compare equal and are ranked in arrival order.
  expect({0.0, -0.0}, 0.0, 0.0);
  expect({-0.0, 0.0}, 0.0, -0.0);
  expect({0.0, -0.0}, 1.0, -0.0);
  expect({-0.0, 0.0}, 1.0, 0.0);

  // The zero that leaves the window is the one that arrived first, not merely an equal one.
  rq::RollingQuantile<double> r(2, 0.0);
  r.update(-0.0);
  CHECK(same(r.update(0.0), -0.0));
  CHECK(same(r.update(5.0), 0.0));
  rq::RollingQuantile<double> s(2, 0.0);
  s.update(0.0);
  CHECK(same(s.update(-0.0), 0.0));
  CHECK(same(s.update(5.0), -0.0));
}

template <class F>
bool throws_invalid_argument(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void interface() {
  CHECK(throws_invalid_argument([] { rq::RollingQuantile<double> r(0, 0.5); }));
  CHECK(throws_invalid_argument([] { rq::RollingQuantile<double> r(8, -0.01); }));
  CHECK(throws_invalid_argument([] { rq::RollingQuantile<double> r(8, 1.01); }));
  CHECK(throws_invalid_argument([] { rq::RollingQuantile<double> r(8, nan); }));
  CHECK(throws_invalid_argument([] { rq::RollingQuantile<double> r(8, 0.5, 1); }));

  rq::RollingQuantile<> r(4);  // double, the median
  CHECK(r.window() == 4 && r.q() == 0.5 && r.size() == 0 && r.empty() && !r.full());
  CHECK(r.sorted().page_capacity() == rq::RollingQuantile<>::default_page_capacity(4));
  CHECK(std::isnan(r.quantile(0.5)));
  CHECK(!r.pop());

  // quantile(p) reports on whatever is held; update() waits for a full window.
  CHECK(std::isnan(r.update(10.0)));
  CHECK(same(r.quantile(0.5), 10.0));
  r.push(20.0);
  CHECK(same(r.quantile(0.5), 15.0));
  CHECK(same(r.quantile(0.0), 10.0) && same(r.quantile(1.0), 20.0));
  CHECK(std::isnan(r.quantile(-0.1)) && std::isnan(r.quantile(1.1)) && std::isnan(r.quantile(nan)));
  CHECK(std::isnan(r.update(30.0)));
  CHECK(same(r.update(40.0), 25.0));
  CHECK(r.full() && r.size() == 4);
  CHECK(same(r.update(50.0), 35.0));  // {20, 30, 40, 50}

  // The sorted view answers rank questions about the same window.
  CHECK(r.sorted().kth(0) == 20.0 && r.sorted().kth(3) == 50.0);
  CHECK(r.sorted().count_less(40.0) == 2 && r.sorted().count_less_equal(40.0) == 3);

  // pop() drops the oldest value; update() then needs a full window again.
  CHECK(r.pop());
  CHECK(r.size() == 3 && !r.full());
  CHECK(same(r.quantile(0.5), 40.0));  // {30, 40, 50}
  CHECK(same(r.update(60.0), 45.0));   // {30, 40, 50, 60}

  // copy_window() and assign() carry the state to another object.
  std::vector<double> state;
  r.copy_window(std::back_inserter(state));
  CHECK(state == (std::vector<double>{30.0, 40.0, 50.0, 60.0}));
  rq::RollingQuantile<> restored(r.window(), r.q());
  restored.assign(state.begin(), state.end());
  CHECK(same(restored.update(70.0), r.update(70.0)));
  CHECK(same(restored.update(5.0), r.update(5.0)));

  // assign() keeps only the most recent window() values.
  const std::vector<double> many = {1, 2, 3, 4, 5, 6, 7};
  restored.assign(many.begin(), many.end());
  CHECK(restored.full() && same(restored.quantile(0.0), 4.0) && same(restored.quantile(1.0), 7.0));

  // Batch update is update() in a loop.
  const std::vector<double> in = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  std::vector<double> out(in.size(), 0.0);
  rq::RollingQuantile<> batch(3, 0.5);
  batch.update(in.data(), in.size(), out.data());
  rq::RollingQuantile<> scalar(3, 0.5);
  for (std::size_t i = 0; i < in.size(); ++i) CHECK(same(out[i], scalar.update(in[i])));
  CHECK(same(out.back(), 5.0));

  // clear() empties the window and keeps the memory.
  const std::size_t bytes = batch.memory_bytes();
  batch.clear();
  CHECK(batch.empty() && batch.nan_count() == 0 && batch.memory_bytes() == bytes);
  CHECK(std::isnan(batch.update(1.0)));

  // Copies are independent.
  rq::RollingQuantile<> original(2, 1.0);
  original.update(1.0);
  rq::RollingQuantile<> copy(original);
  CHECK(same(copy.update(9.0), 9.0));
  CHECK(same(original.update(2.0), 2.0));
  rq::RollingQuantile<> moved(std::move(copy));
  CHECK(same(moved.update(3.0), 9.0));  // {9, 3}

  // Single precision works the same way, with the interpolation carried out in float.
  rq::RollingQuantile<float> f(2, 0.5);
  f.update(1.0F);
  CHECK(same(f.update(2.0F), 1.5F));
}

}  // namespace

int main() {
  worked_examples();
  pandas_rolling_values();
  numpy_values();
  nan_policy();
  infinities_and_overflow();
  signed_zeros();
  interface();
  return test::finish("test_definition");
}
