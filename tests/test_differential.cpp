// SPDX-License-Identifier: Apache-2.0
//
// Randomized differential test. RollingQuantile is driven with random streams and random mixes of
// operations. After every step its results are compared, bit for bit, with the quantile
// definition applied to a plain copy of the window (reference/definition.hpp).
//
//   test_differential [random_cases [seed]]
//
// The defaults keep the run to a few seconds. A failure prints the parameters of the case.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "check.hpp"
#include "reference/definition.hpp"
#include "rolling_quantile/rolling_quantile.hpp"
#include "streams.hpp"

namespace {

using test::below;
using test::Rng;
using test::same;
using test::Shape;
using test::Specials;
using test::unit;

// Which operations a case uses, as percentages per step.
struct Mix {
  const char* name;
  std::size_t update;  // update(x)
  std::size_t batch;   // update(in, count, out)
  std::size_t push;    // push(x)
  std::size_t pop;     // a burst of pop()
  std::size_t clear;   // clear()
  // the remainder: copy_window() into a fresh object via assign(), which then replaces the old
};

const Mix mixes[] = {
    {"steady", 100, 0, 0, 0, 0},
    {"batches", 10, 90, 0, 0, 0},
    {"elastic", 70, 0, 10, 20, 0},
    {"everything", 60, 10, 10, 12, 2},
};

struct Case {
  std::size_t window;
  double q;
  std::size_t page_capacity;  // 0 selects the default
  Shape shape;
  Specials specials;
  std::size_t mix;
  std::uint64_t seed;
  std::size_t arrivals;  // how many stream values the case consumes
};

template <class T>
const char* type_name() {
  return sizeof(T) == sizeof(float) ? "float" : "double";
}

template <class T>
unsigned long long bit_pattern(T v) {
  unsigned long long bits = 0;
  std::memcpy(&bits, &v, sizeof(T));
  return bits;
}

template <class T>
[[noreturn]] void fail(const Case& c, std::size_t step, const char* what, T got, T want) {
  std::fprintf(stderr,
               "test_differential: %s at step %zu\n"
               "  type=%s window=%zu q=%a page_capacity=%zu shape=%s mix=%s seed=%llu\n"
               "  specials: nan=%g infinity=%g zero=%g\n"
               "  got  %.17g (0x%llx)\n  want %.17g (0x%llx)\n",
               what, step, type_name<T>(), c.window, c.q, c.page_capacity, test::name(c.shape),
               mixes[c.mix].name, static_cast<unsigned long long>(c.seed), c.specials.nan,
               c.specials.infinity, c.specials.zero, static_cast<double>(got), bit_pattern(got),
               static_cast<double>(want), bit_pattern(want));
  std::exit(1);
}

// The definition, applied to the model window.
template <class T>
T defined(const std::deque<T>& model, double q) {
  return rq::reference::quantile(std::vector<T>(model.begin(), model.end()), q);
}

template <class T>
void run(const Case& c) {
  const T nan = std::numeric_limits<T>::quiet_NaN();
  rq::RollingQuantile<T> r(c.window, c.q, c.page_capacity);
  std::deque<T> model;  // the values RollingQuantile should be holding, oldest first
  test::Stream<T> stream(c.shape, c.specials, c.seed);
  Rng rng(c.seed * 0x9E3779B97F4A7C15ULL + 12345);
  const Mix& mix = mixes[c.mix];
  std::vector<T> in;
  std::vector<T> out;
  std::vector<T> state;

  std::size_t arrived = 0;
  const auto arrive = [&](T x) {
    ++arrived;
    model.push_back(x);
    if (model.size() > c.window) model.pop_front();
  };
  const auto after_update = [&]() { return model.size() == c.window ? defined(model, c.q) : nan; };

  for (std::size_t step = 0; arrived < c.arrivals; ++step) {
    std::size_t dice = below(rng, 100);
    if (dice < mix.update) {
      const T x = stream.next();
      arrive(x);
      const T got = r.update(x);
      const T want = after_update();
      if (!same(got, want)) fail(c, step, "update(x) differs from the definition", got, want);
    } else if ((dice -= mix.update) < mix.batch) {
      const std::size_t count = 1 + below(rng, std::min<std::size_t>(c.window + 2, 40));
      in.clear();
      for (std::size_t i = 0; i < count; ++i) in.push_back(stream.next());
      out.assign(count, T(0));
      r.update(in.data(), count, out.data());
      for (std::size_t i = 0; i < count; ++i) {
        arrive(in[i]);
        const T want = after_update();
        if (!same(out[i], want)) {
          fail(c, step, "batch update differs from the definition", out[i], want);
        }
      }
    } else if ((dice -= mix.batch) < mix.push) {
      const T x = stream.next();
      arrive(x);
      r.push(x);
    } else if ((dice -= mix.push) < mix.pop) {
      // Usually a few values leave; now and then most of the window does.
      const std::size_t burst = 1 + below(rng, below(rng, 8) == 0 ? c.window + 2 : 4);
      for (std::size_t i = 0; i < burst; ++i) {
        const bool held = !model.empty();
        if (r.pop() != held) fail(c, step, "pop() result is wrong", nan, nan);
        if (held) model.pop_front();
      }
    } else if ((dice -= mix.pop) < mix.clear) {
      r.clear();
      model.clear();
    } else {
      state.clear();
      r.copy_window(std::back_inserter(state));
      rq::RollingQuantile<T> restored(c.window, c.q, c.page_capacity);
      restored.assign(state.begin(), state.end());
      r = std::move(restored);
    }

    // The bookkeeping must match the model after every step.
    const std::size_t nans = static_cast<std::size_t>(
        std::count_if(model.begin(), model.end(), [](T v) { return v != v; }));
    if (r.size() != model.size() || r.nan_count() != nans || r.empty() != model.empty() ||
        r.full() != (model.size() == c.window) || r.sorted().size() != model.size() - nans) {
      fail(c, step, "size(), nan_count(), empty(), full() or sorted().size() is wrong", nan, nan);
    }

    // An arbitrary quantile of whatever is held. The definition costs a sort, so large windows
    // are sampled.
    if (below(rng, 1 + c.window / 64) == 0) {
      const std::size_t pick = below(rng, 8);
      const double p = pick == 0 ? 0.0 : pick == 1 ? 1.0 : pick == 2 ? 0.5 : unit(rng);
      const T got = r.quantile(p);
      const T want = defined(model, p);
      if (!same(got, want)) fail(c, step, "quantile(p) differs from the definition", got, want);
    }

    // Occasionally: the window contents in order, and the invariants of the sorted store.
    if (below(rng, 50 + c.window / 4) == 0 || arrived >= c.arrivals) {
      state.clear();
      r.copy_window(std::back_inserter(state));
      bool equal = state.size() == model.size();
      for (std::size_t i = 0; equal && i < state.size(); ++i) equal = same(state[i], model[i]);
      if (!equal) fail(c, step, "copy_window() differs from the model", nan, nan);
      if (!r.sorted().validate()) fail(c, step, "sorted store invariant violated", nan, nan);
    }
  }
}

// Stream length for a window of n values: enough to fill it, then a number of further values
// that keeps the cost of the definition (a sort per full-window step) bounded.
std::size_t arrivals_for(std::size_t n) {
  const std::size_t further = std::max<std::size_t>(100, 100000 / n);
  return n + std::min<std::size_t>(3 * n + 200, further);
}

Specials random_specials(Rng& rng) {
  const double rates[] = {1.0 / 64, 1.0 / 8, 1.0 / 3};
  Specials s;
  if (below(rng, 2) == 0) return s;
  if (below(rng, 2) == 0) s.nan = rates[below(rng, 3)] / 4;
  if (below(rng, 2) == 0) s.infinity = rates[below(rng, 3)] / 2;
  if (below(rng, 2) == 0) s.zero = rates[below(rng, 3)];
  return s;
}

double random_q(Rng& rng) {
  const double fixed[] = {0.0,  1.0,  0.5,       0.25,      0.75,      1.0 / 3.0, 2.0 / 3.0, 0.1,
                          0.9,  0.01, 0.99,      0.001,     0.999,     1e-9,      1.0 - 1e-9, 0.05};
  if (below(rng, 4) == 0) return unit(rng);
  return fixed[below(rng, 16)];
}

std::size_t random_window(Rng& rng) {
  const std::size_t marked[] = {63,  64,  65,  100, 127,  128,  129,  192,  255,  256, 257, 300,
                                383, 384, 385, 500, 511,  512,  513,  700,  1000, 1024, 1500, 2048,
                                3000, 4096};
  if (below(rng, 2) == 0) return 1 + below(rng, 40);
  return marked[below(rng, 26)];
}

std::size_t random_page_capacity(Rng& rng, std::size_t window) {
  const std::size_t small[] = {2, 3, 4, 5, 8, 16};
  switch (below(rng, 4)) {
    case 0:
      return 0;
    case 1:
      return small[below(rng, 6)];
    case 2:
      return 2 + below(rng, 2 * window + 2);  // from tiny pages to a single page
    default:
      return window > 256 ? 32 + below(rng, 200) : small[below(rng, 6)];
  }
}

template <class T>
void systematic() {
  // Every small window with every operation mix, a spread of quantiles, tiny pages and all
  // stream shapes, with and without special values.
  const double qs[] = {0.0, 0.1, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.9, 0.99, 1.0};
  const std::size_t capacities[] = {0, 2, 3};
  std::uint64_t seed = 1000;
  for (std::size_t window = 1; window <= 10; ++window) {
    for (const double q : qs) {
      for (const std::size_t capacity : capacities) {
        Case c;
        c.window = window;
        c.q = q;
        c.page_capacity = capacity;
        c.shape = static_cast<Shape>(seed % static_cast<std::uint64_t>(test::shape_count));
        if (seed % 2 == 0) c.specials = {1.0 / 16, 1.0 / 16, 1.0 / 8};
        c.mix = static_cast<std::size_t>((seed / 2) % 4);
        c.seed = seed++;
        c.arrivals = 200;
        run<T>(c);
      }
    }
  }
}

template <class T>
void randomized(std::size_t cases, std::uint64_t seed) {
  Rng rng(seed);
  for (std::size_t i = 0; i < cases; ++i) {
    Case c;
    c.window = random_window(rng);
    c.q = random_q(rng);
    c.page_capacity = random_page_capacity(rng, c.window);
    c.shape = static_cast<Shape>(below(rng, static_cast<std::size_t>(test::shape_count)));
    c.specials = random_specials(rng);
    c.mix = below(rng, 4);
    c.seed = rng();
    c.arrivals = arrivals_for(c.window);
    run<T>(c);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t cases = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1500;
  const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  systematic<double>();
  systematic<float>();
  randomized<double>(cases, seed);
  randomized<float>(cases / 3, seed + 1);
  std::printf("test_differential: %zu random cases, seed %llu\n", cases + cases / 3,
              static_cast<unsigned long long>(seed));
  return test::finish("test_differential");
}
