// SPDX-License-Identifier: Apache-2.0
//
// Deterministic input streams for the randomized tests. std::mt19937_64 produces the same
// sequence on every standard library; the conversions to floating point below are explicit, so a
// seed identifies a stream exactly.

#ifndef ROLLING_QUANTILE_TESTS_STREAMS_HPP
#define ROLLING_QUANTILE_TESTS_STREAMS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

namespace test {

using Rng = std::mt19937_64;

// Uniform in [0, 1).
inline double unit(Rng& rng) { return static_cast<double>(rng() >> 11) * 0x1.0p-53; }

// Uniform in {0, ..., n - 1}.
inline std::size_t below(Rng& rng, std::size_t n) { return static_cast<std::size_t>(rng() % n); }

enum class Shape {
  uniform,     // independent, continuous
  walk,        // continuous random walk
  ticks,       // random walk on the integers: long stretches of duplicates
  few,         // five distinct values
  constant,    // one value
  ascending,   // strictly increasing
  descending,  // strictly decreasing
  sawtooth,    // short increasing ramps
  runs,        // monotone runs and plateaus of random length
  extremes,    // huge, tiny and ordinary magnitudes mixed: differences overflow
  wide,        // uniformly distributed exponents, subnormals included
};

constexpr int shape_count = 11;

inline const char* name(Shape s) {
  static const char* const names[shape_count] = {
      "uniform",    "walk",     "ticks", "few",      "constant", "ascending",
      "descending", "sawtooth", "runs",  "extremes", "wide"};
  return names[static_cast<int>(s)];
}

// How often the stream substitutes a special value for the next regular one.
struct Specials {
  double nan = 0.0;
  double infinity = 0.0;  // either sign
  double zero = 0.0;      // either sign
};

template <class T>
class Stream {
 public:
  Stream(Shape shape, Specials specials, std::uint64_t seed)
      : shape_(shape), specials_(specials), rng_(seed) {
    period_ = 2 + below(rng_, 60);
  }

  T next() {
    const double u = unit(rng_);
    if (u < specials_.nan) return std::numeric_limits<T>::quiet_NaN();
    if (u < specials_.nan + specials_.infinity) {
      return below(rng_, 2) == 0 ? std::numeric_limits<T>::infinity()
                                 : -std::numeric_limits<T>::infinity();
    }
    if (u < specials_.nan + specials_.infinity + specials_.zero) {
      return below(rng_, 2) == 0 ? T(0) : -T(0);
    }
    return regular();
  }

 private:
  T regular() {
    ++count_;
    switch (shape_) {
      case Shape::uniform:
        return static_cast<T>((unit(rng_) - 0.5) * 2e5);
      case Shape::walk:
        level_ += unit(rng_) - 0.5;
        return static_cast<T>(level_);
      case Shape::ticks:
        level_ += static_cast<double>(below(rng_, 3)) - 1.0;
        return static_cast<T>(level_);
      case Shape::few: {
        static const double values[5] = {-1.5, 0.0, 0.25, 3.0, 1e9};
        return static_cast<T>(values[below(rng_, 5)]);
      }
      case Shape::constant:
        return T(42);
      case Shape::ascending:
        return static_cast<T>(static_cast<double>(count_));
      case Shape::descending:
        return static_cast<T>(-static_cast<double>(count_));
      case Shape::sawtooth:
        return static_cast<T>(static_cast<double>(count_ % period_));
      case Shape::runs:
        if (run_left_ == 0) {
          run_left_ = 1 + below(rng_, 40);
          run_step_ = static_cast<double>(below(rng_, 5)) - 2.0;  // -2 .. 2; 0 gives a plateau
        }
        --run_left_;
        level_ += run_step_;
        return static_cast<T>(level_);
      case Shape::extremes: {
        const T big = std::numeric_limits<T>::max();
        const T tiny = std::numeric_limits<T>::denorm_min();
        const T values[8] = {big, -big, big / 2, -big / 2, T(1), T(-1), tiny, -tiny};
        return values[below(rng_, 8)];
      }
      case Shape::wide: {
        const int lowest = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits;
        const int span = std::numeric_limits<T>::max_exponent - lowest;
        const int exponent = lowest + static_cast<int>(below(rng_, static_cast<std::size_t>(span)));
        const double mantissa = 0.5 + unit(rng_) / 2;
        const T magnitude = static_cast<T>(std::ldexp(mantissa, exponent));
        return below(rng_, 2) == 0 ? magnitude : -magnitude;
      }
    }
    return T(0);
  }

  Shape shape_;
  Specials specials_;
  Rng rng_;
  std::size_t period_ = 2;
  std::size_t count_ = 0;
  std::size_t run_left_ = 0;
  double run_step_ = 0.0;
  double level_ = 0.0;
};

}  // namespace test

#endif  // ROLLING_QUANTILE_TESTS_STREAMS_HPP
