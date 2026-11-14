// SPDX-License-Identifier: Apache-2.0
//
// The quantile definition of this repository, executed literally. The tests compare everything
// else against quantile() below. Nothing in this directory includes anything from include/.
//
// For n values sorted as x[0] <= ... <= x[n-1] and 0 <= q <= 1:
//
//     h = q * (n - 1),   k = floor(h),   gamma = h - k,
//     Q = x[k]                               if gamma == 0,
//     Q = x[k] + gamma * (x[k+1] - x[k])     otherwise.
//
// This is linear interpolation between closest ranks: NumPy's method="linear", R's type 7.

#ifndef ROLLING_QUANTILE_REFERENCE_DEFINITION_HPP
#define ROLLING_QUANTILE_REFERENCE_DEFINITION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace rq::reference {

// Where the q-quantile of n sorted values lies: at rank k, a fraction gamma of the way to k + 1.
struct Position {
  std::size_t k;
  double gamma;
};

inline Position position(std::size_t n, double q) noexcept {
  const double h = q * static_cast<double>(n - 1);
  const double k = std::floor(h);
  return {static_cast<std::size_t>(k), h - k};
}

// The interpolation step, for 0 < gamma < 1, carried out in T. A difference that is not finite
// (an infinite value, or two finite values further apart than the largest finite T) cannot be
// scaled, so the weighted sum is formed directly in that case.
template <class T>
inline T interpolate(T lower, T upper, double gamma) noexcept {
  const T g = static_cast<T>(gamma);
  const T difference = upper - lower;
  if (std::isfinite(difference)) return lower + g * difference;
  return (T(1) - g) * lower + g * upper;
}

// The q-quantile of `values`, which are given in arrival order. NaN if there are no values or if
// any value is NaN. Values that compare equal, such as +0.0 and -0.0, are ranked in arrival order.
template <class T>
T quantile(std::vector<T> values, double q) {
  if (values.empty()) return std::numeric_limits<T>::quiet_NaN();
  for (const T v : values) {
    if (std::isnan(v)) return std::numeric_limits<T>::quiet_NaN();
  }
  std::stable_sort(values.begin(), values.end());
  const Position pos = position(values.size(), q);
  if (pos.gamma == 0.0) return values[pos.k];
  return interpolate(values[pos.k], values[pos.k + 1], pos.gamma);
}

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_DEFINITION_HPP
