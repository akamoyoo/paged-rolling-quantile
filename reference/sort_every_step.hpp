// SPDX-License-Identifier: Apache-2.0
//
// The most direct rolling quantile: copy the window and sort it at every step.
// O(n log n) per update.

#ifndef ROLLING_QUANTILE_REFERENCE_SORT_EVERY_STEP_HPP
#define ROLLING_QUANTILE_REFERENCE_SORT_EVERY_STEP_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "definition.hpp"
#include "fifo_window.hpp"

namespace rq::reference {

template <class T = double>
class SortEveryStep {
 public:
  SortEveryStep(std::size_t window, double q) : window_(window), pos_(position(window, q)) {
    scratch_.reserve(window);
  }

  T update(T x) {
    T expired = T();
    window_.push(x, expired);
    if (!window_.full() || window_.has_nan()) return std::numeric_limits<T>::quiet_NaN();
    scratch_ = window_.values();
    std::sort(scratch_.begin(), scratch_.end());
    if (pos_.gamma == 0.0) return scratch_[pos_.k];
    return interpolate(scratch_[pos_.k], scratch_[pos_.k + 1], pos_.gamma);
  }

 private:
  FifoWindow<T> window_;
  Position pos_;
  std::vector<T> scratch_;
};

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_SORT_EVERY_STEP_HPP
