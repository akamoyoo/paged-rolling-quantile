// SPDX-License-Identifier: Apache-2.0
//
// Keep the window in one sorted array: binary search, then shift the tail to remove the expired
// value and again to insert the new one. O(n) per update, but the O(n) is a memmove.

#ifndef ROLLING_QUANTILE_REFERENCE_SORTED_ARRAY_HPP
#define ROLLING_QUANTILE_REFERENCE_SORTED_ARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "definition.hpp"
#include "fifo_window.hpp"

namespace rq::reference {

template <class T = double>
class SortedArray {
 public:
  SortedArray(std::size_t window, double q) : window_(window), pos_(position(window, q)) {
    sorted_.reserve(window);
  }

  T update(T x) {
    T expired = T();
    window_.push(x, expired);
    if (!window_.full() || window_.has_nan()) {
      stale_ = true;
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (stale_) {
      // The window has just become full, or its last NaN has just left: sort it once.
      sorted_ = window_.in_arrival_order();
      std::stable_sort(sorted_.begin(), sorted_.end());
      stale_ = false;
    } else {
      sorted_.erase(std::lower_bound(sorted_.begin(), sorted_.end(), expired));
      sorted_.insert(std::upper_bound(sorted_.begin(), sorted_.end(), x), x);
    }
    if (pos_.gamma == 0.0) return sorted_[pos_.k];
    return interpolate(sorted_[pos_.k], sorted_[pos_.k + 1], pos_.gamma);
  }

 private:
  FifoWindow<T> window_;
  Position pos_;
  bool stale_ = true;  // sorted_ does not reflect the window
  std::vector<T> sorted_;
};

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_SORTED_ARRAY_HPP
