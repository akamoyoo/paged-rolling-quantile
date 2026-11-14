// SPDX-License-Identifier: Apache-2.0
//
// Copy the window and select the wanted order statistic with std::nth_element at every step.
// O(n) per update on average.

#ifndef ROLLING_QUANTILE_REFERENCE_NTH_ELEMENT_HPP
#define ROLLING_QUANTILE_REFERENCE_NTH_ELEMENT_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "definition.hpp"
#include "fifo_window.hpp"

namespace rq::reference {

template <class T = double>
class NthElement {
 public:
  NthElement(std::size_t window, double q) : window_(window), pos_(position(window, q)) {
    scratch_.reserve(window);
  }

  T update(T x) {
    T expired = T();
    window_.push(x, expired);
    if (!window_.full() || window_.has_nan()) return std::numeric_limits<T>::quiet_NaN();
    scratch_ = window_.values();
    const auto kth = scratch_.begin() + static_cast<std::ptrdiff_t>(pos_.k);
    std::nth_element(scratch_.begin(), kth, scratch_.end());
    if (pos_.gamma == 0.0) return *kth;
    // Everything after kth is now at least as large as it; the smallest of those is rank k + 1.
    return interpolate(*kth, *std::min_element(kth + 1, scratch_.end()), pos_.gamma);
  }

 private:
  FifoWindow<T> window_;
  Position pos_;
  std::vector<T> scratch_;
};

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_NTH_ELEMENT_HPP
