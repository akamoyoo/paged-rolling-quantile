// SPDX-License-Identifier: Apache-2.0
//
// Keep the window in a std::multiset (a red-black tree) and track an iterator to the element of
// the wanted rank. Each update erases one element and inserts one, which moves that rank by at
// most one position each time. O(log n) per update.

#ifndef ROLLING_QUANTILE_REFERENCE_MULTISET_HPP
#define ROLLING_QUANTILE_REFERENCE_MULTISET_HPP

#include <cstddef>
#include <iterator>
#include <limits>
#include <set>

#include "definition.hpp"
#include "fifo_window.hpp"

namespace rq::reference {

template <class T = double>
class Multiset {
 public:
  Multiset(std::size_t window, double q) : window_(window), pos_(position(window, q)) {}

  T update(T x) {
    T expired = T();
    window_.push(x, expired);
    if (!window_.full() || window_.has_nan()) {
      stale_ = true;
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (stale_) {
      // The window has just become full, or its last NaN has just left: start over.
      set_.clear();
      for (const T v : window_.in_arrival_order()) set_.insert(v);
      kth_ = std::next(set_.begin(), static_cast<std::ptrdiff_t>(pos_.k));
      stale_ = false;
    } else {
      replace(expired, x);
    }
    if (pos_.gamma == 0.0) return *kth_;
    return interpolate(*kth_, *std::next(kth_), pos_.gamma);
  }

 private:
  // std::multiset keeps equal elements in insertion order, so the first element equal to
  // `expired` is the oldest one, the one that is actually leaving the window.
  void replace(T expired, T x) {
    const auto victim = set_.lower_bound(expired);
    // If the victim is at or before the tracked element, rank k passes to the next element.
    if (!(*kth_ < expired)) ++kth_;
    set_.erase(victim);
    // x is inserted after its equals. If that is before the tracked element (or the tracked
    // rank is now one past the end), rank k passes to the previous element.
    const bool before = kth_ == set_.end() || x < *kth_;
    set_.insert(x);
    if (before) --kth_;
  }

  FifoWindow<T> window_;
  Position pos_;
  bool stale_ = true;  // set_ and kth_ do not reflect the window
  std::multiset<T> set_;
  typename std::multiset<T>::iterator kth_;
};

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_MULTISET_HPP
