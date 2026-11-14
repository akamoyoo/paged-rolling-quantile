// SPDX-License-Identifier: Apache-2.0
//
// The textbook two-heap scheme: a max-heap holds the k + 1 smallest values of the window and a
// min-heap the rest, so the two order statistics a quantile needs are the two heap tops. A binary
// heap cannot remove an arbitrary element, so an expired value is only marked dead and is dropped
// when it reaches the top of its heap ("lazy deletion"). Dead values that never reach a top would
// accumulate without bound; once there are more of them than live values, both heaps are rebuilt.
// O(log n) amortised per update, with an O(n log n) rebuild at most once every n updates.

#ifndef ROLLING_QUANTILE_REFERENCE_TWO_HEAPS_HPP
#define ROLLING_QUANTILE_REFERENCE_TWO_HEAPS_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

#include "definition.hpp"
#include "fifo_window.hpp"

namespace rq::reference {

template <class T = double>
class TwoHeaps {
 public:
  TwoHeaps(std::size_t window, double q) : window_(window), pos_(position(window, q)) {}

  T update(T x) {
    T expired = T();
    window_.push(x, expired);
    if (!window_.full() || window_.has_nan()) {
      stale_ = true;
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (stale_) {
      rebuild();
      stale_ = false;
    } else {
      replace(expired, x);
      if (dead_count_ > window_.capacity()) rebuild();
    }
    if (pos_.gamma == 0.0) return low_.top();
    return interpolate(low_.top(), high_.top(), pos_.gamma);
  }

 private:
  using MaxHeap = std::priority_queue<T>;
  using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

  void rebuild() {
    std::vector<T> sorted = window_.values();
    std::sort(sorted.begin(), sorted.end());
    const auto split = sorted.begin() + static_cast<std::ptrdiff_t>(pos_.k + 1);
    low_ = MaxHeap(sorted.begin(), split);
    high_ = MinHeap(split, sorted.end());
    dead_.clear();
    dead_count_ = 0;
  }

  // Invariant between calls: both heap tops are live, low_ holds exactly k + 1 live values, and
  // no live value in low_ exceeds a live value in high_.
  void replace(T expired, T x) {
    int surplus = 0;  // live values in low_, minus k + 1
    if (low_.top() < x) {
      high_.push(x);
    } else {
      low_.push(x);
      ++surplus;
    }
    if (!(low_.top() < expired)) --surplus;  // the expired value counts as one of low_'s
    ++dead_[expired];
    ++dead_count_;
    drop_dead_tops(low_);
    drop_dead_tops(high_);
    if (surplus > 0) {
      high_.push(low_.top());
      low_.pop();
      drop_dead_tops(low_);
    } else if (surplus < 0) {
      low_.push(high_.top());
      high_.pop();
      drop_dead_tops(high_);
    }
  }

  template <class Heap>
  void drop_dead_tops(Heap& heap) {
    while (dead_count_ != 0 && !heap.empty()) {
      const auto it = dead_.find(heap.top());
      if (it == dead_.end()) break;
      if (--it->second == 0) dead_.erase(it);
      --dead_count_;
      heap.pop();
    }
  }

  FifoWindow<T> window_;
  Position pos_;
  bool stale_ = true;  // the heaps do not reflect the window
  MaxHeap low_;
  MinHeap high_;
  std::unordered_map<T, std::size_t> dead_;  // expired values still inside a heap, with counts
  std::size_t dead_count_ = 0;
};

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_TWO_HEAPS_HPP
