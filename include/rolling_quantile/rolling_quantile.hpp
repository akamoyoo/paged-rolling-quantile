// SPDX-License-Identifier: Apache-2.0
//
// RollingQuantile<T>: exact quantiles of the most recent values of a stream.
//
// The window is a ring buffer, which fixes the order in which values leave, plus a PagedMultiset
// holding the same values in sorted order. Quantiles follow the linear-interpolation rule that
// NumPy calls method="linear"; README.md states it in full, including the treatment of NaN,
// infinities and signed zeros.

#ifndef ROLLING_QUANTILE_ROLLING_QUANTILE_HPP
#define ROLLING_QUANTILE_ROLLING_QUANTILE_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "paged_multiset.hpp"

namespace rq {
namespace detail {

// lo + gamma * (hi - lo) for adjacent order statistics lo <= hi and a weight 0 < gamma < 1.
// If hi - lo is not finite (an infinite operand, or finite values further apart than the largest
// finite T) the algebraically equal (1 - gamma) * lo + gamma * hi is evaluated instead, so a
// finite and an infinite neighbour give the infinity and opposite infinities give NaN.
template <class T>
inline T interpolate(T lo, T hi, T gamma) noexcept {
  const T diff = hi - lo;
  if (!std::isfinite(diff)) return (T(1) - gamma) * lo + gamma * hi;
  return lo + gamma * diff;
}

}  // namespace detail

// Not thread-safe. Copyable and movable; a moved-from object may only be destroyed or assigned to.
template <class T = double>
class RollingQuantile {
  static_assert(std::is_floating_point<T>::value, "RollingQuantile needs a floating-point type");

 public:
  using value_type = T;
  using size_type = std::size_t;

  // window         the largest number of values held, at least 1
  // q              the quantile reported by update(), in [0, 1]
  // page_capacity  elements per page of the sorted store; 0 selects default_page_capacity(window)
  //
  // Allocates all the memory the object will ever use. Throws std::invalid_argument if an
  // argument is out of range.
  explicit RollingQuantile(size_type window, double q = 0.5, size_type page_capacity = 0)
      : sorted_(checked_page_capacity(window, q, page_capacity)), ring_(window), q_(q) {
    const double h = q * static_cast<double>(window - 1);
    const double whole = std::floor(h);
    rank_ = static_cast<size_type>(whole);
    gamma_ = h - whole;
    sorted_.reserve(window);
  }

  // Page capacity used when none is given. A window of up to 512 values is kept in a single
  // page, that is, in one sorted array. Larger windows use pages of 128 values, doubled each time
  // the window quadruples beyond 2^17, so that the number of pages grows like the square root of
  // the window. "Choosing the page capacity" in README.md has the measurements behind this.
  static size_type default_page_capacity(size_type window) noexcept {
    if (window <= 512) return window < 2 ? 2 : window;
    size_type capacity = 128;
    for (size_type excess = (window - 1) >> 17; excess != 0; excess >>= 2) capacity *= 2;
    return capacity;
  }

  // Appends x, dropping the oldest value if the window is full, and returns the q-quantile of the
  // window. Returns NaN until window() values are held and while any of them is NaN.
  T update(T x) noexcept {
    push(x);
    if (size_ < ring_.size() || nans_ != 0) return nan();
    return at(rank_, gamma_);
  }

  // update() applied to each of in[0, count), results written to out[0, count).
  void update(const T* in, size_type count, T* out) noexcept {
    for (size_type i = 0; i < count; ++i) out[i] = update(in[i]);
  }

  // Appends x, dropping the oldest value if the window is full. Never allocates.
  void push(T x) noexcept {
    const size_type capacity = ring_.size();
    if (size_ == capacity) {
      const T oldest = ring_[head_];
      ring_[head_] = x;
      head_ = head_ + 1 == capacity ? 0 : head_ + 1;
      if (oldest == oldest && x == x) {
        sorted_.replace(oldest, x);  // the common case: one value out, one in
        return;
      }
      remove(oldest);
    } else {
      const size_type tail = head_ + size_;
      ring_[tail < capacity ? tail : tail - capacity] = x;
      ++size_;
    }
    if (x != x) {
      ++nans_;
    } else {
      sorted_.insert(x);  // within the capacity reserved by the constructor: cannot throw
    }
  }

  // Drops the oldest value. Returns false if the window is empty.
  bool pop() noexcept {
    if (size_ == 0) return false;
    const T oldest = ring_[head_];
    head_ = head_ + 1 == ring_.size() ? 0 : head_ + 1;
    --size_;
    remove(oldest);
    return true;
  }

  // Drops all values. Keeps the allocated memory.
  void clear() noexcept {
    sorted_.clear();
    head_ = 0;
    size_ = 0;
    nans_ = 0;
  }

  // The p-quantile, 0 <= p <= 1, of the values currently held, however many there are.
  // NaN if the window is empty, holds a NaN, or p is outside [0, 1].
  T quantile(double p) const noexcept {
    if (size_ == 0 || nans_ != 0 || !(p >= 0.0 && p <= 1.0)) return nan();
    const double h = p * static_cast<double>(size_ - 1);
    const double whole = std::floor(h);
    return at(static_cast<size_type>(whole), h - whole);
  }

  size_type window() const noexcept { return ring_.size(); }
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  bool full() const noexcept { return size_ == ring_.size(); }
  double q() const noexcept { return q_; }

  // Number of NaNs among the values currently held.
  size_type nan_count() const noexcept { return nans_; }

  // The values currently held, NaNs excluded, as a sorted multiset: order statistics and ranks.
  const PagedMultiset<T>& sorted() const noexcept { return sorted_; }

  // Writes the values currently held, oldest first. Together with window() and q() this is the
  // complete state of the object.
  template <class OutputIt>
  OutputIt copy_window(OutputIt out) const {
    const size_type capacity = ring_.size();
    for (size_type i = 0; i < size_; ++i) {
      const size_type pos = head_ + i;
      *out++ = ring_[pos < capacity ? pos : pos - capacity];
    }
    return out;
  }

  // Equivalent to clear() followed by push() of every element of [first, last).
  template <class InputIt>
  void assign(InputIt first, InputIt last) {
    clear();
    for (; first != last; ++first) push(*first);
  }

  // Bytes of heap memory owned by the object. Constant after construction.
  size_type memory_bytes() const noexcept {
    return sorted_.memory_bytes() + ring_.capacity() * sizeof(T);
  }

 private:
  static T nan() noexcept { return std::numeric_limits<T>::quiet_NaN(); }

  static size_type checked_page_capacity(size_type window, double q, size_type page_capacity) {
    if (window < 1 || window > PagedMultiset<T>::max_size()) {
      throw std::invalid_argument("RollingQuantile: window out of range");
    }
    if (!(q >= 0.0 && q <= 1.0)) {
      throw std::invalid_argument("RollingQuantile: q must be in [0, 1]");
    }
    return page_capacity == 0 ? default_page_capacity(window) : page_capacity;
  }

  void remove(T v) noexcept {
    if (v != v) {
      --nans_;
    } else {
      sorted_.erase(v);
    }
  }

  // The order statistic of rank k, interpolated towards rank k + 1 with weight gamma.
  T at(size_type k, double gamma) const noexcept {
    if (gamma == 0.0) return sorted_.kth(k);
    const std::pair<T, T> pair = sorted_.kth_and_next(k);
    return detail::interpolate(pair.first, pair.second, static_cast<T>(gamma));
  }

  PagedMultiset<T> sorted_;  // the non-NaN values of the window, sorted
  std::vector<T> ring_;      // all values of the window in arrival order, starting at head_
  size_type head_ = 0;       // position of the oldest value in ring_
  size_type size_ = 0;       // number of values held
  size_type nans_ = 0;       // how many of them are NaN
  size_type rank_ = 0;       // floor(q * (window - 1))
  double gamma_ = 0.0;       // q * (window - 1) - rank_
  double q_;
};

}  // namespace rq

#endif  // ROLLING_QUANTILE_ROLLING_QUANTILE_HPP
