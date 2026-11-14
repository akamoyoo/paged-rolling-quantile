// SPDX-License-Identifier: Apache-2.0
//
// The last `capacity` values of a stream. Shared by the simple implementations in this directory.

#ifndef ROLLING_QUANTILE_REFERENCE_FIFO_WINDOW_HPP
#define ROLLING_QUANTILE_REFERENCE_FIFO_WINDOW_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace rq::reference {

template <class T>
class FifoWindow {
 public:
  explicit FifoWindow(std::size_t capacity) : capacity_(capacity) { values_.reserve(capacity); }

  // Appends x. If the window was already full, the oldest value is overwritten, stored in
  // `expired`, and the function returns true.
  bool push(T x, T& expired) {
    if (std::isnan(x)) ++nans_;
    if (values_.size() < capacity_) {
      values_.push_back(x);
      return false;
    }
    expired = values_[oldest_];
    if (std::isnan(expired)) --nans_;
    values_[oldest_] = x;
    oldest_ = oldest_ + 1 == capacity_ ? 0 : oldest_ + 1;
    return true;
  }

  bool full() const noexcept { return values_.size() == capacity_; }
  bool has_nan() const noexcept { return nans_ != 0; }
  std::size_t capacity() const noexcept { return capacity_; }

  // The stored values in storage order, which is not arrival order once the window has wrapped.
  const std::vector<T>& values() const noexcept { return values_; }

  // The stored values, oldest first.
  std::vector<T> in_arrival_order() const {
    std::vector<T> out;
    out.reserve(values_.size());
    for (std::size_t i = oldest_; i < values_.size(); ++i) out.push_back(values_[i]);
    for (std::size_t i = 0; i < oldest_; ++i) out.push_back(values_[i]);
    return out;
  }

 private:
  std::size_t capacity_;
  std::size_t oldest_ = 0;
  std::size_t nans_ = 0;
  std::vector<T> values_;
};

}  // namespace rq::reference

#endif  // ROLLING_QUANTILE_REFERENCE_FIFO_WINDOW_HPP
