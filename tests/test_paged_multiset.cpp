// SPDX-License-Identifier: Apache-2.0
//
// PagedMultiset against a model: a std::vector kept sorted with the same tie rule (insert after
// equal elements, erase the first equal element). Random operation sequences are replayed on
// both; contents, order statistics, ranks and structural invariants must agree throughout.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "rolling_quantile/paged_multiset.hpp"
#include "streams.hpp"

namespace {

using test::below;
using test::Rng;
using test::unit;

// An element type whose ordering ignores part of the value. The tag tells equal keys apart, so a
// wrong tie rule shows up as a tag mismatch against the model.
struct Tagged {
  std::int32_t key;
  std::int32_t tag;
};

bool operator<(const Tagged& a, const Tagged& b) { return a.key < b.key; }

template <class T>
bool identical(const T& a, const T& b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
class Model {
 public:
  void insert(const T& v) {
    values_.insert(std::upper_bound(values_.begin(), values_.end(), v), v);
  }

  bool erase(const T& v) {
    const auto it = std::lower_bound(values_.begin(), values_.end(), v);
    if (it == values_.end() || v < *it) return false;
    values_.erase(it);
    return true;
  }

  void clear() { values_.clear(); }
  const std::vector<T>& values() const { return values_; }

 private:
  std::vector<T> values_;
};

[[noreturn]] void fail(const char* what, const char* type, std::size_t page_capacity,
                       std::uint64_t seed, std::size_t step) {
  std::fprintf(stderr, "test_paged_multiset: %s (type=%s page_capacity=%zu seed=%llu step=%zu)\n",
               what, type, page_capacity, static_cast<unsigned long long>(seed), step);
  std::exit(1);
}

// Everything observable about `set` must match `model`.
template <class T>
const char* compare(const rq::PagedMultiset<T>& set, const Model<T>& model, Rng& rng, bool deep) {
  const std::vector<T>& want = model.values();
  if (set.size() != want.size()) return "size differs";
  if (set.empty() != want.empty()) return "empty() differs";
  if (want.empty()) return deep && !set.validate() ? "invariant violated" : nullptr;

  const std::size_t n = want.size();
  const std::size_t k = below(rng, n);
  if (!identical(set.kth(k), want[k])) return "kth differs";
  if (!identical(set.kth(0), want[0])) return "kth(0) differs";
  if (!identical(set.kth(n - 1), want[n - 1])) return "kth(size - 1) differs";
  if (k + 1 < n) {
    const std::pair<T, T> pair = set.kth_and_next(k);
    if (!identical(pair.first, want[k]) || !identical(pair.second, want[k + 1])) {
      return "kth_and_next differs";
    }
  }

  // Ranks of a stored value and of its neighbours in the order.
  const T probe = want[below(rng, n)];
  const auto lower = std::lower_bound(want.begin(), want.end(), probe);
  const auto upper = std::upper_bound(want.begin(), want.end(), probe);
  if (set.count_less(probe) != static_cast<std::size_t>(lower - want.begin())) {
    return "count_less differs";
  }
  if (set.count_less_equal(probe) != static_cast<std::size_t>(upper - want.begin())) {
    return "count_less_equal differs";
  }

  if (deep) {
    if (!set.validate()) return "invariant violated";
    std::vector<T> got;
    got.reserve(n);
    set.copy_to(std::back_inserter(got));
    if (got.size() != n) return "copy_to length differs";
    for (std::size_t i = 0; i < n; ++i) {
      if (!identical(got[i], want[i])) return "contents differ";
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (!identical(set.kth(i), want[i])) return "kth sweep differs";
    }
    // Pages of a multi-page container are at least a quarter full, which bounds their number.
    const std::size_t quarter = std::max<std::size_t>(1, set.page_capacity() / 4);
    if (set.page_count() > std::max<std::size_t>(1, n / quarter)) return "too many pages";
  }
  return nullptr;
}

template <class T>
struct Values;

template <>
struct Values<double> {
  static const char* type() { return "double"; }
  static double make(Rng& rng, int flavour, std::size_t serial) {
    switch (flavour) {
      case 0:
        return (unit(rng) - 0.5) * 1e6;
      case 1:
        return static_cast<double>(below(rng, 7));
      case 2:
        return static_cast<double>(serial);
      case 3:
        return -static_cast<double>(serial);
      case 4:
        return below(rng, 2) == 0 ? 0.0 : -0.0;
      default: {
        const std::size_t pick = below(rng, 8);
        if (pick == 0) return std::numeric_limits<double>::infinity();
        if (pick == 1) return -std::numeric_limits<double>::infinity();
        return static_cast<double>(below(rng, 5)) - 2.0;
      }
    }
  }
  static constexpr int flavours = 6;
};

template <>
struct Values<float> {
  static const char* type() { return "float"; }
  static float make(Rng& rng, int flavour, std::size_t serial) {
    return static_cast<float>(Values<double>::make(rng, flavour, serial));
  }
  static constexpr int flavours = 6;
};

template <>
struct Values<std::int64_t> {
  static const char* type() { return "int64"; }
  static std::int64_t make(Rng& rng, int flavour, std::size_t serial) {
    switch (flavour) {
      case 0:
        return static_cast<std::int64_t>(rng());
      case 1:
        return static_cast<std::int64_t>(below(rng, 7));
      case 2:
        return static_cast<std::int64_t>(serial);
      default:
        return -static_cast<std::int64_t>(serial);
    }
  }
  static constexpr int flavours = 4;
};

template <>
struct Values<Tagged> {
  static const char* type() { return "tagged"; }
  static Tagged make(Rng& rng, int flavour, std::size_t serial) {
    const std::int32_t tag = static_cast<std::int32_t>(serial);
    switch (flavour) {
      case 0:
        return {static_cast<std::int32_t>(below(rng, 1000)), tag};
      case 1:
        return {static_cast<std::int32_t>(below(rng, 4)), tag};
      default:
        return {7, tag};
    }
  }
  static constexpr int flavours = 3;
};

// A random mix of insertions, removals and replacements. `bias` is the probability of inserting;
// it changes over the run so the container grows, hovers and shrinks. Removals pick their victim
// in several ways: the oldest value (queue order), the newest, the smallest, the largest, a
// random stored value, or a value that may not be stored at all.
template <class T>
void random_operations(std::size_t page_capacity, std::uint64_t seed, std::size_t steps,
                       std::size_t size_limit) {
  Rng rng(seed);
  rq::PagedMultiset<T> set(page_capacity);
  Model<T> model;
  std::deque<T> arrivals;  // stored values in insertion order, as far as removals preserve it
  const int flavour = static_cast<int>(below(rng, static_cast<std::size_t>(Values<T>::flavours)));
  const int victim_rule = static_cast<int>(below(rng, 6));
  if (below(rng, 2) == 0) set.reserve(below(rng, size_limit + 1));

  double bias = 0.9;
  for (std::size_t step = 0; step < steps; ++step) {
    if (step % 97 == 0) {
      const double choices[4] = {0.9, 0.5, 0.5, 0.15};
      bias = choices[below(rng, 4)];
    }
    const std::size_t size = model.values().size();
    const bool insert = size == 0 || (size < size_limit && unit(rng) < bias);
    if (insert) {
      const T v = Values<T>::make(rng, flavour, step);
      set.insert(v);
      model.insert(v);
      arrivals.push_back(v);
    } else {
      T v = model.values()[below(rng, size)];
      bool from_queue = false;
      switch (victim_rule) {
        case 0:
          v = arrivals.front();
          arrivals.pop_front();
          from_queue = true;
          break;
        case 1:
          v = arrivals.back();
          arrivals.pop_back();
          from_queue = true;
          break;
        case 2:
          v = model.values().front();
          break;
        case 3:
          v = model.values().back();
          break;
        case 4:
          break;
        default:
          if (below(rng, 2) == 0) v = Values<T>::make(rng, flavour, step);
          break;
      }
      // A third of the removals are replacements: one value out and another in, in one call.
      const bool swap = below(rng, 3) == 0;
      const T incoming = Values<T>::make(rng, flavour, step);
      const bool erased = model.erase(v);
      const bool reported = swap ? set.replace(v, incoming) : set.erase(v);
      if (reported != erased) {
        fail("erase() or replace() result differs", Values<T>::type(), page_capacity, seed, step);
      }
      if (erased && !from_queue) {
        // Keep the queue consistent with the multiset: drop one equivalent value.
        for (auto it = arrivals.begin(); it != arrivals.end(); ++it) {
          if (!(*it < v) && !(v < *it)) {
            arrivals.erase(it);
            break;
          }
        }
      }
      if (swap) {
        model.insert(incoming);
        arrivals.push_back(incoming);
      }
    }
    if (below(rng, 4000) == 0) {
      set.clear();
      model.clear();
      arrivals.clear();
    }
    const bool deep = step % 61 == 0 || step + 1 == steps;
    if (const char* what = compare(set, model, rng, deep)) {
      fail(what, Values<T>::type(), page_capacity, seed, step);
    }
  }
}

// Fill in one order, drain in another: every page is created, filled, emptied and removed.
template <class T>
void fill_and_drain(std::size_t page_capacity, std::uint64_t seed, std::size_t count) {
  Rng rng(seed);
  rq::PagedMultiset<T> set(page_capacity);
  Model<T> model;
  const std::size_t flavours = static_cast<std::size_t>(Values<T>::flavours);
  const int fill_flavour = static_cast<int>(below(rng, flavours));
  for (std::size_t i = 0; i < count; ++i) {
    const T v = Values<T>::make(rng, fill_flavour, i);
    set.insert(v);
    model.insert(v);
    if (const char* what = compare(set, model, rng, i % 53 == 0)) {
      fail(what, Values<T>::type(), page_capacity, seed, i);
    }
  }
  const std::size_t drain_rule = below(rng, 3);
  for (std::size_t i = 0; i < count; ++i) {
    const std::vector<T>& values = model.values();
    T v = values.front();
    if (drain_rule == 1) v = values.back();
    if (drain_rule == 2) v = values[below(rng, values.size())];
    if (!set.erase(v)) {
      fail("erase() of a stored value failed", Values<T>::type(), page_capacity, seed, i);
    }
    model.erase(v);
    if (const char* what = compare(set, model, rng, i % 53 == 0)) {
      fail(what, Values<T>::type(), page_capacity, seed, count + i);
    }
  }
  CHECK(set.empty());
  CHECK(set.page_count() == 0);
  CHECK(set.validate());
}

template <class T>
void run_type() {
  const std::size_t capacities[] = {2, 3, 4, 5, 7, 8, 9, 16, 31, 64, 128};
  std::uint64_t seed = 1;
  for (const std::size_t capacity : capacities) {
    for (int repeat = 0; repeat < 6; ++repeat) {
      const std::size_t limit = capacity <= 16 ? 400 : 3000;
      random_operations<T>(capacity, seed++, 6000, limit);
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
      fill_and_drain<T>(capacity, seed++, capacity <= 16 ? 300 : 1500);
    }
  }
}

// A large container against std::multiset, which has the same tie rule but no rank queries.
void large_container() {
  Rng rng(2024);
  const std::size_t target = 200000;
  rq::PagedMultiset<double> set;
  std::multiset<double> model;
  std::deque<double> arrivals;
  for (std::size_t step = 0; step < 3 * target; ++step) {
    if (arrivals.size() == target || (step > target && below(rng, 3) == 0 && !arrivals.empty())) {
      const double v = arrivals.front();
      arrivals.pop_front();
      CHECK(set.erase(v));
      model.erase(model.lower_bound(v));
    } else {
      // A coarse grid: about a thousand distinct values, so equal runs span several pages.
      const double v = static_cast<double>(below(rng, 1000)) * 0.125;
      set.insert(v);
      model.insert(v);
      arrivals.push_back(v);
    }
  }
  CHECK(set.size() == model.size());
  CHECK(set.validate());
  CHECK(set.page_count() > 100);
  std::vector<double> got;
  set.copy_to(std::back_inserter(got));
  CHECK(got.size() == model.size());
  CHECK(std::equal(got.begin(), got.end(), model.begin()));
  std::size_t rank = 0;
  for (auto it = model.begin(); it != model.end(); ++it, ++rank) {
    if (rank % 997 == 0) CHECK(set.kth(rank) == *it);
  }
  CHECK(set.count_less(1e9) == set.size());
  CHECK(set.count_less(-1e9) == 0);
}

void interface() {
  bool thrown = false;
  try {
    rq::PagedMultiset<double> bad(1);
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  CHECK(thrown);

  rq::PagedMultiset<int> set(4);
  CHECK(set.page_capacity() == 4);
  CHECK(set.empty() && set.size() == 0 && set.page_count() == 0);
  CHECK(!set.erase(1));
  CHECK(set.count_less(5) == 0 && set.count_less_equal(5) == 0);
  CHECK(set.validate());
  set.clear();
  CHECK(set.validate());

  for (int v : {5, 1, 4, 1, 5, 9, 2, 6, 5, 3}) set.insert(v);
  CHECK(set.size() == 10);
  CHECK(set.kth(0) == 1 && set.kth(1) == 1 && set.kth(2) == 2 && set.kth(9) == 9);
  CHECK(set.count_less(5) == 5 && set.count_less_equal(5) == 8);
  CHECK(!set.erase(7));
  CHECK(set.size() == 10);
  CHECK(set.erase(5) && set.size() == 9 && set.count_less_equal(5) == 7);
  CHECK(set.replace(2, 8) && set.size() == 9 && set.kth(2) == 3 && set.kth(7) == 8);
  // Nothing to remove: replace() only inserts and says so.
  CHECK(!set.replace(7, 2) && set.size() == 10 && set.kth(2) == 2);
  CHECK(set.erase(8) && set.size() == 9);

  // Copies are independent; a moved-to object carries the contents.
  rq::PagedMultiset<int> copy(set);
  CHECK(copy.erase(9));
  CHECK(copy.size() == 8 && set.size() == 9 && set.kth(8) == 9);
  CHECK(copy.validate() && set.validate());
  rq::PagedMultiset<int> moved(std::move(copy));
  CHECK(moved.size() == 8 && moved.kth(7) == 6 && moved.validate());
  copy = set;
  CHECK(copy.size() == 9 && copy.validate());
  moved = std::move(copy);
  CHECK(moved.size() == 9 && moved.kth(8) == 9);

  // reserve() bounds the memory: nothing grows while the size stays within the reservation.
  rq::PagedMultiset<double> reserved(8);
  reserved.reserve(1000);
  const std::size_t bytes = reserved.memory_bytes();
  CHECK(bytes >= 1000 * sizeof(double));
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 1000; ++i) reserved.insert(static_cast<double>((i * 7919) % 1000));
    for (int i = 0; i < 1000; ++i) CHECK(reserved.erase(static_cast<double>(i)));
  }
  CHECK(reserved.empty());
  CHECK(reserved.memory_bytes() == bytes);
}

}  // namespace

int main() {
  interface();
  run_type<double>();
  run_type<float>();
  run_type<std::int64_t>();
  run_type<Tagged>();
  large_container();
  return test::finish("test_paged_multiset");
}
