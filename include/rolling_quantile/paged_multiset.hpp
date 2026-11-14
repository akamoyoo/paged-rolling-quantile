// SPDX-License-Identifier: Apache-2.0
//
// PagedMultiset<T>: a sorted multiset with rank queries, stored as small sorted arrays ("pages").
//
//   pages      Each page is a sorted run of at most page_capacity() elements in one contiguous
//              buffer. Reading the pages in directory order yields the whole multiset in order.
//   directory  Three parallel arrays indexed by page: the page's largest element (the search key),
//              its element count, and the slot of its buffer.
//   tree       A Fenwick tree over the per-page counts. It maps a rank to a page.
//
// An insertion or removal is two binary searches over contiguous arrays (directory, then page), one
// memmove inside a page and one Fenwick update. Elements move between pages only when a page
// overflows or runs low. README.md has the cost model and the measurements.

#ifndef ROLLING_QUANTILE_PAGED_MULTISET_HPP
#define ROLLING_QUANTILE_PAGED_MULTISET_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rq {
namespace detail {

// Index of the first element of the sorted range a[0, n) that is not less than v; n if none.
// Each step is a comparison feeding an addition rather than a jump: on arbitrary data the outcome
// of the comparison is unpredictable, and a mispredicted jump costs more than the step itself.
template <class T>
inline std::size_t lower_bound_index(const T* a, std::size_t n, const T& v) noexcept {
  std::size_t lo = 0;
  while (n > 1) {
    const std::size_t half = n / 2;
    lo += (a[lo + half - 1] < v) ? half : 0;
    n -= half;
  }
  return lo + ((n == 1 && a[lo] < v) ? 1 : 0);
}

// Index of the first element of the sorted range a[0, n) that is greater than v; n if none.
template <class T>
inline std::size_t upper_bound_index(const T* a, std::size_t n, const T& v) noexcept {
  std::size_t lo = 0;
  while (n > 1) {
    const std::size_t half = n / 2;
    lo += (v < a[lo + half - 1]) ? 0 : half;
    n -= half;
  }
  return lo + ((n == 1 && !(v < a[lo])) ? 1 : 0);
}

// Lowest set bit of i.
inline std::size_t low_bit(std::size_t i) noexcept { return i & (~i + 1); }

template <class T>
inline bool is_unordered(const T& v) noexcept {
  if constexpr (std::is_floating_point<T>::value) {
    return v != v;  // NaN
  } else {
    (void)v;
    return false;
  }
}

template <class V>
inline void vector_insert(V& v, std::size_t i, typename V::value_type x) {
  v.insert(v.begin() + static_cast<std::ptrdiff_t>(i), x);
}

template <class V>
inline void vector_erase(V& v, std::size_t i) {
  v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
}

}  // namespace detail

// Requirements on T: trivially copyable, default constructible, ordered by operator< (a strict
// weak ordering). A floating-point NaN is not ordered and must not be inserted.
//
// Equal elements are kept in insertion order: insert() places a value after every element that is
// not greater than it, and erase() removes the first element equal to its argument. Under
// first-in first-out use the container therefore holds exactly the elements a queue would hold,
// down to the sign of a floating-point zero.
//
// Copyable and movable. A moved-from object may only be destroyed or assigned to.
template <class T>
class PagedMultiset {
  static_assert(std::is_trivially_copyable<T>::value, "elements are moved with memmove");
  static_assert(std::is_default_constructible<T>::value, "page buffers are value-initialised");

 public:
  using value_type = T;
  using size_type = std::size_t;

  static constexpr size_type default_page_capacity = 128;

  // page_capacity: the maximum number of elements per page, at least 2.
  explicit PagedMultiset(size_type page_capacity = default_page_capacity)
      : cap_(page_capacity),
        low_(page_capacity / 4),
        merge_limit_(page_capacity - page_capacity / 4),
        spare_limit_(merge_limit_ < page_capacity - 2 ? merge_limit_ : page_capacity - 2) {
    if (page_capacity < 2 || page_capacity > max_count) {
      throw std::invalid_argument("PagedMultiset: page capacity out of range");
    }
  }

  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  size_type page_capacity() const noexcept { return cap_; }
  size_type page_count() const noexcept { return count_.size(); }
  static constexpr size_type max_size() noexcept { return max_count; }

  // Allocates everything needed to hold n elements. Afterwards insert(), erase() and replace()
  // do not allocate as long as size() <= n.
  //
  // The number of page buffers this takes follows from two facts. A page is split only when it
  // is full and another element arrives, so while size() <= page_capacity() no page is ever
  // added. And every page of a multi-page container holds at least max(1, page_capacity / 4)
  // elements, which bounds the page count for larger n.
  void reserve(size_type n) { grow_slots(n <= cap_ ? 1 : n / (low_ > 1 ? low_ : 1)); }

  // Bytes of heap memory owned by the container.
  size_type memory_bytes() const noexcept {
    return (slab_.capacity() + max_.capacity()) * sizeof(T) +
           (count_.capacity() + slot_.capacity() + tree_.capacity() + free_.capacity()) *
               sizeof(std::uint32_t);
  }

  // Inserts v after all elements that are not greater than v.
  // Allocates only when the reserved capacity is exceeded; otherwise it does not throw.
  void insert(const T& v) {
    assert(!detail::is_unordered(v));
    if (size_ == max_count) throw std::length_error("PagedMultiset: too many elements");
    if (count_.empty()) {
      const std::uint32_t s = acquire_slot();
      max_.push_back(v);
      count_.push_back(0);
      slot_.push_back(s);
      rebuild_tree(0);
    }
    insert_into(page_for_insert(v), v);
  }

  // Removes the first element equal to v. Returns false, changing nothing, if there is none.
  bool erase(const T& v) noexcept {
    size_type p = 0;
    size_type i = 0;
    if (!find(v, p, i)) return false;
    erase_at(p, i);
    return true;
  }

  // Same result as erase(old_value) followed by insert(new_value), including the return value
  // of that erase(). When both values belong to the same page, which is always the case while
  // there is only one, the elements between the two positions are shifted once and neither the
  // directory nor the tree is touched.
  bool replace(const T& old_value, const T& new_value) {
    assert(!detail::is_unordered(new_value));
    size_type p = 0;
    size_type i = 0;
    if (!find(old_value, p, i)) {
      insert(new_value);
      return false;
    }
    const size_type q = page_for_insert(new_value);
    if (q != p) {
      // Removing an element leaves q valid unless the directory had to be reorganised.
      const bool reorganised = erase_at(p, i);
      insert_into(reorganised ? page_for_insert(new_value) : q, new_value);
      return true;
    }
    T* const buf = page(p);
    const size_type n = count_[p];
    const size_type j = detail::upper_bound_index(buf, n, new_value);
    if (j > i) {
      std::memmove(buf + i, buf + i + 1, (j - i - 1) * sizeof(T));
      buf[j - 1] = new_value;
    } else {
      std::memmove(buf + j + 1, buf + j, (i - j) * sizeof(T));
      buf[j] = new_value;
    }
    max_[p] = buf[n - 1];
    return true;
  }

  // Removes all elements. Keeps the allocated memory.
  void clear() noexcept {
    free_.clear();
    for (size_type s = slot_count(); s-- > 0;) free_.push_back(static_cast<std::uint32_t>(s));
    max_.clear();
    count_.clear();
    slot_.clear();
    size_ = 0;
    rebuild_tree(0);
  }

  // The element of rank k in sorted order, 0 <= k < size().
  T kth(size_type k) const noexcept {
    assert(k < size_);
    size_type p = 0;
    size_type i = 0;
    locate(k, p, i);
    return page(p)[i];
  }

  // The elements of rank k and k + 1, 0 <= k < size() - 1, for the cost of one rank lookup.
  std::pair<T, T> kth_and_next(size_type k) const noexcept {
    assert(k + 1 < size_);
    size_type p = 0;
    size_type i = 0;
    locate(k, p, i);
    const T* const buf = page(p);
    return {buf[i], i + 1 < count_[p] ? buf[i + 1] : page(p + 1)[0]};
  }

  // Number of elements less than v.
  size_type count_less(const T& v) const noexcept {
    const size_type pages = count_.size();
    const size_type p = detail::lower_bound_index(max_.data(), pages, v);
    if (p == pages) return size_;
    return prefix_count(p) + detail::lower_bound_index(page(p), count_[p], v);
  }

  // Number of elements less than or equal to v.
  size_type count_less_equal(const T& v) const noexcept {
    const size_type pages = count_.size();
    const size_type p = detail::upper_bound_index(max_.data(), pages, v);
    if (p == pages) return size_;
    return prefix_count(p) + detail::upper_bound_index(page(p), count_[p], v);
  }

  // Writes all elements in sorted order.
  template <class OutputIt>
  OutputIt copy_to(OutputIt out) const {
    for (size_type p = 0; p < count_.size(); ++p) {
      const T* const buf = page(p);
      for (size_type i = 0; i < count_[p]; ++i) *out++ = buf[i];
    }
    return out;
  }

  // Checks every structural invariant in O(size()). For tests and debugging.
  bool validate() const {
    const size_type pages = count_.size();
    const size_type slots = slot_count();
    if (max_.size() != pages || slot_.size() != pages) return false;
    if (tree_.size() != (pages == 0 ? 0 : pages + 1)) return false;
    if (free_.size() + pages != slots) return false;
    std::vector<char> used(slots, 0);
    for (const std::uint32_t s : free_) {
      if (s >= slots || used[s] != 0) return false;
      used[s] = 1;
    }
    size_type total = 0;
    for (size_type p = 0; p < pages; ++p) {
      const size_type n = count_[p];
      if (n == 0 || n > cap_ || (pages > 1 && n < low_)) return false;
      if (slot_[p] >= slots || used[slot_[p]] != 0) return false;
      used[slot_[p]] = 1;
      const T* const buf = page(p);
      for (size_type i = 1; i < n; ++i) {
        if (buf[i] < buf[i - 1]) return false;
      }
      if (max_[p] < buf[n - 1] || buf[n - 1] < max_[p]) return false;
      if (p + 1 < pages && page(p + 1)[0] < buf[n - 1]) return false;
      total += n;
    }
    if (total != size_) return false;
    for (size_type i = 1; i <= pages; ++i) {
      size_type sum = 0;
      for (size_type j = i - detail::low_bit(i); j < i; ++j) sum += count_[j];
      if (tree_[i] != sum) return false;
    }
    if (pages == 0) return top_ == 0;
    return top_ <= pages && top_ * 2 > pages && detail::low_bit(top_) == top_;
  }

 private:
  static constexpr size_type max_count = std::numeric_limits<std::uint32_t>::max();
  static constexpr std::uint32_t minus_one = std::numeric_limits<std::uint32_t>::max();

  size_type slot_count() const noexcept { return slab_.size() / cap_; }

  T* page(size_type p) noexcept { return slab_.data() + size_type{slot_[p]} * cap_; }
  const T* page(size_type p) const noexcept { return slab_.data() + size_type{slot_[p]} * cap_; }

  // The page that receives v: the first whose largest element is greater than v, else the last.
  size_type page_for_insert(const T& v) const noexcept {
    const size_type pages = count_.size();
    const size_type p = detail::upper_bound_index(max_.data(), pages, v);
    return p == pages ? pages - 1 : p;
  }

  // Finds the first element equal to v: page p, index i. Returns false if there is none.
  bool find(const T& v, size_type& p, size_type& i) const noexcept {
    const size_type pages = count_.size();
    p = detail::lower_bound_index(max_.data(), pages, v);
    if (p == pages) return false;  // v is greater than every element, or the container is empty
    const T* const buf = page(p);
    i = detail::lower_bound_index(buf, count_[p], v);  // within the page because max_[p] >= v
    return !(v < buf[i]);
  }

  // Inserts v into page p, which must be page_for_insert(v).
  void insert_into(size_type p, const T& v) {
    if (count_[p] == cap_) {
      make_room(p);
      p = page_for_insert(v);
    }
    T* const buf = page(p);
    const size_type n = count_[p];
    const size_type i = detail::upper_bound_index(buf, n, v);
    std::memmove(buf + i + 1, buf + i, (n - i) * sizeof(T));
    buf[i] = v;
    if (i == n) max_[p] = v;
    count_[p] = static_cast<std::uint32_t>(n + 1);
    ++size_;
    tree_add(p, 1);
  }

  // Removes element i of page p. Returns true if that reorganised the directory: a page was
  // removed or elements moved between pages.
  bool erase_at(size_type p, size_type i) noexcept {
    T* const buf = page(p);
    const size_type n = count_[p];
    std::memmove(buf + i, buf + i + 1, (n - i - 1) * sizeof(T));
    --size_;
    if (n == 1) {
      remove_page(p);
      return true;
    }
    count_[p] = static_cast<std::uint32_t>(n - 1);
    if (i == n - 1) max_[p] = buf[n - 2];
    tree_add(p, minus_one);
    if (n - 1 >= low_ || count_.size() == 1) return false;
    refill(p);
    return true;
  }

  // Makes the slab hold at least `slots` page buffers and gives every directory array matching
  // capacity, so that no later operation on that many pages allocates. Strong exception guarantee.
  void grow_slots(size_type slots) {
    const size_type old = slot_count();
    if (slots <= old) return;
    if (slots > max_count) throw std::length_error("PagedMultiset: too many pages");
    max_.reserve(slots);
    count_.reserve(slots);
    slot_.reserve(slots);
    tree_.reserve(slots + 1);
    free_.reserve(slots);
    slab_.resize(slots * cap_);
    for (size_type s = slots; s-- > old;) free_.push_back(static_cast<std::uint32_t>(s));
  }

  // Returns an unused buffer slot. May reallocate the slab: page pointers taken earlier are stale.
  std::uint32_t acquire_slot() {
    if (free_.empty()) grow_slots(slot_count() + slot_count() / 2 + 1);
    const std::uint32_t s = free_.back();
    free_.pop_back();
    return s;
  }

  // Page p is full. Hand elements to the emptier neighbour if it has room to spare, which leaves
  // the directory layout alone; otherwise split the page in two.
  void make_room(size_type p) {
    const size_type pages = count_.size();
    size_type q = pages;
    if (p > 0 && count_[p - 1] <= spare_limit_) q = p - 1;
    if (p + 1 < pages && count_[p + 1] <= spare_limit_) {
      if (q == pages || count_[p + 1] < count_[q]) q = p + 1;
    }
    if (q == pages) {
      split(p);
    } else {
      balance(p < q ? p : q);
    }
  }

  // Moves the upper half of full page p into a new page inserted after it.
  void split(size_type p) {
    const std::uint32_t s = acquire_slot();
    const size_type half = cap_ / 2;
    const T* const src = page(p);
    std::memcpy(slab_.data() + size_type{s} * cap_, src + half, (cap_ - half) * sizeof(T));
    detail::vector_insert(max_, p + 1, max_[p]);
    detail::vector_insert(count_, p + 1, static_cast<std::uint32_t>(cap_ - half));
    detail::vector_insert(slot_, p + 1, s);
    max_[p] = src[half - 1];
    count_[p] = static_cast<std::uint32_t>(half);
    rebuild_tree(p);
  }

  // Page p has fallen below the low-water mark. Merge it with its emptier neighbour if the result
  // leaves room for further insertions, otherwise even the two out.
  void refill(size_type p) noexcept {
    const size_type pages = count_.size();
    size_type q = p + 1;
    if (q == pages || (p > 0 && count_[p - 1] <= count_[q])) q = p - 1;
    const size_type l = p < q ? p : q;
    if (size_type{count_[l]} + count_[l + 1] <= merge_limit_) {
      merge(l);
    } else {
      balance(l);
    }
  }

  // Appends page l + 1 to page l and removes it from the directory.
  void merge(size_type l) noexcept {
    std::memcpy(page(l) + count_[l], page(l + 1), count_[l + 1] * sizeof(T));
    count_[l] += count_[l + 1];
    max_[l] = max_[l + 1];
    free_.push_back(slot_[l + 1]);
    detail::vector_erase(max_, l + 1);
    detail::vector_erase(count_, l + 1);
    detail::vector_erase(slot_, l + 1);
    rebuild_tree(l);
  }

  // Evens out the element counts of pages l and l + 1. The page count does not change, so the
  // Fenwick tree needs two point updates rather than a rebuild.
  void balance(size_type l) noexcept {
    T* const a = page(l);
    T* const b = page(l + 1);
    const size_type na = count_[l];
    const size_type nb = count_[l + 1];
    const size_type target = (na + nb) / 2;  // new size of page l
    if (na > target) {
      const size_type k = na - target;
      std::memmove(b + k, b, nb * sizeof(T));
      std::memcpy(b, a + target, k * sizeof(T));
    } else {
      const size_type k = target - na;
      std::memcpy(a + na, b, k * sizeof(T));
      std::memmove(b, b + k, (nb - k) * sizeof(T));
    }
    count_[l] = static_cast<std::uint32_t>(target);
    count_[l + 1] = static_cast<std::uint32_t>(na + nb - target);
    max_[l] = a[target - 1];
    const std::uint32_t delta = static_cast<std::uint32_t>(target) - static_cast<std::uint32_t>(na);
    tree_add(l, delta);
    tree_add(l + 1, 0U - delta);
  }

  // Removes the page whose last element was just erased.
  void remove_page(size_type p) noexcept {
    free_.push_back(slot_[p]);
    detail::vector_erase(max_, p);
    detail::vector_erase(count_, p);
    detail::vector_erase(slot_, p);
    rebuild_tree(p);
  }

  // Fenwick tree: tree_[i], 1 <= i <= page_count(), is the number of elements in pages
  // [i - low_bit(i), i). Counts are added modulo 2^32, so adding minus_one subtracts one.
  void tree_add(size_type p, std::uint32_t delta) noexcept {
    const size_type pages = count_.size();
    for (size_type i = p + 1; i <= pages; i += detail::low_bit(i)) tree_[i] += delta;
  }

  // Number of elements in pages [0, p).
  size_type prefix_count(size_type p) const noexcept {
    size_type sum = 0;
    for (size_type i = p; i > 0; i -= detail::low_bit(i)) sum += tree_[i];
    return sum;
  }

  // Recomputes the tree after the directory changed at page `first` or later, in time linear in
  // the number of pages from `first` on. Nodes covering only earlier pages are still valid; those
  // among them whose parent is being recomputed (the prefix path of `first`) seed the rebuild.
  void rebuild_tree(size_type first) noexcept {
    const size_type pages = count_.size();
    top_ = 0;
    if (pages == 0) {
      tree_.clear();
      return;
    }
    tree_.resize(pages + 1);
    tree_[0] = 0;
    for (size_type i = first + 1; i <= pages; ++i) tree_[i] = count_[i - 1];
    for (size_type i = first; i > 0; i -= detail::low_bit(i)) {
      const size_type parent = i + detail::low_bit(i);
      if (parent <= pages) tree_[parent] += tree_[i];
    }
    for (size_type i = first + 1; i <= pages; ++i) {
      const size_type parent = i + detail::low_bit(i);
      if (parent <= pages) tree_[parent] += tree_[i];
    }
    top_ = 1;
    while (top_ * 2 <= pages) top_ *= 2;
  }

  // Finds the page p, and the index i within it, of the element of rank k < size().
  void locate(size_type k, size_type& p, size_type& i) const noexcept {
    const size_type pages = count_.size();
    size_type pos = 0;
    for (size_type step = top_; step != 0; step /= 2) {
      const size_type next = pos + step;
      if (next <= pages && tree_[next] <= k) {
        pos = next;
        k -= tree_[next];
      }
    }
    p = pos;
    i = k;
  }

  size_type cap_;          // page capacity
  size_type low_;          // a page with fewer elements than this is merged or refilled
  size_type merge_limit_;  // two pages are merged only if together they hold at most this many
  size_type spare_limit_;  // a full page hands elements to a neighbour holding at most this many
  size_type size_ = 0;
  size_type top_ = 0;                 // largest power of two not exceeding the page count
  std::vector<T> slab_;               // page buffers: slot s is slab_[s * cap_, (s + 1) * cap_)
  std::vector<std::uint32_t> free_;   // unused slots
  std::vector<T> max_;                // directory: largest element of each page
  std::vector<std::uint32_t> count_;  // directory: number of elements in each page
  std::vector<std::uint32_t> slot_;   // directory: buffer slot of each page
  std::vector<std::uint32_t> tree_;   // Fenwick tree over count_
};

}  // namespace rq

#endif  // ROLLING_QUANTILE_PAGED_MULTISET_HPP
