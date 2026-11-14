# paged-rolling-quantile

Exact quantiles of a sliding window over a stream of numbers, in C++17. Header-only, no
dependencies. The sorted window is kept in small sorted arrays ("pages") stored in one contiguous
buffer, so an update is a few binary searches and one short `memmove` rather than a walk through a
pointer-based tree.

This repository also contains the simplest correct ways to compute the same thing, written from
scratch: sort the window at every step, `std::nth_element`, a sorted array with insertion, a
`std::multiset`, and two heaps. All of them are tested against the same definition and benchmarked
with the same harness. Several of them win in some regimes; the results section says where.

```cpp
#include <rolling_quantile/rolling_quantile.hpp>

rq::RollingQuantile<double> median(1000);         // window of 1000 values, q = 0.5
rq::RollingQuantile<double> p99(1000, 0.99);

double m = median.update(x);                      // push x, return the median of the window
                                                  // (NaN until 1000 values have been seen)
p99.update(in, count, out);                       // the same for an array of inputs
double p90 = p99.quantile(0.9);                   // any other quantile of the current window
```

## The problem

Given a window length `n` and a quantile `q`, after each new value `x[t]` report the
`q`-quantile of `x[t-n+1], ..., x[t]`. Each step adds one value and removes the oldest, so the
work is maintaining a sorted multiset under one insertion and one deletion per step, plus a rank
query. Exactness matters: the result must equal what a sort of the window would give, bit for bit.

## The algorithm

`RollingQuantile<T>` is a ring buffer of the last `n` values, which decides which value leaves,
and a `PagedMultiset<T>` holding the same values in sorted order
(`include/rolling_quantile/paged_multiset.hpp`).

`PagedMultiset` stores the sorted sequence as pages of at most `c` elements each:

- **Pages.** Each page is a sorted run in a fixed slot of one slab (`std::vector<T>` of
  `slots * c` elements). Reading the pages in directory order gives the whole multiset in order.
- **Directory.** Three parallel arrays indexed by page position: the page's largest element, its
  element count, and its slot in the slab.
- **Fenwick tree** over the per-page counts, which maps a rank to a page in `O(log P)` for `P`
  pages.

One step of a full window is `replace(oldest, x)`:

1. Binary search the directory's array of page maxima for the page holding `oldest`, then the page
   for its position. Binary search the same array for the page that should receive `x`.
2. If both are the same page (always the case while there is only one page), shift the elements
   between the two positions by one with a single `memmove` and write `x`. The directory and the
   Fenwick tree do not change.
3. Otherwise remove from one page and insert into the other: one `memmove` in each page and two
   Fenwick point updates.
4. The quantile is read by descending the Fenwick tree to the page holding rank `k`, then
   indexing into it; rank `k + 1` is the next element in the same page or the first of the next.

Pages are reorganised only when one overflows or runs low. A full page that receives an element first hands half
the surplus to a neighbour with room; if neither has room it is split. A page that falls below
`c/4` elements is merged with a neighbour if the two fit in `3c/4`, otherwise the two are evened
out. A split, merge or page removal inserts into or erases from the directory arrays and rebuilds
the Fenwick tree from that page onward; that is the expensive, infrequent case.

Equal elements are kept in arrival order (insert after equals, erase the first equal), so the
sorted store holds exactly the values a queue would hold, down to the sign of a zero.

The default page capacity depends on the window: a window of up to 512 values is one page, that
is, one sorted array; larger windows use pages of 128 values, doubled each time the window
quadruples beyond 2^17, so that the page count grows like the square root of the window. It can
be overridden with the third constructor argument. "Choosing the page capacity" below has the
measurements.

### Why it is cache-friendly

Per update the paged structure touches: one slot of the ring buffer; the array of page maxima
(`P` contiguous values, binary searched); one or two pages (`c` contiguous values each, `memmove`d);
and `O(log P)` entries of a contiguous `uint32_t` Fenwick array. For a window of 65,536 values
with `c = 128` that is a directory of about 700 doubles (under 6 KiB) and pages of 1 KiB, all of
which stays in L1/L2 between updates. Nothing is allocated after construction.

A `std::multiset` of the same window does about `log2 n` dependent pointer loads per lookup, to
nodes scattered over the heap, plus one `malloc` and one `free` per update. Once the window no
longer fits in L2 each of those loads is a likely cache miss. Hardware counters were not available
on the benchmark machine, so this argument rests on the access pattern and on the timings below,
not on measured miss counts.

## Complexity

Window `n`, page capacity `c`, page count `P` (between `n/c` and `4n/c`):

| operation | cost |
|---|---|
| update, common case | `O(log n)` comparisons + `O(c)` element moves (one `memmove`) + `O(log P)` |
| update, page split / merge / removal | `O(c + P)`: directory arrays shifted, Fenwick tree rebuilt from that page on |
| `quantile(p)`, `update()` result | `O(log P)` |
| construction | `O(n)`; all memory allocated here |

With the default capacities, `c = n` for `n <= 512` (a single sorted array, `O(n)` moves),
`c = 128` up to `n = 2^17`, and `c` about `0.35 * sqrt(n)` beyond, so a reorganising step costs
`O(sqrt n)` for large windows. Reorganisations are rare (a page has to gain or lose about `c/4`
elements between them), which is why the median cost stays close to the common case while the
tail does not; the latency tables show both.

## Memory

`RollingQuantile` allocates everything in its constructor and never again (checked by
`tests/test_no_allocation.cpp`, which counts calls to `operator new`). Because every page of a
multi-page store holds at least `c/4` elements, the constructor reserves `4n/c` page slots, that
is, room for `4n` values, plus the ring buffer of `n` values. The slab is value-initialised, so
this memory is touched, not just reserved.

Measured peak heap bytes per window value (`bench/memory_footprint.cpp`, `double`, uniform input,
3 window turnovers):

| window | paged (reserved) | paged (pages in use + ring) | multiset | two_heaps | sorted_array | nth_element, sort |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 18.1 | 16.0 | 56.2 | 85.4 | 24.4 | 16.2 |
| 1,024 | 40.8 | 20.0 | 56.1 | 76.9 | 24.0 | 16.0 |
| 65,536 | 40.9 | 19.4 | 56.0 | 78.4 | 24.0 | 16.0 |
| 1,048,576 | 40.2 | 18.9 | 56.0 | 79.0 | 24.0 | 16.0 |

The reservation costs about twice what is used: with pages about 70% full in steady state,
`paged` would need about 19 bytes per value, but it reserves 40 in order to guarantee no
allocation. That is less than `multiset` (one tree node per value, 48 bytes as allocated, plus the ring) and
`two_heaps`, and more than a single sorted array.

## Quantile definition

For `m` values sorted as `x[0] <= ... <= x[m-1]` and `0 <= q <= 1`:

```
h = q * (m - 1),   k = floor(h),   g = h - k
Q = x[k]                          if g == 0
Q = x[k] + g * (x[k+1] - x[k])    otherwise
```

This is linear interpolation between closest ranks: NumPy's default `method="linear"`, R's type 7,
and what `pandas.Series.rolling(n).quantile(q)` computes. `reference/definition.hpp` executes it
literally and every test compares against it.

- **Exactness.** Results are bit-identical to `pandas.Series.rolling(n).quantile(q,
  interpolation="linear")`: `tools/compare_with_pandas.py` found every window identical with
  pandas 1.5.3, 2.3.3 and 3.1.1 (400 random series with NaNs, 84,140 windows per version). NumPy
  evaluates the same rule as
  `x[k+1] - (x[k+1] - x[k]) * (1 - g)` when `g >= 0.5`, which can differ from the expression above
  in the last bit; for `g < 0.5` the two agree.
- **NaN.** A NaN occupies its slot in the window like any other value. While the window holds a
  NaN, `update()` and `quantile()` return NaN; once it leaves, results resume. This is pandas'
  behaviour with the default `min_periods`. NaNs are counted separately and never enter the
  sorted store, so they cost nothing to order. `nan_count()` reports how many are held.
- **Window not yet full.** `update()` returns NaN until `n` values have been pushed.
  `quantile(p)` answers for whatever is held (NaN when empty).
- **Infinities** are ordinary values at the ends of the order. Interpolating between a finite and
  an infinite neighbour gives that infinity; between `-inf` and `+inf` it gives NaN. When
  `x[k+1] - x[k]` is not finite (an infinite operand, or finite values further apart than the
  largest finite `T`), `(1 - g) * x[k] + g * x[k+1]` is evaluated instead, so finite neighbours
  never produce an overflowed result. Here pandas differs: it returns NaN for, e.g., the median of
  `{-inf, 1}`, and NumPy 2.2 returns `-inf` for `q = 0.5` but NaN for `q = 0.25` on the same two
  values. This library returns `-inf` for every `0 < q < 1` there.
- **Signed zeros** compare equal and are ranked in arrival order, as a stable sort would.
- **Arithmetic** is carried out in `T` (`float` or `double`); `g` is computed in `double`.

## Baselines

All in `reference/`, each a few dozen lines, sharing a ring buffer (`fifo_window.hpp`):

| name | method | cost per update |
|---|---|---|
| `sort` | copy the window, `std::sort` | `O(n log n)` |
| `nth_element` | copy the window, `std::nth_element`, then `min_element` for rank `k + 1` | `O(n)` average |
| `sorted_array` | one sorted `std::vector`: `erase` the expired value, `insert` the new one | `O(n)` moves, `O(log n)` compares |
| `multiset` | `std::multiset` with an iterator kept on rank `k`, moved by at most one per step | `O(log n)` |
| `two_heaps` | max-heap of the lowest `k + 1` values, min-heap of the rest, lazy deletion through a hash map of expired values, full rebuild when expired values outnumber live ones | `O(log n)` amortised, `O(n log n)` rebuild about once per `n` updates |

`tests/test_reference.cpp` checks each of them against the definition, and checks
`RollingQuantile` against `multiset` bit for bit on long streams with windows up to 10^6.

## Results

Intel Xeon E5-2650 v3 (Haswell, 2.3 GHz nominal, turbo enabled; 32 KiB L1d and 256 KiB L2 per
core, 25 MiB L3 per socket), Ubuntu 24.04, Linux 6.8, gcc 13.3.0, `-std=c++17 -O3 -DNDEBUG` (CMake
Release, no `-march`). One logical CPU, sibling idle; see "Methodology" for the machine's state.
`q = 0.5`, `double`. Each entry is the median of 5 repetitions. Bold marks a row's fastest entry
when it leads the next one by more than about 10% (the noise level); closer rows are unmarked.

Mean nanoseconds per update (throughput pass):

| window | dist | paged | multiset | two_heaps | sorted_array | nth_element | sort |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | uniform | 61.5 | 117 | 297 | 57.4 | 104 | 91.1 |
| 16 | uniform | 70.4 | 161 | 280 | 74.5 | 217 | 184 |
| 64 | uniform | **86.6** | 208 | 275 | 113 | 621 | 1,416 |
| 256 | uniform | **107** | 244 | 358 | 167 | 1,938 | 8,261 |
| 1,024 | uniform | **182** | 281 | 355 | 293 | 7,627 | 46,525 |
| 4,096 | uniform | **210** | 340 | 357 | 727 | 36,585 | |
| 16,384 | uniform | **239** | 536 | 386 | 4,600 | 180,928 | |
| 65,536 | uniform | **331** | 727 | 423 | 23,919 | | |
| 262,144 | uniform | 443 | 1,301 | 436 | | | |
| 1,048,576 | uniform | 709 | 2,029 | 648 | | | |
| 8 | walk | 55.6 | 112 | 294 | **48.7** | 100 | 92.8 |
| 16 | walk | 63.0 | 158 | 279 | 61.9 | 204 | 183 |
| 64 | walk | **80.1** | 202 | 280 | 96.2 | 546 | 1,357 |
| 256 | walk | **104** | 225 | 364 | 143 | 1,602 | 7,014 |
| 1,024 | walk | **185** | 244 | 364 | 255 | 5,115 | 37,051 |
| 4,096 | walk | **207** | 268 | 364 | 687 | 18,744 | |
| 16,384 | walk | **231** | 322 | 418 | 4,751 | 65,566 | |
| 65,536 | walk | **276** | 487 | 540 | 26,326 | | |
| 262,144 | walk | **336** | 737 | 749 | | | |
| 1,048,576 | walk | **425** | 908 | 792 | | | |
| 8 | ticks | 53.0 | 99.2 | 249 | **42.5** | 81.7 | 77.5 |
| 16 | ticks | 61.7 | 148 | 247 | 56.9 | 163 | 165 |
| 64 | ticks | **79.0** | 189 | 228 | 92.3 | 425 | 1,052 |
| 256 | ticks | **106** | 204 | 224 | 137 | 1,311 | 4,955 |
| 1,024 | ticks | **183** | 213 | 215 | 242 | 4,522 | 23,823 |
| 4,096 | ticks | 202 | 222 | 214 | 667 | 17,806 | |
| 16,384 | ticks | 224 | 243 | 217 | 4,542 | 72,356 | |
| 65,536 | ticks | 257 | 284 | **228** | 26,579 | | |
| 262,144 | ticks | 287 | 350 | **231** | | | |
| 1,048,576 | ticks | 346 | 416 | **244** | | | |
| 8 | ascending | 43.0 | 62.7 | 235 | **24.5** | 40.8 | 37.9 |
| 16 | ascending | 52.3 | 65.7 | 228 | **28.1** | 66.8 | 69.0 |
| 64 | ascending | 70.5 | 77.4 | 240 | **38.2** | 274 | 651 |
| 256 | ascending | 110 | 87.4 | 340 | **61.6** | 1,123 | 3,817 |
| 1,024 | ascending | 145 | **98.0** | 349 | 179 | 4,126 | 20,040 |
| 4,096 | ascending | 163 | **108** | 363 | 669 | 17,850 | |
| 16,384 | ascending | **186** | 216 | 441 | 5,218 | 70,772 | |
| 65,536 | ascending | **214** | 282 | 505 | 28,429 | | |
| 262,144 | ascending | **243** | 337 | 650 | | | |
| 1,048,576 | ascending | **284** | 489 | 779 | | | |

Latency of one update in nanoseconds (latency pass; each sample includes about 8 ns of clock
reading and loop):

| window | dist | implementation | mean | p50 | p99 | p99.9 | max |
|---:|---|---|---:|---:|---:|---:|---:|
| 1,024 | uniform | paged | 182 | 188 | 219 | 269 | 19,415 |
| 1,024 | uniform | multiset | 281 | 282 | 391 | 460 | 42,629 |
| 1,024 | uniform | two_heaps | 355 | 264 | 629 | 18,563 | 104,143 |
| 65,536 | uniform | paged | 331 | 334 | 413 | 521 | 26,246 |
| 65,536 | uniform | multiset | 727 | 714 | 1,071 | 1,487 | 49,351 |
| 65,536 | uniform | two_heaps | 423 | 296 | 638 | 992 | 8,245,002 |
| 1,048,576 | uniform | paged | 709 | 643 | 1,311 | 2,781 | 49,694 |
| 1,048,576 | uniform | multiset | 2,029 | 1,988 | 3,126 | 7,792 | 41,952 |
| 1,048,576 | uniform | two_heaps | 648 | 398 | 949 | 1,752 | 209,090,118 |
| 1,024 | walk | paged | 185 | 187 | 311 | 393 | 26,571 |
| 1,024 | walk | multiset | 244 | 246 | 363 | 427 | 15,423 |
| 1,024 | walk | two_heaps | 364 | 267 | 1,114 | 18,915 | 92,616 |
| 65,536 | walk | paged | 276 | 256 | 805 | 2,175 | 19,043 |
| 65,536 | walk | multiset | 487 | 479 | 1,364 | 2,239 | 70,078 |
| 65,536 | walk | two_heaps | 540 | 333 | 1,970 | 4,932 | 8,434,507 |
| 1,048,576 | walk | paged | 425 | 347 | 988 | 7,358 | 51,645 |
| 1,048,576 | walk | multiset | 908 | 741 | 1,801 | 4,954 | 57,753 |
| 1,048,576 | walk | two_heaps | 792 | 474 | 2,700 | 7,017 | 190,537,269 |

The full tables for every window and distribution are produced by `bench/summarize.py` from the raw
results in `bench/results/`.

A tail quantile, `q = 0.99`, 3 repetitions, mean ns per update:

| window | dist | paged | multiset | two_heaps | sorted_array |
|---:|---|---:|---:|---:|---:|
| 4,096 | uniform | **209** | 336 | 265 | 719 |
| 65,536 | uniform | 331 | 745 | 345 | 23,956 |
| 1,048,576 | uniform | 642 | 1,966 | **401** | |
| 4,096 | walk | **206** | 265 | 288 | 687 |
| 65,536 | walk | **272** | 452 | 354 | 26,160 |
| 1,048,576 | walk | **426** | 935 | 679 | |

### Where the simple implementations win

- **Tiny windows (8 to 16).** `sorted_array` is as fast as `paged` or faster on every input (for
  example 42.5 vs 53.0 ns at `n = 8` on ticks). At `n = 8` on ascending input even `sort` and
  `nth_element` beat `paged` (37.9 and 40.8 vs 43.0 ns). Below a few dozen values, the
  bookkeeping of the paged structure costs more than it saves.
- **Monotone input, small and medium windows.** On strictly ascending input `sorted_array` wins up
  to `n = 256` and `std::multiset` from 1,024 to 4,096 (98 vs 145 ns at 1,024). The tree's
  insertion point is always its rightmost path, which stays in cache.
- **Two heaps on input with many ties, and in the median.** `two_heaps` has the lowest p50 of all
  implementations at large windows on uniform, walk and tick input (398 vs 643 ns at 2^20, uniform). It has the
  lowest mean on tick data from 16,384 up (244 vs 346 ns at 2^20), and also on uniform input at
  2^20 for `q = 0.5` (648 vs 709) and for `q = 0.99` (401 vs 642). It pays for this with a
  rebuild of both heaps about once per window turnover, which shows up as a maximum of 8 ms at
  65,536 and 190 to 210 ms at 2^20. Where that stall is acceptable, it is a reasonable choice.
- **The paged structure's own tail.** At 2^20 on the walk, `paged` has a higher p99.9 than
  `multiset` (7.4 vs 5.0 µs), from page splits and merges whose directory work is `O(P)`.

`paged` is fastest, or within the noise of the fastest, from `n = 64` to 65,536 on uniform and
walk input and from 64 to 16,384 on tick input. It is fastest at every size from 16,384 up on walk and ascending input. Its
largest margins are over `multiset` at large windows (at 2^20: 2.1x to 2.9x on uniform and walk
input, 1.2x to 1.7x on ticks and ascending input; a later run on a quieter machine gave 1.9x to 2.6x
for uniform and walk, see below) and over the `O(n)` methods anywhere above a few hundred values.

### Noise

Median spread between the 5 repetitions of a cell, `(max - min) / median`: 1 to 2% for the mean
and p50 of every implementation, 2 to 13% for p99, 10 to 19% for p99.9. The maximum is not
reproducible (spread 37% to over 100%) and reflects interrupts and the shared machine more than
the code. The worst single cells reached 45% for the mean of `multiset` and `sort`. An earlier
full run on CPUs that shared physical cores with other jobs, and with shorter passes, gave means
up to 30% lower for `multiset` at 2^18 (929 vs 1,301 ns) and a clock overhead of 10.9 instead of
7.8 ns. Only the pinned run is reported here.

### A second run on a quieter machine

The tables above were measured while other users' jobs kept the machine at a load average of 17 to 20.
Twenty-seven cells (windows 1,024, 65,536 and 2^20; uniform, walk and tick input; `paged`, `multiset`
and `two_heaps`) were measured again later with the same binary, core and method, 3 repetitions, at a
load average near 6 (`bench/results/second_run_subset.csv`).

Cells whose working set fits in the core's own caches came out the same: every cell at `n = 1,024` and
every cell on tick input is within 2% of the first run. Cells that depend on the shared last-level cache
were faster in the second run for every implementation, by up to 23%, and `std::multiset` gained the most:

| window | dist | paged | multiset | two_heaps | multiset / paged | in the first run |
|---:|---|---:|---:|---:|---:|---:|
| 65,536 | uniform | 330 | 697 | 407 | 2.1 | 2.2 |
| 1,048,576 | uniform | 603 | 1,558 | 557 | 2.6 | 2.9 |
| 65,536 | walk | 270 | 397 | 427 | 1.5 | 1.8 |
| 1,048,576 | walk | 398 | 775 | 647 | 1.9 | 2.1 |

So the margin over `multiset` at large windows depends on how much of the shared cache the process gets:
1.9x to 2.6x at 2^20 on the quieter machine, against 2.1x to 2.9x in the main tables. The order of the
three implementations did not change in any of the 27 cells.

## Choosing the page capacity

Two costs pull in opposite directions. Every update moves on average a third of a page, so large
pages cost `O(c)`. Every reorganisation shifts the directory and rebuilds the Fenwick tree from
that page on, so small pages cost `O(n/c)` per reorganisation, and they reorganise more often.
Mean nanoseconds per update, `q = 0.5`, 3 repetitions, same machine and method as below (a
capacity at or above the window means a single page):

| window | dist | c=32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | uniform | 172 | 175 | 182 | 187 | 190 | **155** | 155 |
| 4,096 | uniform | **196** | 199 | 210 | 219 | 241 | 270 | 324 |
| 65,536 | uniform | **306** | 310 | 331 | 365 | 432 | 578 | 872 |
| 262,144 | uniform | 415 | **396** | 403 | 450 | 529 | 711 | 1,080 |
| 1,048,576 | uniform | 750 | 710 | 642 | **606** | 622 | 836 | 1,253 |
| 1,024 | walk | 189 | 184 | 185 | 187 | 199 | **175** | 175 |
| 4,096 | walk | 219 | 206 | **207** | 209 | 228 | 263 | 329 |
| 65,536 | walk | 474 | 308 | 270 | **261** | 284 | 320 | 406 |
| 262,144 | walk | 1,220 | 545 | 371 | **324** | 337 | 374 | 477 |
| 1,048,576 | walk | 5,769 | 1,723 | 702 | 464 | **403** | 447 | 572 |

Defaults: `c = 128` at 1,024, 4,096 and 65,536, 256 at 262,144, 512 at 2^20.

On the random walk the insertions concentrate in a few pages, which then split and merge
constantly. Small pages make each of those reorganisations expensive: at `n = 2^20` the p99 of an
update is 126 µs with `c = 32`, 7.4 µs with `c = 128`, and 0.7 µs with `c = 512` (p99.9 7.3 µs).
On independent uniform input, smaller pages are slightly faster. The default follows the walk,
since it is the less forgiving case and closer to price data. On walk input it is the best column
or within 4% of it. On uniform input it is within 9% of the best column, except at `n = 2^18`,
where it is 13% behind (`c = 256` against the best `c = 64`).

The single-page threshold is a trade between typical and worst-case input. On uniform, walk and
tick input a single sorted array beats 128-element pages at `n = 768` and `n = 1,024` (146 vs
180 ns at 768 and 155 vs 182 ns at 1,024, uniform); the two are even at 1,536, and pages win from
2,048. On strictly ascending input, however, every step removes the first element of the array
and appends at the end, so a single page moves all `n` elements: at `n = 1,024` that costs 237 ns
against 144 ns with `c = 128`. The default keeps the single page only up to 512 values, which
bounds the elements moved per step by 512, and gives up about 15% at `n` between 513 and about
1,500 on non-monotone input. Pass `page_capacity = n` to the constructor to choose the single
array.

## Methodology

`bench/bench_rolling_quantile.cpp`, summarised by `bench/summarize.py`.

- **Input.** Four streams from `std::mt19937_64` with seed 1, converted to `double` explicitly so a
  seed identifies the input on any standard library: `uniform` (independent, uniform on [0, 1)),
  `walk` (continuous random walk), `ticks` (random walk on an integer grid: a window holds few
  distinct values and many duplicates), `ascending` (every value is a new maximum and the oldest
  is the minimum).
- **Warm-up.** Each implementation is constructed, the window filled, and 512 further values
  pushed. Implementations whose layout depends on history (`paged`, `multiset`, `two_heaps`) then
  see one more full turnover of the window before timing starts. The first `n + 256` results are
  folded into a checksum that must agree across all implementations, or the run aborts.
- **Throughput.** A pass of `S` further updates timed as a whole with `steady_clock`; mean
  nanoseconds per update. `S` is chosen from the warm-up so the pass takes about 0.5 s, clamped to
  [20,000, 1,000,000]. For `paged`, `multiset` and `two_heaps` a pass is never shorter than two
  window turnovers (`2n` updates), so that amortised work such as the two-heap rebuild, which
  happens about once per turnover, is inside the timed interval. (An earlier run without this rule
  reported `two_heaps` at 241 ns per update for `n = 262,144`, from 20,000 updates that contained no
  rebuild.)
- **Latency.** A second pass of `S` fresh updates reading `rdtsc` after each one. A sample is the
  interval between consecutive readings, so it includes the reading and the loop; with an empty
  update the median sample is 7.8 ns. p50, p99, p99.9 and max are order statistics of
  the `S` samples (max is a single sample and mostly measures interrupts and the machine).
- **Repetitions.** Every cell is measured 5 times, rotating the order of the implementations, and
  the tables report the median of the 5. The spread table reports `(max - min) / median` across
  repetitions.
- **Quantile.** `q = 0.5` unless stated.
- **Excluded cells.** `sort` above 1,024, `nth_element` above 16,384 and `sorted_array` above
  65,536 values are not run; they take milliseconds per update there.
- **Memory.** `bench/memory_footprint.cpp` replaces the global allocation functions and records the
  peak of `malloc_usable_size` over live blocks while each implementation runs 3 window turnovers
  of uniform input.

Pinning. Each timing run was pinned to one logical CPU with `taskset -c 4` (main tables) or
`taskset -c 5` (page-capacity measurements, run at the same time). CPUs 4 and 5 are on different
sockets, so the two runs shared neither a core nor an L3 cache, and their hyperthread siblings
(24 and 25) were kept idle. No other job used those physical cores. The machine as a whole was
shared and not tuned: other users' jobs ran on other cores (load average 17 to 20 on 40 hardware
threads), the frequency governor was `schedutil` with turbo enabled, and interrupts were not
steered away. Treat differences below about 10% as noise; the spread table shows how much each
implementation moved between repetitions. An earlier run on cores shared with other jobs was
discarded. The `paged` rows at `n = 1,024` in the main tables were re-measured afterwards with the
same settings on CPU 4, after the default page capacity for that size was settled; the other rows
did not depend on that choice.

Reproduce:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# CPU 4 is an example: use a core that is idle, with an idle hyperthread sibling
taskset -c 4 build/bench/bench_rolling_quantile --cpu 4 \
    --windows 8,16,64,256,1024,4096,16384,65536,262144,1048576 --out results.csv
python3 bench/summarize.py results.csv
build/bench/memory_footprint 64 1024 16384 65536 1048576
```

## Limitations

- **Count-based windows only.** The window is the last `n` values, not the values of the last `t`
  seconds. A time-based window would need `pop()` driven by timestamps; `push()` and `pop()` exist
  for that, but the default page capacity and the reservation assume a fixed `n`.
- **Not the fastest everywhere.** See "Where the simple implementations win". In short: use a sorted
  array below about 32 values, and consider two heaps for the median on data with many ties if
  occasional stalls of milliseconds are acceptable.
- **Tail latency is not constant-time.** A page split or merge costs `O(c + n/c)`. With the default
  capacities the p99.9 at `n = 2^20` is 2.8 µs (uniform) and 7.4 µs (walk) on this machine.
- **Memory.** About 40 bytes per value for `double`, half of it reserved and not used in steady
  state (see "Memory").
- **Floating-point values only** in `RollingQuantile`. `PagedMultiset` accepts any trivially
  copyable type with a strict weak order.
- **Single-threaded.** Objects are not thread-safe; use one per thread.
- **Measurements on one machine and one compiler.** gcc 13 on a shared Haswell server, without
  `-march=native`. clang is built and tested in CI, but the benchmark numbers come from gcc only.
  Hardware performance counters were not available, so cache behaviour is inferred from timings.
- **The pandas comparison is not part of the test suite** because it needs Python, numpy and
  pandas; it was run for this release with pandas 1.5.3, 2.3.3 and 3.1.1.

## Build and test

Requires CMake 3.16 and a C++17 compiler. The library itself is two headers; to use it, add
`include/` to the include path or `add_subdirectory()` this repository and link
`rq::rolling_quantile`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRQ_WARNINGS_AS_ERRORS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# AddressSanitizer and UndefinedBehaviorSanitizer, assertions enabled
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=RelWithDebInfo -DRQ_SANITIZE=ON
cmake --build build-san -j
ctest --test-dir build-san --output-on-failure
```

Programs are built with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`. The
tests are compiled with `-ffp-contract=off` because they compare floating-point results bit for
bit and the two sides must not be contracted into FMAs differently.

| test | what it checks |
|---|---|
| `test_definition` | worked examples; pandas and NumPy values written out; NaN, infinities, overflow, signed zeros; the interface contract |
| `test_differential` | randomized differential test: random windows (1 to 4,096, with page-boundary sizes), quantiles (including 0, 1, 1e-9 from either end), page capacities (2 to beyond one page), 11 stream shapes (uniform, random walk, ticks, five distinct values, constant, ascending, descending, sawtooth, monotone runs with plateaus, overflowing magnitudes, subnormals), injected NaN / ±inf / ±0, and mixes of `update`, batch `update`, `push`, bursts of `pop`, `clear` and state transfer through `copy_window`/`assign`; after every step the result is compared bit for bit with the definition applied to a copy of the window, and the store's invariants are validated periodically. `test_differential [cases [seed]]` runs more |
| `test_paged_multiset` | the sorted store against a sorted-vector model for `double`, `float`, `int64_t` and a type with ties distinguishable by a tag, page capacities 2 to 128, every removal pattern; and a 200,000-element store against `std::multiset` |
| `test_reference` | each baseline against the definition; `RollingQuantile` against `multiset` on streams of up to 1.6 million values with windows up to 10^6 |
| `test_no_allocation` | no heap allocation after construction, under every operation |
| `bench_smoke`, `memory_footprint_smoke` | short runs of the benchmark programs, so warnings and sanitizers cover them too |

`examples/rolling_quantile_cli.cpp` reads numbers from standard input and prints the rolling
quantile; `tools/compare_with_pandas.py` uses it to compare against pandas (needs numpy and
pandas, not part of the test suite).

## License

Apache License 2.0. See `LICENSE`.
