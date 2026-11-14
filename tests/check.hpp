// SPDX-License-Identifier: Apache-2.0
//
// Test support: a CHECK macro and exact floating-point comparison. No framework, no dependency.

#ifndef ROLLING_QUANTILE_TESTS_CHECK_HPP
#define ROLLING_QUANTILE_TESTS_CHECK_HPP

#include <cmath>
#include <cstdio>
#include <cstring>

namespace test {

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline void report_failure(const char* file, int line, const char* what) {
  std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, what);
  ++failure_count();
}

// Prints the verdict for one test program and returns its exit code.
inline int finish(const char* name) {
  if (failure_count() != 0) {
    std::fprintf(stderr, "%s: FAILED (%d checks)\n", name, failure_count());
    return 1;
  }
  std::printf("%s: ok\n", name);
  return 0;
}

// True if a and b are the same result: both NaN, or identical bit patterns.
template <class T>
bool same(T a, T b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// As same(), except that a zero matches a zero of either sign.
template <class T>
bool same_value(T a, T b) {
  return same(a, b) || (a == T(0) && b == T(0));
}

}  // namespace test

#define CHECK(condition)                                                  \
  do {                                                                    \
    if (!(condition)) ::test::report_failure(__FILE__, __LINE__, #condition); \
  } while (false)

#endif  // ROLLING_QUANTILE_TESTS_CHECK_HPP
