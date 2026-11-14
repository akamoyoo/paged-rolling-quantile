// SPDX-License-Identifier: Apache-2.0
//
// Reads numbers from standard input, one per line, and after each one writes the Q-quantile of
// the last WINDOW of them; "nan" until that many have been read.
//
//   rolling_quantile_cli WINDOW [Q]        Q defaults to 0.5
//
// Results are written with 17 significant digits, which identifies a double exactly.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "rolling_quantile/rolling_quantile.hpp"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fputs("usage: rolling_quantile_cli WINDOW [Q]\n", stderr);
    return 2;
  }
  const std::size_t window = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  const double q = argc == 3 ? std::strtod(argv[2], nullptr) : 0.5;
  try {
    rq::RollingQuantile<double> rolling(window, q);
    char line[128];
    while (std::fgets(line, sizeof(line), stdin) != nullptr) {
      std::printf("%.17g\n", rolling.update(std::strtod(line, nullptr)));
    }
  } catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
  return 0;
}
