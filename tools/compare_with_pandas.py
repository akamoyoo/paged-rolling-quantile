#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare examples/rolling_quantile_cli with pandas, bit for bit.

    compare_with_pandas.py path/to/rolling_quantile_cli [cases]

Random series are piped through the command-line example and through
pandas.Series.rolling(window).quantile(q, interpolation="linear"). Every output must have the
same bit pattern, and NaN must appear at the same positions. Series contain NaNs but no
infinities: pandas interpolates towards an infinite neighbour with plain arithmetic, which this
library deliberately does not (see "Quantile definition" in README.md).

Needs numpy and pandas. It is not part of the test suite.
"""

import subprocess
import sys

import numpy as np
import pandas as pd

QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 1 / 3, 0.5, 0.6, 2 / 3, 0.75, 0.9, 0.99, 0.999, 1.0]


def run_cli(cli, values, window, q):
    text = "".join(f"{float(v)!r}\n" for v in values)
    done = subprocess.run([cli, str(window), repr(q)], input=text, capture_output=True,
                          text=True, check=True)
    return np.array([float(token) for token in done.stdout.split()])


def series(rng, kind, length):
    if kind == 0:
        values = rng.normal(size=length)
    elif kind == 1:
        values = rng.uniform(-1e5, 1e5, size=length)
    elif kind == 2:
        values = np.cumsum(rng.integers(-1, 2, size=length)).astype(float)  # a tick grid
    else:
        values = np.cumsum(rng.normal(size=length)) * 10.0 ** rng.integers(-30, 30)
    if rng.integers(0, 3) == 0:
        values[rng.random(length) < 0.02] = np.nan
    return values


def main(cli, cases):
    rng = np.random.default_rng(2024)
    compared = 0
    for case in range(cases):
        window = int(rng.integers(1, 300))
        values = series(rng, case % 4, window + int(rng.integers(0, 600)))
        q = float(rng.choice(QUANTILES)) if rng.integers(0, 4) else float(rng.random())
        got = run_cli(cli, values, window, q)
        want = pd.Series(values).rolling(window).quantile(q, interpolation="linear").to_numpy()
        same_nan = np.array_equal(np.isnan(got), np.isnan(want))
        held = ~np.isnan(want)
        if not same_nan or not np.array_equal(got[held].view(np.uint64),
                                              want[held].view(np.uint64)):
            sys.exit(f"MISMATCH: case {case}, window {window}, q {q!r}")
        compared += int(held.sum())
    print(f"pandas {pd.__version__}: {cases} series, {compared} windows, all identical")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
