#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn the CSV written by bench_rolling_quantile into Markdown tables.

    summarize.py results.csv [more.csv ...]

Every (distribution, window, implementation) cell is measured several times. A table entry is
the median over those repetitions; the spread of a cell is (largest - smallest) / median.
Only the standard library is used.
"""

import csv
import statistics
import sys
from collections import defaultdict

METRICS = ["ns_per_update", "p50_ns", "p99_ns", "p999_ns", "max_ns"]
HEADINGS = {"ns_per_update": "mean", "p50_ns": "p50", "p99_ns": "p99", "p999_ns": "p99.9",
            "max_ns": "max"}


def read(paths):
    notes, cells, order = [], defaultdict(lambda: defaultdict(list)), {"dist": [], "impl": []}
    for path in paths:
        with open(path, newline="") as handle:
            lines = handle.readlines()
        notes.extend(line[1:].strip() for line in lines if line.startswith("#"))
        for row in csv.DictReader(line for line in lines if not line.startswith("#")):
            key = (row["dist"], int(row["window"]), row["impl"])
            for metric in METRICS:
                cells[key][metric].append(float(row[metric]))
            cells[key]["samples"].append(int(row["samples"]))
            cells[key]["page_capacity"].append(int(row["page_capacity"]))
            for field in ("dist", "impl"):
                if row[field] not in order[field]:
                    order[field].append(row[field])
    return notes, cells, order


def number(value):
    if value < 100:
        return f"{value:.1f}"
    return f"{value:,.0f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def main(paths):
    notes, cells, order = read(paths)
    windows = sorted({key[1] for key in cells})
    for note in dict.fromkeys(notes):
        print(f"- {note}")
    print()

    for dist in order["dist"]:
        impls = [i for i in order["impl"] if any((dist, w, i) in cells for w in windows)]
        print(f"### {dist}: mean cost per update, ns\n")
        rows = []
        for window in windows:
            row = [f"{window:,}"]
            for impl in impls:
                cell = cells.get((dist, window, impl))
                row.append(number(statistics.median(cell["ns_per_update"])) if cell else "")
            rows.append(row)
        table(["window"] + impls, rows)

        print(f"### {dist}: latency of one update, ns\n")
        rows = []
        for window in windows:
            for impl in impls:
                cell = cells.get((dist, window, impl))
                if not cell:
                    continue
                rows.append([f"{window:,}", impl] +
                            [number(statistics.median(cell[m])) for m in METRICS] +
                            [f"{min(cell['samples']):,}"])
        table(["window", "implementation"] + [HEADINGS[m] for m in METRICS] +
              ["samples per pass"], rows)

    print("### Spread between repetitions\n")
    rows = []
    for impl in order["impl"]:
        row = [impl]
        for metric in METRICS:
            spreads = []
            for key, cell in cells.items():
                if key[2] == impl and len(cell[metric]) > 1:
                    middle = statistics.median(cell[metric])
                    spreads.append((max(cell[metric]) - min(cell[metric])) / middle)
            if spreads:
                row.append(f"{100 * statistics.median(spreads):.1f}% / {100 * max(spreads):.0f}%")
            else:
                row.append("")
        rows.append(row)
    table(["implementation"] + [HEADINGS[m] + " (median / worst cell)" for m in METRICS], rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
