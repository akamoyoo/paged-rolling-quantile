# Raw benchmark results

The tables in the top-level README are computed from these files by `bench/summarize.py`. Each CSV starts
with comment lines that record the CPU, the pinned core, the compiler and its flags, and the clock overhead.
One row is one repetition of one cell.

| File | What it holds | Used for |
|---|---|---|
| `main_q50.csv` | every implementation, 10 window sizes, 4 input distributions, `q = 0.5`, 5 repetitions | "Mean nanoseconds per update" and the latency table |
| `q99.csv` | `q = 0.99`, 3 repetitions | the tail-quantile table |
| `page_capacity_sweep.csv` | `paged@c` for page capacities 32 to 2,048, 3 repetitions | "Choosing the page capacity" (the rows named `paged@<c>`) |
| `single_page_crossover.csv` | a single page against 128-element pages around `n = 1,024` | the single-page threshold |
| `memory.csv` | peak heap bytes per implementation and window, from `bench/memory_footprint` | "Memory" |
| `second_run_subset.csv` | 27 cells measured again later on a quieter machine, 3 repetitions | "A second run on a quieter machine" |

Columns of the timing files: `ns_per_update` is the mean over the throughput pass; `p50_ns`, `p99_ns`,
`p999_ns` and `max_ns` are order statistics of the latency pass; `checksum` is a hash of the first results
and must be equal across implementations within a cell.
