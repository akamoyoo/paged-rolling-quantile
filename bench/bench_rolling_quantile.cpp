// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: cost per update of RollingQuantile and of the simple implementations in reference/.
//
// For every (input distribution, window) pair each implementation is constructed, warmed up, and
// measured twice over fresh stretches of the same input stream:
//   - a throughput pass, timed as a whole, giving the mean cost per update;
//   - a latency pass that reads a clock after every update, giving the distribution.
// This is repeated --reps times, cycling through the implementations, so that drift in the state
// of the machine shows up as spread between repetitions instead of as a bias for one of them.
// One CSV row is written per measurement; bench/summarize.py turns the rows into tables.
// README.md describes the method and its limits. Run with --help for the options.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "reference/multiset.hpp"
#include "reference/nth_element.hpp"
#include "reference/sort_every_step.hpp"
#include "reference/sorted_array.hpp"
#include "reference/two_heaps.hpp"
#include "rolling_quantile/rolling_quantile.hpp"

#ifndef RQ_BENCH_FLAGS
#define RQ_BENCH_FLAGS "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

// ---- Clock for the latency pass -------------------------------------------------------------
//
// On x86-64 the time stamp counter is read directly (about 25 cycles); its rate is measured
// against steady_clock at start-up. Elsewhere steady_clock itself is used.

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
inline std::uint64_t ticks() { return __builtin_ia32_rdtsc(); }
const char* const tick_source = "rdtsc";
#else
inline std::uint64_t ticks() {
  return static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
}
const char* const tick_source = "steady_clock";
#endif

double seconds_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

double measure_ns_per_tick() {
  const Clock::time_point t0 = Clock::now();
  const std::uint64_t c0 = ticks();
  while (seconds_between(t0, Clock::now()) < 0.2) {
  }
  const std::uint64_t c1 = ticks();
  const Clock::time_point t1 = Clock::now();
  return seconds_between(t0, t1) * 1e9 / static_cast<double>(c1 - c0);
}

// ---- Input streams --------------------------------------------------------------------------
//
// std::mt19937_64 yields the same sequence on every standard library, and the conversions below
// are explicit, so a seed identifies the input exactly.

const char* const distribution_names[] = {"uniform", "walk", "ticks", "ascending"};

std::vector<double> make_input(const std::string& distribution, std::size_t length,
                               std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const auto unit = [&rng] { return static_cast<double>(rng() >> 11) * 0x1.0p-53; };
  std::vector<double> values(length);
  double level = 0.0;
  if (distribution == "uniform") {
    // Independent, uniform on [0, 1).
    for (double& v : values) v = unit();
  } else if (distribution == "walk") {
    // A continuous random walk: each value is close to the previous one.
    for (double& v : values) v = level += unit() - 0.5;
  } else if (distribution == "ticks") {
    // A random walk on a grid, like a price in ticks: a window holds few distinct values.
    for (double& v : values) v = level += static_cast<double>(rng() % 3) - 1.0;
  } else {
    // Ascending: every value is a new maximum and the oldest value is the minimum.
    for (std::size_t i = 0; i < length; ++i) values[i] = static_cast<double>(i);
  }
  return values;
}

// ---- One measurement ------------------------------------------------------------------------

struct Job {
  std::size_t window = 0;
  double q = 0.5;
  const std::vector<double>* input = nullptr;
  std::size_t checked = 0;      // outputs covered by the checksum
  std::size_t warmup = 0;       // updates before the first timed one
  std::size_t max_samples = 0;  // per pass
  std::size_t min_samples = 0;
  double pass_seconds = 0.0;    // target duration of one pass
  double ns_per_tick = 1.0;
};

struct Result {
  std::size_t samples = 0;
  double ns_per_update = 0.0;  // throughput pass
  double p50 = 0.0;            // latency pass, nanoseconds
  double p99 = 0.0;
  double p999 = 0.0;
  double max = 0.0;
  std::uint64_t checksum = 0;  // over the first Job::checked outputs
};

inline std::uint64_t mix(std::uint64_t sum, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return ((sum << 1) | (sum >> 63)) ^ bits;
}

// The smallest sample that at least the fraction p of all samples do not exceed.
double percentile(const std::vector<std::uint32_t>& sorted, double p, double ns_per_tick) {
  const std::size_t n = sorted.size();
  std::size_t rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(n)));
  rank = rank == 0 ? 0 : rank - 1;
  return static_cast<double>(sorted[rank < n ? rank : n - 1]) * ns_per_tick;
}

template <class Make>
Result measure(const Job& job, Make&& make) {
  const double* const x = job.input->data();
  auto impl = make(job.window, job.q);
  Result result;

  // Warm-up. The first `checked` outputs are folded into a checksum that must be the same for
  // every implementation. The duration of the tail of the warm-up sets the sample count.
  std::size_t i = 0;
  for (; i < job.checked; ++i) result.checksum = mix(result.checksum, impl.update(x[i]));
  std::uint64_t sink = 0;
  const std::size_t probe = std::min<std::size_t>(256, job.warmup - job.checked);
  for (; i < job.warmup - probe; ++i) sink = mix(sink, impl.update(x[i]));
  const Clock::time_point probe_start = Clock::now();
  for (; i < job.warmup; ++i) sink = mix(sink, impl.update(x[i]));
  const double probe_seconds = seconds_between(probe_start, Clock::now());
  const double estimate = probe_seconds / static_cast<double>(probe);
  double wanted = job.pass_seconds / (estimate > 0.0 ? estimate : 1e-9);
  wanted = std::min(wanted, static_cast<double>(job.max_samples));
  wanted = std::max(wanted, static_cast<double>(job.min_samples));
  const std::size_t samples = static_cast<std::size_t>(wanted);
  result.samples = samples;
  if (job.warmup + 2 * samples > job.input->size()) {
    std::fputs("input stream too short for this measurement\n", stderr);
    std::exit(1);
  }

  // Throughput pass.
  const Clock::time_point start = Clock::now();
  for (std::size_t j = 0; j < samples; ++j) sink = mix(sink, impl.update(x[i + j]));
  const Clock::time_point stop = Clock::now();
  result.ns_per_update = seconds_between(start, stop) * 1e9 / static_cast<double>(samples);
  i += samples;

  // Latency pass: one clock reading per update. A sample is the time from one reading to the
  // next, so it includes the reading itself and the loop around the update.
  std::vector<std::uint32_t> latency(samples);
  std::uint64_t previous = ticks();
  for (std::size_t j = 0; j < samples; ++j) {
    const double y = impl.update(x[i + j]);
    const std::uint64_t now = ticks();
    const std::uint64_t elapsed = now - previous;
    latency[j] = elapsed < 0xFFFFFFFFU ? static_cast<std::uint32_t>(elapsed) : 0xFFFFFFFFU;
    previous = now;
    sink = mix(sink, y);
  }
  std::sort(latency.begin(), latency.end());
  result.p50 = percentile(latency, 0.50, job.ns_per_tick);
  result.p99 = percentile(latency, 0.99, job.ns_per_tick);
  result.p999 = percentile(latency, 0.999, job.ns_per_tick);
  result.max = static_cast<double>(latency.back()) * job.ns_per_tick;

  // Keep the compiler from discarding the updates whose results only reach `sink`.
  volatile std::uint64_t keep = sink;
  (void)keep;
  return result;
}

// ---- Implementations ------------------------------------------------------------------------

struct Subject {
  std::string name;
  std::size_t page_capacity = 0;  // RollingQuantile only; 0 is its default
  std::size_t max_window = 0;     // 0: no limit
  bool evolves = false;           // internal layout depends on history: warm up for a full turnover
  std::function<Result(const Job&)> run;
};

// An update that does nothing, to measure what the latency pass itself costs.
struct Nothing {
  Nothing(std::size_t, double) {}
  double update(double x) { return x; }
};

template <class Impl>
std::function<Result(const Job&)> runner() {
  return [](const Job& job) {
    return measure(job, [](std::size_t window, double q) { return Impl(window, q); });
  };
}

// Recognised names: paged, paged@<page capacity>, multiset, two_heaps, sorted_array,
// nth_element, sort.
bool make_subject(const std::string& name, Subject& subject) {
  subject = Subject();
  subject.name = name;
  if (name.rfind("paged", 0) == 0) {
    std::size_t capacity = 0;
    if (name.size() > 5) {
      if (name[5] != '@') return false;
      capacity = static_cast<std::size_t>(std::strtoull(name.c_str() + 6, nullptr, 10));
      if (capacity < 2) return false;
    }
    subject.page_capacity = capacity;
    subject.evolves = true;
    subject.run = [capacity](const Job& job) {
      return measure(job, [capacity](std::size_t window, double q) {
        return rq::RollingQuantile<double>(window, q, capacity);
      });
    };
  } else if (name == "multiset") {
    subject.evolves = true;
    subject.run = runner<rq::reference::Multiset<double>>();
  } else if (name == "two_heaps") {
    subject.evolves = true;
    subject.run = runner<rq::reference::TwoHeaps<double>>();
  } else if (name == "sorted_array") {
    subject.max_window = 65536;
    subject.run = runner<rq::reference::SortedArray<double>>();
  } else if (name == "nth_element") {
    subject.max_window = 16384;
    subject.run = runner<rq::reference::NthElement<double>>();
  } else if (name == "sort") {
    subject.max_window = 1024;
    subject.run = runner<rq::reference::SortEveryStep<double>>();
  } else {
    return false;
  }
  return true;
}

// ---- Command line ---------------------------------------------------------------------------

struct Options {
  std::vector<std::size_t> windows = {16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
  std::vector<std::string> distributions = {"uniform", "walk", "ticks", "ascending"};
  std::vector<std::string> subjects = {"paged",        "multiset",    "two_heaps",
                                       "sorted_array", "nth_element", "sort"};
  double q = 0.5;
  int reps = 5;
  std::uint64_t seed = 1;
  double pass_seconds = 0.5;
  std::size_t min_samples = 20000;
  std::size_t max_samples = 1000000;
  int cpu = -1;
  std::string out;
};

void usage() {
  std::puts(
      "bench_rolling_quantile [options]\n"
      "  --windows a,b,...   window sizes (default 16,64,...,1048576)\n"
      "  --dists a,b,...     input distributions: uniform, walk, ticks, ascending (default all)\n"
      "  --impls a,b,...     paged, paged@<page capacity>, multiset, two_heaps, sorted_array,\n"
      "                      nth_element, sort (default all)\n"
      "  --q x               quantile (default 0.5)\n"
      "  --reps n            repetitions of every measurement (default 5)\n"
      "  --seed n            seed of the input streams (default 1)\n"
      "  --pass-seconds x    target duration of one timed pass (default 0.5)\n"
      "  --min-samples n     lower bound on updates per pass (default 20000)\n"
      "  --max-samples n     upper bound on updates per pass (default 1000000)\n"
      "  --cpu n             pin the process to this CPU (Linux)\n"
      "  --out file          write the CSV there instead of standard output");
}

std::vector<std::string> split(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ',')) {
    if (!part.empty()) parts.push_back(part);
  }
  return parts;
}

bool parse(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || i + 1 >= argc) return false;
    const std::string value = argv[++i];
    if (arg == "--windows") {
      options.windows.clear();
      for (const std::string& w : split(value)) {
        options.windows.push_back(static_cast<std::size_t>(std::strtoull(w.c_str(), nullptr, 10)));
      }
    } else if (arg == "--dists") {
      options.distributions = split(value);
    } else if (arg == "--impls") {
      options.subjects = split(value);
    } else if (arg == "--q") {
      options.q = std::strtod(value.c_str(), nullptr);
    } else if (arg == "--reps") {
      options.reps = std::atoi(value.c_str());
    } else if (arg == "--seed") {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (arg == "--pass-seconds") {
      options.pass_seconds = std::strtod(value.c_str(), nullptr);
    } else if (arg == "--min-samples") {
      options.min_samples = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (arg == "--max-samples") {
      options.max_samples = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (arg == "--cpu") {
      options.cpu = std::atoi(value.c_str());
    } else if (arg == "--out") {
      options.out = value;
    } else {
      return false;
    }
  }
  for (const std::string& d : options.distributions) {
    if (std::find(std::begin(distribution_names), std::end(distribution_names), d) ==
        std::end(distribution_names)) {
      return false;
    }
  }
  for (const std::size_t w : options.windows) {
    if (w == 0) return false;
  }
  return options.reps > 0 && options.q >= 0.0 && options.q <= 1.0 && options.min_samples > 0 &&
         options.max_samples >= options.min_samples && options.pass_seconds > 0.0;
}

bool pin_to_cpu(int cpu) {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<unsigned>(cpu), &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
  (void)cpu;
  return false;
#endif
}

std::string cpu_model() {
  std::ifstream info("/proc/cpuinfo");
  std::string line;
  while (std::getline(info, line)) {
    if (line.rfind("model name", 0) == 0) {
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) return line.substr(colon + 2);
    }
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse(argc, argv, options)) {
    usage();
    return 2;
  }
  std::vector<Subject> subjects(options.subjects.size());
  for (std::size_t s = 0; s < subjects.size(); ++s) {
    if (!make_subject(options.subjects[s], subjects[s])) {
      std::fprintf(stderr, "unknown implementation: %s\n", options.subjects[s].c_str());
      return 2;
    }
  }
  if (options.cpu >= 0 && !pin_to_cpu(options.cpu)) {
    std::fprintf(stderr, "cannot pin to cpu %d\n", options.cpu);
    return 2;
  }
  std::FILE* out = stdout;
  if (!options.out.empty()) {
    out = std::fopen(options.out.c_str(), "w");
    if (out == nullptr) {
      std::fprintf(stderr, "cannot write %s\n", options.out.c_str());
      return 2;
    }
  }

  const double ns_per_tick = measure_ns_per_tick();

  // What a latency sample costs when the update does nothing.
  double clock_overhead = 0.0;
  {
    const std::vector<double> input = make_input("uniform", 400000, options.seed);
    Job job;
    job.window = 1;
    job.input = &input;
    job.checked = 1;
    job.warmup = 1000;
    job.min_samples = job.max_samples = 100000;
    job.pass_seconds = 1.0;
    job.ns_per_tick = ns_per_tick;
    clock_overhead = measure(job, [](std::size_t w, double q) { return Nothing(w, q); }).p50;
  }

  std::fprintf(out, "# cpu: %s\n", cpu_model().c_str());
  std::fprintf(out, "# pinned to cpu: %d\n", options.cpu);
  std::fprintf(out, "# compiler: %s\n", __VERSION__);
  std::fprintf(out, "# flags: %s\n", RQ_BENCH_FLAGS);
  std::fprintf(out, "# latency clock: %s, %.4f ns per tick\n", tick_source, ns_per_tick);
  std::fprintf(out, "# latency sample with an empty update, median: %.1f ns\n", clock_overhead);
  std::fprintf(out, "# seed: %llu  pass seconds: %g  samples per pass: %zu to %zu\n",
               static_cast<unsigned long long>(options.seed), options.pass_seconds,
               options.min_samples, options.max_samples);
  std::fprintf(out,
               "dist,window,q,impl,page_capacity,rep,warmup,samples,ns_per_update,p50_ns,p99_ns,"
               "p999_ns,max_ns,checksum\n");

  for (const std::string& distribution : options.distributions) {
    for (const std::size_t window : options.windows) {
      // Warm-up: the window fills and takes 512 further values. Implementations whose layout
      // depends on history then see the whole window turn over once more; the others are in
      // their steady state as soon as they are full.
      const std::size_t checked = window + 256;
      const std::size_t short_warmup = window + 512;
      const std::size_t long_warmup = 2 * window + 512;
      // Implementations with amortised costs (two_heaps rebuilds about once per window turnover,
      // paged reorganises its pages) are timed over at least two full turnovers per pass, so that
      // a pass cannot miss the expensive steps.
      const std::size_t turnovers = 2 * window;
      const std::size_t length = long_warmup + 2 * std::max(options.max_samples, turnovers);
      const std::vector<double> input = make_input(distribution, length, options.seed);
      bool have_checksum = false;
      std::uint64_t checksum = 0;

      for (int rep = 0; rep < options.reps; ++rep) {
        for (std::size_t turn = 0; turn < subjects.size(); ++turn) {
          const std::size_t index = (turn + static_cast<std::size_t>(rep)) % subjects.size();
          const Subject& subject = subjects[index];
          if (subject.max_window != 0 && window > subject.max_window) continue;

          Job job;
          job.window = window;
          job.q = options.q;
          job.input = &input;
          job.checked = checked;
          job.warmup = subject.evolves ? long_warmup : short_warmup;
          job.min_samples = options.min_samples;
          job.max_samples = options.max_samples;
          if (subject.evolves) {
            job.min_samples = std::max(job.min_samples, turnovers);
            job.max_samples = std::max(job.max_samples, turnovers);
          }
          job.pass_seconds = options.pass_seconds;
          job.ns_per_tick = ns_per_tick;
          const Result r = subject.run(job);

          if (have_checksum && r.checksum != checksum) {
            std::fprintf(stderr, "%s disagrees with the other implementations (%s, window %zu)\n",
                         subject.name.c_str(), distribution.c_str(), window);
            return 1;
          }
          have_checksum = true;
          checksum = r.checksum;

          std::size_t page_capacity = subject.page_capacity;
          if (subject.name.rfind("paged", 0) == 0 && page_capacity == 0) {
            page_capacity = rq::RollingQuantile<double>::default_page_capacity(window);
          }
          std::fprintf(out, "%s,%zu,%g,%s,%zu,%d,%zu,%zu,%.2f,%.1f,%.1f,%.1f,%.1f,%016llx\n",
                       distribution.c_str(), window, options.q, subject.name.c_str(),
                       page_capacity, rep, job.warmup, r.samples, r.ns_per_update, r.p50, r.p99,
                       r.p999, r.max, static_cast<unsigned long long>(r.checksum));
          std::fflush(out);
        }
      }
    }
  }
  if (out != stdout) std::fclose(out);
  return 0;
}
