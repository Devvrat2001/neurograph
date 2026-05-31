// Performance benchmarks for the predict.exe CLI (P1-07 AC2 + AC3).
//
// AC2: "Cold start (load bundle + predict 1 row) < 500 ms on a 100k-param
//       model."
// AC3: "Throughput > 5k rows/sec for titanic-sized model on single core."
//
// Approach mirrors tests/integration/test_predict_binary.cpp:
//   1. Build a real bundle in-process via the orchestrator.
//   2. Shell out to $<TARGET_FILE:neurograph_predict> for the timing run.
//   3. Measure wall time with std::chrono::steady_clock.
//
// P1-15 follow-up: the orchestrator now ships SON-trained bundles
// (multi-layer MLP via `training::TrainingEngine`), so these gates
// measure the real production path. Baselines on a dev laptop:
//   - cold start (1-row predict):  ~80 ms (under the 500 ms spec).
//   - throughput (10k rows):       ~30k rows/s (over the 5k spec).
// The gates below stay loose to absorb CI noise; tighten when a
// dedicated perf runner exists.
//
// Gating policy (matches test_perf_classification_metrics.cpp):
//   - NEUROGRAPH_PERF_PREDICT_COLD_GATE_MS overrides cold-start gate.
//   - NEUROGRAPH_PERF_PREDICT_THROUGHPUT_MIN overrides throughput gate.
//   - NEUROGRAPH_PERF_SKIP_GATE=1 records timings without failing.

#include "neurograph/orchestration/auto_pipeline.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef NEUROGRAPH_PREDICT_EXE
#  error "NEUROGRAPH_PREDICT_EXE must be defined by the test build"
#endif

namespace fs = std::filesystem;
using neurograph::orchestration::AutoOptions;
using neurograph::orchestration::run;

namespace {

long env_long_or(const char* name, long fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    try {
        return std::stol(v);
    } catch (...) {
        return fallback;
    }
}

double env_double_or(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    try {
        return std::stod(v);
    } catch (...) {
        return fallback;
    }
}

bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}

// Same cmd.exe quoting workaround as test_predict_binary.cpp — the
// outer pair stops cmd from stripping the inner quote pair when both
// the exe path and arg paths contain spaces.
std::string q(const fs::path& p) {
    return "\"" + p.string() + "\"";
}

int run_predict_silently(const std::string& args) {
    // Redirect stdout/stderr to NUL so per-row predict logging doesn't
    // pollute test output during the throughput run. Errors that would
    // normally print to stderr are still surfaced via the non-zero
    // exit code we assert on.
    std::string cmd = std::string("\"\"") + NEUROGRAPH_PREDICT_EXE
                    + "\" " + args + " >NUL 2>&1\"";
    return std::system(cmd.c_str());
}

fs::path write_train_csv(const fs::path& dir, int n_rows = 200) {
    const auto path = dir / "train.csv";
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3,label\n";
    for (int i = 0; i < n_rows; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << ','
          << ((x1 + x2 - x3 > 0) ? "A" : "B") << '\n';
    }
    return path;
}

fs::path write_test_csv(const fs::path& dir, const char* name, int n) {
    const auto path = dir / name;
    std::mt19937_64 rng(99);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3\n";
    for (int i = 0; i < n; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << '\n';
    }
    return path;
}

// PerfPredictBinaryTest builds the bundle once per fixture instance —
// gtest constructs a fresh fixture per TEST_F, so each test gets its
// own scratch dir but pays the bundle-build cost (a few seconds via
// the orchestrator) once. Acceptable: we run two perf tests, total
// added wall time on the order of single-digit seconds.
class PerfPredictBinaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()
                                ->current_test_info();
        tmp_ = fs::temp_directory_path() / "neurograph_perf_predict"
             / info.test_suite_name() / info.name();
        std::error_code ec;
        fs::remove_all(tmp_, ec);
        fs::create_directories(tmp_, ec);

        const auto train_csv = write_train_csv(tmp_);
        AutoOptions opts;
        opts.bundle_out_dir    = tmp_ / "bundle";
        opts.do_learning_curve = false;  // shave bundle-build time
        opts.seed              = 42;
        auto r = run(train_csv, "label", opts);
        ASSERT_TRUE(r.ok()) << r.error().message;
        bundle_ = opts.bundle_out_dir;
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_, ec);
    }

    fs::path tmp_;
    fs::path bundle_;
};

void report_ms_gate(const char* label, double elapsed_ms,
                    long default_gate_ms, const char* gate_env_name) {
    const long gate_ms = env_long_or(gate_env_name, default_gate_ms);
    const bool skip    = env_truthy("NEUROGRAPH_PERF_SKIP_GATE");

    std::cout << "[perf] " << label
              << " elapsed=" << elapsed_ms << "ms"
              << " gate=" << gate_ms << "ms"
              << (skip ? " (gate skipped)" : "") << '\n';
    ::testing::Test::RecordProperty(std::string(label) + "_ms",
                                    static_cast<int>(elapsed_ms));

    if (!skip) {
        EXPECT_LT(elapsed_ms, static_cast<double>(gate_ms))
            << label << " perf regression: " << elapsed_ms
            << " ms exceeded gate of " << gate_ms << " ms. "
            << "Set NEUROGRAPH_PERF_SKIP_GATE=1 to record without failing, "
            << "or " << gate_env_name << "=<ms> to override the gate.";
    }
}

void report_throughput_gate(const char* label, double rows_per_sec,
                            double default_min, const char* gate_env_name) {
    const double min_rps = env_double_or(gate_env_name, default_min);
    const bool   skip    = env_truthy("NEUROGRAPH_PERF_SKIP_GATE");

    std::cout << "[perf] " << label
              << " throughput=" << rows_per_sec << " rows/s"
              << " min=" << min_rps << " rows/s"
              << (skip ? " (gate skipped)" : "") << '\n';
    ::testing::Test::RecordProperty(std::string(label) + "_rps",
                                    static_cast<int>(rows_per_sec));

    if (!skip) {
        EXPECT_GT(rows_per_sec, min_rps)
            << label << " throughput regression: " << rows_per_sec
            << " rows/s below minimum of " << min_rps << " rows/s. "
            << "Set NEUROGRAPH_PERF_SKIP_GATE=1 to record without failing, "
            << "or " << gate_env_name << "=<rps> to override the gate.";
    }
}

double time_system_ms(const std::string& args) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const int  rc = run_predict_silently(args);
    const auto t1 = clock::now();
    EXPECT_EQ(rc, 0) << "predict.exe returned non-zero exit code " << rc;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

// ---------------------------------------------------------------------------
// P1-07 AC2 — Cold start.
//
// The "1 row" workload exercises the full critical path: process spawn,
// VCRuntime init, bundle load, pipeline.ngpipe deserialize, model.ngm
// deserialize, single forward pass, CSV write, exit.
//
// Spec target: < 500 ms on a 100k-param model.
// Default gate: 5000 ms (10× the spec) — Windows process spawn alone
// can be ~100-300 ms before our code even runs, and CI runners vary
// wildly. Tighten once we have a perf runner.
// ---------------------------------------------------------------------------

TEST_F(PerfPredictBinaryTest, ColdStartPredictsOneRowFast) {
    const auto in  = write_test_csv(tmp_, "one.csv", 1);
    const auto out = tmp_ / "one_pred.csv";
    const std::string args = "--model " + q(bundle_)
                           + " --in "    + q(in)
                           + " --out "   + q(out);

    // Warm filesystem caches with one untimed run.
    (void) run_predict_silently(args);

    const double elapsed = time_system_ms(args);
    ASSERT_TRUE(fs::exists(out));
    report_ms_gate("predict_cold_start_1row", elapsed,
                   /*default_gate_ms=*/5000,
                   "NEUROGRAPH_PERF_PREDICT_COLD_GATE_MS");
}

// ---------------------------------------------------------------------------
// P1-07 AC3 — Throughput.
//
// Runs the same exe over a 10000-row CSV. Reports rows/sec where the
// denominator is the *full* wall time including process spawn —
// pessimistic by design, since real users aren't going to amortize
// spawn cost across millions of rows from one shell invocation.
//
// Spec target: > 5000 rows/sec single-core.
// Default gate: 1000 rows/sec (5× looser than the spec) so we don't
// flake on cold CI VMs. SON-trained bundle baseline: ~30k rows/sec
// — 6× spec, 30× the gate.
// ---------------------------------------------------------------------------

TEST_F(PerfPredictBinaryTest, ThroughputAcrossTenKRows) {
    constexpr int n_rows = 10000;
    const auto in  = write_test_csv(tmp_, "many.csv", n_rows);
    const auto out = tmp_ / "many_pred.csv";
    const std::string args = "--model " + q(bundle_)
                           + " --in "    + q(in)
                           + " --out "   + q(out);

    (void) run_predict_silently(args);  // warm

    const double elapsed_ms = time_system_ms(args);
    ASSERT_TRUE(fs::exists(out));

    // Guard against zero-duration measurements on absurdly fast hosts;
    // a 1ms floor avoids divide-by-zero and reports a believable upper
    // bound rather than infinity.
    const double safe_ms      = (elapsed_ms < 1.0) ? 1.0 : elapsed_ms;
    const double rows_per_sec = (n_rows * 1000.0) / safe_ms;

    std::cout << "[perf] predict_throughput_10k elapsed=" << elapsed_ms
              << "ms (" << rows_per_sec << " rows/s incl. spawn overhead)\n";
    report_throughput_gate("predict_throughput_10k", rows_per_sec,
                           /*default_min=*/1000.0,
                           "NEUROGRAPH_PERF_PREDICT_THROUGHPUT_MIN");
}
