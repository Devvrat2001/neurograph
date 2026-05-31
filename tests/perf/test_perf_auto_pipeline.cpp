// Performance benchmark for the full AutoPipeline::run flow (DoD perf gate).
//
// Spec target from docs/phase1-tabular-mvp.md DoD:
//   "AutoPipeline::run on titanic < 60 s default budget."
//
// We approximate "titanic" with a synthesized 2-class CSV of similar
// shape (~1000 rows × 5 features). The orchestrator runs all 9 stages
// end-to-end including the learning-curve sub-runs, so this test
// exercises the full happy path the wizard's "Train" button drives.
//
// P1-15: orchestrator now drives the SON `TrainingEngine` (Adam +
// auto-hidden ReLU layer + LINEAR cls outputs + bounded growth-only
// self-org). The full end-to-end run on a 1000-row × 5-feature
// synthetic CSV with learning curve enabled lands at ~5 seconds —
// well under the 60 s spec but two orders of magnitude over the
// pre-SON linear-probe baseline (37 ms). The budget gate is the
// real ship-blocker now: tighten only after CI hardware stabilizes.
//
// Gating policy:
//   - Default gate: 120 000 ms (2× the 60 s spec) — generous enough
//     to absorb cold caches and CI noise; will be tightened once we
//     have a dedicated perf runner.
//   - Override via NEUROGRAPH_PERF_PIPELINE_GATE_MS.
//   - NEUROGRAPH_PERF_SKIP_GATE=1 records the timing without failing.

#include "neurograph/orchestration/auto_pipeline.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

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

bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}

// A 5-feature 2-class synthetic dataset of `n_rows` rows. Linear
// separability is preserved so the SON network converges quickly —
// we're measuring orchestration overhead + SON training cost, not
// trainer stress. (For trainer-stress benchmarking use a harder
// fixture in a follow-up perf test.)
fs::path write_titanic_like_csv(const fs::path& dir, int n_rows = 1000) {
    const auto path = dir / "synth.csv";
    std::mt19937_64 rng(2026);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3,f4,f5,survived\n";
    for (int i = 0; i < n_rows; ++i) {
        const double x1 = nd(rng), x2 = nd(rng);
        const double x3 = nd(rng), x4 = nd(rng), x5 = nd(rng);
        // Decision boundary mixes the first three features so all three
        // matter for accuracy; x4/x5 are noise the preprocessor must
        // tolerate without throwing.
        const bool   pos = (1.5 * x1 - x2 + 0.5 * x3) > 0.0;
        f << x1 << ',' << x2 << ',' << x3 << ',' << x4 << ',' << x5
          << ',' << (pos ? "yes" : "no") << '\n';
    }
    return path;
}

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

class PerfAutoPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()
                                ->current_test_info();
        tmp_ = fs::temp_directory_path() / "neurograph_perf_pipeline"
             / info.test_suite_name() / info.name();
        std::error_code ec;
        fs::remove_all(tmp_, ec);
        fs::create_directories(tmp_, ec);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_, ec);
    }
    fs::path tmp_;
};

} // namespace

// ---------------------------------------------------------------------------
// Full pipeline budget — single-shot end-to-end timing.
//
// Includes scan, preprocess, auto-config, train, evaluate, baseline,
// learning curve, and packaging — i.e. everything the wizard's Train
// button drives. Bundle write is enabled since packaging cost is part
// of the spec budget.
// ---------------------------------------------------------------------------

TEST_F(PerfAutoPipelineTest, FullRunFitsInDefaultBudget) {
    const auto csv = write_titanic_like_csv(tmp_, /*n_rows=*/1000);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = true;   // spec budget includes the curve
    opts.do_export_bundle  = true;
    opts.seed              = 42;

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto r = run(csv, "survived", opts);
    const auto t1 = clock::now();

    ASSERT_TRUE(r.ok()) << r.error().message;
    ASSERT_FALSE(r.value().bundle_path.empty());
    ASSERT_TRUE(fs::exists(r.value().bundle_path));

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    report_ms_gate("auto_pipeline_full_run", elapsed_ms,
                   /*default_gate_ms=*/120000,
                   "NEUROGRAPH_PERF_PIPELINE_GATE_MS");
}

// ---------------------------------------------------------------------------
// Learning-curve-off variant — useful as a fast smoke for orchestrator
// startup overhead. When SON lands and the budget gets tighter, this
// measurement is the floor (curve adds ~3-4× the train cost).
// ---------------------------------------------------------------------------

TEST_F(PerfAutoPipelineTest, FastRunWithoutLearningCurve) {
    const auto csv = write_titanic_like_csv(tmp_, /*n_rows=*/1000);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;
    opts.do_export_bundle  = true;
    opts.seed              = 42;

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto r = run(csv, "survived", opts);
    const auto t1 = clock::now();

    ASSERT_TRUE(r.ok()) << r.error().message;

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    report_ms_gate("auto_pipeline_no_curve", elapsed_ms,
                   /*default_gate_ms=*/30000,
                   "NEUROGRAPH_PERF_PIPELINE_NOCURVE_GATE_MS");
}
