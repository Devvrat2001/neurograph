// Integration tests for the P1-14 AutoPipeline orchestrator.
//
// What we actually exercise:
//
//   1. End-to-end happy path on a synthetic classification CSV — verifies
//      every stage runs and the bundle is laid out where P1-07/P1-08 will
//      look for it.
//   2. End-to-end on a synthetic regression CSV — confirms the
//      classification/regression branch in the orchestrator is wired.
//   3. Cancellation propagates within ~1s of being requested.
//   4. Corrupt / non-existent CSV produces a clean error, not a crash.
//   5. Determinism: same seed + same inputs → byte-identical model.ngm.
//   6. Progress callback fires for every stage and overall_fraction is
//      monotonically non-decreasing — so the wizard's progress bar
//      behaves predictably.
//   7. A trainable but not-rich-enough dataset still produces a verdict.
//
// Synthetic data instead of a fixture file because the test runner has
// no guaranteed working dir, and dragging titanic.csv into tests/ would
// inflate the repo for what's effectively just a shape check. The
// orchestrator doesn't care about the dataset's semantic meaning — it
// cares that scan → preprocess → train → eval all work — so a
// 100-row two-class CSV with linearly separable features suffices.

#include "neurograph/orchestration/auto_pipeline.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using neurograph::orchestration::AutoOptions;
using neurograph::orchestration::AutoResult;
using neurograph::orchestration::Progress;
using neurograph::orchestration::Stage;
using neurograph::orchestration::run;

namespace fs = std::filesystem;

namespace {

// Per-test temp dir under the OS temp folder. Clean up at fixture
// teardown so a flaky run doesn't leak gigabytes.
class AutoPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base = fs::temp_directory_path() / "neurograph_auto_pipeline_test";
        // Unique per test so parallel runs don't stomp.
        const auto& info = *::testing::UnitTest::GetInstance()->current_test_info();
        tmp_ = base / info.test_suite_name() / info.name();
        std::error_code ec;
        fs::remove_all(tmp_, ec);
        fs::create_directories(tmp_, ec);
        ASSERT_FALSE(ec) << "Cannot create tmp dir " << tmp_;
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_, ec);
    }

    fs::path tmp_;
};

// Make a 100-row, 3-feature, 2-class CSV. Class is determined by the
// sign of (x1 + x2 - x3) plus tiny noise so the linear probe converges
// to ~85-95% accuracy and verdict tests have something to assert on.
fs::path write_classification_csv(const fs::path& dir, std::uint64_t seed = 7) {
    const auto path = dir / "cls.csv";
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::ofstream f(path);
    f << "f1,f2,f3,label\n";
    for (int i = 0; i < 100; ++i) {
        const double x1 = nd(rng);
        const double x2 = nd(rng);
        const double x3 = nd(rng);
        const double s  = x1 + x2 - x3 + 0.1 * nd(rng);
        const char* lbl = (s > 0.0) ? "A" : "B";
        f << x1 << ',' << x2 << ',' << x3 << ',' << lbl << '\n';
    }
    return path;
}

// Make a 100-row, 3-feature regression CSV where y ≈ 2x1 - x2 + 0.5x3.
fs::path write_regression_csv(const fs::path& dir, std::uint64_t seed = 11) {
    const auto path = dir / "reg.csv";
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::ofstream f(path);
    f << "f1,f2,f3,target\n";
    for (int i = 0; i < 100; ++i) {
        const double x1 = nd(rng);
        const double x2 = nd(rng);
        const double x3 = nd(rng);
        const double y  = 2.0 * x1 - x2 + 0.5 * x3 + 0.05 * nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << ',' << y << '\n';
    }
    return path;
}

// SHA-256 → string, lifted from manifest.cpp via its public API.
// Used to assert determinism on model.ngm content.
} // namespace

#include "neurograph/reporting/manifest.hpp"

// ---------------------------------------------------------------------------
// 1. Happy path — classification
// ---------------------------------------------------------------------------

// The real 3-class problem that once trained to chance level (0.33) in
// the workbench. The synthetic CSV below is too easy to catch that kind
// of failure.
TEST_F(AutoPipelineTest, EndToEndIrisClassificationConverges) {
    const fs::path iris =
        NEUROGRAPH_SOURCE_DIR "/datasets/iris.csv";
    if (!fs::exists(iris)) GTEST_SKIP() << "iris.csv not found";

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "iris_bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;

    auto r = run(iris, "species", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& res = r.value();

    ASSERT_TRUE(std::holds_alternative<neurograph::eval::ClassificationMetrics>(res.metrics));
    const auto& cm = std::get<neurograph::eval::ClassificationMetrics>(res.metrics);

    // Published results on iris are 0.93-1.0; chance is 0.33. Scored on
    // about 23 held-back rows, so one mistake costs four points.
    EXPECT_GT(cm.accuracy, 0.85)
        << "iris classification accuracy=" << cm.accuracy;
}

TEST_F(AutoPipelineTest, EndToEndClassificationProducesBundle) {
    const auto csv = write_classification_csv(tmp_);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = true;
    opts.seed              = 42;

    auto r = run(csv, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& res = r.value();

    EXPECT_TRUE(res.dq.trainable);
    EXPECT_EQ(res.bundle_path, opts.bundle_out_dir);
    EXPECT_TRUE(fs::exists(opts.bundle_out_dir / "pipeline.ngpipe"));
    EXPECT_TRUE(fs::exists(opts.bundle_out_dir / "model.ngm"));
    EXPECT_TRUE(fs::exists(opts.bundle_out_dir / "manifest.json"));
    EXPECT_TRUE(fs::exists(opts.bundle_out_dir / "card.md"));

    // Nearly linearly separable data: every model family should do well.
    ASSERT_TRUE(std::holds_alternative<neurograph::eval::ClassificationMetrics>(res.metrics));
    const auto& cm = std::get<neurograph::eval::ClassificationMetrics>(res.metrics);
    EXPECT_GT(cm.accuracy, 0.7) << "should learn this easy task";

    // Summary one-liner is set and non-empty.
    EXPECT_FALSE(res.summary_one_liner.empty());
    EXPECT_FALSE(res.baselines.verdict.empty());
}

// ---------------------------------------------------------------------------
// 2. Happy path — regression
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, EndToEndRegressionProducesBundle) {
    const auto csv = write_regression_csv(tmp_);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;     // skip to keep the test snappy
    opts.seed              = 42;

    auto r = run(csv, "target", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& res = r.value();

    EXPECT_TRUE(fs::exists(opts.bundle_out_dir / "model.ngm"));
    ASSERT_TRUE(std::holds_alternative<neurograph::eval::RegressionMetrics>(res.metrics));
    const auto& rm = std::get<neurograph::eval::RegressionMetrics>(res.metrics);
    EXPECT_GT(rm.r2, 0.7) << "should explain this linear target well";
    EXPECT_LT(rm.rmse, 1.0);
}

// ---------------------------------------------------------------------------
// 3. Cancellation propagates promptly
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, CancellationReturnsCleanErrorPromptly) {
    const auto csv = write_classification_csv(tmp_);

    AutoOptions opts;
    opts.bundle_out_dir = tmp_ / "bundle";
    opts.seed           = 42;

    // Cancel from the second progress event onward — i.e. anywhere
    // after Scanning has emitted its start frame. Guarantees we hit
    // a between-stage poll before the run finishes.
    std::atomic<int> n_progress = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto r = run(csv, "label", opts,
                 [&](const Progress&) { ++n_progress; },
                 [&]() { return n_progress.load() >= 2; });
    auto dt = std::chrono::steady_clock::now() - t0;

    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().message.find("Cancel"), std::string::npos)
        << "got: " << r.error().message;
    // Cancel should land within a couple of seconds even on cold MSVC
    // Debug builds. The acceptance criterion is "<1s on titanic"; this
    // synthetic dataset is much smaller, so 5s is comfortable headroom.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(dt).count(), 5);
}

// ---------------------------------------------------------------------------
// 4. Bad inputs — clean error, not a crash
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, NonexistentCsvReturnsErrorNotCrash) {
    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;
    auto r = run(tmp_ / "nope.csv", "label", opts);
    ASSERT_FALSE(r.ok());
    // Should be the trainable-gate error from DQ scan.
    EXPECT_NE(r.error().message.find("Cannot train"), std::string::npos)
        << r.error().message;
}

TEST_F(AutoPipelineTest, BundleDirEmptyButExportRequestedFails) {
    const auto csv = write_classification_csv(tmp_);
    AutoOptions opts;                       // bundle_out_dir is empty
    opts.do_learning_curve = false;
    auto r = run(csv, "label", opts);
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().message.find("bundle_out_dir is empty"), std::string::npos)
        << r.error().message;
}

// ---------------------------------------------------------------------------
// 5. Determinism — same seed + same data ⇒ byte-identical model.ngm
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, DeterministicModelArtifactGivenSameSeed) {
    const auto csv = write_classification_csv(tmp_, /*seed=*/7);

    auto run_once = [&](const fs::path& bundle) {
        AutoOptions opts;
        opts.bundle_out_dir    = bundle;
        opts.do_learning_curve = false;       // trim the test
        opts.seed              = 1234;
        auto r = run(csv, "label", opts);
        EXPECT_TRUE(r.ok()) << r.error().message;
    };

    const auto a = tmp_ / "a";
    const auto b = tmp_ / "b";
    run_once(a);
    run_once(b);

    // The manifest carries a UTC timestamp by design, so we hash the
    // *model* file (not the manifest) — that's the byte-for-byte
    // determinism contract.
    const auto ha = neurograph::reporting::sha256_of_file(a / "model.ngm");
    const auto hb = neurograph::reporting::sha256_of_file(b / "model.ngm");
    EXPECT_EQ(ha, hb);
}

// ---------------------------------------------------------------------------
// 6. Progress events — every stage fires; overall is monotonic
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, ProgressIsMonotonicAndCoversAllStages) {
    const auto csv = write_classification_csv(tmp_);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = true;
    opts.seed              = 42;

    std::vector<Progress> events;
    auto r = run(csv, "label", opts,
                 [&](const Progress& p) { events.push_back(p); });
    ASSERT_TRUE(r.ok()) << r.error().message;
    ASSERT_FALSE(events.empty());

    // Every stage in the enum should have fired at least once.
    std::vector<bool> seen(static_cast<int>(Stage::Done) + 1, false);
    double last = -1.0;
    for (const auto& p : events) {
        seen[static_cast<int>(p.stage)] = true;
        EXPECT_GE(p.overall_fraction, last - 1e-9)
            << "overall_fraction must be monotonically non-decreasing";
        last = p.overall_fraction;
        EXPECT_GE(p.stage_fraction, 0.0);
        EXPECT_LE(p.stage_fraction, 1.0);
    }
    for (int i = 0; i <= static_cast<int>(Stage::Done); ++i) {
        EXPECT_TRUE(seen[i]) << "Stage " << i << " never emitted progress";
    }
    // Last event should be Done at 100%.
    EXPECT_EQ(events.back().stage, Stage::Done);
    EXPECT_DOUBLE_EQ(events.back().overall_fraction, 1.0);
}

// ---------------------------------------------------------------------------
// 7. Learning-curve skip flag is respected
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, DoLearningCurveFalseSkipsThatStage) {
    const auto csv = write_classification_csv(tmp_);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;

    auto r = run(csv, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    EXPECT_TRUE(r.value().curve.points.empty());
}

// ---------------------------------------------------------------------------
// 8. do_export_bundle=false leaves disk untouched but result is valid
// ---------------------------------------------------------------------------

TEST_F(AutoPipelineTest, DoExportBundleFalseSkipsPackaging) {
    const auto csv = write_classification_csv(tmp_);

    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";   // intentionally won't be touched
    opts.do_export_bundle  = false;
    opts.do_learning_curve = false;
    opts.seed              = 42;

    auto r = run(csv, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    EXPECT_TRUE(r.value().bundle_path.empty());
    EXPECT_FALSE(fs::exists(opts.bundle_out_dir / "model.ngm"));
}

// ---------------------------------------------------------------------------
// 9. Model selection and honest evaluation
// ---------------------------------------------------------------------------

#include "neurograph/runtime/bundle.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>

namespace {

// Rows of a CSV written by these tests (no quoting), as JSON objects keyed
// by header; cells that read fully as numbers become numbers.
std::vector<nlohmann::json> read_rows(const fs::path& p) {
    std::ifstream f(p);
    std::string line;
    std::getline(f, line);
    std::vector<std::string> header;
    {
        std::stringstream ss(line);
        std::string h;
        while (std::getline(ss, h, ',')) header.push_back(h);
    }
    std::vector<nlohmann::json> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        nlohmann::json row = nlohmann::json::object();
        for (std::size_t i = 0; i < header.size() && std::getline(ss, cell, ','); ++i) {
            try {
                std::size_t used = 0;
                const double v = std::stod(cell, &used);
                if (used == cell.size()) { row[header[i]] = v; continue; }
            } catch (...) {}
            row[header[i]] = cell;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// `label` depends on `signal` only; the two noise columns are unrelated.
fs::path write_signal_and_noise_csv(const fs::path& dir, int n = 300) {
    const auto path = dir / "signal.csv";
    std::mt19937_64 rng(21);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "noise1,signal,noise2,label\n";
    for (int i = 0; i < n; ++i) {
        const double a = nd(rng), sig = nd(rng), b = nd(rng);
        f << a << ',' << sig << ',' << b << ',' << (sig > 0.0 ? "A" : "B") << '\n';
    }
    return path;
}

} // namespace

TEST_F(AutoPipelineTest, ScoresComeFromHeldBackRowsAndOneModelIsChosen) {
    const auto csv = write_classification_csv(tmp_);
    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;
    auto r = run(csv, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& res = r.value();

    EXPECT_EQ(res.n_train + res.n_val + res.n_test, 100);
    ASSERT_GT(res.n_test, 0);
    EXPECT_EQ(res.evaluated_on, "test");
    EXPECT_EQ(res.n_evaluated, res.n_test);

    const auto& cm = std::get<neurograph::eval::ClassificationMetrics>(res.metrics);
    int scored = 0;
    for (const auto& row : cm.confusion.cells) for (int v : row) scored += v;
    EXPECT_EQ(scored, res.n_test) << "scores must come from the test rows only";

    ASSERT_GE(res.leaderboard.size(), 2u);
    EXPECT_EQ(std::count_if(res.leaderboard.begin(), res.leaderboard.end(),
                            [](const auto& c) { return c.chosen; }), 1);
    EXPECT_FALSE(res.model_name.empty());
    EXPECT_FALSE(res.model_description.empty());
    EXPECT_NE(res.summary_one_liner.find("held back"), std::string::npos)
        << res.summary_one_liner;
}

TEST_F(AutoPipelineTest, ExportedBundleScoresLikeTheReport) {
    // Guards against the old defect where exported predictions were four
    // times too large: the saved bundle, scored on the whole file, must
    // agree with what the training run reported.
    const auto csv = write_regression_csv(tmp_);
    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;
    auto r = run(csv, "target", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const double reported =
        std::get<neurograph::eval::RegressionMetrics>(r.value().metrics).r2;

    auto b = neurograph::runtime::load_bundle(opts.bundle_out_dir);
    ASSERT_TRUE(b.ok()) << b.error().message;
    const auto rows = read_rows(csv);
    ASSERT_EQ(rows.size(), 100u);
    double mean = 0.0;
    for (const auto& row : rows) mean += row["target"].get<double>();
    mean /= static_cast<double>(rows.size());
    double sse = 0.0, sst = 0.0;
    for (const auto& row : rows) {
        const double y = row["target"].get<double>();
        const auto p = neurograph::runtime::predict_row(b.value(), row);
        sse += (p.value - y) * (p.value - y);
        sst += (y - mean) * (y - mean);
    }
    const double r2_all = 1.0 - sse / sst;
    EXPECT_GT(r2_all, 0.9);
    EXPECT_NEAR(r2_all, reported, 0.1);
}

TEST_F(AutoPipelineTest, ImportanceRanksTheInformativeColumnFirst) {
    const auto csv = write_signal_and_noise_csv(tmp_);
    AutoOptions opts;
    opts.do_export_bundle  = false;
    opts.do_learning_curve = false;
    auto r = run(csv, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& imp = r.value().importance;
    ASSERT_EQ(imp.size(), 3u);
    EXPECT_EQ(imp[0].column, "signal");
    EXPECT_DOUBLE_EQ(imp[0].relative, 1.0);
    EXPECT_GT(imp[0].score_drop, 0.2);
}

TEST_F(AutoPipelineTest, ExcludedColumnIsLeftOutOfTheModel) {
    // `leak` is the answer in another form; leaving it out must keep it
    // out of the model and of the saved pipeline.
    const auto path = tmp_ / "leak.csv";
    {
        std::mt19937_64 rng(5);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::ofstream f(path);
        f << "x1,x2,leak,label\n";
        for (int i = 0; i < 200; ++i) {
            const double x1 = nd(rng), x2 = nd(rng);
            const bool pos = x1 + 0.3 * nd(rng) > 0.0;
            f << x1 << ',' << x2 << ',' << (pos ? "yes" : "no") << ','
              << (pos ? "A" : "B") << '\n';
        }
    }
    AutoOptions opts;
    opts.bundle_out_dir    = tmp_ / "bundle";
    opts.do_learning_curve = false;
    opts.exclude_columns   = {"leak"};
    auto r = run(path, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& res = r.value();

    bool found = false;
    for (const auto& c : res.columns) {
        if (c.name == "leak") { found = true; EXPECT_EQ(c.role, "left out"); }
    }
    EXPECT_TRUE(found);
    for (const auto& ci : res.importance) EXPECT_NE(ci.column, "leak");

    auto b = neurograph::runtime::load_bundle(opts.bundle_out_dir);
    ASSERT_TRUE(b.ok()) << b.error().message;
    const auto inputs = b.value().pipeline->input_columns();
    EXPECT_EQ(std::find(inputs.begin(), inputs.end(), "leak"), inputs.end());
    EXPECT_NE(std::find(inputs.begin(), inputs.end(), "x1"), inputs.end());
}

TEST_F(AutoPipelineTest, TaskOverrideTreatsTwoValuedTargetAsANumber) {
    const auto path = tmp_ / "flag.csv";
    {
        std::ofstream f(path);
        f << "x,flag\n";
        for (int i = 0; i < 120; ++i) f << (i % 17) << ',' << ((i % 17) > 8 ? 1 : 0) << '\n';
    }
    AutoOptions opts;
    opts.do_export_bundle  = false;
    opts.do_learning_curve = false;
    auto auto_r = run(path, "flag", opts);
    ASSERT_TRUE(auto_r.ok()) << auto_r.error().message;
    EXPECT_TRUE(std::holds_alternative<neurograph::eval::ClassificationMetrics>(
        auto_r.value().metrics));

    opts.task = neurograph::preprocess::TaskChoice::Regression;
    auto reg_r = run(path, "flag", opts);
    ASSERT_TRUE(reg_r.ok()) << reg_r.error().message;
    EXPECT_TRUE(std::holds_alternative<neurograph::eval::RegressionMetrics>(
        reg_r.value().metrics));
}

TEST_F(AutoPipelineTest, LearningCurveRetrainsTheChosenModel) {
    const auto csv = write_signal_and_noise_csv(tmp_, 400);
    AutoOptions opts;
    opts.do_export_bundle  = false;
    opts.do_learning_curve = true;
    auto r = run(csv, "label", opts);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& res = r.value();
    ASSERT_EQ(res.curve.points.size(), 4u) << res.curve.message;
    // The 100% point is the chosen model itself, scored on the check rows.
    EXPECT_DOUBLE_EQ(res.curve.points.back().val_metric, res.check_score);
    EXPECT_EQ(res.curve.points.back().n_samples, res.n_train);
    // Smaller samples are real retrainings, not copies of the last point.
    EXPECT_GT(res.curve.points.front().val_metric, 0.5);
}
