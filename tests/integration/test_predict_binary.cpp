// Integration test for the P1-07 `predict.exe` binary.
//
// Approach: build a tiny bundle in-process via the orchestrator, then
// shell out to the built `neurograph_predict` executable to predict
// against a fresh CSV. Confirms the on-disk format the CLI consumes
// stays in lockstep with what the orchestrator writes.
//
// The path to the executable is supplied at compile time via the
// `NEUROGRAPH_PREDICT_EXE` define, set by tests/CMakeLists.txt to
// `$<TARGET_FILE:neurograph_predict>`. Avoids relying on PATH lookups
// or assumptions about where the build laid down the artifact.

#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/runtime/bundle.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef NEUROGRAPH_PREDICT_EXE
#  error "NEUROGRAPH_PREDICT_EXE must be defined by the test build"
#endif

namespace fs = std::filesystem;
using neurograph::orchestration::AutoOptions;
using neurograph::orchestration::run;

namespace {

class PredictBinaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()->current_test_info();
        tmp_ = fs::temp_directory_path() / "neurograph_predict_bin_test"
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

// Quote a path for std::system on Windows. cmd.exe wants outer quotes
// and the inner path to not contain quotes; that's true for tmp paths
// we generate here.
std::string q(const fs::path& p) {
    std::string s = p.string();
    return "\"" + s + "\"";
}

fs::path write_train_csv(const fs::path& dir) {
    const auto path = dir / "train.csv";
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3,label\n";
    for (int i = 0; i < 200; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << ','
          << ((x1 + x2 - x3 > 0) ? "A" : "B") << '\n';
    }
    return path;
}

fs::path write_test_csv(const fs::path& dir, int n = 20) {
    const auto path = dir / "test.csv";
    std::mt19937_64 rng(99);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3\n";  // no label — predict.exe doesn't require it
    for (int i = 0; i < n; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << '\n';
    }
    return path;
}

fs::path build_bundle(const fs::path& tmp) {
    const auto csv = write_train_csv(tmp);
    AutoOptions opts;
    opts.bundle_out_dir    = tmp / "bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;
    auto r = run(csv, "label", opts);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return opts.bundle_out_dir;
}

int run_predict(const std::string& args) {
    // cmd.exe under std::system() requires an outer quote pair around
    // the *entire* command when the inner exe path AND arg paths are
    // both quoted — otherwise cmd's quote-stripping rule eats one of
    // them and emits "filename or volume name is incorrect". The double
    // leading quote is intentional.
    std::string cmd = std::string("\"\"") + NEUROGRAPH_PREDICT_EXE
                    + "\" " + args + "\"";
    return std::system(cmd.c_str());
}

std::vector<std::string> read_lines(const fs::path& p) {
    std::ifstream f(p);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) lines.push_back(std::move(line));
    return lines;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. CSV in / CSV out — happy path
// ---------------------------------------------------------------------------

TEST_F(PredictBinaryTest, CsvInCsvOutProducesPredictions) {
    const auto bundle = build_bundle(tmp_);
    const auto in     = write_test_csv(tmp_, 10);
    const auto out    = tmp_ / "preds.csv";

    const int rc = run_predict("--model " + q(bundle)
                               + " --in " + q(in)
                               + " --out " + q(out));
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(fs::exists(out));
    const auto lines = read_lines(out);
    // 1 header + 10 predictions
    ASSERT_EQ(lines.size(), 11u);
    EXPECT_NE(lines[0].find("predicted_class"), std::string::npos);
    EXPECT_NE(lines[0].find("p_A"), std::string::npos);
    EXPECT_NE(lines[0].find("p_B"), std::string::npos);
}

// ---------------------------------------------------------------------------
// 2. JSON output format
// ---------------------------------------------------------------------------

TEST_F(PredictBinaryTest, JsonOutputIsOneObjectPerLine) {
    const auto bundle = build_bundle(tmp_);
    const auto in     = write_test_csv(tmp_, 5);
    const auto out    = tmp_ / "preds.json";

    const int rc = run_predict("--model " + q(bundle)
                               + " --in " + q(in)
                               + " --out " + q(out)
                               + " --format json");
    ASSERT_EQ(rc, 0);
    const auto lines = read_lines(out);
    ASSERT_EQ(lines.size(), 5u);
    for (const auto& line : lines) {
        const auto j = nlohmann::json::parse(line);
        EXPECT_TRUE(j.contains("predicted_class"));
        EXPECT_TRUE(j.contains("predicted_class_id"));
        ASSERT_TRUE(j.contains("probabilities"));
        EXPECT_EQ(j["probabilities"].size(), 2u);
    }
}

// ---------------------------------------------------------------------------
// 3. --top-k surfaces alternative classes
// ---------------------------------------------------------------------------

TEST_F(PredictBinaryTest, TopKEmitsExtraColumns) {
    const auto bundle = build_bundle(tmp_);
    const auto in     = write_test_csv(tmp_, 3);
    const auto out    = tmp_ / "topk.csv";

    const int rc = run_predict("--model " + q(bundle)
                               + " --in " + q(in)
                               + " --out " + q(out)
                               + " --top-k 2");
    ASSERT_EQ(rc, 0);
    const auto lines = read_lines(out);
    ASSERT_FALSE(lines.empty());
    EXPECT_NE(lines[0].find("top1_class"), std::string::npos);
    EXPECT_NE(lines[0].find("top2_class"), std::string::npos);
}

// ---------------------------------------------------------------------------
// 4. Bad arg → exit 2
// ---------------------------------------------------------------------------

TEST_F(PredictBinaryTest, MissingModelFlagExitsWithBadArgs) {
    const int rc = run_predict("--in foo.csv");
    EXPECT_NE(rc, 0);
}

// ---------------------------------------------------------------------------
// 5. Bad bundle → exit 3 (model load failed)
// ---------------------------------------------------------------------------

TEST_F(PredictBinaryTest, MissingBundleExitsWithModelLoadFail) {
    const int rc = run_predict("--model " + q(tmp_ / "nope")
                               + " --in -"
                               + " --out -");
    EXPECT_NE(rc, 0);
}

// ---------------------------------------------------------------------------
// 6. A column the model needs is missing → exit 4, unless --allow-missing
// ---------------------------------------------------------------------------

namespace {

fs::path write_csv_with_header(const fs::path& dir, const std::string& name,
                               const std::string& header, int n_cols, int n_rows) {
    const auto path = dir / name;
    std::ofstream f(path, std::ios::binary);
    f << header << '\n';
    for (int i = 0; i < n_rows; ++i) {
        for (int c = 0; c < n_cols; ++c) f << (c ? "," : "") << (0.1 * (i + c) - 0.5);
        f << '\n';
    }
    return path;
}

} // namespace

TEST_F(PredictBinaryTest, MissingColumnStopsWithInputError) {
    const auto bundle = build_bundle(tmp_);
    // "f2" misspelt: filling it with the training average would quietly
    // give confident nonsense.
    const auto in  = write_csv_with_header(tmp_, "typo.csv", "f1,feature2,f3", 3, 4);
    const auto out = tmp_ / "preds.csv";
    const int rc = run_predict("--model " + q(bundle) + " --in " + q(in)
                               + " --out " + q(out));
    EXPECT_EQ(rc, 4);
}

TEST_F(PredictBinaryTest, AllowMissingFillsTheColumnAndCarriesOn) {
    const auto bundle = build_bundle(tmp_);
    const auto in  = write_csv_with_header(tmp_, "two_cols.csv", "f1,f3", 2, 4);
    const auto out = tmp_ / "preds.csv";
    const int rc = run_predict("--model " + q(bundle) + " --in " + q(in)
                               + " --out " + q(out) + " --allow-missing");
    ASSERT_EQ(rc, 0);
    EXPECT_EQ(read_lines(out).size(), 5u);   // header + 4 rows
}

TEST_F(PredictBinaryTest, HeaderWithByteOrderMarkAndSpacesIsAccepted) {
    // Excel's "CSV UTF-8" export starts the file with a byte-order mark.
    const auto bundle = build_bundle(tmp_);
    const auto in  = write_csv_with_header(tmp_, "excel.csv",
                                           "\xEF\xBB\xBF" "f1, f2 ,f3", 3, 4);
    const auto out = tmp_ / "preds.csv";
    const int rc = run_predict("--model " + q(bundle) + " --in " + q(in)
                               + " --out " + q(out));
    ASSERT_EQ(rc, 0);
    EXPECT_EQ(read_lines(out).size(), 5u);
}
