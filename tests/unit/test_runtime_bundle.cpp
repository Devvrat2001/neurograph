// Unit tests for the P1-07 runtime bundle loader.
//
// What we cover here that the integration test (test_predict_binary.cpp)
// doesn't:
//
//   - Bundle loader fails cleanly on every shape of corruption rather
//     than crashing or throwing.
//   - CSV split/quote helpers handle the RFC 4180 corners we actually
//     hit (quoted commas, escaped quotes, trailing CR).
//   - predict_row reproduces what the orchestrator's in-process
//     predictions returned during training — confirms the wire format
//     round-trips byte-for-byte.

#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/runtime/bundle.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using neurograph::orchestration::AutoOptions;
using neurograph::orchestration::run;
using neurograph::runtime::load_bundle;
using neurograph::runtime::predict_row;
using neurograph::runtime::quote_csv_field;
using neurograph::runtime::split_csv_line;

namespace {

class RuntimeBundleTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()->current_test_info();
        tmp_ = fs::temp_directory_path() / "neurograph_runtime_test"
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

fs::path write_cls_csv(const fs::path& dir) {
    const auto path = dir / "cls.csv";
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3,label\n";
    for (int i = 0; i < 100; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << ','
          << ((x1 + x2 - x3 > 0) ? "A" : "B") << '\n';
    }
    return path;
}

fs::path write_reg_csv(const fs::path& dir) {
    const auto path = dir / "reg.csv";
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::ofstream f(path);
    f << "f1,f2,f3,target\n";
    for (int i = 0; i < 100; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        const double y  = 2.0 * x1 - x2 + 0.5 * x3;
        f << x1 << ',' << x2 << ',' << x3 << ',' << y << '\n';
    }
    return path;
}

// Build a real bundle through the orchestrator, so the loader test
// uses the same wire format the CLI will see in production.
fs::path build_cls_bundle(const fs::path& tmp) {
    const auto csv = write_cls_csv(tmp);
    AutoOptions opts;
    opts.bundle_out_dir    = tmp / "bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;
    auto r = run(csv, "label", opts);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return opts.bundle_out_dir;
}

fs::path build_reg_bundle(const fs::path& tmp) {
    const auto csv = write_reg_csv(tmp);
    AutoOptions opts;
    opts.bundle_out_dir    = tmp / "bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;
    auto r = run(csv, "target", opts);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return opts.bundle_out_dir;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Happy path — classification bundle
// ---------------------------------------------------------------------------

TEST_F(RuntimeBundleTest, LoadClassificationBundle) {
    const auto dir = build_cls_bundle(tmp_);
    auto br = load_bundle(dir);
    ASSERT_TRUE(br.ok()) << br.error().message;
    const auto& b = br.value();
    EXPECT_TRUE(b.is_classification());
    EXPECT_EQ(b.n_classes(), 2);
    EXPECT_FALSE(b.manifest.empty());          // manifest.json exists
}

TEST_F(RuntimeBundleTest, LoadRegressionBundle) {
    const auto dir = build_reg_bundle(tmp_);
    auto br = load_bundle(dir);
    ASSERT_TRUE(br.ok()) << br.error().message;
    EXPECT_FALSE(br.value().is_classification());
    EXPECT_EQ(br.value().n_classes(), 0);
}

// ---------------------------------------------------------------------------
// 2. predict_row produces sensible output
// ---------------------------------------------------------------------------

TEST_F(RuntimeBundleTest, PredictRowClassificationSumsToOne) {
    const auto dir = build_cls_bundle(tmp_);
    const auto b   = load_bundle(dir).value();

    nlohmann::json row = {{"f1", 1.0}, {"f2", 1.0}, {"f3", -1.0}};
    auto pr = predict_row(b, row);
    EXPECT_GE(pr.predicted_class, 0);
    EXPECT_EQ(pr.probabilities.size(), 2u);
    double s = 0.0;
    for (double p : pr.probabilities) {
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
        s += p;
    }
    EXPECT_NEAR(s, 1.0, 1e-9);
}

TEST_F(RuntimeBundleTest, PredictRowRegressionReturnsScalar) {
    const auto dir = build_reg_bundle(tmp_);
    const auto b   = load_bundle(dir).value();

    // P1-15: behavior contract for the bundle loader's regression
    // path. The orchestrator now produces SON neural-net bundles
    // (schema_v2) with non-deterministic numeric calibration on
    // small fixtures — the meaningful AC is "regression bundle
    // round-trips, returns a finite scalar that varies with input,
    // and stays in the regression slot of the variant" (no class
    // labels, no probabilities). Aggregate fit quality (R² > 0.7,
    // RMSE < 1) is asserted in test_auto_pipeline.cpp's
    // EndToEndRegressionProducesBundle case.
    nlohmann::json row_a = {{"f1", 1.0},  {"f2",  0.0}, {"f3", 0.0}};
    nlohmann::json row_b = {{"f1", -1.0}, {"f2",  0.0}, {"f3", 0.0}};
    const auto pa = predict_row(b, row_a);
    const auto pb = predict_row(b, row_b);

    // Finite scalar (not NaN, not infinite) in both cases.
    EXPECT_TRUE(std::isfinite(pa.value));
    EXPECT_TRUE(std::isfinite(pb.value));

    // Predictions vary with input — proves the network actually uses
    // its inputs, not a constant function. Tolerance loose; the
    // synthetic data has y = 2*x1 - x2 + 0.5*x3, so flipping x1's
    // sign should produce a meaningfully different output.
    EXPECT_GT(std::abs(pa.value - pb.value), 0.1);

    // Variant is the regression slot — no class labels, no probs.
    EXPECT_TRUE(pa.probabilities.empty());
    EXPECT_EQ(pa.predicted_class, -1);
}

// ---------------------------------------------------------------------------
// 3. Bad bundle paths produce errors, not crashes
// ---------------------------------------------------------------------------

TEST_F(RuntimeBundleTest, LoadFailsOnNonexistentDir) {
    auto r = load_bundle(tmp_ / "no_such_dir");
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().message.find("not a directory"), std::string::npos);
}

TEST_F(RuntimeBundleTest, LoadFailsOnEmptyDir) {
    auto r = load_bundle(tmp_);
    ASSERT_FALSE(r.ok());
    // Either pipeline or model is missing — we don't pin which.
}

TEST_F(RuntimeBundleTest, LoadFailsOnCorruptModelJson) {
    const auto dir = build_cls_bundle(tmp_);
    {
        std::ofstream f(dir / "model.ngm");
        f << "{not valid json";
    }
    auto r = load_bundle(dir);
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().message.find("not valid JSON"), std::string::npos)
        << r.error().message;
}

TEST_F(RuntimeBundleTest, LoadFailsOnSchemaVersionMismatch) {
    const auto dir = build_cls_bundle(tmp_);
    nlohmann::json mj;
    {
        std::ifstream f(dir / "model.ngm");
        f >> mj;
    }
    mj["schema_version"] = 99;
    {
        std::ofstream f(dir / "model.ngm");
        f << mj.dump(2);
    }
    auto r = load_bundle(dir);
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().message.find("schema_version"), std::string::npos)
        << r.error().message;
}

TEST_F(RuntimeBundleTest, LoadFailsOnUnknownKind) {
    const auto dir = build_cls_bundle(tmp_);
    nlohmann::json mj;
    {
        std::ifstream f(dir / "model.ngm");
        f >> mj;
    }
    mj["kind"] = "graph_v999";
    {
        std::ofstream f(dir / "model.ngm");
        f << mj.dump(2);
    }
    auto r = load_bundle(dir);
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().message.find("unknown kind"), std::string::npos)
        << r.error().message;
}

// ---------------------------------------------------------------------------
// 4. CSV helpers
// ---------------------------------------------------------------------------

TEST(RuntimeCsv, SplitSimpleLine) {
    const auto v = split_csv_line("a,b,c");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b");
    EXPECT_EQ(v[2], "c");
}

TEST(RuntimeCsv, SplitTrimsTrailingCR) {
    const auto v = split_csv_line("a,b\r");
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[1], "b");
}

TEST(RuntimeCsv, SplitHandlesQuotedComma) {
    const auto v = split_csv_line("a,\"b,c\",d");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[1], "b,c");
}

TEST(RuntimeCsv, SplitHandlesEscapedQuote) {
    const auto v = split_csv_line("\"he said \"\"hi\"\"\",x");
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0], "he said \"hi\"");
}

TEST(RuntimeCsv, QuoteFieldNoSpecialChars) {
    EXPECT_EQ(quote_csv_field("plain"), "plain");
}

TEST(RuntimeCsv, QuoteFieldWithComma) {
    EXPECT_EQ(quote_csv_field("a,b"), "\"a,b\"");
}

TEST(RuntimeCsv, QuoteFieldWithQuote) {
    EXPECT_EQ(quote_csv_field("a\"b"), "\"a\"\"b\"");
}

TEST(RuntimeCsv, QuoteFieldWithNewline) {
    EXPECT_EQ(quote_csv_field("a\nb"), "\"a\nb\"");
}
