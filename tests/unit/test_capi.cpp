// Unit tests for the P1-08 C ABI.
//
// The example program (examples/capi/example.c) covers the "compiles
// as pure C" acceptance criterion by virtue of being built as a C
// target. These tests cover the runtime behavior:
//
//   - Every public function is reachable from a statically-linked build.
//   - Bad arguments produce the right status + populate ng_last_error.
//   - Predict results match the C++ orchestrator's in-process
//     predictions (confirms no drift between the two API surfaces).
//   - Multi-threaded torture: 8 threads × 10k predictions concurrent on
//     the same model produce identical results and don't crash.
//   - ng_last_error is thread-local: a failure on thread A doesn't
//     pollute thread B's last-error slot.
//
// The tests link `neurograph_rt` statically (via the library's
// `NEUROGRAPH_RT_STATIC` linkage path) so we don't have to chase the
// shared DLL around at test run time. The exported functions are the
// same either way.

#define NEUROGRAPH_RT_STATIC 1
#include "neurograph/capi/neurograph_rt.h"

#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/runtime/bundle.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

class CapiTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()->current_test_info();
        tmp_ = fs::temp_directory_path() / "neurograph_capi_test"
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
    for (int i = 0; i < 150; ++i) {
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
    for (int i = 0; i < 150; ++i) {
        const double x1 = nd(rng), x2 = nd(rng), x3 = nd(rng);
        f << x1 << ',' << x2 << ',' << x3 << ','
          << (2.0 * x1 - x2 + 0.5 * x3) << '\n';
    }
    return path;
}

fs::path build_cls_bundle(const fs::path& tmp) {
    const auto csv = write_cls_csv(tmp);
    neurograph::orchestration::AutoOptions opts;
    opts.bundle_out_dir    = tmp / "bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;
    auto r = neurograph::orchestration::run(csv, "label", opts);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return opts.bundle_out_dir;
}

fs::path build_reg_bundle(const fs::path& tmp) {
    const auto csv = write_reg_csv(tmp);
    neurograph::orchestration::AutoOptions opts;
    opts.bundle_out_dir    = tmp / "bundle";
    opts.do_learning_curve = false;
    opts.seed              = 42;
    auto r = neurograph::orchestration::run(csv, "target", opts);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return opts.bundle_out_dir;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Lifecycle + metadata
// ---------------------------------------------------------------------------

TEST_F(CapiTest, LoadClassificationBundleAndReadMetadata) {
    const auto dir = build_cls_bundle(tmp_);

    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK)
        << ng_last_error();
    ASSERT_NE(m, nullptr);

    EXPECT_EQ(ng_model_num_classes(m), 2);
    EXPECT_GT(ng_model_num_features(m), 0);
    EXPECT_NE(ng_model_class_label(m, 0), nullptr);
    EXPECT_NE(ng_model_class_label(m, 1), nullptr);
    EXPECT_EQ(ng_model_class_label(m, 99), nullptr);   // out of range
    EXPECT_STREQ(ng_model_version(m), "0.1.0");

    ng_model_free(m);
}

TEST_F(CapiTest, LoadRegressionBundleReportsZeroClasses) {
    const auto dir = build_reg_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);
    EXPECT_EQ(ng_model_num_classes(m), 0);
    EXPECT_GT(ng_model_num_features(m), 0);
    ng_model_free(m);
}

TEST_F(CapiTest, FreeOnNullIsNoOp) {
    ng_model_free(nullptr);   // must not crash
}

TEST_F(CapiTest, RtVersionIsNonEmpty) {
    EXPECT_NE(ng_rt_version(), nullptr);
    EXPECT_NE(std::string(ng_rt_version()).size(), 0u);
}

// ---------------------------------------------------------------------------
// 2. Bad arguments populate ng_last_error + return the right status
// ---------------------------------------------------------------------------

TEST_F(CapiTest, LoadNullDirReturnsBadArg) {
    ng_model* m = nullptr;
    EXPECT_EQ(ng_model_load(nullptr, &m), NG_ERR_BAD_ARG);
    EXPECT_EQ(m, nullptr);
    EXPECT_NE(std::string(ng_last_error()).find("NULL"), std::string::npos);
}

TEST_F(CapiTest, LoadNullOutReturnsBadArg) {
    EXPECT_EQ(ng_model_load("anything", nullptr), NG_ERR_BAD_ARG);
    EXPECT_NE(std::string(ng_last_error()).find("out"), std::string::npos);
}

TEST_F(CapiTest, LoadMissingDirReturnsIoError) {
    ng_model* m = nullptr;
    const auto bad = (tmp_ / "does_not_exist").string();
    EXPECT_EQ(ng_model_load(bad.c_str(), &m), NG_ERR_IO);
    EXPECT_EQ(m, nullptr);
    EXPECT_FALSE(std::string(ng_last_error()).empty());
}

TEST_F(CapiTest, LoadCorruptModelReturnsCorruptError) {
    const auto dir = build_cls_bundle(tmp_);
    {
        std::ofstream f(dir / "model.ngm");
        f << "not json";
    }
    ng_model* m = nullptr;
    EXPECT_EQ(ng_model_load(dir.string().c_str(), &m), NG_ERR_CORRUPT);
    EXPECT_EQ(m, nullptr);
}

TEST_F(CapiTest, PredictJsonWithBadJsonReturnsBadArg) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);

    int pred = -1;
    double probs[2] = {0.0, 0.0};
    EXPECT_EQ(ng_predict_json(m, "{not json", probs, &pred), NG_ERR_BAD_ARG);
    EXPECT_FALSE(std::string(ng_last_error()).empty());
    ng_model_free(m);
}

TEST_F(CapiTest, PredictJsonNullRowReturnsBadArg) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);
    EXPECT_EQ(ng_predict_json(m, nullptr, nullptr, nullptr), NG_ERR_BAD_ARG);
    ng_model_free(m);
}

TEST_F(CapiTest, PredictVecNullFeaturesReturnsBadArg) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);
    double probs[2];
    EXPECT_EQ(ng_predict_vec(m, nullptr, 3, probs, nullptr), NG_ERR_BAD_ARG);
    ng_model_free(m);
}

// ---------------------------------------------------------------------------
// 3. Predict outputs are sensible + match the C++ path
// ---------------------------------------------------------------------------

TEST_F(CapiTest, PredictJsonClassificationMatchesCxxPath) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);

    // Run the same row through the C++ bundle directly and compare.
    auto br = neurograph::runtime::load_bundle(dir);
    ASSERT_TRUE(br.ok());
    nlohmann::json row = {{"f1", 1.0}, {"f2", 1.0}, {"f3", -1.0}};
    const auto cxx_pred = neurograph::runtime::predict_row(br.value(), row);

    int pred = -1;
    double probs[2] = {0.0, 0.0};
    const std::string row_s = row.dump();
    ASSERT_EQ(ng_predict_json(m, row_s.c_str(), probs, &pred), NG_OK);

    EXPECT_EQ(pred, cxx_pred.predicted_class);
    ASSERT_EQ(cxx_pred.probabilities.size(), 2u);
    EXPECT_NEAR(probs[0], cxx_pred.probabilities[0], 1e-12);
    EXPECT_NEAR(probs[1], cxx_pred.probabilities[1], 1e-12);
    ng_model_free(m);
}

TEST_F(CapiTest, PredictJsonRegressionWritesValueIntoProbs0) {
    const auto dir = build_reg_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);

    double out = 0.0;
    ASSERT_EQ(ng_predict_json(m, "{\"f1\":1.0,\"f2\":0.0,\"f3\":0.0}", &out, nullptr), NG_OK);
    EXPECT_GT(out, 0.5);     // y ≈ 2*x1 on our synthetic data
    ng_model_free(m);
}

TEST_F(CapiTest, PredictVecMatchesPredictJsonForSameInput) {
    // If the caller preprocesses once and reuses the vector, they
    // should get the same answer they'd get from json-then-preprocess.
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);

    // Build the transformed feature vector via the C++ bundle.
    auto br = neurograph::runtime::load_bundle(dir);
    ASSERT_TRUE(br.ok());
    nlohmann::json row = {{"f1", 0.5}, {"f2", -0.2}, {"f3", 0.7}};
    const auto x = neurograph::preprocess::apply_pipeline(*br.value().pipeline, row);

    int    pred_vec = -1, pred_json = -1;
    double probs_vec[2]  = {0.0, 0.0};
    double probs_json[2] = {0.0, 0.0};
    ASSERT_EQ(ng_predict_vec(m, x.data(), static_cast<int>(x.size()),
                              probs_vec, &pred_vec), NG_OK);
    const std::string row_s = row.dump();
    ASSERT_EQ(ng_predict_json(m, row_s.c_str(), probs_json, &pred_json), NG_OK);

    EXPECT_EQ(pred_vec, pred_json);
    EXPECT_NEAR(probs_vec[0], probs_json[0], 1e-12);
    EXPECT_NEAR(probs_vec[1], probs_json[1], 1e-12);
    ng_model_free(m);
}

// ---------------------------------------------------------------------------
// 4. Optional-output pointers — caller can skip probs or pred
// ---------------------------------------------------------------------------

TEST_F(CapiTest, PredictJsonPredOnlyWorks) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);
    int pred = -1;
    ASSERT_EQ(ng_predict_json(m, "{\"f1\":0,\"f2\":0,\"f3\":0}",
                               /*out_probs=*/nullptr, &pred), NG_OK);
    EXPECT_GE(pred, 0);
    EXPECT_LT(pred, 2);
    ng_model_free(m);
}

TEST_F(CapiTest, PredictJsonProbsOnlyWorks) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);
    double probs[2] = {0.0, 0.0};
    ASSERT_EQ(ng_predict_json(m, "{\"f1\":0,\"f2\":0,\"f3\":0}",
                               probs, /*out_pred=*/nullptr), NG_OK);
    EXPECT_NEAR(probs[0] + probs[1], 1.0, 1e-9);
    ng_model_free(m);
}

// ---------------------------------------------------------------------------
// 5. Batch predict
// ---------------------------------------------------------------------------

TEST_F(CapiTest, PredictBatchMatchesPerRowCalls) {
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);

    auto br = neurograph::runtime::load_bundle(dir);
    ASSERT_TRUE(br.ok());

    constexpr int N = 8;
    std::vector<std::vector<double>> rows;
    std::mt19937_64 rng(123);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        nlohmann::json row = {{"f1", nd(rng)}, {"f2", nd(rng)}, {"f3", nd(rng)}};
        rows.push_back(neurograph::preprocess::apply_pipeline(*br.value().pipeline, row));
    }
    const int D = static_cast<int>(rows[0].size());

    std::vector<double> flat;
    flat.reserve(static_cast<std::size_t>(N) * D);
    for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());

    std::vector<double> probs(static_cast<std::size_t>(N) * 2, 0.0);
    std::vector<int>    preds(N, -1);
    ASSERT_EQ(ng_predict_batch_vec(m, flat.data(), N, D, probs.data(), preds.data()),
              NG_OK);

    for (int i = 0; i < N; ++i) {
        double single_probs[2] = {0.0, 0.0};
        int    single_pred     = -1;
        ASSERT_EQ(ng_predict_vec(m, rows[i].data(), D, single_probs, &single_pred), NG_OK);
        EXPECT_EQ(preds[i], single_pred);
        EXPECT_NEAR(probs[i * 2 + 0], single_probs[0], 1e-12);
        EXPECT_NEAR(probs[i * 2 + 1], single_probs[1], 1e-12);
    }
    ng_model_free(m);
}

// ---------------------------------------------------------------------------
// 6. Concurrency — same model, many threads
// ---------------------------------------------------------------------------

TEST_F(CapiTest, ConcurrentPredictIsSafeAndDeterministic) {
    // Acceptance criterion: 8 threads × 10k predictions = 80k calls on
    // one model handle, no crashes, no data races. We also assert each
    // thread sees the same answer as a single-threaded reference run —
    // if there were a shared-mutable scratch buffer we'd see drift.
    const auto dir = build_cls_bundle(tmp_);
    ng_model* m = nullptr;
    ASSERT_EQ(ng_model_load(dir.string().c_str(), &m), NG_OK);

    auto br = neurograph::runtime::load_bundle(dir);
    ASSERT_TRUE(br.ok());
    nlohmann::json row = {{"f1", 0.25}, {"f2", -0.1}, {"f3", 0.8}};
    const auto ref_pred = neurograph::runtime::predict_row(br.value(), row);
    const std::string row_s = row.dump();

    constexpr int T = 8;
    constexpr int N = 10000;
    std::atomic<int> failures = 0;
    std::atomic<int> mismatches = 0;
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&] {
            double probs[2];
            int    pred = -1;
            for (int i = 0; i < N; ++i) {
                if (ng_predict_json(m, row_s.c_str(), probs, &pred) != NG_OK) {
                    ++failures;
                    return;
                }
                if (pred != ref_pred.predicted_class) ++mismatches;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(mismatches.load(), 0);

    ng_model_free(m);
}

// ---------------------------------------------------------------------------
// 7. ng_last_error is thread-local
// ---------------------------------------------------------------------------

TEST_F(CapiTest, LastErrorIsThreadLocal) {
    // Thread A triggers an error. Thread B calls ng_last_error() and
    // must see an empty string (never touched anything on this
    // thread). If TLS is broken and we're using a shared global, B
    // will see A's message.
    std::thread a([] {
        ng_model_load(nullptr, nullptr);   // forces set_err on thread A
    });
    a.join();

    std::string from_b;
    std::thread b([&] {
        from_b = ng_last_error();
    });
    b.join();
    EXPECT_TRUE(from_b.empty()) << "saw bleed-through: " << from_b;
}
