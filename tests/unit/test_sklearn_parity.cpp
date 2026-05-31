// Sklearn-parity test for evaluate_classification (P1-04 AC1).
//
// Loads tests/fixtures/eval/sklearn_golden.json — a deterministic
// classification problem where the expected metrics were computed by
// sklearn (regenerable via tools/gen_sklearn_golden.py). Asserts that
// evaluate_classification produces matching values within 1e-6.
//
// What this catches that the hand-crafted closed-form tests in
// test_classification_metrics.cpp don't:
//   - Numerical-stability bugs that only surface on >40-sample inputs.
//   - Tied-rank handling in ROC AUC under realistic prob distributions
//     (the closed-form fixtures use exact 0.0/1.0 probabilities which
//     never trigger the tied-rank path).
//   - Macro-vs-weighted F1 split (the closed-form fixtures have
//     balanced classes where the two coincide).
//   - Argmax stability for rows where two classes are within 1e-12 of
//     each other (the noise-injected fixture exercises this).
//
// Tolerance: 1e-6 per the AC. Most metrics actually agree to 1e-12;
// the looser bound exists because:
//   - sklearn's `roc_auc_score(multi_class='ovr')` uses a different
//     tied-rank averaging convention from our Mann-Whitney U formula
//     in some edge cases (off by ≤ 1e-9 on this fixture).
//   - macro_f1 sums per-class F1s and divides — accumulated FP errors
//     can be ~1e-15, well inside 1e-6.
//
// Path resolution: the JSON's location is baked in at compile time via
// NEUROGRAPH_SKLEARN_GOLDEN_PATH. Same idiom as
// tests/integration/test_predict_binary.cpp's NEUROGRAPH_PREDICT_EXE
// — keeps the test runnable from any CWD.

#include "neurograph/eval/classification.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef NEUROGRAPH_SKLEARN_GOLDEN_PATH
#  error "NEUROGRAPH_SKLEARN_GOLDEN_PATH must be defined by the test build"
#endif

namespace fs = std::filesystem;
using neurograph::eval::evaluate_classification;

namespace {

constexpr double kSklearnTolerance = 1e-6;

nlohmann::json load_fixture() {
    const fs::path path = NEUROGRAPH_SKLEARN_GOLDEN_PATH;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "fixture not found at " << path.string();
    nlohmann::json j;
    f >> j;
    return j;
}

} // namespace

class SklearnParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixture_ = load_fixture();

        // Reconstruct inputs from the JSON. We materialize into the
        // same containers evaluate_classification expects rather than
        // duck-typing through nlohmann's iterators — clearer failure
        // messages when the fixture schema drifts.
        for (const auto& v : fixture_["y_true"])  y_true_.push_back(v.get<int>());
        for (const auto& row : fixture_["probs"]) {
            std::vector<double> r;
            for (const auto& v : row) r.push_back(v.get<double>());
            probs_.push_back(std::move(r));
        }
        for (const auto& v : fixture_["class_labels"])
            labels_.push_back(v.get<std::string>());
    }

    nlohmann::json                   fixture_;
    std::vector<int>                 y_true_;
    std::vector<std::vector<double>> probs_;
    std::vector<std::string>         labels_;
};

TEST_F(SklearnParityTest, FixtureHasExpectedShape) {
    // Sanity check on the loader before we trust anything downstream.
    EXPECT_EQ(y_true_.size(), fixture_["n_samples"].get<size_t>());
    EXPECT_EQ(labels_.size(), fixture_["n_classes"].get<size_t>());
    EXPECT_EQ(probs_.size(), y_true_.size());
    for (const auto& row : probs_) {
        EXPECT_EQ(row.size(), labels_.size());
    }
}

TEST_F(SklearnParityTest, AccuracyMatchesSklearn) {
    const auto m = evaluate_classification(y_true_, probs_, labels_);
    EXPECT_NEAR(m.accuracy,
                fixture_["expected"]["accuracy"].get<double>(),
                kSklearnTolerance);
}

TEST_F(SklearnParityTest, MacroAndWeightedF1MatchSklearn) {
    const auto m = evaluate_classification(y_true_, probs_, labels_);
    EXPECT_NEAR(m.macro_f1,
                fixture_["expected"]["macro_f1"].get<double>(),
                kSklearnTolerance);
    EXPECT_NEAR(m.weighted_f1,
                fixture_["expected"]["weighted_f1"].get<double>(),
                kSklearnTolerance);
}

TEST_F(SklearnParityTest, RocAucMacroMatchesSklearn) {
    const auto m = evaluate_classification(y_true_, probs_, labels_);
    EXPECT_NEAR(m.roc_auc_macro,
                fixture_["expected"]["roc_auc_macro"].get<double>(),
                kSklearnTolerance);
}

TEST_F(SklearnParityTest, PerClassPrecisionRecallF1MatchSklearn) {
    const auto m = evaluate_classification(y_true_, probs_, labels_);
    const auto& exp = fixture_["expected"]["per_class"];

    ASSERT_EQ(m.per_class.size(), exp.size())
        << "per-class array size mismatch — fixture vs evaluate_classification";

    for (size_t c = 0; c < exp.size(); ++c) {
        const std::string label = exp[c]["label"].get<std::string>();
        SCOPED_TRACE("class " + std::to_string(c) + " (" + label + ")");

        EXPECT_EQ(m.per_class[c].class_label, label);
        EXPECT_NEAR(m.per_class[c].precision,
                    exp[c]["precision"].get<double>(), kSklearnTolerance);
        EXPECT_NEAR(m.per_class[c].recall,
                    exp[c]["recall"].get<double>(),    kSklearnTolerance);
        EXPECT_NEAR(m.per_class[c].f1,
                    exp[c]["f1"].get<double>(),        kSklearnTolerance);
        EXPECT_EQ(m.per_class[c].support,
                  exp[c]["support"].get<int>());
    }
}

TEST_F(SklearnParityTest, ConfusionMatrixMatchesSklearn) {
    const auto m = evaluate_classification(y_true_, probs_, labels_);
    const auto& exp = fixture_["expected"]["confusion"];

    ASSERT_EQ(m.confusion.cells.size(), exp.size());
    for (size_t r = 0; r < exp.size(); ++r) {
        ASSERT_EQ(m.confusion.cells[r].size(), exp[r].size())
            << "confusion row " << r << " width mismatch";
        for (size_t c = 0; c < exp[r].size(); ++c) {
            // Exact match — confusion cells are integer counts, no
            // floating-point tolerance needed.
            EXPECT_EQ(m.confusion.cells[r][c], exp[r][c].get<int>())
                << "confusion[" << r << "][" << c << "]";
        }
    }
}
