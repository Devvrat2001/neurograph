// Unit tests for neurograph::eval::evaluate_classification (ticket P1-04).
//
// Strategy: small hand-crafted fixtures where every metric has a
// closed-form answer we can write down. Bigger sklearn-golden JSON
// parity test is deferred to its own fixture file once we have python
// tooling in CI.

#include "neurograph/eval/classification.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

using neurograph::eval::ClassificationMetrics;
using neurograph::eval::evaluate_classification;

namespace {

const std::vector<std::string> kBinaryLabels = {"neg", "pos"};

// Perfect binary classifier: probabilities concentrated on the true class.
TEST(ClassificationMetrics, PerfectBinaryClassifier) {
    const std::vector<int> y{0, 1, 0, 1};
    const std::vector<std::vector<double>> probs{
        {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};

    const auto m = evaluate_classification(y, probs, kBinaryLabels);
    EXPECT_DOUBLE_EQ(m.accuracy,     1.0);
    EXPECT_DOUBLE_EQ(m.macro_f1,     1.0);
    EXPECT_DOUBLE_EQ(m.weighted_f1,  1.0);
    EXPECT_DOUBLE_EQ(m.roc_auc_macro, 1.0);
    EXPECT_DOUBLE_EQ(m.pr_auc_macro,  1.0);
    EXPECT_DOUBLE_EQ(m.brier_score,   0.0);
    EXPECT_EQ(m.confusion.cells[0][0], 2);
    EXPECT_EQ(m.confusion.cells[1][1], 2);
    EXPECT_EQ(m.confusion.cells[0][1], 0);
    EXPECT_EQ(m.confusion.cells[1][0], 0);
}

// Totally wrong predictions: accuracy 0, AUC 0 (perfectly anti-correlated).
TEST(ClassificationMetrics, CompletelyWrong) {
    const std::vector<int> y{0, 1, 0, 1};
    const std::vector<std::vector<double>> probs{
        {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}};

    const auto m = evaluate_classification(y, probs, kBinaryLabels);
    EXPECT_DOUBLE_EQ(m.accuracy, 0.0);
    EXPECT_DOUBLE_EQ(m.roc_auc_macro, 0.0);
}

// Ties across all samples → AUC = 0.5 (true chance-level), verifying we
// handle tied scores via mean-rank assignment instead of blowing up the
// Mann-Whitney sum.
TEST(ClassificationMetrics, TiedProbabilitiesGiveChanceAUC) {
    const std::vector<int> y{0, 1, 0, 1};
    const std::vector<std::vector<double>> probs{
        {0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}};

    const auto m = evaluate_classification(y, probs, kBinaryLabels);
    EXPECT_NEAR(m.roc_auc_macro, 0.5, 1e-12);
}

// 3-class problem: hand-computed confusion + per-class F1.
TEST(ClassificationMetrics, ThreeClassPerClassMetrics) {
    const std::vector<std::string> labels{"a", "b", "c"};
    const std::vector<int> y{0, 0, 1, 1, 2, 2};
    // Predictions: a,a, a,b, b,c  (diagonal = 1,1,1; 2 off-diagonals)
    const std::vector<std::vector<double>> probs{
        {0.9, 0.1, 0.0},  // pred a, true a  ✓
        {0.8, 0.1, 0.1},  // pred a, true a  ✓
        {0.7, 0.2, 0.1},  // pred a, true b  ✗
        {0.1, 0.8, 0.1},  // pred b, true b  ✓
        {0.1, 0.7, 0.2},  // pred b, true c  ✗
        {0.0, 0.1, 0.9},  // pred c, true c  ✓
    };

    const auto m = evaluate_classification(y, probs, labels);
    EXPECT_NEAR(m.accuracy, 4.0 / 6.0, 1e-12);
    // class a: TP=2, FP=1, FN=0 → P=2/3, R=1.0, F1=0.8
    EXPECT_NEAR(m.per_class[0].precision, 2.0 / 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(m.per_class[0].recall, 1.0);
    EXPECT_NEAR(m.per_class[0].f1, 0.8, 1e-12);
    // class b: TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
    EXPECT_NEAR(m.per_class[1].f1, 0.5, 1e-12);
    // class c: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1≈0.6667
    EXPECT_NEAR(m.per_class[2].f1, 2.0 / 3.0, 1e-12);
    // support sums should equal n
    int total_support = 0;
    for (const auto& pc : m.per_class) total_support += pc.support;
    EXPECT_EQ(total_support, static_cast<int>(y.size()));
}

// top_5_accuracy only defined for n_classes >= 5.
TEST(ClassificationMetrics, Top5AccuracyNaNForSmallNClasses) {
    const std::vector<int> y{0, 1};
    const std::vector<std::vector<double>> probs{{0.9, 0.1}, {0.2, 0.8}};
    const auto m = evaluate_classification(y, probs, kBinaryLabels);
    EXPECT_TRUE(std::isnan(m.top_5_accuracy));
}

// 6-class top-5: the true class must be in the top 5 probs — so only an
// argmax of true-class = 6th highest should miss.
TEST(ClassificationMetrics, Top5AccuracyHitWhenTrueClassInTop5) {
    const std::vector<std::string> labels{"c0","c1","c2","c3","c4","c5"};
    // Sample 1: true=0, probs put class 0 = 2nd highest (hit).
    // Sample 2: true=5, probs put class 5 = lowest (miss).
    const std::vector<int> y{0, 5};
    const std::vector<std::vector<double>> probs{
        {0.2, 0.4, 0.1, 0.1, 0.1, 0.1},
        {0.4, 0.2, 0.15, 0.1, 0.1, 0.05},
    };
    const auto m = evaluate_classification(y, probs, labels);
    EXPECT_NEAR(m.top_5_accuracy, 0.5, 1e-12);
}

// Brier score for confidently-correct predictions → 0; for confidently-wrong → 2.
// (Sum over classes: (p - one_hot)^2; for p=[1,0] true=1 → (1)^2 + (1)^2 = 2.)
TEST(ClassificationMetrics, BrierBoundsKnownCases) {
    const std::vector<int> y_correct{1};
    const std::vector<std::vector<double>> probs_correct{{0.0, 1.0}};
    EXPECT_DOUBLE_EQ(
        evaluate_classification(y_correct, probs_correct, kBinaryLabels).brier_score,
        0.0);

    const std::vector<int> y_wrong{1};
    const std::vector<std::vector<double>> probs_wrong{{1.0, 0.0}};
    EXPECT_DOUBLE_EQ(
        evaluate_classification(y_wrong, probs_wrong, kBinaryLabels).brier_score,
        2.0);
}

// Confusion diagonal sum should always equal correct count = accuracy * n.
TEST(ClassificationMetrics, ConfusionDiagonalMatchesAccuracy) {
    const std::vector<int> y{0, 1, 0, 1, 0};
    const std::vector<std::vector<double>> probs{
        {0.9, 0.1}, {0.4, 0.6}, {0.7, 0.3}, {0.2, 0.8}, {0.6, 0.4}};
    const auto m = evaluate_classification(y, probs, kBinaryLabels);
    int diag = 0;
    for (int c = 0; c < m.confusion.n_classes; ++c)
        diag += m.confusion.cells[c][c];
    EXPECT_EQ(diag, static_cast<int>(std::round(m.accuracy * y.size())));
}

// Contract: empty input returns NaN accuracy, doesn't throw.
TEST(ClassificationMetrics, EmptyInputReturnsNaNs) {
    const auto m = evaluate_classification({}, {}, kBinaryLabels);
    EXPECT_TRUE(std::isnan(m.accuracy));
    EXPECT_TRUE(std::isnan(m.brier_score));
    // per_class slots still initialized so UI code doesn't segfault.
    ASSERT_EQ(m.per_class.size(), kBinaryLabels.size());
    EXPECT_EQ(m.per_class[0].class_label, "neg");
}

// Invalid inputs should throw invalid_argument — callers will upgrade
// these to friendly error messages at the UI layer.
TEST(ClassificationMetrics, InvalidInputsThrow) {
    const std::vector<int> y{0, 1};
    const std::vector<std::vector<double>> probs{{0.9, 0.1}, {0.2, 0.8}};

    // Single class label.
    EXPECT_THROW(evaluate_classification(y, probs, {"only"}),
                 std::invalid_argument);
    // Size mismatch.
    EXPECT_THROW(evaluate_classification({0}, probs, kBinaryLabels),
                 std::invalid_argument);
    // Row length mismatch.
    const std::vector<std::vector<double>> bad_probs{{0.9}, {0.2}};
    EXPECT_THROW(evaluate_classification(y, bad_probs, kBinaryLabels),
                 std::invalid_argument);
    // Out-of-range label.
    EXPECT_THROW(evaluate_classification({0, 5}, probs, kBinaryLabels),
                 std::invalid_argument);
}

// Reliability bins should sum (in count) to the number of samples.
TEST(ClassificationMetrics, ReliabilityCountsSumToN) {
    const std::vector<int> y{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    std::vector<std::vector<double>> probs;
    for (int i = 0; i < 10; ++i) {
        const double p = 0.1 * (i + 1);
        probs.push_back({1.0 - p, p});  // confidence varies across bins
    }
    const auto m = evaluate_classification(y, probs, kBinaryLabels);
    int total = 0;
    for (const auto& b : m.reliability) total += b.count;
    EXPECT_EQ(total, static_cast<int>(y.size()));
    EXPECT_EQ(m.reliability.size(), 10u);
}

} // namespace
