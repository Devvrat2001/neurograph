// Unit tests for neurograph::eval::evaluate_regression (ticket P1-05).
//
// These cover the happy path (sklearn-equivalent numbers) and the
// degenerate-input contracts we explicitly promise not to blow up on:
// empty vectors, constant targets, zero-valued targets (MAPE).

#include "neurograph/eval/regression.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using neurograph::eval::evaluate_regression;
using neurograph::eval::evaluate_regression_quantile;
using neurograph::eval::RegressionMetrics;

namespace {

// Values hand-computed against Python reference; guards sklearn parity.
TEST(RegressionMetrics, KnownGoldenValues) {
    // y  = [1, 2, 3, 4, 5], ŷ = [1.1, 1.9, 3.2, 3.8, 5.1]
    // residuals = [0.1, -0.1, 0.2, -0.2, 0.1] → MSE = (0.01+0.01+0.04+0.04+0.01)/5
    const std::vector<double> y    = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> yhat = {1.1, 1.9, 3.2, 3.8, 5.1};

    const auto m = evaluate_regression(y, yhat);
    EXPECT_NEAR(m.mse,  0.022, 1e-9);
    EXPECT_NEAR(m.rmse, std::sqrt(0.022), 1e-9);
    EXPECT_NEAR(m.mae,  0.14,  1e-9);
    // SS_tot = Var(y)*n = 2.0*5 = 10. SSE = 0.11. R² = 1 - 0.11/10 = 0.989.
    EXPECT_NEAR(m.r2,   0.989, 1e-6);
    EXPECT_EQ  (m.n,    5u);
}

TEST(RegressionMetrics, PerfectPredictionsGiveZeroErrorAndUnitR2) {
    const std::vector<double> y{1.0, 2.0, 3.0};
    const auto m = evaluate_regression(y, y);
    EXPECT_DOUBLE_EQ(m.mse,  0.0);
    EXPECT_DOUBLE_EQ(m.rmse, 0.0);
    EXPECT_DOUBLE_EQ(m.mae,  0.0);
    EXPECT_DOUBLE_EQ(m.r2,   1.0);
}

// Constant target has zero variance so R² is undefined. Contract: we return
// NaN rather than the "infinite improvement" you'd get from a naive 1-0/0.
TEST(RegressionMetrics, ConstantTargetR2IsNaN) {
    const std::vector<double> y{3.0, 3.0, 3.0, 3.0};
    const std::vector<double> yhat{3.0, 3.0, 3.0, 3.0};
    const auto m = evaluate_regression(y, yhat);
    EXPECT_DOUBLE_EQ(m.mse, 0.0);
    EXPECT_TRUE(std::isnan(m.r2));
    EXPECT_TRUE(std::isnan(m.explained_variance));
}

TEST(RegressionMetrics, MapeIsNanWhenAnyTrueIsZero) {
    const std::vector<double> y   {0.0, 1.0, 2.0};
    const std::vector<double> yhat{0.1, 1.0, 2.0};
    const auto m = evaluate_regression(y, yhat);
    EXPECT_TRUE(std::isnan(m.mape));
}

TEST(RegressionMetrics, MapeDefinedWhenAllTrueNonZero) {
    const std::vector<double> y   {1.0, 2.0, 4.0};
    const std::vector<double> yhat{1.1, 1.8, 4.4};
    // errors/|y| = 0.1, 0.1, 0.1 → MAPE = 0.1
    const auto m = evaluate_regression(y, yhat);
    EXPECT_NEAR(m.mape, 0.1, 1e-12);
}

TEST(RegressionMetrics, EmptyInputsReturnNaNNotThrow) {
    const auto m = evaluate_regression({}, {});
    EXPECT_EQ(m.n, 0u);
    EXPECT_TRUE(std::isnan(m.mse));
    EXPECT_TRUE(std::isnan(m.r2));
}

TEST(RegressionMetrics, SizeMismatchThrows) {
    EXPECT_THROW(evaluate_regression({1.0, 2.0}, {1.0}), std::invalid_argument);
}

TEST(RegressionMetrics, ResidualMeanCapturesBias) {
    // Constant +0.5 bias on all predictions.
    const std::vector<double> y   {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> yhat{1.5, 2.5, 3.5, 4.5};
    const auto m = evaluate_regression(y, yhat);
    EXPECT_NEAR(m.residual_mean, 0.5, 1e-12);
}

// Pinball loss of a perfect median forecast on symmetric residuals should
// equal MAE/2 — a standard sanity check for quantile metrics.
TEST(QuantileRegressionMetrics, Q50PinballEqualsHalfMAE) {
    const std::vector<double> y  {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> q50{1.2, 1.8, 3.1, 3.9};  // median == point fcst
    const std::vector<double> q10{0.5, 1.0, 2.0, 3.0};
    const std::vector<double> q90{2.0, 3.0, 4.0, 5.0};

    const auto m = evaluate_regression_quantile(y, q10, q50, q90);
    const auto point = evaluate_regression(y, q50);
    EXPECT_NEAR(m.q50_loss, point.mae / 2.0, 1e-12);
}

} // namespace
