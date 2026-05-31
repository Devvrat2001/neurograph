#pragma once

// P1-05 — Regression metrics.
//
// Scope: everything a customer asks when they see "this is a regression
// model." MSE / RMSE / MAE / MAPE / R² / explained-variance / residual stats,
// plus pinball loss for quantile regressors.
//
// Design:
//   - All functions are pure (no hidden state) so they're trivial to test
//     and safe to call from any thread.
//   - Degenerate inputs (empty, size mismatch, zero-variance target) never
//     throw unless size mismatch makes the call nonsensical — that's a
//     programmer error, not a data error.
//   - NaN is the sentinel for "undefined" (e.g. MAPE when y_true has zeros).

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace neurograph::eval {

struct RegressionMetrics {
    double mse                = 0.0;
    double rmse               = 0.0;
    double mae                = 0.0;
    double r2                 = 0.0;
    double explained_variance = 0.0;
    double mape               = 0.0;   // NaN if |y_true| < eps anywhere
    double q10_loss           = 0.0;   // pinball losses — NaN when not provided
    double q50_loss           = 0.0;
    double q90_loss           = 0.0;
    double residual_mean      = 0.0;
    double residual_stddev    = 0.0;
    std::size_t n             = 0;
};

/// Point-prediction metrics. `y_pred.size()` must equal `y_true.size()`.
/// Quantile losses in the returned struct are NaN (use the quantile overload).
RegressionMetrics evaluate_regression(const std::vector<double>& y_true,
                                      const std::vector<double>& y_pred);

/// Quantile-regression metrics. All four vectors must be the same length.
/// Point-prediction metrics are computed using the median prediction.
RegressionMetrics evaluate_regression_quantile(const std::vector<double>& y_true,
                                               const std::vector<double>& q10_pred,
                                               const std::vector<double>& q50_pred,
                                               const std::vector<double>& q90_pred);

} // namespace neurograph::eval
