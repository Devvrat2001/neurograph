#include "neurograph/eval/regression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace neurograph::eval {

namespace {

constexpr double kQuietNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kMapeEps  = 1e-9;

void ensure_same_size(std::size_t a, std::size_t b, const char* what) {
    if (a != b) {
        throw std::invalid_argument(
            std::string("evaluate_regression: size mismatch (") + what + ")");
    }
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Single-quantile pinball (a.k.a. tilted-absolute) loss.
//   L_q(y, ŷ) = max(q*(y - ŷ), (q-1)*(y - ŷ))
// Averaged over the sample.
double pinball(const std::vector<double>& y,
               const std::vector<double>& y_hat,
               double q) {
    if (y.empty()) return kQuietNaN;
    double sum = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double diff = y[i] - y_hat[i];
        sum += std::max(q * diff, (q - 1.0) * diff);
    }
    return sum / static_cast<double>(y.size());
}

} // namespace

RegressionMetrics evaluate_regression(const std::vector<double>& y_true,
                                      const std::vector<double>& y_pred) {
    ensure_same_size(y_true.size(), y_pred.size(), "y_true vs y_pred");

    RegressionMetrics m;
    m.n = y_true.size();
    m.q10_loss = m.q50_loss = m.q90_loss = kQuietNaN;

    if (m.n == 0) {
        m.mse = m.rmse = m.mae = kQuietNaN;
        m.r2 = m.explained_variance = kQuietNaN;
        m.mape = kQuietNaN;
        m.residual_mean = m.residual_stddev = kQuietNaN;
        return m;
    }

    // First pass: mean of truth (needed for R²), residuals for MAPE.
    const double y_mean = mean(y_true);

    double sse = 0.0;      // sum squared error
    double sae = 0.0;      // sum abs error
    double ss_tot = 0.0;   // total sum of squares
    double res_sum = 0.0;  // sum of residuals (for mean residual)
    bool   mape_defined = true;
    double mape_sum = 0.0;

    for (std::size_t i = 0; i < m.n; ++i) {
        const double err = y_pred[i] - y_true[i];
        const double abs_err = std::abs(err);
        sse += err * err;
        sae += abs_err;
        const double dev = y_true[i] - y_mean;
        ss_tot += dev * dev;
        res_sum += err;

        if (mape_defined) {
            const double denom = std::abs(y_true[i]);
            if (denom < kMapeEps) {
                mape_defined = false;
            } else {
                mape_sum += abs_err / denom;
            }
        }
    }

    const double n_d = static_cast<double>(m.n);
    m.mse  = sse / n_d;
    m.rmse = std::sqrt(m.mse);
    m.mae  = sae / n_d;
    m.mape = mape_defined ? (mape_sum / n_d) : kQuietNaN;

    // R²: 1 - SSE/SS_tot. Degenerate when target is constant — return NaN
    // so the caller can surface "target has no variance" rather than a
    // meaningless "infinite improvement."
    if (ss_tot < kMapeEps) {
        m.r2 = kQuietNaN;
        m.explained_variance = kQuietNaN;
    } else {
        m.r2 = 1.0 - sse / ss_tot;
        // Explained variance = 1 - Var(residuals) / Var(truth). Same answer
        // as R² only when residuals have zero mean; diverges for biased
        // predictors and is the more useful metric in that case.
        const double res_mean = res_sum / n_d;
        double res_var = 0.0;
        for (std::size_t i = 0; i < m.n; ++i) {
            const double e = (y_pred[i] - y_true[i]) - res_mean;
            res_var += e * e;
        }
        res_var /= n_d;
        const double y_var = ss_tot / n_d;
        m.explained_variance = 1.0 - res_var / y_var;
    }

    // Residual mean/stddev — useful for spotting systematic bias.
    m.residual_mean = res_sum / n_d;
    double rss = 0.0;
    for (std::size_t i = 0; i < m.n; ++i) {
        const double e = (y_pred[i] - y_true[i]) - m.residual_mean;
        rss += e * e;
    }
    m.residual_stddev = (m.n > 1) ? std::sqrt(rss / static_cast<double>(m.n - 1)) : 0.0;

    return m;
}

RegressionMetrics evaluate_regression_quantile(const std::vector<double>& y_true,
                                               const std::vector<double>& q10_pred,
                                               const std::vector<double>& q50_pred,
                                               const std::vector<double>& q90_pred) {
    ensure_same_size(y_true.size(), q10_pred.size(), "y_true vs q10_pred");
    ensure_same_size(y_true.size(), q50_pred.size(), "y_true vs q50_pred");
    ensure_same_size(y_true.size(), q90_pred.size(), "y_true vs q90_pred");

    // Point-metrics use the median forecast — that's the standard convention
    // and gives a meaningful "central prediction" summary.
    RegressionMetrics m = evaluate_regression(y_true, q50_pred);
    m.q10_loss = pinball(y_true, q10_pred, 0.1);
    m.q50_loss = pinball(y_true, q50_pred, 0.5);
    m.q90_loss = pinball(y_true, q90_pred, 0.9);
    return m;
}

} // namespace neurograph::eval
