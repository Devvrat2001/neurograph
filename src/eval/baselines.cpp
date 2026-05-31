#include "neurograph/eval/baselines.hpp"
#include "neurograph/eval/classification.hpp"
#include "neurograph/eval/regression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace neurograph::eval {

namespace {

// ---------------------------------------------------------------------------
// Small helpers reused across baselines.
// ---------------------------------------------------------------------------

// argmax of a probability / one-hot vector. Returns -1 on empty.
int argmax(const std::vector<double>& v) {
    if (v.empty()) return -1;
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

// Convert a list of one-hot rows to integer class indices.
std::vector<int> argmax_rows(const std::vector<std::vector<double>>& rows) {
    std::vector<int> out;
    out.reserve(rows.size());
    for (const auto& r : rows) out.push_back(argmax(r));
    return out;
}

// Per-class frequency over a list of one-hot rows. Returns the class
// counts as ints; index = class id.
std::vector<int> class_counts(const std::vector<std::vector<double>>& rows,
                              int n_classes) {
    std::vector<int> c(n_classes, 0);
    for (const auto& r : rows) {
        const int k = argmax(r);
        if (k >= 0 && k < n_classes) ++c[k];
    }
    return c;
}

// Build dummy class_labels — the metrics module only uses size + names for
// display, not for the math. Names are "0", "1", ... so logging stays sane.
std::vector<std::string> dummy_labels(int n) {
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(std::to_string(i));
    return out;
}

// Build a fixed K-dim one-hot probability vector at index `k`.
std::vector<double> one_hot(int k, int K) {
    std::vector<double> v(static_cast<std::size_t>(K), 0.0);
    if (k >= 0 && k < K) v[k] = 1.0;
    return v;
}

// Flatten a list of single-element rows into a scalar vector. Used for
// regression metrics, which take std::vector<double>.
std::vector<double> flatten_scalar(const std::vector<std::vector<double>>& rows) {
    std::vector<double> out;
    out.reserve(rows.size());
    for (const auto& r : rows) out.push_back(r.empty() ? 0.0 : r[0]);
    return out;
}

// Mean / stddev over a scalar vector. Welford-free; vector is small.
struct MeanStd { double mean = 0.0; double stddev = 0.0; };
MeanStd mean_std(const std::vector<double>& v) {
    MeanStd r;
    if (v.empty()) return r;
    double sum = 0.0;
    for (double x : v) sum += x;
    r.mean = sum / v.size();
    if (v.size() < 2) return r;
    double ssq = 0.0;
    for (double x : v) {
        const double d = x - r.mean;
        ssq += d * d;
    }
    r.stddev = std::sqrt(ssq / (v.size() - 1));
    return r;
}

// ---------------------------------------------------------------------------
// Linear probe — classification (multiclass softmax via 100 GD iters)
// ---------------------------------------------------------------------------
//
// Model: z_k = W_k · x + b_k,  p = softmax(z),  loss = -Σ y_k · log p_k.
// W is K×D, b is K. Full-batch GD for determinism; 100 iters is enough on
// a normalized feature matrix to converge to within a percent or two of
// the optimum on typical Phase-1 datasets.
//
// Returns p̂ on the eval set (one row per eval sample).
struct LinearProbeFitCls {
    std::vector<std::vector<double>> probs_eval;     // |eval| × K
    std::vector<double>              loss_history;   // length = iters
};

LinearProbeFitCls linear_probe_classification(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>&                 y_train_idx,
    const std::vector<std::vector<double>>& X_eval,
    int                                     n_classes,
    int                                     n_iters       = 100,
    double                                  learning_rate = 0.1) {

    LinearProbeFitCls out;
    if (X_train.empty() || X_eval.empty() || n_classes < 2) {
        out.probs_eval.assign(X_eval.size(),
                              std::vector<double>(n_classes,
                                                  1.0 / std::max(1, n_classes)));
        return out;
    }
    const std::size_t N = X_train.size();
    const std::size_t D = X_train.front().size();
    const int         K = n_classes;

    // W[k][d], b[k]. Zero init keeps the iteration reproducible without an
    // RNG and is fine for a 1-layer linear model.
    std::vector<std::vector<double>> W(K, std::vector<double>(D, 0.0));
    std::vector<double>              b(K, 0.0);

    auto softmax_inplace = [](std::vector<double>& z) {
        const double m = *std::max_element(z.begin(), z.end());
        double s = 0.0;
        for (double& v : z) { v = std::exp(v - m); s += v; }
        if (s > 0.0) for (double& v : z) v /= s;
    };

    out.loss_history.reserve(n_iters);
    for (int it = 0; it < n_iters; ++it) {
        // Forward: probs[N][K], plus accumulate loss.
        std::vector<std::vector<double>> probs(N, std::vector<double>(K, 0.0));
        double loss = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<double> z(K, 0.0);
            for (int k = 0; k < K; ++k) {
                double s = b[k];
                for (std::size_t d = 0; d < D; ++d) s += W[k][d] * X_train[i][d];
                z[k] = s;
            }
            softmax_inplace(z);
            probs[i] = z;
            const int y = y_train_idx[i];
            const double p = std::max(z[y], 1e-12);
            loss += -std::log(p);
        }
        loss /= N;
        out.loss_history.push_back(loss);

        // Backward: full-batch gradient.
        std::vector<std::vector<double>> dW(K, std::vector<double>(D, 0.0));
        std::vector<double>              db(K, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            for (int k = 0; k < K; ++k) {
                const double err = probs[i][k] - ((y_train_idx[i] == k) ? 1.0 : 0.0);
                db[k] += err;
                for (std::size_t d = 0; d < D; ++d) dW[k][d] += err * X_train[i][d];
            }
        }
        const double inv_n = 1.0 / N;
        for (int k = 0; k < K; ++k) {
            b[k] -= learning_rate * inv_n * db[k];
            for (std::size_t d = 0; d < D; ++d) W[k][d] -= learning_rate * inv_n * dW[k][d];
        }
    }

    // Predict on eval.
    out.probs_eval.assign(X_eval.size(), std::vector<double>(K, 0.0));
    for (std::size_t i = 0; i < X_eval.size(); ++i) {
        std::vector<double> z(K, 0.0);
        for (int k = 0; k < K; ++k) {
            double s = b[k];
            for (std::size_t d = 0; d < D; ++d) s += W[k][d] * X_eval[i][d];
            z[k] = s;
        }
        softmax_inplace(z);
        out.probs_eval[i] = std::move(z);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Linear probe — regression (ridge-flavored, GD)
// ---------------------------------------------------------------------------
//
// Model: ŷ = w · x + b. Loss = mean((ŷ - y)²) + l2_reg · ||w||².
// l2_reg is small to avoid driving w to zero on small features.
struct LinearProbeFitReg {
    std::vector<double> y_eval_pred;
    std::vector<double> loss_history;
};

LinearProbeFitReg linear_probe_regression(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<double>&              y_train,
    const std::vector<std::vector<double>>& X_eval,
    int                                     n_iters       = 100,
    double                                  learning_rate = 0.05,
    double                                  l2_reg        = 1e-4) {

    LinearProbeFitReg out;
    if (X_train.empty() || X_eval.empty()) {
        out.y_eval_pred.assign(X_eval.size(), 0.0);
        return out;
    }
    const std::size_t N = X_train.size();
    const std::size_t D = X_train.front().size();
    std::vector<double> w(D, 0.0);
    double              b = 0.0;

    out.loss_history.reserve(n_iters);
    for (int it = 0; it < n_iters; ++it) {
        // Forward + loss.
        double loss = 0.0;
        std::vector<double> errs(N, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            double yhat = b;
            for (std::size_t d = 0; d < D; ++d) yhat += w[d] * X_train[i][d];
            const double e = yhat - y_train[i];
            errs[i] = e;
            loss += e * e;
        }
        loss /= N;
        double reg = 0.0;
        for (double wi : w) reg += wi * wi;
        loss += l2_reg * reg;
        out.loss_history.push_back(loss);

        // Backward.
        std::vector<double> dw(D, 0.0);
        double              db_acc = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            db_acc += errs[i];
            for (std::size_t d = 0; d < D; ++d) dw[d] += errs[i] * X_train[i][d];
        }
        const double inv_n = 1.0 / N;
        b -= learning_rate * 2.0 * inv_n * db_acc;
        for (std::size_t d = 0; d < D; ++d) {
            w[d] -= learning_rate * (2.0 * inv_n * dw[d] + 2.0 * l2_reg * w[d]);
        }
    }

    out.y_eval_pred.resize(X_eval.size());
    for (std::size_t i = 0; i < X_eval.size(); ++i) {
        double yhat = b;
        for (std::size_t d = 0; d < D; ++d) yhat += w[d] * X_eval[i][d];
        out.y_eval_pred[i] = yhat;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Verdict — exact strings come from the spec; keep verbatim.
// ---------------------------------------------------------------------------
std::string make_verdict(double lift) {
    if (lift > 0.15) return "Strong result — model is substantially better than simple baselines.";
    if (lift > 0.05) return "Good result — model beats baselines by a meaningful margin.";
    if (lift > 0.0)  return "Marginal — model is only slightly better than guessing. Consider more data or features.";
    return "⚠ Model is not learning — it performs no better than a trivial baseline.";
}

// Compute (primary, secondary) for a classification baseline given its
// prob predictions on the val set.
std::pair<double, double> cls_metrics(
    const std::vector<int>&                 y_true,
    const std::vector<std::vector<double>>& probs,
    int                                     n_classes) {
    if (y_true.empty()) return {0.0, 0.0};
    const auto m = evaluate_classification(y_true, probs, dummy_labels(n_classes));
    return { m.accuracy, m.macro_f1 };
}

// Compute (primary=R², secondary=RMSE) for a regression baseline.
std::pair<double, double> reg_metrics(const std::vector<double>& y_true,
                                       const std::vector<double>& y_pred) {
    if (y_true.empty()) return {0.0, 0.0};
    const auto m = evaluate_regression(y_true, y_pred);
    return { m.r2, m.rmse };
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
BaselineReport compare_to_baselines(
    const preprocess::TransformedDataset&    ds,
    const std::vector<std::vector<double>>&  model_val_preds,
    bool                                     is_classification,
    std::uint64_t                            seed) {

    BaselineReport report;

    // Sanity: empty val set is a hard stop, no baselines computable.
    if (ds.X_val.empty() || ds.y_val.empty()) {
        report.verdict = "Cannot evaluate — val set is empty.";
        return report;
    }

    if (is_classification) {
        const int n_classes = ds.y_val.front().empty()
                            ? 0
                            : static_cast<int>(ds.y_val.front().size());
        if (n_classes < 2) {
            report.verdict = "Cannot evaluate — val set has only one class.";
            return report;
        }

        const auto y_val_idx   = argmax_rows(ds.y_val);
        const auto y_train_idx = argmax_rows(ds.y_train);
        const auto train_counts = class_counts(ds.y_train, n_classes);

        // Single-class val set: every val row has the same target. Metrics
        // (especially F1) become degenerate / NaN. Bail with the dedicated
        // verdict so the wizard surfaces a clear message.
        const bool val_single_class = std::all_of(
            y_val_idx.begin() + 1, y_val_idx.end(),
            [&](int x) { return x == y_val_idx.front(); });
        if (val_single_class) {
            report.verdict = "Cannot evaluate — val set has only one class.";
            return report;
        }

        // --- Majority baseline ---
        const int majority = static_cast<int>(
            std::distance(train_counts.begin(),
                          std::max_element(train_counts.begin(), train_counts.end())));
        std::vector<std::vector<double>> majority_probs(
            ds.X_val.size(), one_hot(majority, n_classes));
        BaselineResult maj;
        maj.name = "majority";
        std::tie(maj.primary_metric, maj.secondary_metric) =
            cls_metrics(y_val_idx, majority_probs, n_classes);
        report.baselines.push_back(maj);

        // --- Random baseline (sample by training prior, fixed seed) ---
        std::vector<double> prior(n_classes, 0.0);
        const int total = std::accumulate(train_counts.begin(), train_counts.end(), 0);
        if (total > 0) {
            for (int k = 0; k < n_classes; ++k) {
                prior[k] = static_cast<double>(train_counts[k]) / total;
            }
        } else {
            std::fill(prior.begin(), prior.end(), 1.0 / n_classes);
        }
        std::mt19937_64 rng(seed);
        std::discrete_distribution<int> dist(prior.begin(), prior.end());
        std::vector<std::vector<double>> random_probs(ds.X_val.size());
        for (std::size_t i = 0; i < ds.X_val.size(); ++i) {
            random_probs[i] = one_hot(dist(rng), n_classes);
        }
        BaselineResult rnd;
        rnd.name = "random";
        std::tie(rnd.primary_metric, rnd.secondary_metric) =
            cls_metrics(y_val_idx, random_probs, n_classes);
        report.baselines.push_back(rnd);

        // --- Linear probe ---
        const auto probe = linear_probe_classification(
            ds.X_train, y_train_idx, ds.X_val, n_classes);
        BaselineResult lin;
        lin.name = "linear_probe";
        std::tie(lin.primary_metric, lin.secondary_metric) =
            cls_metrics(y_val_idx, probe.probs_eval, n_classes);
        report.baselines.push_back(lin);

        // --- Model ---
        std::tie(report.model_primary, report.model_secondary) =
            cls_metrics(y_val_idx, model_val_preds, n_classes);

    } else {
        const auto y_val   = flatten_scalar(ds.y_val);
        const auto y_train = flatten_scalar(ds.y_train);
        const auto y_pred  = flatten_scalar(model_val_preds);

        const auto stats = mean_std(y_train);

        // --- Mean baseline ---
        std::vector<double> mean_pred(y_val.size(), stats.mean);
        BaselineResult m;
        m.name = "mean";
        std::tie(m.primary_metric, m.secondary_metric) = reg_metrics(y_val, mean_pred);
        report.baselines.push_back(m);

        // --- Random baseline ---
        std::vector<double> rand_pred(y_val.size(), 0.0);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nrm(stats.mean, std::max(stats.stddev, 1e-9));
        for (std::size_t i = 0; i < y_val.size(); ++i) rand_pred[i] = nrm(rng);
        BaselineResult r;
        r.name = "random";
        std::tie(r.primary_metric, r.secondary_metric) = reg_metrics(y_val, rand_pred);
        report.baselines.push_back(r);

        // --- Linear probe ---
        const auto probe = linear_probe_regression(ds.X_train, y_train, ds.X_val);
        BaselineResult l;
        l.name = "linear_probe";
        std::tie(l.primary_metric, l.secondary_metric) = reg_metrics(y_val, probe.y_eval_pred);
        report.baselines.push_back(l);

        // --- Model ---
        std::tie(report.model_primary, report.model_secondary) = reg_metrics(y_val, y_pred);
    }

    // --- Lift + verdict ---
    double best_baseline = -std::numeric_limits<double>::infinity();
    for (const auto& b : report.baselines) {
        // R² and accuracy can be negative (R²) or in [0,1] (accuracy). Use
        // them at face value — anything negative for R² is "worse than
        // mean" so it correctly never wins the max.
        if (std::isfinite(b.primary_metric) && b.primary_metric > best_baseline) {
            best_baseline = b.primary_metric;
        }
    }
    if (!std::isfinite(best_baseline)) best_baseline = 0.0;
    report.lift_vs_best_baseline = report.model_primary - best_baseline;
    report.verdict = make_verdict(report.lift_vs_best_baseline);
    return report;
}

} // namespace neurograph::eval
