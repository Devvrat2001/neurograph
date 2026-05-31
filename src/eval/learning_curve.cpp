#include "neurograph/eval/learning_curve.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace neurograph::eval {

namespace {

// Verdict thresholds — applied to b * log(2), the projected lift one
// data-doubling out. Keep these in lockstep with the spec strings below.
constexpr double kPlateauThreshold   = 0.005;   // < 0.5% lift
constexpr double kImprovingThreshold = 0.020;   // > 2.0% lift

constexpr const char* kMsgPlateau =
    "More data will likely not help much (projected +<0.5% at 2× data).";
constexpr const char* kMsgImproving =
    "Collecting more data should meaningfully improve accuracy "
    "(projected +2% or more at 2× data).";
constexpr const char* kMsgInconclusive =
    "Curve is noisy; run again with a larger eval set to get a clearer signal.";
constexpr const char* kMsgTooFewSamples =
    "Not enough training data to estimate a learning curve "
    "(need at least 4 rows).";
constexpr const char* kMsgTooFewPoints =
    "Not enough valid fractions produced points to fit a curve.";
constexpr const char* kMsgEmptyVal =
    "Cannot evaluate — val set is empty.";

// Sanitize the user's fractions: drop ≤0 and >1, dedup, sort, ensure 1.0
// is present so we always have a 100% anchor.
std::vector<double> sanitize_fractions(std::vector<double> fs) {
    fs.erase(std::remove_if(fs.begin(), fs.end(),
                            [](double f) { return f <= 0.0 || f > 1.0; }),
             fs.end());
    std::sort(fs.begin(), fs.end());
    fs.erase(std::unique(fs.begin(), fs.end(),
                         [](double a, double b) {
                             return std::abs(a - b) < 1e-9;
                         }),
             fs.end());
    if (fs.empty() || std::abs(fs.back() - 1.0) > 1e-9) {
        fs.push_back(1.0);
    }
    return fs;
}

// Returns a permutation of [0, n) seeded by `seed`. Used to draw the
// training subset for each point — the same seed picks the same order,
// so the 25% subset is always a prefix of the 50% subset, etc., matching
// the "incremental data" narrative.
std::vector<std::size_t> shuffled_indices(std::size_t n, std::uint64_t seed) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    return idx;
}

// Least-squares fit y = a + b*x. Returns (a, b). Caller guarantees
// xs.size() == ys.size() >= 2 and xs has nonzero variance.
std::pair<double, double> linear_fit(const std::vector<double>& xs,
                                     const std::vector<double>& ys) {
    const std::size_t n = xs.size();
    const double mean_x = std::accumulate(xs.begin(), xs.end(), 0.0) / n;
    const double mean_y = std::accumulate(ys.begin(), ys.end(), 0.0) / n;
    double sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = xs[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (ys[i] - mean_y);
    }
    if (sxx < 1e-18) {
        // Degenerate: all xs equal. Slope undefined; treat as flat.
        return {mean_y, 0.0};
    }
    const double b = sxy / sxx;
    const double a = mean_y - b * mean_x;
    return {a, b};
}

LearningCurveReport make_inconclusive(const char* msg) {
    LearningCurveReport r;
    r.verdict = LearningCurveReport::Verdict::Inconclusive;
    r.message = msg;
    return r;
}

} // namespace

LearningCurveReport compute_learning_curve(
    const preprocess::TransformedDataset& ds,
    bool                                  /*is_classification*/,
    const LearningCurveConfig&            lc_config,
    TrainerFn                             trainer)
{
    if (!trainer) {
        spdlog::warn("compute_learning_curve called with null trainer");
        return make_inconclusive(kMsgInconclusive);
    }
    if (ds.X_val.empty()) {
        return make_inconclusive(kMsgEmptyVal);
    }
    const std::size_t n_train = ds.X_train.size();
    if (n_train < 4) {
        return make_inconclusive(kMsgTooFewSamples);
    }

    const auto fractions = sanitize_fractions(lc_config.fractions);
    const auto perm      = shuffled_indices(n_train, lc_config.seed);

    LearningCurveReport report;
    report.points.reserve(fractions.size());

    for (std::size_t fi = 0; fi < fractions.size(); ++fi) {
        const double f = fractions[fi];
        const int    n = static_cast<int>(std::floor(f * static_cast<double>(n_train)));
        if (n <= 0) continue;

        // Build the subset by taking the first `n` indices from the shared
        // shuffle. This nesting (25% ⊂ 50% ⊂ 75% ⊂ 100%) matches what the
        // user actually experiences when they collect more rows — each
        // point measures the marginal value of the *added* data, not a
        // fresh random draw.
        std::vector<std::vector<double>> X_sub, y_sub;
        X_sub.reserve(static_cast<std::size_t>(n));
        y_sub.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const std::size_t src = perm[static_cast<std::size_t>(i)];
            X_sub.push_back(ds.X_train[src]);
            y_sub.push_back(ds.y_train[src]);
        }

        // Per-point seed: derive deterministically from the master seed
        // so re-runs reproduce, but each fraction gets its own RNG stream.
        const std::uint64_t point_seed =
            lc_config.seed + 0x9E3779B97F4A7C15ull * (fi + 1);

        const auto [train_m, val_m] =
            trainer(X_sub, y_sub, ds.X_val, ds.y_val,
                    lc_config.epochs_per_point, point_seed);

        LearningCurvePoint p;
        p.fraction     = f;
        p.n_samples    = n;
        p.train_metric = train_m;
        p.val_metric   = val_m;
        report.points.push_back(p);
    }

    if (report.points.size() < 2) {
        report.verdict = LearningCurveReport::Verdict::Inconclusive;
        report.message = kMsgTooFewPoints;
        return report;
    }

    // Fit val_metric = a + b * log(n_samples). log here is the natural
    // log; the verdict thresholds are calibrated against b * log(2)
    // (also natural-log) so the units stay consistent.
    std::vector<double> log_n, ys;
    log_n.reserve(report.points.size());
    ys.reserve(report.points.size());
    for (const auto& p : report.points) {
        log_n.push_back(std::log(static_cast<double>(p.n_samples)));
        ys.push_back(p.val_metric);
    }
    const auto [a, b] = linear_fit(log_n, ys);

    const double n_full     = static_cast<double>(report.points.back().n_samples);
    const double val_at_100 = report.points.back().val_metric;
    report.slope_at_100pct      = (n_full > 0.0) ? (b / n_full) : 0.0;
    report.projected_at_2x_data = val_at_100 + b * std::log(2.0);

    const double lift = b * std::log(2.0);
    if (lift < kPlateauThreshold) {
        report.verdict = LearningCurveReport::Verdict::Plateau;
        report.message = kMsgPlateau;
    } else if (lift > kImprovingThreshold) {
        report.verdict = LearningCurveReport::Verdict::Improving;
        report.message = kMsgImproving;
    } else {
        report.verdict = LearningCurveReport::Verdict::Inconclusive;
        report.message = kMsgInconclusive;
    }
    return report;
}

} // namespace neurograph::eval
