#pragma once

// P1-06 — Learning Curve Analysis.
//
// After a model trains, the most useful question the user can ask is:
// "Will collecting more data make this better?" The answer comes from
// re-training the same model on {25%, 50%, 75%, 100%} of the training
// data and watching how the val metric responds. If the curve is still
// climbing, more data helps. If it's flat, the user is at the data
// ceiling and should focus elsewhere (better features, different model).
//
// We fit `val_metric = a + b * log(n_samples)` to the four points and
// extrapolate one doubling beyond. The verdict is a plain-English
// sentence the wizard shows on the results screen.
//
// Design note — injectable trainer:
//
// The public API takes a `TrainerFn` callable rather than a
// `training::SupervisedConfig`. Two reasons:
//
//   1. Testability. A mock trainer that returns a programmed curve lets
//      us unit-test the slope math, verdict thresholds, and edge cases
//      without burning seconds per test on real GD.
//   2. Decoupling. `eval/` shouldn't depend on `training/`. The wizard
//      (or P1-14 AutoPipeline) supplies a thin adapter that wraps
//      `TrainingEngine`.

#include "neurograph/preprocess/pipeline.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace neurograph::eval {

struct LearningCurvePoint {
    double fraction     = 0.0;   // 0.25, 0.5, 0.75, 1.0
    int    n_samples    = 0;     // floor(fraction * n_train)
    double train_metric = 0.0;   // primary metric on the training subset
    double val_metric   = 0.0;   // primary metric on the full val set
};

struct LearningCurveReport {
    std::vector<LearningCurvePoint> points;

    /// Per-sample slope at the 100% point, i.e. d(val_metric)/dn evaluated
    /// at n = n_full. Equals `b / n_full` from the log fit.
    double slope_at_100pct = 0.0;

    /// Linear-in-log extrapolation to 2× the current training set size.
    /// Equals `val_metric[100%] + b * log(2)`.
    double projected_at_2x_data = 0.0;

    enum class Verdict {
        Plateau,        // more data won't help much
        Improving,      // more data should meaningfully help
        Inconclusive    // curve too noisy or insufficient points
    };
    Verdict     verdict = Verdict::Inconclusive;

    /// One-sentence plain-English summary the wizard shows.
    std::string message;
};

struct LearningCurveConfig {
    /// Training-set fractions to evaluate. Must be sorted ascending and
    /// end at 1.0; values outside (0, 1] are clamped/dropped.
    std::vector<double> fractions = {0.25, 0.5, 0.75, 1.0};

    /// Cap on training epochs per point. Keeps total extra compute bounded
    /// (4 × epochs_per_point ≤ 3× base training in practice).
    int                 epochs_per_point = 20;

    /// Seeds the deterministic shuffle used to draw subsets and is
    /// forwarded to the trainer for its own RNG.
    std::uint64_t       seed = 42;
};

/// Trainer callback. Implementations:
///   - Train a fresh model on (X_subset, y_subset) for `epochs` epochs.
///   - Evaluate on (X_val, y_val) and return (train_metric, val_metric).
///   - The metric is whatever the caller treats as "primary": accuracy
///     for classification, R² for regression. Compute it consistently.
///
/// `seed` is the seed for *this point's* training run — the curve
/// implementation derives a per-point seed from `LearningCurveConfig::seed`
/// so different points get reproducible-but-distinct RNG streams.
using TrainerFn = std::function<std::pair<double, double>(
    const std::vector<std::vector<double>>& X_subset,
    const std::vector<std::vector<double>>& y_subset,
    const std::vector<std::vector<double>>& X_val,
    const std::vector<std::vector<double>>& y_val,
    int                                     epochs,
    std::uint64_t                           seed)>;

/// Compute the learning curve and verdict.
///
/// Edge cases:
///   - n_train < 4 or fewer than 2 valid fractions → Inconclusive,
///     `message` explains.
///   - Training set is empty after subsampling (n_samples == 0) →
///     point is skipped.
///   - All val_metrics identical → b = 0 → Plateau.
///
/// Determinism: the same `seed` + same trainer + same dataset reproduces
/// identical reports.
LearningCurveReport compute_learning_curve(
    const preprocess::TransformedDataset& ds,
    bool                                  is_classification,
    const LearningCurveConfig&            lc_config,
    TrainerFn                             trainer);

} // namespace neurograph::eval
