#pragma once

// P1-03 — Baseline comparison.
//
// After training, we want to answer one user-facing question: "Did my
// neural net actually learn anything?" The answer is meaningless without
// reference points. This module computes three trivial baselines on the
// same val split and produces a single-sentence verdict the wizard shows
// on the final screen:
//
//   "Strong result — model is substantially better than simple baselines."
//   "Good result — model beats baselines by a meaningful margin."
//   "Marginal — model is only slightly better than guessing."
//   "⚠ Model is not learning — it performs no better than a trivial baseline."
//
// Without this, an 80% accuracy could mean "great 4-class classifier" or
// "garbage majority-class predictor on a 90/10 dataset" — and the user has
// no way to tell.
//
// Three baselines:
//   - majority      : predict argmax(class freq)  (cls)
//                     predict mean(y_train)        (reg)
//   - random        : sample from class prior      (cls)
//                     sample from N(mean, stddev)  (reg)
//   - linear_probe  : 100 iters of GD on a 1-layer linear model. Measures
//                     whether the data is linearly separable — if the probe
//                     already gets 95%, the neural net adds nothing.

#include "neurograph/preprocess/pipeline.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace neurograph::eval {

struct BaselineResult {
    std::string name;
    double      primary_metric   = 0.0;  // accuracy (cls) or R² (reg)
    double      secondary_metric = 0.0;  // macro F1 (cls) or RMSE (reg)
};

struct BaselineReport {
    std::vector<BaselineResult> baselines;        // one entry per baseline
    double                      model_primary         = 0.0;
    double                      model_secondary       = 0.0;
    /// model_primary − max(baseline.primary_metric), the linear probe
    /// included. Positive means the model beats every baseline.
    double                      lift_vs_best_baseline = 0.0;
    /// model_primary − the better of the two trivial baselines (majority
    /// or mean, and random). Positive means the model learned something.
    /// This, not lift_vs_best_baseline, decides whether the verdict says
    /// "not learning": matching a straight-line model is not failure.
    double                      lift_vs_trivial       = 0.0;
    /// True when the straight-line probe scores within one point of the
    /// model, i.e. the pattern in the data is mostly linear.
    bool                        linear_is_enough      = false;
    /// One-sentence plain-English summary the wizard shows the user.
    std::string                 verdict;
};

/// Compare the model's val-set predictions against three trivial baselines
/// trained on the same split.
///
/// Inputs:
///   ds                — full transformed dataset (we use X_train, X_val,
///                       y_train, y_val).
///   model_val_preds   — model's predictions on the val set. Same shape as
///                       y_val: K-dim probability vectors for classification,
///                       1-dim point estimates for regression.
///   is_classification — explicit flag because the dataset shape alone
///                       isn't always enough to tell (e.g. binary regression).
///
/// Determinism: random baseline + linear probe both use a fixed seed, so
/// repeated calls produce identical reports.
///
/// Edge cases:
///   - Empty val set → verdict "Cannot evaluate — val set is empty."
///   - Single-class val (cls) → verdict "Cannot evaluate — val set has only
///     one class." Baselines still computed where defined.
BaselineReport compare_to_baselines(
    const preprocess::TransformedDataset&    ds,
    const std::vector<std::vector<double>>&  model_val_preds,
    bool                                     is_classification,
    std::uint64_t                            seed = 1234567ull);

/// The same comparison on explicit rows. Baselines learn from
/// (X_train, y_train) and everything is scored on (X_eval, y_eval); the
/// orchestrator passes the untouched test rows here. `y_*` rows are
/// one-hot (classification) or one value (regression), as in
/// TransformedDataset.
BaselineReport compare_to_baselines(
    const std::vector<std::vector<double>>&  X_train,
    const std::vector<std::vector<double>>&  y_train,
    const std::vector<std::vector<double>>&  X_eval,
    const std::vector<std::vector<double>>&  y_eval,
    const std::vector<std::vector<double>>&  model_eval_preds,
    bool                                     is_classification,
    std::uint64_t                            seed = 1234567ull);

} // namespace neurograph::eval
