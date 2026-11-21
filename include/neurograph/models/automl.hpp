#pragma once

// Automatic model selection.
//
// Trains each model family on the same train/validation split inside one
// wall-clock budget and keeps whichever has the lowest validation loss
// (log-loss for classification, squared error for regression). If
// averaging the two best beats the single best by a clear margin, the
// blend is kept instead.
//
// Nothing here looks at the test split. The caller reports final
// numbers on rows this function never saw.

#include "neurograph/models/model.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace neurograph::models {

/// One row of the leaderboard shown on the results screen.
struct Candidate {
    std::string name;                  // "Gradient-boosted trees"
    std::string kind;                  // "gbt_v1"
    std::string detail;                // "143 trees"
    double      val_loss   = std::numeric_limits<double>::infinity();
    double      val_score  = 0.0;      // accuracy (classification) or R² (regression)
    double      seconds    = 0.0;
    int         iterations = 0;
    bool        chosen          = false;
    bool        stopped_by_time = false;
};

struct AutoFitOptions {
    /// Total wall-clock budget for all candidates. 0 means unlimited; each
    /// family then runs until its own early stopping.
    double        time_budget_seconds = 300.0;
    std::uint64_t seed                = 42;

    bool use_linear = true;
    bool use_gbt    = true;
    bool use_mlp    = true;
    bool try_blend  = true;

    /// Also try a larger configuration of the tree and network families.
    /// Worth it only with a generous budget.
    bool thorough   = false;
};

struct AutoFitResult {
    /// The selected model; null when nothing could be trained (for
    /// example, cancelled before the first candidate finished).
    std::shared_ptr<const Model> model;
    std::vector<Candidate>       leaderboard;   // in the order trained
    bool                         cancelled = false;

    /// Retrains the selected recipe (the same family and settings, or the
    /// same two-model blend) on other rows. The learning curve uses it to
    /// ask "how good would this model be with less data?".
    std::function<FitReport(const TrainSet&, const FitControl&)> refit;

    /// Seconds the selected recipe took (both members, for a blend).
    double refit_seconds = 0.0;

    /// True when the selected recipe stopped because its time ran out
    /// rather than because it had stopped improving.
    bool   refit_stopped_by_time = false;

    /// Why this model was picked, in one plain sentence.
    std::string choice_reason;
};

AutoFitResult auto_fit(const TrainSet& data,
                       const AutoFitOptions& opts,
                       std::function<bool()> cancelled = {},
                       std::function<void(double, const std::string&)> progress = {});

/// Accuracy (classification) or R² (regression) of `model` on (X, y).
double evaluate_score(const Model& model,
                      const Matrix& X,
                      const std::vector<int>&    y_cls,
                      const std::vector<double>& y_reg);

} // namespace neurograph::models
