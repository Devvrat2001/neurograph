#pragma once

// AutoPipeline — the one call behind the wizard's Train button and the
// neurograph_train command:
//
//   scan → preprocess → choose a model → evaluate → compare with
//   baselines → learning curve → write the bundle
//
// Every stage reports progress and polls the cancel callback, and every
// failure comes back as one sentence a user can act on (see `run`).
//
// How the rows are used. The file is split once, before anything is
// fitted:
//
//   train  (70%) — every candidate model learns from these rows;
//   check  (15%) — early stopping, and the choice between candidates;
//   test   (15%) — used once, at the end, for the reported scores.
//
// So the scores shown to the user come from rows that neither training nor
// model selection saw. (A very small file can end up with no test rows;
// the scores then come from the check rows and `evaluated_on` says so.)
//
// Bundle layout (`bundle_out_dir`):
//
//   pipeline.ngpipe — the fitted preprocessing (preprocess::save_pipeline)
//   model.ngm       — schema 3: the chosen model (models::Model::to_json)
//   manifest.json   — file hashes, settings and every model tried
//   card.md         — the model report, in plain language
//
// Headless: nothing here includes GUI code. Callers wire a UI in through
// the ProgressCb / CancelCb function objects.

#include "neurograph/dataqc/report.hpp"
#include "neurograph/eval/baselines.hpp"
#include "neurograph/eval/classification.hpp"
#include "neurograph/eval/learning_curve.hpp"
#include "neurograph/eval/regression.hpp"
#include "neurograph/models/automl.hpp"
#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/util/result.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace neurograph::orchestration {

/// Stages fired in order by `run`. Done is dispatched once at the end so
/// callers can set their progress bar to 100% before reading the result.
enum class Stage {
    Scanning,
    Preprocessing,
    AutoConfig,
    Training,
    Evaluating,
    Baselining,
    LearningCurve,
    Packaging,
    Done,
};

/// One progress update. `overall_fraction` never decreases during a run;
/// `stage_fraction` is progress within the current stage.
struct Progress {
    Stage       stage            = Stage::Scanning;
    double      stage_fraction   = 0.0;   // 0..1 within current stage
    double      overall_fraction = 0.0;   // 0..1 across the full run
    std::string message;                  // one-line status for the UI
};

/// Called from the thread running `run` (normally a worker thread), so
/// it must be safe with respect to the caller's UI loop. May be empty.
using ProgressCb = std::function<void(const Progress&)>;

/// Polled between stages and often within them (every training epoch or
/// boosting round). Return true to stop; `run` then returns
/// "Cancelled by user.". May be empty.
using CancelCb = std::function<bool()>;

struct AutoOptions {
    /// Soft limit on total wall time, in seconds. About 80% goes to
    /// training the candidate models, the rest to evaluation and the
    /// learning curve. Small files finish far sooner: each model stops on
    /// its own once it stops improving. 0 or less means no limit.
    int                   time_budget_seconds = 300;

    /// When false the LearningCurve stage is skipped and `curve` keeps its
    /// default (Inconclusive, no points).
    bool                  do_learning_curve   = true;

    /// When false nothing is written and `bundle_path` stays empty; the
    /// in-memory result is otherwise complete.
    bool                  do_export_bundle    = true;

    /// Required when `do_export_bundle` is true; created if missing.
    /// Existing bundle files in it are overwritten without asking.
    std::filesystem::path bundle_out_dir;

    /// Same file, same options and same seed give an identical model.ngm,
    /// as long as no model hits the time limit.
    std::uint64_t         seed = 42;

    /// Columns to leave out of the model, by header name.
    std::vector<std::string> exclude_columns;

    /// Predict a category or a number. Auto decides from the target column.
    preprocess::TaskChoice task = preprocess::TaskChoice::Auto;

    /// Also try larger tree and network configurations. Worth it with a
    /// budget of several minutes.
    bool                  thorough = false;
};

/// How much the model relies on one input column, measured by shuffling
/// that column's values between rows and seeing how much worse it does.
struct ColumnImportance {
    std::string column;
    /// Increase in log-loss (classification) or in squared error as a
    /// share of the target's variance (regression).
    double      loss_increase = 0.0;
    /// The same thing as a fall in accuracy or R².
    double      score_drop    = 0.0;
    /// `loss_increase` scaled so the most important column is 1.
    double      relative      = 0.0;
};

/// Everything the results screen needs, held by value.
struct AutoResult {
    /// Data quality scan of the input file.
    dataqc::DataQualityReport   dq;

    /// The model against trivial baselines, on the same rows as `metrics`.
    eval::BaselineReport        baselines;

    /// Scores on the rows named by `evaluated_on`. Holds the alternative
    /// that matches the task.
    std::variant<eval::ClassificationMetrics, eval::RegressionMetrics> metrics;

    /// "Would more data help?" Default (Inconclusive, no points) when the
    /// stage was skipped; `curve.message` then says why.
    eval::LearningCurveReport   curve;

    /// The bundle directory when one was written, otherwise empty.
    std::filesystem::path       bundle_path;

    /// One line for the success message.
    std::string                 summary_one_liner;

    /// Every model tried, in the order trained; exactly one is `chosen`.
    std::vector<models::Candidate> leaderboard;

    /// Name and one-line description of the chosen model, and why it won.
    std::string                 model_name;
    std::string                 model_description;
    std::string                 model_choice_reason;

    /// Input columns, most important first. Empty if cancelled first.
    std::vector<ColumnImportance> importance;

    /// What happened to each CSV column (used, left out, and why).
    std::vector<preprocess::ColumnDecision> columns;

    /// Rows in each split, and rows skipped because the target was blank.
    int n_train = 0;
    int n_val   = 0;
    int n_test  = 0;
    int rows_dropped_missing_target = 0;

    /// "test" (rows held back from everything) or "check" (the rows used
    /// to choose the model; only when the file was too small for a test
    /// split), and how many rows that was.
    std::string evaluated_on;
    int         n_evaluated = 0;

    /// Accuracy or R² of the chosen model on the check rows.
    double      check_score = 0.0;
};

/// Run the whole pipeline.
///
/// On success the bundle (when requested) is on disk and the result is
/// complete. On failure the message can be shown to the user as it is:
///
///   "Cancelled by user."
///   "Cannot train — <reason>."      the data cannot be trained on
///   "Training failed: <detail>"     no model could be fitted
///   "Bundle write failed: <path>: <reason>"
util::Result<AutoResult>
run(const std::filesystem::path& csv,
    const std::string&           target_column,
    const AutoOptions&           opts,
    ProgressCb                   progress = {},
    CancelCb                     cancel   = {});

} // namespace neurograph::orchestration
