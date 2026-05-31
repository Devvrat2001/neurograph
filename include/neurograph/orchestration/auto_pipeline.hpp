#pragma once

// P1-14 — AutoPipeline Orchestrator.
//
// Single top-level entry point that drives the full happy path the wizard
// runs from a "Train" click:
//
//   scan → validate → preprocess → auto-configure → train → evaluate →
//   baseline → learning curve → package
//
// Every stage already exists as a focused module (P1-01..P1-13). This
// orchestrator is the glue: it sequences them, normalizes their error
// shapes into a single Result<AutoResult>, fires progress callbacks for
// the wizard's progress bar, polls a cooperative cancel callback between
// stages, and writes the resulting bundle to disk in a format that
// `predict.exe` (P1-07) and the C-ABI DLL (P1-08) can re-open.
//
// What the wizard needed before this ticket:
//
//   if (auto dq = scan_csv(...)) { ... } else show_error(...);
//   show_progress("Preprocessing");
//   if (cancel) return;
//   auto pipe = fit_pipeline(...);
//   show_progress("Training");
//   if (cancel) return;
//   auto trained = train(...);
//   ... eight more stages, each with its own cancel/progress dance ...
//
// What it needs now:
//
//   auto r = orchestration::run(csv, target, opts, on_progress, is_cancelled);
//   if (r.ok()) show_results(r.value()); else show_error(r.error().message);
//
// All sequencing, time budgeting, cancellation, and packaging live here
// so the wizard stays a thin shell over this function.
//
// Trainer note (interim):
//
// The "training" stage uses the same in-house linear-probe trainer that
// powers the baseline comparison module. This is a deliberate stand-in:
// it makes the orchestrator end-to-end runnable today, which unblocks
// P1-09 (wizard) and P1-07 (predict.exe). Swapping in a real
// `TrainingEngine` adapter is a localized change in
// `src/orchestration/auto_pipeline.cpp` — the public API here doesn't
// need to move when that swap happens.
//
// Bundle layout (`bundle_out_dir`):
//
//   pipeline.ngpipe   — preprocess::save_pipeline output
//   model.ngm         — minimal JSON: {kind, weights, biases, classes...}
//                       Schema lives in auto_pipeline.cpp; predict.exe
//                       reads the same struct.
//   manifest.json     — reporting::save_manifest output (P1-13). Hashes
//                       pipeline+model so verify_against catches any
//                       post-training tampering.
//   card.md           — reporting::save_model_card output (P1-12).
//
// Headlessness contract: this header includes nothing from `workbench/`,
// `imgui/`, or any GUI library. Callers wire ImGui (or anything else)
// only via the ProgressCb / CancelCb function objects.

#include "neurograph/dataqc/report.hpp"
#include "neurograph/eval/baselines.hpp"
#include "neurograph/eval/classification.hpp"
#include "neurograph/eval/learning_curve.hpp"
#include "neurograph/eval/regression.hpp"
#include "neurograph/util/result.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <variant>

namespace neurograph::orchestration {

/// Stages fired in order by `run`. The Done sentinel is dispatched
/// once after Packaging so callers can flip their progress bar to 100%
/// before they look at the result struct.
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

/// One progress update. Fires at the start of each stage and at any
/// internal milestone the stage chooses to surface (training epochs,
/// learning-curve points). `overall_fraction` is the wizard's
/// monotonically-increasing master progress bar; `stage_fraction` is the
/// inner secondary bar.
struct Progress {
    Stage       stage           = Stage::Scanning;
    double      stage_fraction   = 0.0;   // 0..1 within current stage
    double      overall_fraction = 0.0;   // 0..1 across the full run
    std::string message;                  // one-line status for the UI
};

/// Caller-supplied progress sink. Called from the orchestrator's thread
/// (typically a worker thread the wizard spawned). The callback must be
/// thread-safe with respect to the caller's UI loop. Pass `{}` to
/// suppress progress events.
using ProgressCb = std::function<void(const Progress&)>;

/// Caller-supplied cancellation poll. Returns true when the user has
/// asked to abort. Polled between stages and at coarse internal points
/// (per training epoch, per learning-curve point). Cooperative cancel
/// returns a `Cancelled` error from `run`. Pass `{}` for "never cancel".
using CancelCb = std::function<bool()>;

/// Knobs the wizard surfaces or that the orchestrator's defaults set.
struct AutoOptions {
    /// Soft cap on total wall time. Internally split across stages by
    /// fixed weights (see `kStageWeights` in auto_pipeline.cpp). Stages
    /// that finish under their slice donate the remainder forward.
    int                   time_budget_seconds = 300;

    /// When false, skips the LearningCurve stage entirely and leaves
    /// `AutoResult::curve` at its default-constructed (Inconclusive)
    /// value. Useful for fast iteration during development.
    bool                  do_learning_curve   = true;

    /// When false, skips Packaging — `bundle_path` will be empty. The
    /// in-memory result is otherwise complete.
    bool                  do_export_bundle    = true;

    /// Required when `do_export_bundle` is true. Created if missing.
    /// Existing files in this directory are overwritten without
    /// confirmation — caller is responsible for "do you want to
    /// overwrite?" prompts.
    std::filesystem::path bundle_out_dir;

    /// Master seed. Forwarded into every stochastic stage so the run is
    /// reproducible bit-for-bit when re-invoked with the same inputs +
    /// seed on the same machine.
    std::uint64_t         seed = 42;
};

/// Everything the wizard's results screen needs to render. Held by
/// value rather than by handle so the caller can drop the orchestrator
/// instance immediately.
struct AutoResult {
    /// From the P1-01 scanner. Includes findings the wizard echoes on
    /// the results screen (e.g. "we dropped 2 constant columns").
    dataqc::DataQualityReport   dq;

    /// From the P1-03 baseline module. `verdict` is the one-liner.
    eval::BaselineReport        baselines;

    /// Exactly one alternative is set, matching `dq` task kind. The
    /// wizard `std::visit`s on it to render the right metric grid.
    std::variant<eval::ClassificationMetrics, eval::RegressionMetrics> metrics;

    /// From the P1-06 learning-curve module. When
    /// `AutoOptions::do_learning_curve` was false this is a
    /// default-constructed report (verdict == Inconclusive, no points).
    eval::LearningCurveReport   curve;

    /// Path to the bundle directory (same as `AutoOptions::bundle_out_dir`)
    /// when packaging ran. Empty when packaging was skipped or failed.
    /// Use this with `predict.exe` (P1-07) and the C-ABI DLL (P1-08).
    std::filesystem::path       bundle_path;

    /// One-line summary the wizard's success toast shows. Composed from
    /// `baselines.verdict` plus the headline metric.
    std::string                 summary_one_liner;
};

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

/// Execute the full pipeline.
///
/// On success the bundle (when `do_export_bundle`) is on disk and the
/// returned `AutoResult` is fully populated. On failure the error
/// message is suitable for showing the user directly (e.g. "Cannot
/// train — input has 3 rows, minimum is 50.") rather than a stack
/// trace. Categories of failures the caller should expect:
///
///   - "Cancelled by user." — cooperative cancel fired.
///   - "Cannot train — <reason>." — DQ scan rejected the dataset.
///   - "Pipeline fit failed: <detail>." — preprocessing exception.
///   - "Bundle write failed: <path>: <reason>." — packaging IO error.
///
/// Determinism: same `csv`, same `target_column`, same `opts.seed`, same
/// build → byte-identical `model.ngm`, byte-identical metrics. The
/// manifest's `timestamp_iso8601` is the only varying field, by design.
util::Result<AutoResult>
run(const std::filesystem::path& csv,
    const std::string&           target_column,
    const AutoOptions&           opts,
    ProgressCb                   progress = {},
    CancelCb                     cancel   = {});

} // namespace neurograph::orchestration
