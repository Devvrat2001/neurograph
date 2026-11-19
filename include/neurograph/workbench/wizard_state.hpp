#pragma once

// P1-09 — Wizard state machine.
//
// The wizard is a 4-step flow: Load → Quality → Train → Results. The UI
// layer (ImGui widgets + D3D11 + window plumbing) lives in
// `src/workbench/wizard/` and is not unit-testable in a sensible way.
// All the decision logic — "can the user advance?", "did applying this
// fix change the pipeline?", "what time budget does 'Balanced' mean?" —
// lives here, in a pure-data struct with pure transition functions, so
// the behavior can be exercised end-to-end under GTest without running
// a graphics context.
//
// Concretely this header depends only on the runtime + dataqc +
// orchestration types. No ImGui, no threading primitives in the
// *declaration* (state contains atomics so the UI thread and the
// worker thread can share it safely — see individual field docs).
//
// Thread model:
//
//   - Main (UI) thread: mutates via the transition functions below,
//     reads everything for rendering.
//   - Training worker: fires the orchestrator's ProgressCb/CancelCb;
//     these write to `last_progress` and `status` through helpers
//     (`on_progress`, `on_training_finished`). The fields they touch
//     are either atomics or live under `progress_mutex` — see docs on
//     the fields themselves.
//
// Design note on using std::optional: every "produced by the previous
// step" field is optional precisely because "back" and "reset_to_load"
// are legal. Making them optional means `back` can just clear them
// rather than caring about holding a default-constructed stand-in that
// the UI would then mis-render.

#include "neurograph/dataqc/report.hpp"
#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/util/result.hpp"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace neurograph::workbench {

/// Which step of the wizard the user is currently viewing.
enum class WizardStep {
    Load     = 0,
    Quality  = 1,
    Train    = 2,
    Results  = 3,
};

/// User-facing time allowance. Maps to seconds via `time_budget_seconds`.
/// These are upper limits: each model stops on its own once it stops
/// improving, so small files finish in seconds whatever is picked.
enum class TimeBudget {
    Quick    = 0,   // up to 30 s, no learning curve
    Balanced = 1,   // up to 5 min (default)
    Best     = 2,   // up to 30 min, also tries larger models
};

/// Task-kind override. Auto decides from the target column (text or a
/// few whole numbers: categories; otherwise a number). The other two
/// force it; the pipeline refuses to treat text as a number.
enum class TaskOverride {
    Auto           = 0,
    Classification = 1,
    Regression     = 2,
};

/// Lifecycle of the background orchestrator run (Step 3). The UI
/// reads this every frame to drive the progress widget; the worker
/// thread writes it through `on_training_finished`.
enum class TrainingStatus {
    Idle       = 0,
    Running    = 1,
    Succeeded  = 2,
    Failed     = 3,
    Cancelled  = 4,
};

/// Summary of a CSV file loaded from Step 1. Produced by
/// `peek_csv_metadata`; does not own the file data (just the header
/// columns and a short preview).
///
/// This type deliberately does not carry the full dataset — the
/// orchestrator re-reads the CSV when training starts. The preview is
/// UI-only; 5 rows is enough for the user to confirm "yes, that's my
/// data" without the wizard holding a million rows in RAM.
struct CsvMetadata {
    std::filesystem::path                   path;
    std::vector<std::string>                columns;
    std::size_t                             row_count    = 0;   // data rows, excluding header
    std::vector<std::vector<std::string>>   preview_rows;        // up to 5 data rows
};

/// Complete wizard state. One instance lives on AppState; the UI
/// reads from it every frame, and transition functions below mutate
/// it in response to user input.
///
/// Copyable by value for unit tests and undo-style "snapshot + compare"
/// scenarios, except for the atomic / mutex fields which are
/// intentionally not copyable — those are explicit carve-outs via
/// deleted copy/move, see below.
struct WizardState {
    WizardStep current_step = WizardStep::Load;

    // Step 1 — Load ------------------------------------------------------
    std::optional<CsvMetadata> csv;
    std::string                target_column;   // empty until user picks / default applied
    std::string                load_error;      // non-empty when most recent load failed

    // Step 2 — Data Quality ---------------------------------------------
    std::optional<dataqc::DataQualityReport> dq_report;

    /// What the pipeline will do with each column (used / left out and
    /// why), for the current file, target and task. Empty until the
    /// Quality step computes it; cleared when any of those change.
    std::vector<preprocess::ColumnDecision>  column_plan;
    std::string                              column_plan_error;

    /// Columns the user chose to leave out of the model.
    std::set<std::string>                    excluded_columns;

    // Step 3 — Train -----------------------------------------------------
    TimeBudget      time_budget    = TimeBudget::Balanced;
    TaskOverride    task_override  = TaskOverride::Auto;
    bool            show_advanced  = false;

    /// Live status written by the worker thread. Reads in the UI are
    /// single-load-and-done so an atomic is enough; we never need a
    /// multi-field consistent snapshot here.
    std::atomic<TrainingStatus> status{TrainingStatus::Idle};
    /// Flipped true by the "Stop" button; the orchestrator's
    /// CancelCb polls it between stages. `atomic<bool>` rather than a
    /// mutex because this is a single boolean.
    std::atomic<bool>           cancel_requested{false};

    /// Latest progress the orchestrator pushed. Guarded by
    /// `progress_mutex` because `Progress` contains a std::string —
    /// an atomic would be wrong for non-trivially-copyable types.
    mutable std::mutex          progress_mutex;
    orchestration::Progress     last_progress;

    /// Error message when status == Failed. Written once by the
    /// worker thread before flipping status; the UI reads it only
    /// after observing a terminal status so no lock is needed.
    std::string                 error_message;

    // Step 4 — Results ---------------------------------------------------
    /// The orchestrator's return value. Populated before status flips
    /// to Succeeded; read-only to the UI thereafter.
    std::optional<orchestration::AutoResult> result;

    /// Working directory the wizard suggests for "Save model bundle".
    /// Defaults on construction to a per-run temp location; the user
    /// can override via the export dialog on Step 4.
    std::filesystem::path export_dir;

    // --- Non-copyable because of the mutex + atomics ---
    WizardState() = default;
    WizardState(const WizardState&)            = delete;
    WizardState& operator=(const WizardState&) = delete;
    WizardState(WizardState&&)                 = delete;
    WizardState& operator=(WizardState&&)      = delete;
};

// ---------------------------------------------------------------------------
// Step transitions
// ---------------------------------------------------------------------------

/// Is it legal for the user to click "Next" right now? Returns an
/// empty string when yes, otherwise a reason string to surface next to
/// the disabled button ("Pick a target column", "Fix the blocking
/// data quality error", etc.). Never returns nullptr.
std::string can_advance(const WizardState& s);

/// Move to the next step if `can_advance` would allow. Returns true
/// on transition. No-op (returns false) when already on Results or
/// when blocked.
bool advance(WizardState& s);

/// Move to the previous step. True on transition, false when already
/// on Load. "Back" never discards prior state — the user can go back
/// to Load, change the target column, and forward again without
/// losing their Quality-step "Fix automatically" picks.
bool back(WizardState& s);

/// Reset to Load and clear everything after it. Used by "Start over".
void reset_to_load(WizardState& s);

/// Reset to Train and clear Results. Used by "Train again" after a
/// successful run — keeps the loaded file + DQ report + fixes.
void reset_to_train(WizardState& s);

// ---------------------------------------------------------------------------
// Step 1 — Load helpers
// ---------------------------------------------------------------------------

/// Accept a CSV metadata struct (loaded via the CSV-peek helper in a
/// separate translation unit) and default the target column to the
/// last column per spec. Clears any prior DQ report + fixes because
/// they're stale for the new dataset.
void set_csv(WizardState& s, CsvMetadata meta);

/// Clear the loaded CSV and everything downstream. Leaves the current
/// step alone — the UI decides whether to jump back to Load.
void clear_csv(WizardState& s);

/// Override the auto-picked target column. Rejected (returns false)
/// when `col` is not in `s.csv->columns`.
bool set_target_column(WizardState& s, std::string col);

/// Default target column for a freshly loaded CSV: the last column.
/// Returns "" for an empty columns list.
std::string default_target_column(const CsvMetadata& meta);

// ---------------------------------------------------------------------------
// Step 2 — Data quality helpers
// ---------------------------------------------------------------------------

/// Attach a quality report.
void set_dq_report(WizardState& s, dataqc::DataQualityReport report);

/// Leave a column out of the model, or put it back. The target cannot
/// be left out; returns false (and changes nothing) for it or for a
/// name that is not in the file. Otherwise returns the new state:
/// true = left out.
bool toggle_excluded_column(WizardState& s, const std::string& column);

bool is_excluded(const WizardState& s, const std::string& column);

/// Choose classification / regression / automatic. Clears the column
/// plan, which depends on it.
void set_task_override(WizardState& s, TaskOverride t);

/// Does the report contain any Error-severity finding? The UI uses
/// this to disable the Next button with a blocking-error message.
bool has_blocking_error(const WizardState& s);

// ---------------------------------------------------------------------------
// Step 3 — Training knobs / orchestrator inputs
// ---------------------------------------------------------------------------

/// Map a time-budget pick to the orchestrator's `time_budget_seconds`.
int time_budget_seconds(TimeBudget b);

/// Build the orchestrator's AutoOptions from the current wizard state.
/// The caller passes in a bundle output directory; this function
/// wires every user-facing pick into the orchestrator struct.
orchestration::AutoOptions build_options(const WizardState& s,
                                         std::filesystem::path bundle_dir);

/// Called from the worker thread via the orchestrator's ProgressCb.
/// Thread-safe: uses `progress_mutex` to store the string message.
void on_progress(WizardState& s, const orchestration::Progress& p);

/// Called from the worker thread once after the orchestrator returns.
/// Writes `result`, `error_message`, and flips `status` atomically.
void on_training_finished(WizardState& s, util::Result<orchestration::AutoResult> r);

/// Ask the orchestrator to cancel (just sets the flag — the worker
/// thread polls it). Safe from any thread.
void request_cancel(WizardState& s);

// ---------------------------------------------------------------------------
// Display helpers (pure formatting, no side effects)
// ---------------------------------------------------------------------------

/// Human name for a step, e.g. "Load data".
const char* step_label(WizardStep s);

/// Human name for a time budget with its rough wall-time cap.
const char* time_budget_label(TimeBudget b);

} // namespace neurograph::workbench
