// P1-09 — Wizard state transitions.
//
// Pure logic: no ImGui, no threads spawned here (the worker thread
// is owned by the UI layer). See wizard_state.hpp for the contract.

#include "neurograph/workbench/wizard_state.hpp"

#include <algorithm>
#include <utility>

namespace neurograph::workbench {

namespace {

void clear_plan(WizardState& s) {
    s.column_plan.clear();
    s.column_plan_error.clear();
}

} // namespace

// ---------------------------------------------------------------------------
// Step transitions
// ---------------------------------------------------------------------------

std::string can_advance(const WizardState& s) {
    switch (s.current_step) {
        case WizardStep::Load:
            if (!s.csv) {
                // User hasn't dropped a file yet. The UI greys the
                // button; this message isn't normally displayed, but
                // keeps the contract honest for headless callers.
                return "Load a CSV file to continue.";
            }
            if (s.target_column.empty()) {
                return "Pick a target column to continue.";
            }
            // Defense-in-depth: default_target_column should have
            // already picked a column, but if someone cleared it
            // manually we want the message to match what the UI shows.
            {
                const auto& cols = s.csv->columns;
                if (std::find(cols.begin(), cols.end(), s.target_column) == cols.end()) {
                    return "Selected target column is not in this file.";
                }
            }
            return "";
        case WizardStep::Quality:
            if (!s.dq_report) {
                return "Data quality report missing.";
            }
            if (has_blocking_error(s)) {
                return "Fix the blocking data quality error to continue.";
            }
            return "";
        case WizardStep::Train:
            // Can only move forward from Train once training has
            // finished successfully. The UI normally auto-advances on
            // success — this guard catches "user hit Next after a
            // failure" etc.
            if (s.status.load() != TrainingStatus::Succeeded) {
                return "Training must finish successfully before viewing results.";
            }
            return "";
        case WizardStep::Results:
            return "Already on the last step.";
    }
    return "";   // unreachable; GCC wants the terminator
}

bool advance(WizardState& s) {
    if (!can_advance(s).empty()) return false;
    switch (s.current_step) {
        case WizardStep::Load:    s.current_step = WizardStep::Quality; return true;
        case WizardStep::Quality: s.current_step = WizardStep::Train;   return true;
        case WizardStep::Train:   s.current_step = WizardStep::Results; return true;
        case WizardStep::Results: return false;
    }
    return false;
}

bool back(WizardState& s) {
    switch (s.current_step) {
        case WizardStep::Load:    return false;
        case WizardStep::Quality: s.current_step = WizardStep::Load;    return true;
        case WizardStep::Train:
            // If a run is in flight, the UI must cancel first — we
            // won't step off the Train screen while a worker thread
            // is still touching state. UI guarantees this by disabling
            // Back while status == Running; we assert defensively.
            if (s.status.load() == TrainingStatus::Running) return false;
            s.current_step = WizardStep::Quality;
            return true;
        case WizardStep::Results:
            // Going back from Results means "I want to change
            // something and retrain". Drop results so the UI's Train
            // step redisplays the initial "Start training" call to
            // action instead of the stale "last run finished" chrome.
            s.result.reset();
            s.status.store(TrainingStatus::Idle);
            s.error_message.clear();
            s.current_step = WizardStep::Train;
            return true;
    }
    return false;
}

void reset_to_load(WizardState& s) {
    s.current_step = WizardStep::Load;
    s.csv.reset();
    s.target_column.clear();
    s.load_error.clear();
    s.dq_report.reset();
    clear_plan(s);
    s.excluded_columns.clear();
    s.status.store(TrainingStatus::Idle);
    s.cancel_requested.store(false);
    {
        std::lock_guard<std::mutex> lk(s.progress_mutex);
        s.last_progress = {};
    }
    s.error_message.clear();
    s.result.reset();
}

void reset_to_train(WizardState& s) {
    // Keep csv / dq_report / fixes; just clear the just-finished run
    // so Step 3's UI re-opens on the "ready to train" state.
    s.current_step = WizardStep::Train;
    s.status.store(TrainingStatus::Idle);
    s.cancel_requested.store(false);
    {
        std::lock_guard<std::mutex> lk(s.progress_mutex);
        s.last_progress = {};
    }
    s.error_message.clear();
    s.result.reset();
}

// ---------------------------------------------------------------------------
// Step 1 — Load
// ---------------------------------------------------------------------------

std::string default_target_column(const CsvMetadata& meta) {
    if (meta.columns.empty()) return "";
    return meta.columns.back();
}

void set_csv(WizardState& s, CsvMetadata meta) {
    // Loading a new file invalidates every downstream computation.
    // The user expects "drop a new CSV → everything after Step 1
    // resets"; surfacing a stale DQ report from a different dataset
    // would be worse than making them scan again.
    s.load_error.clear();
    s.target_column = default_target_column(meta);
    s.csv = std::move(meta);
    s.dq_report.reset();
    clear_plan(s);
    s.excluded_columns.clear();
    s.status.store(TrainingStatus::Idle);
    s.error_message.clear();
    s.result.reset();
}

void clear_csv(WizardState& s) {
    s.csv.reset();
    s.target_column.clear();
    s.dq_report.reset();
    clear_plan(s);
    s.excluded_columns.clear();
    s.status.store(TrainingStatus::Idle);
    s.error_message.clear();
    s.result.reset();
}

bool set_target_column(WizardState& s, std::string col) {
    if (!s.csv) return false;
    const auto& cols = s.csv->columns;
    if (std::find(cols.begin(), cols.end(), col) == cols.end()) return false;
    s.target_column = std::move(col);
    // Changing the target invalidates the DQ report (class imbalance and
    // leakage are measured against the target) and the column plan.
    s.excluded_columns.erase(s.target_column);
    s.dq_report.reset();
    clear_plan(s);
    return true;
}

// ---------------------------------------------------------------------------
// Step 2 — Data quality
// ---------------------------------------------------------------------------

void set_dq_report(WizardState& s, dataqc::DataQualityReport report) {
    s.dq_report = std::move(report);
}

bool toggle_excluded_column(WizardState& s, const std::string& column) {
    if (!s.csv || column == s.target_column) return false;
    const auto& cols = s.csv->columns;
    if (std::find(cols.begin(), cols.end(), column) == cols.end()) return false;
    if (s.excluded_columns.erase(column) > 0) return false;
    s.excluded_columns.insert(column);
    return true;
}

bool is_excluded(const WizardState& s, const std::string& column) {
    return s.excluded_columns.count(column) > 0;
}

void set_task_override(WizardState& s, TaskOverride t) {
    if (s.task_override == t) return;
    s.task_override = t;
    clear_plan(s);
}

bool has_blocking_error(const WizardState& s) {
    if (!s.dq_report) return false;
    return s.dq_report->count_of(dataqc::Severity::Error) > 0;
}

// ---------------------------------------------------------------------------
// Step 3 — Training
// ---------------------------------------------------------------------------

int time_budget_seconds(TimeBudget b) {
    switch (b) {
        case TimeBudget::Quick:    return 30;
        case TimeBudget::Balanced: return 300;
        case TimeBudget::Best:     return 1800;
    }
    return 300;   // defensive default
}

orchestration::AutoOptions build_options(const WizardState& s,
                                         std::filesystem::path bundle_dir) {
    orchestration::AutoOptions opts;
    opts.time_budget_seconds = time_budget_seconds(s.time_budget);
    // The learning curve retrains the chosen model on 25/50/75% of the
    // rows (about 1.5x the cost of the model itself, and skipped when the
    // budget cannot cover it). Worth it except on Quick.
    opts.do_learning_curve   = (s.time_budget != TimeBudget::Quick);
    opts.thorough            = (s.time_budget == TimeBudget::Best);
    opts.do_export_bundle    = true;
    opts.bundle_out_dir      = std::move(bundle_dir);
    opts.exclude_columns.assign(s.excluded_columns.begin(), s.excluded_columns.end());
    switch (s.task_override) {
        case TaskOverride::Auto:           opts.task = preprocess::TaskChoice::Auto;           break;
        case TaskOverride::Classification: opts.task = preprocess::TaskChoice::Classification; break;
        case TaskOverride::Regression:     opts.task = preprocess::TaskChoice::Regression;     break;
    }
    // Seed stays at the orchestrator default (42), so the same file and
    // choices give the same model.
    return opts;
}

void on_progress(WizardState& s, const orchestration::Progress& p) {
    std::lock_guard<std::mutex> lk(s.progress_mutex);
    s.last_progress = p;
}

void on_training_finished(WizardState& s, util::Result<orchestration::AutoResult> r) {
    if (r.ok()) {
        s.result = std::move(r).value();
        s.error_message.clear();
        // Advance to Results automatically — the UI is watching
        // `status`, so flipping it last after the payload is ready
        // avoids a race where the Results step reads a half-populated
        // `result`.
        s.current_step = WizardStep::Results;
        s.status.store(TrainingStatus::Succeeded);
    } else {
        s.error_message = r.error().message;
        // Cancellation is surfaced separately so the Train step can
        // render "Cancelled — click Train to try again" rather than
        // the failure chrome with a red error icon.
        if (s.cancel_requested.load()) {
            s.status.store(TrainingStatus::Cancelled);
        } else {
            s.status.store(TrainingStatus::Failed);
        }
    }
    s.cancel_requested.store(false);
}

void request_cancel(WizardState& s) {
    s.cancel_requested.store(true);
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

const char* step_label(WizardStep s) {
    switch (s) {
        case WizardStep::Load:    return "Load data";
        case WizardStep::Quality: return "Data quality";
        case WizardStep::Train:   return "Train";
        case WizardStep::Results: return "Results";
    }
    return "?";
}

const char* time_budget_label(TimeBudget b) {
    switch (b) {
        case TimeBudget::Quick:    return "Quick (up to 30 s)";
        case TimeBudget::Balanced: return "Balanced (up to 5 min)";
        case TimeBudget::Best:     return "Best quality (up to 30 min, tries larger models)";
    }
    return "?";
}

} // namespace neurograph::workbench
