// P1-09 — Step 2: Data Quality.
//
// Renders the scanner report grouped by severity. Each finding that
// the preprocessor knows how to fix gets a "Fix automatically"
// checkbox that toggles the code into `state.auto_fixes_applied`.
//
// This step runs the scanner synchronously on enter. For a 1M-row
// dataset that's a few hundred milliseconds — blocking the UI for
// one frame is acceptable per the "renders in <16 ms with a 1M row
// dataset loaded" acceptance criterion (that criterion is about the
// *render* loop, not the initial scan which is a one-shot).

#include "wizard_ui.hpp"
#include "neurograph/dataqc/scanner.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

namespace neurograph::workbench {

namespace {

ImVec4 color_for_severity(dataqc::Severity s) {
    switch (s) {
        case dataqc::Severity::Info:    return ImVec4(0.6f, 0.8f, 1.0f, 1.0f);
        case dataqc::Severity::Warning: return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
        case dataqc::Severity::Error:   return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

const char* severity_label(dataqc::Severity s) {
    switch (s) {
        case dataqc::Severity::Info:    return "INFO";
        case dataqc::Severity::Warning: return "WARN";
        case dataqc::Severity::Error:   return "ERROR";
    }
    return "?";
}

void run_scan_if_needed(WizardState& state) {
    if (state.dq_report) return;
    if (!state.csv) return;

    // Note: scan_csv treats the last column as the target. If the
    // user chose a different target from the Load step, the
    // class-imbalance / leakage findings will be against the last
    // column instead — a known limitation of the P1-01 scanner that
    // we'll tighten once the scanner takes a target-column arg.
    dataqc::ScanOptions opts;
    auto report = dataqc::scan_csv(state.csv->path, opts);
    set_dq_report(state, std::move(report));
}

} // namespace

void render_step_quality(WizardState& state, WizardRuntime& /*rt*/) {
    run_scan_if_needed(state);
    if (!state.dq_report) {
        // Defensive — should never happen after run_scan_if_needed.
        ImGui::TextUnformatted("Scanning your data...");
        return;
    }
    const auto& dq = *state.dq_report;

    ImGui::TextWrapped("%s", dq.summary_line.c_str());
    ImGui::Spacing();
    ImGui::Text("Rows: %d   |   Columns: %d   |   Class imbalance: %.2f",
                dq.n_rows, dq.n_cols, dq.class_imbalance_ratio);
    ImGui::Spacing();

    if (dq.findings.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f),
                           "Everything checks out. You can proceed.");
    }

    for (const auto& f : dq.findings) {
        ImGui::PushStyleColor(ImGuiCol_Text, color_for_severity(f.severity));
        ImGui::Text("[%s] %s", severity_label(f.severity), f.code.c_str());
        ImGui::PopStyleColor();

        if (!f.column.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(column: %s)", f.column.c_str());
        }
        ImGui::TextWrapped("  %s", f.message.c_str());
        if (!f.remedy.empty()) {
            ImGui::TextDisabled("  Suggested fix: %s", f.remedy.c_str());
        }

        if (fix_is_applicable(f)) {
            bool applied = state.auto_fixes_applied.count(f.code) > 0;
            // Per-code unique-id suffix so two findings with the same
            // code (rare but possible — one per column) each have
            // their own ImGui ID.
            const std::string id = "##fix_" + f.code + "_" + f.column;
            if (ImGui::Checkbox(("Fix automatically" + id).c_str(), &applied)) {
                toggle_fix(state, f.code);
            }
        }
        ImGui::Separator();
    }

    if (has_blocking_error(state)) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                           "Training is disabled until the blocking errors are resolved.");
    }
}

} // namespace neurograph::workbench
