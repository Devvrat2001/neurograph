// Step 2 — Data quality and columns.
//
// Scans the file against the chosen target and shows what was found and
// what will be done about it. Below that, every column with what the
// pipeline will do with it (number, category, date, left out and why) and
// a tick box to leave it out of the model.
//
// The scan and the column plan run synchronously the first time the step
// is shown (a few hundred milliseconds for large files), then are cached
// in WizardState until the file, target or task changes.

#include "wizard_ui.hpp"
#include "neurograph/dataqc/scanner.hpp"
#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <algorithm>
#include <exception>

namespace neurograph::workbench {

namespace {

ImVec4 color_for_severity(dataqc::Severity s) {
    switch (s) {
        case dataqc::Severity::Info:    return ImVec4(0.6f, 0.8f, 1.0f, 1.0f);
        case dataqc::Severity::Warning: return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
        case dataqc::Severity::Error:   return ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

const char* severity_label(dataqc::Severity s) {
    switch (s) {
        case dataqc::Severity::Info:    return "Note";
        case dataqc::Severity::Warning: return "Warning";
        case dataqc::Severity::Error:   return "Blocking";
    }
    return "?";
}

preprocess::TaskChoice task_choice(TaskOverride t) {
    switch (t) {
        case TaskOverride::Classification: return preprocess::TaskChoice::Classification;
        case TaskOverride::Regression:     return preprocess::TaskChoice::Regression;
        case TaskOverride::Auto:           break;
    }
    return preprocess::TaskChoice::Auto;
}

void compute_if_needed(WizardState& state) {
    if (!state.csv) return;
    if (!state.dq_report) {
        dataqc::ScanOptions opts;
        opts.target_column = state.target_column;
        set_dq_report(state, dataqc::scan_csv(state.csv->path, opts));
    }
    if (state.column_plan.empty() && state.column_plan_error.empty()) {
        preprocess::PipelineConfig cfg;
        cfg.target_column = state.target_column;
        cfg.task          = task_choice(state.task_override);
        try {
            state.column_plan = preprocess::plan_columns(state.csv->path, cfg);
        } catch (const std::exception& e) {
            state.column_plan_error = e.what();
        }
    }
}

void render_findings(const dataqc::DataQualityReport& dq) {
    if (dq.findings.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "No problems found.");
        return;
    }
    for (const auto& f : dq.findings) {
        ImGui::PushStyleColor(ImGuiCol_Text, color_for_severity(f.severity));
        ImGui::TextUnformatted(severity_label(f.severity));
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextWrapped("%s", f.message.c_str());
        if (!f.remedy.empty()) {
            ImGui::Indent();
            ImGui::TextDisabled("%s", f.remedy.c_str());
            ImGui::Unindent();
        }
    }
}

void render_columns(WizardState& state) {
    if (!state.column_plan_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                           state.column_plan_error.c_str());
        return;
    }
    if (state.column_plan.empty()) return;

    ImGui::TextWrapped("Untick a column to leave it out, for example one that is only "
                       "known after the outcome, or an identifier the automatic check "
                       "missed.");
    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    const float height = ImGui::GetTextLineHeightWithSpacing() *
                         static_cast<float>(std::min<std::size_t>(state.column_plan.size() + 1, 14));
    if (!ImGui::BeginTable("columns", 4, flags, ImVec2(0.0f, height))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Use", ImGuiTableColumnFlags_WidthFixed, 36.0f);
    ImGui::TableSetupColumn("Column");
    ImGui::TableSetupColumn("Used as");
    ImGui::TableSetupColumn("Why", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableHeadersRow();

    for (const auto& c : state.column_plan) {
        ImGui::TableNextRow();
        ImGui::PushID(c.name.c_str());
        const bool is_target  = (c.role == "target");
        const bool auto_out   = (c.role == "left out");
        const bool user_out   = is_excluded(state, c.name);

        ImGui::TableSetColumnIndex(0);
        if (is_target) {
            ImGui::TextDisabled("-");
        } else {
            bool use = !auto_out && !user_out;
            ImGui::BeginDisabled(auto_out);
            if (ImGui::Checkbox("##use", &use)) toggle_excluded_column(state, c.name);
            ImGui::EndDisabled();
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(c.name.c_str());
        ImGui::TableSetColumnIndex(2);
        if (is_target)      ImGui::TextColored(ImVec4(0.3f, 0.75f, 1.0f, 1.0f), "to predict");
        else if (user_out)  ImGui::TextDisabled("left out");
        else                ImGui::TextUnformatted(c.role.c_str());
        ImGui::TableSetColumnIndex(3);
        if (user_out)            ImGui::TextDisabled("left out by you");
        else if (!c.note.empty()) ImGui::TextWrapped("%s", c.note.c_str());
        ImGui::PopID();
    }
    ImGui::EndTable();
}

} // namespace

void render_step_quality(WizardState& state, WizardRuntime& /*rt*/) {
    compute_if_needed(state);
    if (!state.dq_report) {
        ImGui::TextUnformatted("Scanning your data...");
        return;
    }
    const auto& dq = *state.dq_report;

    ImGui::TextWrapped("%s", dq.summary_line.c_str());
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("What the check found", ImGuiTreeNodeFlags_DefaultOpen)) {
        render_findings(dq);
    }
    if (has_blocking_error(state)) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                           "Training cannot start until the blocking problem is fixed in "
                           "the file (or a different column is chosen to predict).");
    }
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Columns", ImGuiTreeNodeFlags_DefaultOpen)) {
        render_columns(state);
    }
}

} // namespace neurograph::workbench
