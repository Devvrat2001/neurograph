// P1-09 — Step 4: Results & Export.
//
// Hero line + headline metric, expandable detail sections, export
// buttons. Export paths are placeholders right now — they point at
// the same bundle directory the orchestrator already wrote. A real
// "Export .exe" that bundles predict.exe alongside the bundle is a
// Phase-1-post ticket; for now the user gets the bundle directory
// and can run predict.exe against it via the CLI.

#include "wizard_ui.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <cstdio>
#include <variant>

#ifdef _WIN32
#  include <windows.h>
#  include <shellapi.h>
#endif

namespace neurograph::workbench {

namespace {

void open_folder_in_explorer(const std::filesystem::path& p) {
#ifdef _WIN32
    // ShellExecuteW via the "open" verb on a directory opens it in
    // Explorer. Best-effort — failure is silent; the user sees the
    // path next to the button either way.
    ShellExecuteW(nullptr, L"open", p.wstring().c_str(), nullptr, nullptr,
                  SW_SHOWNORMAL);
#else
    (void)p;
#endif
}

void render_classification_metrics(const eval::ClassificationMetrics& m) {
    if (ImGui::CollapsingHeader("Classification metrics",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Accuracy:     %.4f", m.accuracy);
        ImGui::Text("Macro F1:     %.4f", m.macro_f1);
        ImGui::Text("Weighted F1:  %.4f", m.weighted_f1);
        if (m.roc_auc_macro == m.roc_auc_macro) {   // NaN check
            ImGui::Text("ROC AUC:      %.4f", m.roc_auc_macro);
        }
        if (m.brier_score == m.brier_score) {
            ImGui::Text("Brier score:  %.4f", m.brier_score);
        }
        if (!m.per_class.empty()) {
            ImGui::Text("Per-class (%zu classes):", m.per_class.size());
            if (ImGui::BeginTable("per_class", 4,
                                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("class");
                ImGui::TableSetupColumn("precision");
                ImGui::TableSetupColumn("recall");
                ImGui::TableSetupColumn("f1");
                ImGui::TableHeadersRow();
                for (const auto& pc : m.per_class) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(pc.class_label.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", pc.precision);
                    ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", pc.recall);
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", pc.f1);
                }
                ImGui::EndTable();
            }
        }
    }
}

void render_regression_metrics(const eval::RegressionMetrics& m) {
    if (ImGui::CollapsingHeader("Regression metrics",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("R²:       %.4f", m.r2);
        ImGui::Text("RMSE:     %.4f", m.rmse);
        ImGui::Text("MAE:      %.4f", m.mae);
        ImGui::Text("Expl var: %.4f", m.explained_variance);
        ImGui::Text("n:        %zu",  m.n);
    }
}

void render_baseline_report(const eval::BaselineReport& b) {
    if (ImGui::CollapsingHeader("How this compares to simple baselines",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("%s", b.verdict.c_str());
        ImGui::Spacing();
        ImGui::Text("Lift vs best baseline: %+.4f", b.lift_vs_best_baseline);
        if (ImGui::BeginTable("baselines", 3,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("baseline");
            ImGui::TableSetupColumn("primary");
            ImGui::TableSetupColumn("secondary");
            ImGui::TableHeadersRow();
            for (const auto& r : b.baselines) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(r.name.c_str());
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.4f", r.primary_metric);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%.4f", r.secondary_metric);
            }
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "your model");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.4f", b.model_primary);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.4f", b.model_secondary);
            ImGui::EndTable();
        }
    }
}

} // namespace

void render_step_results(WizardState& state, WizardRuntime& /*rt*/) {
    if (!state.result) {
        ImGui::TextUnformatted("No result available. Go back and train.");
        return;
    }
    const auto& r = *state.result;

    // Hero card
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.9f, 0.4f, 1.0f));
    ImGui::TextWrapped("%s", r.summary_one_liner.c_str());
    ImGui::PopStyleColor();
    ImGui::Spacing();

    // Headline metric from the variant
    std::visit([&](auto&& m) {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, eval::ClassificationMetrics>) {
            ImGui::Text("Validation accuracy: %.4f", m.accuracy);
        } else {
            ImGui::Text("Validation R²: %.4f", m.r2);
        }
    }, r.metrics);

    ImGui::Separator();
    render_baseline_report(r.baselines);

    std::visit([&](auto&& m) {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, eval::ClassificationMetrics>) {
            render_classification_metrics(m);
        } else {
            render_regression_metrics(m);
        }
    }, r.metrics);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextUnformatted("Export");
    ImGui::Spacing();

    if (ImGui::Button("Open bundle folder", ImVec2(220, 36))) {
        if (!r.bundle_path.empty()) {
            open_folder_in_explorer(r.bundle_path);
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s",
        r.bundle_path.empty() ? "(no bundle)" : r.bundle_path.string().c_str());

    ImGui::Spacing();
    // These two buttons document the intended "Export .exe" /
    // "Export DLL" packaging flows — today the bundle folder
    // contains everything a user needs to hand to predict.exe
    // (P1-07) or neurograph_rt.dll (P1-08). The wrapper-packaging
    // scripts themselves are on the follow-up list.
    ImGui::BeginDisabled(true);
    if (ImGui::Button("Export as standalone .exe (coming soon)", ImVec2(340, 28))) {}
    if (ImGui::Button("Export DLL/SDK (coming soon)",            ImVec2(340, 28))) {}
    ImGui::EndDisabled();

    ImGui::Spacing();
    if (ImGui::Button("Train again with a different configuration", ImVec2(340, 28))) {
        reset_to_train(state);
    }
}

} // namespace neurograph::workbench
