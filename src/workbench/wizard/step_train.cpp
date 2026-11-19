// Step 3 — Train.
//
// Time allowance and task picker, one Train button, then a progress bar
// with a Stop button. Failure and cancellation come back to this screen
// with the reason and a way to try again.

#include "wizard_ui.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

namespace neurograph::workbench {

namespace {

const char* stage_label(orchestration::Stage s) {
    switch (s) {
        case orchestration::Stage::Scanning:      return "Checking the data";
        case orchestration::Stage::Preprocessing: return "Preparing the columns";
        case orchestration::Stage::AutoConfig:    return "Setting up";
        case orchestration::Stage::Training:      return "Training and comparing models";
        case orchestration::Stage::Evaluating:    return "Measuring on held-back rows";
        case orchestration::Stage::Baselining:    return "Comparing with simple guesses";
        case orchestration::Stage::LearningCurve: return "Checking whether more data would help";
        case orchestration::Stage::Packaging:     return "Saving the model";
        case orchestration::Stage::Done:          return "Done";
    }
    return "?";
}

const char* budget_explanation(TimeBudget b) {
    switch (b) {
        case TimeBudget::Quick:
            return "Trains a straight-line model, boosted decision trees and a neural "
                   "network, and keeps the best.";
        case TimeBudget::Balanced:
            return "The same models with more time each, plus a check of whether "
                   "collecting more data would help. Small files still finish in seconds.";
        case TimeBudget::Best:
            return "Also tries larger tree and network models. Worth it for files with "
                   "tens of thousands of rows.";
    }
    return "";
}

void render_options(WizardState& state) {
    ImGui::TextUnformatted("How long may training take at most?");
    for (const auto opt : {TimeBudget::Quick, TimeBudget::Balanced, TimeBudget::Best}) {
        if (ImGui::RadioButton(time_budget_label(opt), state.time_budget == opt)) {
            state.time_budget = opt;
        }
    }
    ImGui::TextDisabled("%s", budget_explanation(state.time_budget));
    ImGui::Spacing();

    ImGui::TextUnformatted("What should be predicted?");
    const char* opts[] = {"Decide from the column (recommended)",
                          "A category (classification)",
                          "A number (regression)"};
    int current = static_cast<int>(state.task_override);
    ImGui::SetNextItemWidth(320);
    if (ImGui::Combo("##task_override", &current, opts, 3)) {
        set_task_override(state, static_cast<TaskOverride>(current));
    }

    if (!state.excluded_columns.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("Leaving out %zu column(s) you unticked on the previous step.",
                            state.excluded_columns.size());
    }
}

void render_progress(const WizardState& state) {
    orchestration::Progress snap;
    {
        std::lock_guard<std::mutex> lk(state.progress_mutex);
        snap = state.last_progress;
    }
    ImGui::Text("%s", stage_label(snap.stage));
    ImGui::ProgressBar(static_cast<float>(snap.overall_fraction), ImVec2(-1.0f, 0.0f));
    if (!snap.message.empty()) ImGui::TextDisabled("%s", snap.message.c_str());
}

} // namespace

void render_step_train(WizardState& state, WizardRuntime& rt) {
    switch (state.status.load()) {
        case TrainingStatus::Idle: {
            render_options(state);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Train", ImVec2(220, 40))) start_training(state, rt);
            ImGui::SameLine();
            ImGui::TextDisabled("(you can stop at any time)");
            break;
        }
        case TrainingStatus::Running: {
            render_progress(state);
            ImGui::Spacing();
            if (ImGui::Button("Stop", ImVec2(160, 32))) request_cancel(state);
            ImGui::SameLine();
            ImGui::TextDisabled("Stops within a second or two.");
            break;
        }
        case TrainingStatus::Succeeded: {
            // Normally the wizard moves to Results on its own; this shows
            // when the user came back with Back.
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "The last run succeeded.");
            if (ImGui::Button("View results", ImVec2(200, 0))) advance(state);
            ImGui::SameLine();
            if (ImGui::Button("Train again", ImVec2(200, 0))) reset_to_train(state);
            break;
        }
        case TrainingStatus::Failed: {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Training did not complete:");
            ImGui::TextWrapped("%s", state.error_message.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Try again", ImVec2(200, 0))) reset_to_train(state);
            break;
        }
        case TrainingStatus::Cancelled: {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Training was stopped.");
            ImGui::Spacing();
            if (ImGui::Button("Train again", ImVec2(200, 0))) reset_to_train(state);
            break;
        }
    }
}

} // namespace neurograph::workbench
