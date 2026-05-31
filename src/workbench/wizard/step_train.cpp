// P1-09 — Step 3: Train.
//
// Budget + task-override picker up front. One big "Train" button.
// Once running, a progress bar + current stage + "Stop" button.
// Failure and cancellation both land back on this screen with a
// colored banner and a "Train" button the user can click again.

#include "wizard_ui.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

namespace neurograph::workbench {

namespace {

const char* stage_label(orchestration::Stage s) {
    switch (s) {
        case orchestration::Stage::Scanning:      return "Scanning data";
        case orchestration::Stage::Preprocessing: return "Preprocessing";
        case orchestration::Stage::AutoConfig:    return "Auto-configuring";
        case orchestration::Stage::Training:      return "Training model";
        case orchestration::Stage::Evaluating:    return "Evaluating";
        case orchestration::Stage::Baselining:    return "Running baselines";
        case orchestration::Stage::LearningCurve: return "Learning curve";
        case orchestration::Stage::Packaging:     return "Packaging bundle";
        case orchestration::Stage::Done:          return "Done";
    }
    return "?";
}

void render_budget_picker(WizardState& state) {
    ImGui::TextUnformatted("How long should training take?");

    const TimeBudget options[] = {
        TimeBudget::Quick, TimeBudget::Balanced, TimeBudget::Best,
    };
    for (const auto opt : options) {
        const bool selected = (state.time_budget == opt);
        if (ImGui::RadioButton(time_budget_label(opt), selected)) {
            state.time_budget = opt;
        }
    }
    ImGui::Spacing();

    ImGui::TextUnformatted("Task type:");
    const char* opts[] = { "Auto-detect", "Classification", "Regression" };
    int current = static_cast<int>(state.task_override);
    ImGui::SetNextItemWidth(220);
    if (ImGui::Combo("##task_override", &current, opts, 3)) {
        state.task_override = static_cast<TaskOverride>(current);
    }
    if (state.task_override != TaskOverride::Auto) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                           "(auto-detect is usually correct)");
    }
}

void render_progress(const WizardState& state) {
    orchestration::Progress snap;
    {
        std::lock_guard<std::mutex> lk(state.progress_mutex);
        snap = state.last_progress;
    }
    ImGui::Text("Stage: %s", stage_label(snap.stage));
    ImGui::ProgressBar(static_cast<float>(snap.overall_fraction),
                       ImVec2(-1.0f, 0.0f));
    if (!snap.message.empty()) {
        ImGui::TextDisabled("%s", snap.message.c_str());
    }
}

} // namespace

void render_step_train(WizardState& state, WizardRuntime& rt) {
    const auto status = state.status.load();

    switch (status) {
        case TrainingStatus::Idle: {
            render_budget_picker(state);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Train", ImVec2(220, 40))) {
                start_training(state, rt);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(you can Stop at any time)");
            break;
        }
        case TrainingStatus::Running: {
            render_progress(state);
            ImGui::Spacing();
            if (ImGui::Button("Stop", ImVec2(160, 32))) {
                request_cancel(state);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Cancellation takes effect between stages.");
            break;
        }
        case TrainingStatus::Succeeded: {
            // Normally we auto-advance to Results; if the user came
            // back via "Back", they see a "View results" button.
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f),
                               "Last run succeeded.");
            if (ImGui::Button("View results", ImVec2(200, 0))) {
                advance(state);
            }
            ImGui::SameLine();
            if (ImGui::Button("Train again", ImVec2(200, 0))) {
                reset_to_train(state);
            }
            break;
        }
        case TrainingStatus::Failed: {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "Training failed:");
            ImGui::TextWrapped("%s", state.error_message.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Try again", ImVec2(200, 0))) {
                reset_to_train(state);
            }
            break;
        }
        case TrainingStatus::Cancelled: {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                               "Training was cancelled.");
            ImGui::Spacing();
            if (ImGui::Button("Train again", ImVec2(200, 0))) {
                reset_to_train(state);
            }
            break;
        }
    }
}

} // namespace neurograph::workbench
