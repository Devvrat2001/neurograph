// Step 3 — Train.
//
// Time allowance and task picker, one Train button, then a progress bar
// with a Stop button. Failure and cancellation come back to this screen
// with the reason and a way to try again.

#include "wizard_ui.hpp"
#include "theme.hpp"
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
    text_hint(budget_explanation(state.time_budget));
    ImGui::Spacing();

    ImGui::TextUnformatted("What should be predicted?");
    const struct { TaskOverride task; const char* label; } tasks[] = {
        {TaskOverride::Auto,           "Decide from the column (recommended)"},
        {TaskOverride::Classification, "A category (classification)"},
        {TaskOverride::Regression,     "A number (regression)"},
    };
    for (const auto& t : tasks) {
        if (ImGui::RadioButton(t.label, state.task_override == t.task)) {
            set_task_override(state, t.task);
        }
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

// A large button with a quiet note beside it, centred on the button.
bool big_button_with_note(const char* label, bool accent, const char* note) {
    const ImVec2 size(ImGui::GetFontSize() * 12.0f, ImGui::GetFrameHeight() * 1.5f);
    const bool clicked = accent ? accent_button(label, size) : ImGui::Button(label, size);
    // Text on the same line as a button is already lowered by the frame
    // padding; move it the rest of the way to the button's middle.
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (size.y - ImGui::GetTextLineHeight()) * 0.5f -
                         ImGui::GetStyle().FramePadding.y);
    ImGui::TextDisabled("%s", note);
    return clicked;
}

} // namespace

void render_step_train(WizardState& state, WizardRuntime& rt) {
    const ImVec2 button(ImGui::GetFontSize() * 10.0f, 0.0f);
    switch (state.status.load()) {
        case TrainingStatus::Idle: {
            render_options(state);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (big_button_with_note("Train", true, "You can stop it at any time.")) {
                start_training(state, rt);
            }
            break;
        }
        case TrainingStatus::Running: {
            render_progress(state);
            ImGui::Spacing();
            if (big_button_with_note("Stop", false, "Stops within a second or two.")) {
                request_cancel(state);
            }
            break;
        }
        case TrainingStatus::Succeeded: {
            // Normally the wizard moves to Results on its own; this shows
            // when the user came back with Back.
            ImGui::TextColored(colors::good, "The last run succeeded.");
            if (accent_button("View results", button)) advance(state);
            ImGui::SameLine();
            if (ImGui::Button("Train again", button)) reset_to_train(state);
            break;
        }
        case TrainingStatus::Failed: {
            ImGui::TextColored(colors::bad, "Training did not complete:");
            ImGui::TextWrapped("%s", state.error_message.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Try again", button)) reset_to_train(state);
            break;
        }
        case TrainingStatus::Cancelled: {
            ImGui::TextColored(colors::caution, "Training was stopped.");
            ImGui::Spacing();
            if (ImGui::Button("Train again", button)) reset_to_train(state);
            break;
        }
    }
}

} // namespace neurograph::workbench
