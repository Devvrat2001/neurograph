// P1-09 — Wizard top-level render + worker-thread plumbing.
//
// Drawing for each step is split across step_load/quality/train/
// results.cpp. This file handles the step indicator, navigation
// buttons, and the background thread that runs the orchestrator
// while the user stares at the progress bar.

#include "wizard_ui.hpp"
#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <filesystem>

namespace neurograph::workbench {

// Step-render functions live in step_*.cpp. Forward-declared here so
// wizard_ui.cpp is the only caller responsible for dispatch.
void render_step_load   (WizardState&, WizardRuntime&);
void render_step_quality(WizardState&, WizardRuntime&);
void render_step_train  (WizardState&, WizardRuntime&);
void render_step_results(WizardState&, WizardRuntime&);

std::unique_ptr<WizardRuntime> make_wizard_runtime() {
    return std::make_unique<WizardRuntime>();
}

// ---------------------------------------------------------------------------
// Worker thread management
// ---------------------------------------------------------------------------

void finalize_worker_if_done(WizardRuntime& rt, WizardState& state) {
    if (!rt.worker_active) return;
    // The worker calls `on_training_finished` before it returns.
    // Observing a terminal status is the handshake that says "the
    // atomics have been published and we can safely join".
    const auto st = state.status.load();
    if (st == TrainingStatus::Succeeded
     || st == TrainingStatus::Failed
     || st == TrainingStatus::Cancelled) {
        if (rt.worker.joinable()) rt.worker.join();
        rt.worker_active = false;
    }
}

void start_training(WizardState& state, WizardRuntime& rt) {
    if (rt.worker_active) return;
    if (!state.csv) return;

    // Default the bundle output directory if the user hasn't picked
    // one. A temp path is fine here — the Results step's "Save
    // model bundle" button copies the bundle somewhere permanent.
    if (state.export_dir.empty()) {
        state.export_dir = std::filesystem::temp_directory_path()
                           / "neurograph_wizard_bundle";
    }

    state.status.store(TrainingStatus::Running);
    state.cancel_requested.store(false);
    state.error_message.clear();
    state.result.reset();

    const auto csv_path    = state.csv->path;
    const auto target_col  = state.target_column;
    const auto opts        = build_options(state, state.export_dir);

    rt.worker_active = true;
    rt.worker = std::thread([&state, csv_path, target_col, opts]() {
        auto result = orchestration::run(
            csv_path,
            target_col,
            opts,
            /* progress */ [&state](const orchestration::Progress& p) {
                on_progress(state, p);
            },
            /* cancel */ [&state]() {
                return state.cancel_requested.load();
            });
        on_training_finished(state, std::move(result));
    });
}

// ---------------------------------------------------------------------------
// Step indicator (breadcrumbs across the top of the wizard window)
// ---------------------------------------------------------------------------

namespace {

void render_step_indicator(const WizardState& state) {
    ImGui::BeginGroup();
    const WizardStep steps[] = {
        WizardStep::Load, WizardStep::Quality,
        WizardStep::Train, WizardStep::Results,
    };
    for (std::size_t i = 0; i < 4; ++i) {
        const auto s = steps[i];
        const bool active   = (s == state.current_step);
        const bool complete = (static_cast<int>(s) < static_cast<int>(state.current_step));

        // Circle + label; colored by state. These text lines are
        // intentionally cheap — a polished version would use
        // ImDrawList primitives, but plain ImGui::Text works today.
        if (active) {
            ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "%zu. %s",
                               i + 1, step_label(s));
        } else if (complete) {
            ImGui::TextColored(ImVec4(0.4f, 0.7f, 0.4f, 1.0f), "%zu. %s",
                               i + 1, step_label(s));
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%zu. %s",
                               i + 1, step_label(s));
        }
        if (i < 3) {
            ImGui::SameLine();
            ImGui::TextUnformatted("  >  ");
            ImGui::SameLine();
        }
    }
    ImGui::EndGroup();
    ImGui::Separator();
}

// Navigation buttons — Back / Next. "Next" is a tap on `advance`;
// it's disabled + carries a tooltip when `can_advance` says no.
// Returns whether navigation happened this frame (caller can use
// this to short-circuit extra work, but currently ignored).
void render_nav_buttons(WizardState& state, WizardRuntime& rt) {
    ImGui::Separator();

    const bool training_running =
        state.status.load() == TrainingStatus::Running;

    // Back
    const bool back_disabled =
        (state.current_step == WizardStep::Load) || training_running;
    ImGui::BeginDisabled(back_disabled);
    if (ImGui::Button("< Back")) {
        back(state);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    // Next / Finish
    const auto reason = can_advance(state);
    const bool next_disabled = !reason.empty() || training_running;
    const char* next_label =
        state.current_step == WizardStep::Train    ? "Next: Results >" :
        state.current_step == WizardStep::Results  ? "Done" :
        "Next >";
    ImGui::BeginDisabled(next_disabled);
    if (ImGui::Button(next_label)) {
        advance(state);
    }
    ImGui::EndDisabled();
    if (next_disabled && !reason.empty() && ImGui::IsItemHovered(
            ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", reason.c_str());
    }

    ImGui::SameLine();
    ImGui::Dummy(ImVec2(20, 0));
    ImGui::SameLine();

    if (ImGui::Button("Start over")) {
        if (!training_running) {
            // Stop-then-restart is gated behind the Stop button on
            // the Train step; from other steps we just wipe state.
            reset_to_load(state);
        }
    }
    (void)rt; // reserved for future "save progress" hooks
}

} // namespace

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

bool render_wizard(WizardState& state, WizardRuntime& rt) {
    // Reap a finished worker once per frame so the state never shows
    // "Running" for longer than one frame after the orchestrator
    // returns.
    finalize_worker_if_done(rt, state);

    bool user_wants_advanced = false;

    // One big window, docked to the main viewport. We don't use
    // ImGui::Begin with a title bar because the window frame itself
    // is the wizard; the step indicator + advanced toggle live at
    // the top inside the window.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoDocking;
    if (ImGui::Begin("NeuroGraph — Guided Setup", nullptr, flags)) {
        // Advanced-mode escape hatch at the top-right.
        const float btn_w = 160.0f;
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - btn_w);
        if (ImGui::SmallButton("Show advanced UI")) {
            user_wants_advanced = true;
        }

        render_step_indicator(state);

        // Dispatch to the step renderer; each step owns its own
        // child region + scrolling.
        switch (state.current_step) {
            case WizardStep::Load:    render_step_load(state, rt);    break;
            case WizardStep::Quality: render_step_quality(state, rt); break;
            case WizardStep::Train:   render_step_train(state, rt);   break;
            case WizardStep::Results: render_step_results(state, rt); break;
        }

        render_nav_buttons(state, rt);
    }
    ImGui::End();

    return user_wants_advanced;
}

} // namespace neurograph::workbench
