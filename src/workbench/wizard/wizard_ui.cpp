// P1-09 — Wizard top-level render + worker-thread plumbing.
//
// Drawing for each step is split across step_load/quality/train/
// results.cpp. This file draws the frame around them: the step
// indicator at the top, the scrolling area each step draws into, and the
// Back / Next buttons pinned to the bottom. It also owns the background
// thread that runs the orchestrator while the user watches the progress
// bar.

#include "wizard_ui.hpp"
#include "theme.hpp"
#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

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

std::string with_commas(std::size_t n) {
    std::string digits = std::to_string(n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) {
        digits.insert(static_cast<std::size_t>(i), 1, ',');
    }
    return digits;
}

void text_hint(const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
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
// Frame: step indicator, scrolling step area, buttons
// ---------------------------------------------------------------------------

namespace {

// Numbered circles joined by short lines, each followed by the step's
// name: blue for the current step, green for the ones done.
void render_step_indicator(const WizardState& state) {
    ImDrawList* dl      = ImGui::GetWindowDrawList();
    const ImGuiStyle& s = ImGui::GetStyle();
    const float h       = ImGui::GetFrameHeight();
    const float radius  = h * 0.42f;
    const float joint   = h * 1.4f;
    const int   current = static_cast<int>(state.current_step);

    const ImU32 done_col  = ImGui::GetColorU32(colors::good);
    const ImU32 now_col   = ImGui::GetColorU32(colors::accent);
    const ImU32 todo_col  = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    const ImU32 white     = IM_COL32(255, 255, 255, 255);

    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            ImGui::SameLine(0.0f, s.ItemInnerSpacing.x);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            dl->AddLine(ImVec2(p.x, p.y + h * 0.5f), ImVec2(p.x + joint, p.y + h * 0.5f),
                        i <= current ? done_col : todo_col, 1.0f);
            ImGui::Dummy(ImVec2(joint, h));
            ImGui::SameLine(0.0f, s.ItemInnerSpacing.x);
        }
        const ImU32 col = i < current ? done_col : (i == current ? now_col : todo_col);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 c(p.x + radius, p.y + h * 0.5f);
        char num[4];
        std::snprintf(num, sizeof num, "%d", i + 1);
        const ImVec2 ts = ImGui::CalcTextSize(num);
        if (i <= current) dl->AddCircleFilled(c, radius, col);
        else              dl->AddCircle(c, radius, col, 0, 1.5f);
        dl->AddText(ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), i <= current ? white : col, num);
        ImGui::Dummy(ImVec2(radius * 2.0f, h));

        ImGui::SameLine(0.0f, s.ItemInnerSpacing.x);
        ImGui::AlignTextToFramePadding();
        const char* label = step_label(static_cast<WizardStep>(i));
        if (i == current) ImGui::TextUnformatted(label);
        else              ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), "%s", label);
    }
}

// Back and Start over on the left; Next on the right, in the accent
// colour, with the reason next to it when it cannot be used yet.
void render_nav_buttons(WizardState& state) {
    const bool running = state.status.load() == TrainingStatus::Running;
    const float right  = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    const float min_w  = ImGui::GetFontSize() * 6.0f;

    ImGui::BeginDisabled(state.current_step == WizardStep::Load || running);
    if (ImGui::Button("< Back", ImVec2(min_w, 0.0f))) back(state);
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(running || (!state.csv && state.current_step == WizardStep::Load));
    if (ImGui::Button("Start over", ImVec2(min_w, 0.0f))) reset_to_load(state);
    ImGui::EndDisabled();

    // Results is the last step: nothing comes after it.
    if (state.current_step == WizardStep::Results) return;

    const char* label = state.current_step == WizardStep::Train ? "Results >" : "Next >";
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float w = std::max(min_w, ImGui::CalcTextSize(label).x +
                                    ImGui::GetStyle().FramePadding.x * 2.0f);
    const std::string reason = running ? std::string() : can_advance(state);
    // The reason goes just left of the button, if the window is wide
    // enough to keep it clear of the buttons on the left.
    const float left_end = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x;
    const float reason_x = right - w - spacing - ImGui::CalcTextSize(reason.c_str()).x;
    if (!reason.empty() && reason_x > left_end + spacing) {
        ImGui::SameLine(reason_x);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", reason.c_str());
    }
    ImGui::SameLine(right - w);
    ImGui::BeginDisabled(running || !reason.empty());
    if (accent_button(label, ImVec2(w, 0.0f))) advance(state);
    ImGui::EndDisabled();
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

    // One window filling the main viewport. The application window's own
    // title bar names the app, so this one has none.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::Begin("Guided setup", nullptr, flags)) {
        const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
        render_step_indicator(state);

        // Escape hatch to the 0.1 panels, at the right end of the same row.
        const char* adv = "Show advanced UI";
        const float adv_w = ImGui::CalcTextSize(adv).x + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SameLine(right - adv_w);
        if (ImGui::Button(adv)) user_wants_advanced = true;
        ImGui::Separator();

        // The step scrolls on its own; the buttons below stay in view.
        // Each step has its own scroll position, so a long page scrolled
        // down does not open the next one part-way down.
        const float footer = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
        char child_id[16];
        std::snprintf(child_id, sizeof child_id, "##step%d", static_cast<int>(state.current_step));
        if (ImGui::BeginChild(child_id, ImVec2(0.0f, -footer))) {
            switch (state.current_step) {
                case WizardStep::Load:    render_step_load(state, rt);    break;
                case WizardStep::Quality: render_step_quality(state, rt); break;
                case WizardStep::Train:   render_step_train(state, rt);   break;
                case WizardStep::Results: render_step_results(state, rt); break;
            }
        }
        ImGui::EndChild();
        ImGui::Separator();
        render_nav_buttons(state);
    }
    ImGui::End();
    ImGui::PopStyleVar(2);

    return user_wants_advanced;
}

} // namespace neurograph::workbench
