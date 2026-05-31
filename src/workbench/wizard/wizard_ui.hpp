#pragma once

// P1-09 — Wizard UI render glue.
//
// This is the ImGui-bound side of the wizard. The pure state machine
// in neurograph/workbench/wizard_state.hpp has no rendering code; this
// file wires ImGui widgets to the transition functions there.
//
// Why separated:
//   - The state machine is unit-testable without a D3D11 context.
//   - This file needs imgui.h, which unit-test translation units
//     shouldn't have to pull in.
//   - If we ever swap from ImGui to a web UI, only this file has to
//     be rewritten.
//
// Only consumed by `workbench_app.cpp`.

#include "neurograph/workbench/wizard_state.hpp"

#include <memory>
#include <thread>

namespace neurograph::workbench {

/// Owns the orchestrator worker thread. One per wizard instance.
/// Kept behind an opaque declaration here so the header doesn't have
/// to include <thread> / <atomic> / <mutex> from the global workbench
/// header fanout.
struct WizardRuntime {
    // The training worker. The UI is the only thread that spawns or
    // joins it; the worker thread only touches WizardState through
    // the published helpers (on_progress, on_training_finished).
    std::thread worker;

    // True while `worker` is still running. The UI polls this each
    // frame and calls `finalize_worker_if_done` to reap the thread.
    // `joinable()` alone isn't enough because a finished thread is
    // still joinable until you call .join() on it.
    bool worker_active = false;
};

/// Construct a fresh runtime. Moving requires manual work — the
/// thread is owned; callers should just not move this.
std::unique_ptr<WizardRuntime> make_wizard_runtime();

/// Render one frame of the wizard. Expects an active ImGui frame.
///
/// Returns true when the user toggled "Show advanced" in the current
/// frame — the caller uses that signal to switch back to the legacy
/// panels.
bool render_wizard(WizardState& state, WizardRuntime& rt);

/// If the worker thread is alive, check whether it's done (status
/// flipped out of Running), and if so join it. Safe to call every
/// frame regardless of worker state.
void finalize_worker_if_done(WizardRuntime& rt, WizardState& state);

/// Spawn a worker thread that invokes `orchestration::run` with the
/// current wizard state. No-op when a worker is already active.
void start_training(WizardState& state, WizardRuntime& rt);

} // namespace neurograph::workbench
