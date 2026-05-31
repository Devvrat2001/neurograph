#pragma once

// P1-11 -- License dialog for the workbench.
//
// Two interrelated concerns:
//   1. On startup, decide whether the user can use the app at all.
//      Either an activated license is on disk, or we're inside the
//      30-day trial window. If neither, the wizard is blocked behind
//      the modal -- no main UI render until they enter a key (or we
//      determine the trial is still alive).
//   2. Provide an "Enter license key..." menu item that re-opens the
//      dialog so existing users can upgrade Trial -> paid mid-session.
//
// This file is workbench-only because it pulls in <imgui.h>. The
// licensing crypto + persistence layer (`neurograph::licensing`)
// stays in neurograph_lib and is fully unit-testable without ImGui.

#include "neurograph/licensing/license.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace neurograph::workbench {

/// Snapshot of "what tier is this user on right now". Held in
/// AppState so the menu bar / status bar can render it without
/// re-querying the disk every frame.
struct LicenseStatus {
    /// The license currently in force -- either the activated key or
    /// a synthesized trial license.
    std::optional<licensing::License> license;

    /// True iff `license` is a trial-tier synthesized one, vs. an
    /// activated paid key. Drives the "Buy" upsell prompt.
    bool is_trial = false;

    /// Days left in the trial. 0 for non-trial states. -1 if trial
    /// is over and there's no activated key (gates the whole UI).
    int trial_days_left = 0;
};

/// Run the boot-time license probe: load any activated key, else
/// load-or-init the trial. Should be called once at startup before
/// the main loop.
LicenseStatus boot_license_state();

/// Re-render the LicenseStatus from disk after the user entered or
/// cleared a key in the dialog.
LicenseStatus reload_license_state();

/// State kept between frames while the dialog is on screen. The
/// caller owns one of these inside AppState.
///
/// The key edit buffer is a fixed-size char array because
/// `ImGui::InputTextMultiline` writes raw chars into a caller buffer
/// and we don't want to depend on imgui_stdlib.h being in the vcpkg
/// build. 1024 bytes is plenty for a ~140-char base32 key plus
/// generous formatting whitespace.
struct LicenseDialogState {
    static constexpr std::size_t kKeyBufBytes = 1024;

    bool        open = false;
    char        input_key[kKeyBufBytes] = {0};
    std::string last_error;          // shown red if non-empty
    bool        modal_block = false; // when true, the dialog is the only
                                     // thing rendered -- the main UI is
                                     // gated until activation succeeds
};

/// Request the dialog be shown (e.g. menu click).
void open_license_dialog(LicenseDialogState& d);

/// Render one frame. If the user activated a key successfully this
/// frame, mutates `status` to reflect the new license and returns
/// true so the caller can re-enable the main UI.
bool render_license_dialog(LicenseDialogState& d, LicenseStatus& status);

} // namespace neurograph::workbench
