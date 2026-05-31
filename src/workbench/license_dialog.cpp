// P1-11 -- License dialog implementation. ImGui-bound; lives in the
// workbench EXE only. The licensing logic itself is in
// neurograph::licensing (neurograph_lib) so the unit tests can
// exercise it without spinning up D3D11.

#include "license_dialog.hpp"

#include "imgui.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace neurograph::workbench {

namespace {

// Pretty-print a system_clock time as YYYY-MM-DD UTC (no time of day
// -- it would just be confusing in the dialog).
std::string format_date_utc(std::chrono::system_clock::time_point tp) {
    auto t = std::chrono::system_clock::to_time_t(tp);
    std::ostringstream os;
    os << std::put_time(std::gmtime(&t), "%Y-%m-%d");
    return os.str();
}

const char* tier_label(licensing::Tier t) {
    switch (t) {
        case licensing::Tier::Trial:        return "Trial";
        case licensing::Tier::Personal:     return "Personal";
        case licensing::Tier::Professional: return "Professional";
        case licensing::Tier::Enterprise:   return "Enterprise";
    }
    return "?";
}

// Compose the LicenseStatus from whatever's on disk right now. Used by
// both boot_license_state and reload_license_state.
LicenseStatus probe_disk() {
    LicenseStatus s;

    if (auto activated = licensing::current_license()) {
        s.license = *activated;
        s.is_trial = false;
        s.trial_days_left = 0;
        return s;
    }

    auto trial = licensing::load_or_init_trial();
    if (trial.is_error()) {
        // Trial file unwritable (e.g. AppData not creatable). Block
        // the UI -- the dialog will surface the error to the user.
        s.license.reset();
        s.is_trial = false;
        s.trial_days_left = -1;
        return s;
    }

    if (licensing::trial_active(trial.value())) {
        s.license = licensing::make_trial_license(trial.value());
        s.is_trial = true;
        s.trial_days_left = licensing::trial_days_remaining(trial.value());
    } else {
        // Trial elapsed and no key activated -> hard gate.
        s.license.reset();
        s.is_trial = false;
        s.trial_days_left = -1;
    }
    return s;
}

} // namespace

LicenseStatus boot_license_state() {
    return probe_disk();
}

LicenseStatus reload_license_state() {
    return probe_disk();
}

void open_license_dialog(LicenseDialogState& d) {
    d.open = true;
    d.last_error.clear();
    // Don't wipe input_key on reopen -- if the user closed and
    // reopened by accident their typed key is still there.
}

bool render_license_dialog(LicenseDialogState& d, LicenseStatus& status) {
    // The modal_block flag is set by the caller when there is no
    // active license AND no live trial -- in that case the dialog is
    // the only thing on screen and Cancel is hidden.
    if (!d.open && !d.modal_block) return false;

    // Centre on screen.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, {0.5f, 0.5f});
    ImGui::SetNextWindowSize({560, 0}, ImGuiCond_Always);

    const char* title = d.modal_block
        ? "NeuroGraph -- License required"
        : "NeuroGraph -- License";

    bool activated_this_frame = false;

    // ImGui::OpenPopup is idempotent per id; safe to call every frame
    // while we want it open.
    ImGui::OpenPopup(title);

    if (ImGui::BeginPopupModal(title, nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize |
                               ImGuiWindowFlags_NoCollapse)) {

        // ---- Current state summary -----------------------------------
        if (status.license) {
            const auto& l = *status.license;
            ImGui::Text("Current state: %s", tier_label(l.tier));
            if (status.is_trial) {
                if (status.trial_days_left > 0) {
                    ImGui::TextColored({1.0f, 0.85f, 0.4f, 1.0f},
                        "Trial: %d day%s remaining (until %s)",
                        status.trial_days_left,
                        status.trial_days_left == 1 ? "" : "s",
                        format_date_utc(l.expires).c_str());
                } else {
                    ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f},
                        "Trial expired on %s. Enter a license key to continue.",
                        format_date_utc(l.expires).c_str());
                }
            } else {
                ImGui::Text("Holder:  %s", l.holder.c_str());
                ImGui::Text("Expires: %s",
                            format_date_utc(l.expires).c_str());
                if (!l.features.empty()) {
                    ImGui::Text("Features:");
                    for (const auto& f : l.features) {
                        ImGui::BulletText("%s", f.c_str());
                    }
                }
            }
        } else if (status.trial_days_left < 0) {
            ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f},
                "Your free trial has ended. Enter a license key to "
                "continue using NeuroGraph.");
        } else {
            ImGui::Text("No license currently installed.");
        }

        ImGui::Separator();

        // ---- Key entry ----------------------------------------------
        ImGui::TextWrapped(
            "Paste your license key below. Keys look like "
            "AEAQ-CAJB-EAQC-... and are case-insensitive.");

        // ImGui::InputTextMultiline writes raw chars into our buffer.
        // The buffer is a fixed-size array on LicenseDialogState so we
        // don't need imgui_stdlib.h's std::string overload.
        ImGui::InputTextMultiline("##key", d.input_key,
                                  LicenseDialogState::kKeyBufBytes,
                                  {520, ImGui::GetTextLineHeight() * 4});

        if (!d.last_error.empty()) {
            ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "%s",
                               d.last_error.c_str());
        }

        ImGui::Spacing();

        // ---- Buttons -------------------------------------------------
        if (ImGui::Button("Activate", {120, 0})) {
            const std::string typed(d.input_key);
            auto r = licensing::activate(typed);
            if (r.is_error()) {
                d.last_error = r.error().message;
            } else {
                d.last_error.clear();
                d.input_key[0] = '\0';
                status = reload_license_state();
                d.open = false;
                d.modal_block = false;
                activated_this_frame = true;
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::SameLine();

        // Cancel is suppressed when modal_block is on -- the user
        // can't dismiss the dialog without entering a valid key.
        if (!d.modal_block) {
            if (ImGui::Button("Close", {100, 0})) {
                d.open = false;
                d.last_error.clear();
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();

            // Sign-out is only meaningful when an activated (non-trial)
            // license exists.
            if (status.license && !status.is_trial) {
                if (ImGui::Button("Sign out", {100, 0})) {
                    licensing::clear_persisted_license();
                    status = reload_license_state();
                    d.last_error.clear();
                    // If sign-out left us with no usable state at all
                    // (trial also elapsed), the dialog must become a
                    // hard gate -- otherwise the user could click
                    // Close and end up using an unlicensed workbench.
                    if (!status.license) {
                        d.modal_block = true;
                        d.open        = true;
                    }
                }
            }
        }

        // Helpful link target -- not actually clickable from ImGui in
        // a portable way without OS shellopen, but useful as text.
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled(
            "Need a key? Visit https://neurograph.ai/buy");

        ImGui::EndPopup();
    }

    return activated_this_frame;
}

} // namespace neurograph::workbench
