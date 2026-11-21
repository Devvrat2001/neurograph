#pragma once

// Fonts and colours for the workbench window. Shared with the
// wizard_snapshots tool so its pictures match what the app draws.
// load_fonts and apply_style need a current ImGui context and run once,
// before the first frame.

#include "imgui.h"

namespace neurograph::workbench {

/// Load the interface font, sized for `dpi_scale` (1.0 = 96 DPI): Segoe
/// UI with Segoe UI Symbol for any symbols it lacks, or the font built
/// into ImGui when Windows has neither.
void load_fonts(float dpi_scale);

/// Dark theme with blue accents; spacing and rounding scaled by
/// `dpi_scale`.
void apply_style(float dpi_scale);

/// Colours shared by screens that draw their own highlights.
namespace colors {
inline constexpr ImVec4 accent  {0.16f, 0.47f, 0.87f, 1.0f};   // main action, current step
inline constexpr ImVec4 good    {0.45f, 0.85f, 0.45f, 1.0f};   // finished, succeeded
inline constexpr ImVec4 caution {1.00f, 0.80f, 0.30f, 1.0f};   // warnings
inline constexpr ImVec4 bad     {1.00f, 0.42f, 0.42f, 1.0f};   // errors, blocking problems
} // namespace colors

/// A button in the accent colour, for the one main action on a screen.
/// Draws as an ordinary button inside BeginDisabled, so a disabled main
/// action does not look clickable.
bool accent_button(const char* label, const ImVec2& size = ImVec2(0.0f, 0.0f));

} // namespace neurograph::workbench
