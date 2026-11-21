// Fonts and colours for the workbench window. See theme.hpp.

#include "theme.hpp"

#include "imgui_internal.h"

#include <windows.h>

#include <filesystem>
#include <string>

namespace neurograph::workbench {

namespace {

constexpr float kFontSize = 16.0f;   // pixels at 96 DPI

// A file in the Windows fonts folder, as UTF-8 (what ImGui expects), or
// empty when it is not there.
std::string windows_font(const wchar_t* file) {
    wchar_t dir[MAX_PATH];
    const UINT n = GetWindowsDirectoryW(dir, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    const std::filesystem::path p = std::filesystem::path(dir) / L"Fonts" / file;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) return {};
    const auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

} // namespace

void load_fonts(float dpi_scale) {
    ImGuiIO& io = ImGui::GetIO();
    // Segoe UI is on every Windows installation since Vista and covers the
    // punctuation the wizard's messages use (dashes, ellipses, superscript
    // two); ImGui's own font covers ASCII and Latin-1 only, which showed
    // every dash as "?". Glyphs are rasterised on demand at any size.
    const std::string text    = windows_font(L"segoeui.ttf");
    const std::string symbols = windows_font(L"seguisym.ttf");
    if (!text.empty() && io.Fonts->AddFontFromFileTTF(text.c_str(), kFontSize)) {
        if (!symbols.empty()) {
            ImFontConfig merge;
            merge.MergeMode = true;
            io.Fonts->AddFontFromFileTTF(symbols.c_str(), kFontSize, &merge);
        }
    } else {
        io.Fonts->AddFontDefault();
    }
    ImGuiStyle& style  = ImGui::GetStyle();
    style.FontSizeBase = kFontSize;
    style.FontScaleDpi = dpi_scale;
}

bool accent_button(const char* label, const ImVec2& size) {
    // Inside BeginDisabled, the item about to be drawn carries this flag.
    if (GImGui->CurrentItemFlags & ImGuiItemFlags_Disabled) return ImGui::Button(label, size);
    ImGui::PushStyleColor(ImGuiCol_Button,        colors::accent);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.12f, 0.40f, 0.78f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(4);
    return clicked;
}

void apply_style(float dpi_scale) {
    const auto S = [dpi_scale](float v) { return v * dpi_scale; };

    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();

    // Geometry - scaled by DPI
    s.WindowRounding    = S(6);
    s.FrameRounding     = S(4);
    s.GrabRounding      = S(4);
    s.TabRounding       = S(4);
    s.ScrollbarRounding = S(6);
    s.ChildRounding     = S(4);
    s.PopupRounding     = S(4);

    s.WindowPadding     = ImVec2(S(12), S(12));
    s.FramePadding      = ImVec2(S(10), S(6));
    s.ItemSpacing       = ImVec2(S(10), S(8));
    s.ItemInnerSpacing  = ImVec2(S(8), S(6));
    s.IndentSpacing     = S(20);
    s.ScrollbarSize     = S(14);
    s.GrabMinSize       = S(12);
    s.WindowBorderSize  = 1.0f;
    // A thin outline, so empty tick boxes, radio buttons and text fields
    // stand out from the dark background.
    s.FrameBorderSize   = 1.0f;
    s.TabBorderSize     = 0.0f;
    s.SeparatorTextBorderSize = S(2);

    // Color palette - high contrast dark theme with blue accents
    ImVec4* c = s.Colors;

    // Backgrounds
    c[ImGuiCol_WindowBg]             = ImVec4(0.06f, 0.07f, 0.11f, 1.00f);
    c[ImGuiCol_ChildBg]              = ImVec4(0.06f, 0.07f, 0.11f, 0.00f);
    c[ImGuiCol_PopupBg]              = ImVec4(0.08f, 0.09f, 0.14f, 0.97f);

    // Borders
    c[ImGuiCol_Border]               = ImVec4(0.27f, 0.30f, 0.40f, 1.00f);
    c[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input fields, etc)
    c[ImGuiCol_FrameBg]              = ImVec4(0.10f, 0.12f, 0.18f, 1.00f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(0.14f, 0.16f, 0.24f, 1.00f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(0.16f, 0.20f, 0.30f, 1.00f);

    // Title bars
    c[ImGuiCol_TitleBg]              = ImVec4(0.05f, 0.06f, 0.09f, 1.00f);
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.08f, 0.10f, 0.16f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]     = ImVec4(0.04f, 0.05f, 0.08f, 0.80f);

    // Menu bar
    c[ImGuiCol_MenuBarBg]            = ImVec4(0.07f, 0.08f, 0.12f, 1.00f);

    // Scrollbar
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.05f, 0.06f, 0.09f, 0.60f);
    c[ImGuiCol_ScrollbarGrab]        = ImVec4(0.20f, 0.24f, 0.35f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.28f, 0.32f, 0.45f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.34f, 0.40f, 0.55f, 1.00f);

    // Buttons
    c[ImGuiCol_Button]               = ImVec4(0.14f, 0.18f, 0.28f, 1.00f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(0.20f, 0.26f, 0.40f, 1.00f);
    c[ImGuiCol_ButtonActive]         = ImVec4(0.16f, 0.44f, 0.82f, 1.00f);

    // Headers (collapsing headers, tree nodes, selectable)
    c[ImGuiCol_Header]               = ImVec4(0.12f, 0.16f, 0.26f, 1.00f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(0.18f, 0.24f, 0.38f, 1.00f);
    c[ImGuiCol_HeaderActive]         = ImVec4(0.16f, 0.44f, 0.82f, 1.00f);

    // Separator
    c[ImGuiCol_Separator]            = ImVec4(0.18f, 0.20f, 0.28f, 1.00f);
    c[ImGuiCol_SeparatorHovered]     = ImVec4(0.30f, 0.50f, 0.80f, 0.80f);
    c[ImGuiCol_SeparatorActive]      = ImVec4(0.30f, 0.50f, 0.80f, 1.00f);

    // Resize grip
    c[ImGuiCol_ResizeGrip]           = ImVec4(0.20f, 0.30f, 0.50f, 0.40f);
    c[ImGuiCol_ResizeGripHovered]    = ImVec4(0.30f, 0.50f, 0.80f, 0.60f);
    c[ImGuiCol_ResizeGripActive]     = ImVec4(0.30f, 0.50f, 0.80f, 0.90f);

    // Tabs
    c[ImGuiCol_Tab]                  = ImVec4(0.08f, 0.10f, 0.16f, 1.00f);
    c[ImGuiCol_TabHovered]           = ImVec4(0.16f, 0.44f, 0.82f, 0.80f);
    c[ImGuiCol_TabSelected]          = ImVec4(0.14f, 0.36f, 0.68f, 1.00f);
    c[ImGuiCol_TabDimmed]            = ImVec4(0.06f, 0.08f, 0.12f, 1.00f);
    c[ImGuiCol_TabDimmedSelected]    = ImVec4(0.10f, 0.22f, 0.42f, 1.00f);

    // Docking
    c[ImGuiCol_DockingPreview]       = ImVec4(0.16f, 0.44f, 0.82f, 0.70f);
    c[ImGuiCol_DockingEmptyBg]       = ImVec4(0.04f, 0.05f, 0.08f, 1.00f);

    // Slider
    c[ImGuiCol_SliderGrab]           = ImVec4(0.25f, 0.55f, 0.95f, 1.00f);
    c[ImGuiCol_SliderGrabActive]     = ImVec4(0.35f, 0.65f, 1.00f, 1.00f);

    // Check mark
    c[ImGuiCol_CheckMark]            = ImVec4(0.30f, 0.65f, 1.00f, 1.00f);

    // Plot
    c[ImGuiCol_PlotLines]            = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
    c[ImGuiCol_PlotLinesHovered]     = ImVec4(0.50f, 0.80f, 1.00f, 1.00f);
    c[ImGuiCol_PlotHistogram]        = ImVec4(0.25f, 0.55f, 0.95f, 1.00f);
    c[ImGuiCol_PlotHistogramHovered] = ImVec4(0.35f, 0.65f, 1.00f, 1.00f);

    // Text
    c[ImGuiCol_Text]                 = ImVec4(0.85f, 0.88f, 0.95f, 1.00f);
    c[ImGuiCol_TextDisabled]         = ImVec4(0.40f, 0.42f, 0.50f, 1.00f);
    c[ImGuiCol_TextSelectedBg]       = ImVec4(0.16f, 0.44f, 0.82f, 0.40f);

    // Nav
    c[ImGuiCol_NavHighlight]         = ImVec4(0.30f, 0.60f, 1.00f, 1.00f);
}

} // namespace neurograph::workbench
