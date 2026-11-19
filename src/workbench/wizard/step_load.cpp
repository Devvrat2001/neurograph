// Step 1 — Load data.
//
// Three ways in: the Browse button (Windows file dialog), dragging a file
// onto the window (WM_DROPFILES, forwarded through WizardRuntime), or
// pasting a path and pressing Enter. Then the user picks the column to
// predict; the last column is the default.

#include "wizard_ui.hpp"
#include "native_dialogs.hpp"
#include "neurograph/workbench/csv_preview.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace neurograph::workbench {

namespace {

// Backing store for the path field; ImGui keeps only a pointer to it.
char g_path_buf[2048] = "";

// Tables wider than this become unreadable; the rest are summarised.
constexpr std::size_t kMaxPreviewColumns = 40;

void load_path(WizardState& state, const std::filesystem::path& p) {
    std::snprintf(g_path_buf, sizeof(g_path_buf), "%s", path_utf8(p).c_str());
    auto result = peek_csv_metadata(p, /*preview=*/5);
    if (result.ok()) {
        set_csv(state, std::move(result).value());
    } else {
        // Keep the previous file loaded; the user can fix the path.
        state.load_error = result.error().message;
    }
}

} // namespace

void render_step_load(WizardState& state, WizardRuntime& rt) {
    ImGui::TextWrapped(
        "Choose a CSV file with your data: one row per example, and one column "
        "holding what you want to predict. Excel can save any sheet as CSV "
        "(File > Save As > CSV UTF-8).");
    ImGui::Spacing();

    if (!rt.dropped_file.empty()) {
        const std::filesystem::path dropped = rt.dropped_file;
        rt.dropped_file.clear();
        load_path(state, dropped);
    }

    if (ImGui::Button("Browse...", ImVec2(120, 0))) {
        if (auto p = pick_csv_file()) load_path(state, *p);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-130.0f);
    const bool entered = ImGui::InputTextWithHint(
        "##csv_path", "or paste a path here, or drag a file onto this window",
        g_path_buf, sizeof(g_path_buf), ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    const bool load_clicked = ImGui::Button("Load", ImVec2(120, 0));
    if ((load_clicked || entered) && g_path_buf[0] != '\0') {
        load_path(state, path_from_utf8(g_path_buf));
    }

    if (!state.load_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Could not load the file: %s",
                           state.load_error.c_str());
    }
    if (!state.csv) return;

    const auto& csv = *state.csv;
    ImGui::Separator();
    ImGui::Text("File: %s", path_utf8(csv.path.filename()).c_str());
    ImGui::Text("%zu rows, %zu columns", csv.row_count, csv.columns.size());
    ImGui::Spacing();

    const std::size_t shown = std::min(csv.columns.size(), kMaxPreviewColumns);
    if (!csv.preview_rows.empty() && shown > 0
        && ImGui::BeginTable("preview", static_cast<int>(shown),
                             ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit,
                             ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 7.5f))) {
        for (std::size_t i = 0; i < shown; ++i) {
            ImGui::TableSetupColumn(csv.columns[i].c_str());
        }
        ImGui::TableHeadersRow();
        for (const auto& row : csv.preview_rows) {
            ImGui::TableNextRow();
            for (std::size_t i = 0; i < shown; ++i) {
                ImGui::TableSetColumnIndex(static_cast<int>(i));
                if (i < row.size()) ImGui::TextUnformatted(row[i].c_str());
            }
        }
        ImGui::EndTable();
    }
    if (csv.columns.size() > shown) {
        ImGui::TextDisabled("(%zu more columns not shown)", csv.columns.size() - shown);
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Column to predict:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(260);
    if (ImGui::BeginCombo("##target", state.target_column.c_str())) {
        for (const auto& col : csv.columns) {
            const bool selected = (col == state.target_column);
            if (ImGui::Selectable(col.c_str(), selected)) {
                set_target_column(state, col);
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::TextDisabled("NeuroGraph learns to predict this column from the others.");
}

} // namespace neurograph::workbench
