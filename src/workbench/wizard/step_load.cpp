// P1-09 — Step 1: Load Data.
//
// The spec asks for a drop zone; we use a path input + "Load" button
// for now. Drag-drop in ImGui on Windows requires Win32 RegisterDragDrop
// which is a larger chunk of work and can land as a follow-up. The
// path-input flow is keyboard-navigable (matches "keyboard nav"
// acceptance criterion) and lets QA paste test paths without a mouse.

#include "wizard_ui.hpp"
#include "neurograph/workbench/csv_preview.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <cstring>

namespace neurograph::workbench {

namespace {

// Buffer for the file path input. ImGui::InputText needs a char
// buffer; we back it with a static so the field keeps its contents
// across frames.
char g_path_buf[1024] = "";

} // namespace

void render_step_load(WizardState& state, WizardRuntime& /*rt*/) {
    ImGui::TextWrapped("Point NeuroGraph at a CSV file with your training data. "
                       "The last column is used as the label by default — you can "
                       "change that below.");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-180.0f);
    ImGui::InputTextWithHint("##csv_path", "C:/path/to/data.csv",
                              g_path_buf, sizeof(g_path_buf));
    ImGui::SameLine();
    const bool load_clicked = ImGui::Button("Load", ImVec2(170, 0));

    if (load_clicked && g_path_buf[0] != '\0') {
        auto result = peek_csv_metadata(g_path_buf, /*preview=*/5);
        if (result.ok()) {
            set_csv(state, std::move(result).value());
        } else {
            state.load_error = result.error().message;
            // Leaving state.csv untouched — the user can correct the
            // path and try again without losing their prior load.
        }
    }

    if (!state.load_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s",
                           state.load_error.c_str());
    }

    if (state.csv) {
        const auto& csv = *state.csv;
        ImGui::Separator();
        ImGui::Text("File: %s", csv.path.string().c_str());
        ImGui::Text("Rows: %zu   |   Columns: %zu",
                    csv.row_count, csv.columns.size());
        ImGui::Spacing();

        // Preview table — first 5 rows.
        if (!csv.preview_rows.empty()
            && ImGui::BeginTable("preview", static_cast<int>(csv.columns.size()),
                                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
                                | ImGuiTableFlags_SizingStretchProp)) {
            for (const auto& col : csv.columns) {
                ImGui::TableSetupColumn(col.c_str());
            }
            ImGui::TableHeadersRow();
            for (const auto& row : csv.preview_rows) {
                ImGui::TableNextRow();
                for (std::size_t i = 0; i < csv.columns.size(); ++i) {
                    ImGui::TableSetColumnIndex(static_cast<int>(i));
                    if (i < row.size()) {
                        ImGui::TextUnformatted(row[i].c_str());
                    }
                }
            }
            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("Target column:");
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

        ImGui::TextDisabled("NeuroGraph will predict this column from the rest.");
    }
}

} // namespace neurograph::workbench
