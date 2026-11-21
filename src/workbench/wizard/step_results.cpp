// Step 4 — Results and using the model.
//
// The headline is the score on rows held back from training and model
// selection, next to what always guessing would score. Straight after it
// come the buttons to keep and use the model (save it to a folder, score
// another CSV file with it, open its report), so they are in view without
// scrolling. Below them: which models were tried, what the model relies
// on, whether more data would help, and the detailed scores.

#include "wizard_ui.hpp"
#include "native_dialogs.hpp"
#include "theme.hpp"
#include "neurograph/runtime/bundle.hpp"
#include "neurograph/runtime/score.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <system_error>
#include <variant>

namespace neurograph::workbench {

namespace {

namespace fs = std::filesystem;

// Result of the last Save / Score action, shown under the buttons.
std::string g_action_message;
bool        g_action_ok = true;
bool        g_allow_missing = false;

std::string pct(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f%%", v * 100.0);
    return b;
}

std::string num(double v, int digits) {
    if (!std::isfinite(v)) return "n/a";
    char b[48];
    std::snprintf(b, sizeof b, "%.*f", digits, v);
    return b;
}

// A quantity in the target's own units, to about three significant
// figures: 48.5451 -> "48.5", 0.0123456 -> "0.0123".
std::string sig3(double v) {
    if (!std::isfinite(v)) return "n/a";
    const double a = std::fabs(v);
    char b[48];
    if (a >= 100.0)     std::snprintf(b, sizeof b, "%.0f", v);
    else if (a >= 10.0) std::snprintf(b, sizeof b, "%.1f", v);
    else if (a >= 1.0)  std::snprintf(b, sizeof b, "%.2f", v);
    else                std::snprintf(b, sizeof b, "%.3g", v);
    return b;
}

std::string seconds(double s) {
    if (s < 0.05) return "under 0.1 s";
    char b[32];
    if (s < 60.0) std::snprintf(b, sizeof b, "%.1f s", s);
    else          std::snprintf(b, sizeof b, "%d min %02d s", static_cast<int>(s) / 60,
                                static_cast<int>(s) % 60);
    return b;
}

// Share of the most important column, for the importance list.
std::string share(double relative) {
    if (relative <= 0.0)  return "0%";
    if (relative < 0.005) return "under 1%";
    char b[16];
    std::snprintf(b, sizeof b, "%.0f%%", relative * 100.0);
    return b;
}

// Wide enough for the longest label, so a row of buttons can share a width
// without cutting any label off.
float button_width(std::initializer_list<const char*> labels) {
    float w = 0.0f;
    for (const char* l : labels) w = std::max(w, ImGui::CalcTextSize(l).x);
    return w + ImGui::GetStyle().FramePadding.x * 2.0f;
}

const char* friendly_baseline(const std::string& name) {
    if (name == "majority")     return "Always the most common class";
    if (name == "mean")         return "Always the average";
    if (name == "random")       return "Random guess";
    if (name == "linear_probe") return "Quick straight-line fit";
    return name.c_str();
}

void render_headline(const orchestration::AutoResult& r) {
    const bool cls = std::holds_alternative<eval::ClassificationMetrics>(r.metrics);
    std::string where = (r.evaluated_on == "test")
        ? "on " + with_commas(static_cast<std::size_t>(r.n_evaluated)) +
          " rows held back from training"
        : "on " + with_commas(static_cast<std::size_t>(r.n_evaluated)) +
          " check rows (the file was too small to hold back separate test rows)";

    // The score in a larger size; where it was measured below it.
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.6f);
    if (cls) {
        const auto& m = std::get<eval::ClassificationMetrics>(r.metrics);
        ImGui::Text("Accuracy %s", pct(m.accuracy).c_str());
    } else {
        const auto& m = std::get<eval::RegressionMetrics>(r.metrics);
        ImGui::Text("R\xc2\xb2 %s", num(m.r2, 3).c_str());
        where += ", typical error " + sig3(m.rmse);
    }
    ImGui::PopFont();
    ImGui::TextUnformatted(where.c_str());
    for (const auto& b : r.baselines.baselines) {
        if (b.name == "majority") {
            ImGui::TextDisabled("Always guessing the most common class would score %s.",
                                pct(b.primary_metric).c_str());
        }
    }
    ImGui::Spacing();
    ImGui::TextWrapped("%s", r.baselines.verdict.c_str());
    ImGui::TextWrapped("Model: %s", r.model_description.c_str());
    if (!r.model_choice_reason.empty()) text_hint(r.model_choice_reason);
}

void save_bundle_to(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::create_directories(to, ec);
    for (const char* f : {"pipeline.ngpipe", "model.ngm", "manifest.json", "card.md"}) {
        if (ec) break;
        fs::copy_file(from / f, to / f, fs::copy_options::overwrite_existing, ec);
    }
    g_action_ok = !ec;
    g_action_message = ec ? "Could not save the model: " + ec.message()
                          : "Saved the model to " + path_utf8(to);
}

void score_a_file(const fs::path& bundle_dir) {
    const auto in = pick_csv_file();
    if (!in) return;
    fs::path suggested = in->parent_path() / (in->stem().wstring() + L"_predictions.csv");
    const auto out = pick_save_csv(suggested);
    if (!out) return;

    auto bundle = runtime::load_bundle(bundle_dir);
    if (!bundle.ok()) {
        g_action_ok = false;
        g_action_message = "Could not load the model: " + bundle.error().message;
        return;
    }
    runtime::ScoreOptions so;
    so.allow_missing = g_allow_missing;
    auto r = runtime::score_file(bundle.value(), *in, *out, so);
    if (!r.ok()) {
        g_action_ok = false;
        g_action_message = "Could not score the file: " + r.error().message;
        return;
    }
    g_action_ok = true;
    g_action_message = "Wrote " + with_commas(r.value().rows) + " predictions to " +
                       path_utf8(*out);
    if (!r.value().filled_columns.empty()) {
        g_action_message += " (" + std::to_string(r.value().filled_columns.size()) +
                            " missing column(s) filled with typical values)";
    }
}

void render_use(WizardState& state, const orchestration::AutoResult& r) {
    ImGui::SeparatorText("Use the model");
    const char* save   = "Save model as...";
    const char* score  = "Score a CSV file...";
    const char* report = "Open model report";
    const char* folder = "Open model folder";
    const ImVec2 b(button_width({save, score, report, folder}), 0.0f);

    ImGui::BeginDisabled(r.bundle_path.empty());
    if (accent_button(save, b)) {
        if (auto dir = pick_folder(L"Choose a folder for the model")) save_bundle_to(r.bundle_path, *dir);
    }
    ImGui::SameLine();
    if (ImGui::Button(score, b)) score_a_file(r.bundle_path);
    ImGui::SameLine();
    if (ImGui::Button(report, b)) open_with_shell(r.bundle_path / "card.md");
    ImGui::SameLine();
    if (ImGui::Button(folder, b)) open_with_shell(r.bundle_path);
    ImGui::Checkbox("When scoring a file, fill in columns it lacks with typical values",
                    &g_allow_missing);
    ImGui::EndDisabled();

    if (!g_action_message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, g_action_ok ? colors::good : colors::bad);
        ImGui::TextWrapped("%s", g_action_message.c_str());
        ImGui::PopStyleColor();
    }
    text_hint("The model folder also works with neurograph_predict.exe (command line) "
              "and neurograph_rt.dll (from your own programs).");
    if (ImGui::Button("Train again with different settings")) {
        g_action_message.clear();
        reset_to_train(state);
    }
    ImGui::Spacing();
}

void render_leaderboard(const orchestration::AutoResult& r, bool cls) {
    if (!ImGui::CollapsingHeader("Models tried", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::TextDisabled("Scores on the %s check rows used to choose between them.",
                        with_commas(static_cast<std::size_t>(r.n_val)).c_str());
    if (!ImGui::BeginTable("leaderboard", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                             ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn("Model");
    ImGui::TableSetupColumn(cls ? "Accuracy" : "R\xc2\xb2");
    ImGui::TableSetupColumn("Time");
    ImGui::TableSetupColumn("Detail", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableHeadersRow();
    for (const auto& c : r.leaderboard) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        if (c.chosen) ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "%s (chosen)", c.name.c_str());
        else          ImGui::TextUnformatted(c.name.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(cls ? pct(c.val_score).c_str() : num(c.val_score, 3).c_str());
        ImGui::TableSetColumnIndex(2);
        if (c.seconds > 0.0) {
            ImGui::Text("%s%s", seconds(c.seconds).c_str(), c.stopped_by_time ? " (time limit)" : "");
        } else {
            ImGui::TextDisabled("-");
        }
        ImGui::TableSetColumnIndex(3);
        ImGui::TextWrapped("%s", c.detail.c_str());
    }
    ImGui::EndTable();
}

// The bar is the same quantity as the number beside it: for regression
// both are the fall in R²; for classification, the rise in log loss as a
// share of the most important column's (accuracy moves in steps of one
// row, too coarsely to rank columns; the model report lists it too).
void render_importance(const orchestration::AutoResult& r, bool cls) {
    if (r.importance.empty()) return;
    if (!ImGui::CollapsingHeader("What the model relies on", ImGuiTreeNodeFlags_DefaultOpen)) return;
    text_hint(cls
        ? "How much worse the model's predictions get when a column's values are shuffled "
          "between rows, compared with the column it relies on most."
        : "How much R\xc2\xb2 falls when a column's values are shuffled between rows.");
    const std::size_t n = std::min<std::size_t>(r.importance.size(), 12);
    const float bar_w = ImGui::GetFontSize() * 15.0f;
    const float line  = ImGui::GetTextLineHeight();
    const float bar_h = line * 0.6f;
    const ImU32 track = ImGui::GetColorU32(ImGuiCol_FrameBg);
    const ImU32 fill  = ImGui::GetColorU32(ImGuiCol_PlotHistogram);
    if (ImGui::BeginTable("importance", 3, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthFixed, bar_w);
        ImGui::TableSetupColumn("Value");
        for (std::size_t i = 0; i < n; ++i) {
            const auto& c = r.importance[i];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(c.column.c_str());
            // Drawn directly rather than as a ProgressBar, whose frame
            // padding would push the text in the next cell down.
            ImGui::TableSetColumnIndex(1);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float  y = p.y + (line - bar_h) * 0.5f;
            const float  w = bar_w * static_cast<float>(std::clamp(c.relative, 0.0, 1.0));
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + bar_w, y + bar_h), track, 2.0f);
            if (w >= 1.0f) dl->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + w, y + bar_h), fill, 2.0f);
            ImGui::Dummy(ImVec2(bar_w, line));
            ImGui::TableSetColumnIndex(2);
            if (cls) ImGui::TextDisabled("%s", share(c.relative).c_str());
            else     ImGui::TextDisabled("R\xc2\xb2 falls %s", num(c.score_drop, 3).c_str());
        }
        ImGui::EndTable();
    }
    if (r.importance.size() > n) {
        ImGui::TextDisabled("%zu more columns matter less.", r.importance.size() - n);
    }
}

void render_curve(const orchestration::AutoResult& r, bool cls) {
    if (r.curve.message.empty()) return;
    if (!ImGui::CollapsingHeader("Would more data help?")) return;
    ImGui::TextWrapped("%s", r.curve.message.c_str());
    if (r.curve.points.empty()) return;
    if (ImGui::BeginTable("curve", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Training rows");
        ImGui::TableSetupColumn(cls ? "Accuracy on check rows" : "R\xc2\xb2 on check rows");
        ImGui::TableHeadersRow();
        for (const auto& p : r.curve.points) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(with_commas(static_cast<std::size_t>(p.n_samples)).c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(cls ? pct(p.val_metric).c_str() : num(p.val_metric, 3).c_str());
        }
        ImGui::EndTable();
    }
}

void render_details(const orchestration::AutoResult& r, bool cls) {
    if (!ImGui::CollapsingHeader("Detailed scores")) return;
    std::visit([&](const auto& m) {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, eval::ClassificationMetrics>) {
            ImGui::Text("Accuracy %s, macro F1 %s, ROC AUC %s", pct(m.accuracy).c_str(),
                        num(m.macro_f1, 3).c_str(), num(m.roc_auc_macro, 3).c_str());
            if (!m.per_class.empty()
                && ImGui::BeginTable("per_class", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Class");
                ImGui::TableSetupColumn("Precision");
                ImGui::TableSetupColumn("Recall");
                ImGui::TableSetupColumn("F1");
                ImGui::TableSetupColumn("Rows");
                ImGui::TableHeadersRow();
                for (const auto& pc : m.per_class) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(pc.class_label.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", pc.precision);
                    ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", pc.recall);
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", pc.f1);
                    ImGui::TableSetColumnIndex(4); ImGui::Text("%d", pc.support);
                }
                ImGui::EndTable();
            }
        } else {
            ImGui::Text("R\xc2\xb2 %s, typical error (RMSE) %s, average error (MAE) %s",
                        num(m.r2, 3).c_str(), sig3(m.rmse).c_str(), sig3(m.mae).c_str());
        }
    }, r.metrics);

    ImGui::Spacing();
    ImGui::TextUnformatted("Compared with simple guesses on the same rows:");
    const auto score = [cls](double v) { return cls ? pct(v) : num(v, 3); };
    if (ImGui::BeginTable("baselines", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_SizingFixedFit)) {
        for (const auto& b : r.baselines.baselines) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(friendly_baseline(b.name));
            ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(score(b.primary_metric).c_str());
        }
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Your model");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(score(r.baselines.model_primary).c_str());
        ImGui::EndTable();
    }
}

} // namespace

void render_step_results(WizardState& state, WizardRuntime& /*rt*/) {
    if (!state.result) {
        ImGui::TextUnformatted("No result yet. Go back and train.");
        return;
    }
    const auto& r = *state.result;
    const bool cls = std::holds_alternative<eval::ClassificationMetrics>(r.metrics);

    render_headline(r);
    ImGui::Spacing();
    render_use(state, r);
    render_leaderboard(r, cls);
    render_importance(r, cls);
    render_curve(r, cls);
    render_details(r, cls);
}

} // namespace neurograph::workbench
