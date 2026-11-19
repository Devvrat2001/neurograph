// Step 4 — Results and using the model.
//
// The headline is the score on rows held back from training and model
// selection, next to what always guessing would score. Below: which models
// were tried, what the model relies on, whether more data would help, the
// detailed scores, and the buttons to keep and use the model: save it to
// a folder, open its report, and score another CSV file with it.

#include "wizard_ui.hpp"
#include "native_dialogs.hpp"
#include "neurograph/runtime/bundle.hpp"
#include "neurograph/runtime/score.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

const char* friendly_baseline(const std::string& name) {
    if (name == "majority")     return "Always the most common class";
    if (name == "mean")         return "Always the average";
    if (name == "random")       return "Random guess";
    if (name == "linear_probe") return "Quick straight-line fit";
    return name.c_str();
}

void render_headline(const orchestration::AutoResult& r) {
    const bool cls = std::holds_alternative<eval::ClassificationMetrics>(r.metrics);
    const std::string where = (r.evaluated_on == "test")
        ? "on " + std::to_string(r.n_evaluated) + " rows held back from training"
        : "on " + std::to_string(r.n_evaluated) +
          " check rows (the file was too small to hold back separate test rows)";

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.9f, 0.45f, 1.0f));
    if (cls) {
        const auto& m = std::get<eval::ClassificationMetrics>(r.metrics);
        ImGui::Text("Accuracy %s %s", pct(m.accuracy).c_str(), where.c_str());
    } else {
        const auto& m = std::get<eval::RegressionMetrics>(r.metrics);
        ImGui::Text("R\xc2\xb2 %s %s (typical error %s)", num(m.r2, 3).c_str(),
                    where.c_str(), num(m.rmse, 4).c_str());
    }
    ImGui::PopStyleColor();
    for (const auto& b : r.baselines.baselines) {
        if (b.name == "majority") {
            ImGui::TextDisabled("Always guessing the most common class would score %s.",
                                pct(b.primary_metric).c_str());
        }
    }
    ImGui::TextWrapped("%s", r.baselines.verdict.c_str());
    ImGui::Spacing();
    ImGui::TextWrapped("Model: %s", r.model_description.c_str());
    if (!r.model_choice_reason.empty()) ImGui::TextDisabled("%s", r.model_choice_reason.c_str());
}

void render_leaderboard(const orchestration::AutoResult& r, bool cls) {
    if (!ImGui::CollapsingHeader("Models tried", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::TextDisabled("Scores on the %d check rows used to choose between them.", r.n_val);
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
        if (c.seconds > 0.0) ImGui::Text("%.1f s%s", c.seconds, c.stopped_by_time ? " (time limit)" : "");
        else                 ImGui::TextDisabled("-");
        ImGui::TableSetColumnIndex(3);
        ImGui::TextWrapped("%s", c.detail.c_str());
    }
    ImGui::EndTable();
}

void render_importance(const orchestration::AutoResult& r, bool cls) {
    if (r.importance.empty()) return;
    if (!ImGui::CollapsingHeader("What the model relies on", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::TextDisabled("How much worse the model does when a column's values are shuffled.");
    const std::size_t n = std::min<std::size_t>(r.importance.size(), 12);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = r.importance[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::SetNextItemWidth(240);
        ImGui::ProgressBar(static_cast<float>(std::clamp(c.relative, 0.0, 1.0)), ImVec2(240, 0), "");
        ImGui::SameLine();
        ImGui::Text("%s", c.column.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("(%s %s when shuffled)", cls ? "accuracy falls" : "R\xc2\xb2 falls",
                            cls ? pct(c.score_drop).c_str() : num(c.score_drop, 3).c_str());
        ImGui::PopID();
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
            ImGui::TableSetColumnIndex(0); ImGui::Text("%d", p.n_samples);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(cls ? pct(p.val_metric).c_str() : num(p.val_metric, 3).c_str());
        }
        ImGui::EndTable();
    }
}

void render_details(const orchestration::AutoResult& r) {
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
            ImGui::Text("R\xc2\xb2 %s   RMSE %s   MAE %s", num(m.r2, 4).c_str(),
                        num(m.rmse, 4).c_str(), num(m.mae, 4).c_str());
        }
    }, r.metrics);

    ImGui::Spacing();
    ImGui::TextUnformatted("Compared with simple guesses on the same rows:");
    if (ImGui::BeginTable("baselines", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_SizingFixedFit)) {
        for (const auto& b : r.baselines.baselines) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(friendly_baseline(b.name));
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", b.primary_metric);
        }
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Your model");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", r.baselines.model_primary);
        ImGui::EndTable();
    }
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
    g_action_message = "Wrote " + std::to_string(r.value().rows) + " predictions to " +
                       path_utf8(*out);
    if (!r.value().filled_columns.empty()) {
        g_action_message += " (" + std::to_string(r.value().filled_columns.size()) +
                            " missing column(s) filled with typical values)";
    }
}

void render_use(WizardState& state, const orchestration::AutoResult& r) {
    ImGui::Separator();
    ImGui::TextUnformatted("Use the model");
    ImGui::Spacing();
    const bool have_bundle = !r.bundle_path.empty();
    ImGui::BeginDisabled(!have_bundle);
    if (ImGui::Button("Score a CSV file...", ImVec2(200, 32))) score_a_file(r.bundle_path);
    ImGui::SameLine();
    ImGui::Checkbox("Fill missing columns with typical values", &g_allow_missing);
    if (ImGui::Button("Save model as...", ImVec2(200, 32))) {
        if (auto dir = pick_folder(L"Choose a folder for the model")) save_bundle_to(r.bundle_path, *dir);
    }
    ImGui::SameLine();
    if (ImGui::Button("Open model report", ImVec2(200, 32))) open_with_shell(r.bundle_path / "card.md");
    ImGui::SameLine();
    if (ImGui::Button("Open model folder", ImVec2(200, 32))) open_with_shell(r.bundle_path);
    ImGui::EndDisabled();

    if (!g_action_message.empty()) {
        ImGui::TextColored(g_action_ok ? ImVec4(0.45f, 0.9f, 0.45f, 1.0f)
                                       : ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "%s", g_action_message.c_str());
    }
    ImGui::TextDisabled("The model folder also works with neurograph_predict.exe (command line) "
                        "and neurograph_rt.dll (from your own programs).");
    ImGui::Spacing();
    if (ImGui::Button("Train again with different settings", ImVec2(300, 28))) {
        g_action_message.clear();
        reset_to_train(state);
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
    render_leaderboard(r, cls);
    render_importance(r, cls);
    render_curve(r, cls);
    render_details(r);
    ImGui::Spacing();
    render_use(state, r);
}

} // namespace neurograph::workbench
