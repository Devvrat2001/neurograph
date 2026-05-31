// Performance benchmark for the P1-09 wizard's per-frame render cost.
//
// Closes P1-09 AC2 + DoD line 1473:
//   "Each wizard step renders in < 16 ms (60 FPS) with a 1M-row
//    dataset loaded."
//
// Approach: headless ImGui (no backend, no GPU) — for this AC the
// "render" cost is dominated by ImGui's draw-list construction +
// the wizard's per-frame iteration over its own state. Actual GPU
// submit time isn't part of the per-frame budget the AC measures
// (the ImGui Test Engine library would be overkill — it's designed
// for click/keyboard event scripting, which we don't need here).
//
// Note on the "1M-row dataset" caveat: the wizard's CSV preview is
// deliberately capped at 5 rows by `peek_csv_metadata` (see
// `wizard_state.hpp:99`). Render cost is therefore O(preview_cap),
// not O(rows). The dataset's `row_count` field is set to 1,000,000 so
// any code path that iterates `row_count` (rather than the preview)
// gets exercised at scale.
//
// Gating policy mirrors the other tests in this directory:
//   - Default gate: 16 ms per frame (the spec target).
//   - Override via NEUROGRAPH_PERF_WIZARD_GATE_MS.
//   - Set NEUROGRAPH_PERF_SKIP_GATE=1 to record without failing.

#include "neurograph/dataqc/report.hpp"
#include "neurograph/eval/baselines.hpp"
#include "neurograph/eval/classification.hpp"
#include "neurograph/eval/learning_curve.hpp"
#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "wizard_ui.hpp"   // render_step_* + WizardRuntime

#include "imgui.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using neurograph::workbench::CsvMetadata;
using neurograph::workbench::WizardRuntime;
using neurograph::workbench::WizardState;
using neurograph::workbench::WizardStep;
using neurograph::workbench::TrainingStatus;
using neurograph::workbench::render_wizard;

namespace {

long env_long_or(const char* name, long fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    try {
        return std::stol(v);
    } catch (...) {
        return fallback;
    }
}

bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}

// Headless ImGui context — sized + initialized once, torn down on exit.
// The font atlas is built CPU-side; we hand ImGui a non-null fake
// `TexID` so its draw calls don't assert. No GPU backend is registered
// because `ImGui::Render()` only produces ImDrawData; nothing here ever
// submits it.
class HeadlessImGui {
public:
    HeadlessImGui() {
        ctx_ = ImGui::CreateContext();
        ImGui::SetCurrentContext(ctx_);
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize             = ImVec2(1280.0f, 720.0f);
        io.DeltaTime               = 1.0f / 60.0f;
        io.IniFilename             = nullptr;   // don't try to write imgui.ini
        io.LogFilename             = nullptr;
        io.BackendPlatformName     = "headless";
        io.BackendRendererName     = "headless";

        // Build font atlas manually so the renderer doesn't assert
        // on a missing texture.
        unsigned char* tex_pixels = nullptr;
        int tex_w = 0, tex_h = 0;
        io.Fonts->AddFontDefault();
        io.Fonts->GetTexDataAsAlpha8(&tex_pixels, &tex_w, &tex_h);
        // Set a non-null tex id so any code that asserts
        // tex_id != 0 doesn't fire. Value is opaque; we never
        // dereference it because we don't render.
        // Modern ImGui (1.91+) uses an integer-typed ImTextureID
        // (was a void*); use a static_cast to cover both ABIs.
        io.Fonts->SetTexID(static_cast<ImTextureID>(1));
        ImGui::StyleColorsDark();
    }

    ~HeadlessImGui() {
        ImGui::DestroyContext(ctx_);
    }

    // One headless frame around `body`. Returns wall time in ms.
    template <typename F>
    double time_one_frame(F body) {
        using clock = std::chrono::steady_clock;
        ImGui::NewFrame();
        // Wrap the body in a top-level window — ImGui requires
        // widgets to be inside a Begin/End pair. Fixed window so
        // layout doesn't drift across runs.
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(1280, 720));
        ImGui::Begin("##perf_root", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
                   | ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoCollapse);

        const auto t0 = clock::now();
        body();
        const auto t1 = clock::now();

        ImGui::End();
        ImGui::Render();
        // Touch the draw data to defeat dead-code elimination.
        [[maybe_unused]] volatile int sink =
            ImGui::GetDrawData()->TotalVtxCount;
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

private:
    ImGuiContext* ctx_ = nullptr;
};

void report_and_gate(const char* label, double elapsed_ms,
                     long default_gate_ms, const char* gate_env_name) {
    const long gate_ms = env_long_or(gate_env_name, default_gate_ms);
    const bool skip    = env_truthy("NEUROGRAPH_PERF_SKIP_GATE");

    std::cout << "[perf] " << label
              << " elapsed=" << elapsed_ms << "ms"
              << " gate=" << gate_ms << "ms"
              << (skip ? " (gate skipped)" : "") << '\n';
    ::testing::Test::RecordProperty(std::string(label) + "_ms",
                                    static_cast<int>(elapsed_ms));
    if (!skip) {
        EXPECT_LT(elapsed_ms, static_cast<double>(gate_ms))
            << label << " perf regression: " << elapsed_ms
            << " ms exceeded gate of " << gate_ms << " ms. "
            << "Set NEUROGRAPH_PERF_SKIP_GATE=1 to record without failing, "
            << "or " << gate_env_name << "=<ms> to override.";
    }
}

// Fill a WizardState that looks like a freshly-loaded 1M-row CSV with
// a passing data-quality scan + a couple of warnings + fixes available.
// Mutates `s` in place because WizardState is non-movable (atomic +
// mutex members).
void fill_loaded_state(WizardState& s) {
    CsvMetadata meta;
    meta.path      = "C:/synth/big.csv";
    meta.columns   = {"f1", "f2", "f3", "f4", "f5", "label"};
    meta.row_count = 1'000'000;
    // The preview is capped at 5 rows by design — this is what the
    // render layer actually iterates over.
    for (int i = 0; i < 5; ++i) {
        meta.preview_rows.push_back({
            "1.234", "5.678", "9.012", "0.345", "6.789",
            (i % 2 == 0) ? "A" : "B"
        });
    }
    s.csv = std::move(meta);
    s.target_column = "label";

    // Synthesize a DataQualityReport with two warnings so Step 2
    // renders a non-empty findings list + fix buttons (the wizard's
    // fix-applicable detector picks up MISSING_VALUES + CONSTANT_COLUMN).
    neurograph::dataqc::DataQualityReport dq;
    dq.n_rows       = 1'000'000;
    dq.n_cols       = 6;
    dq.summary_line = "1,000,000 rows, 5 features, 2 classes - looks good.";
    dq.trainable    = true;
    dq.findings.push_back({
        neurograph::dataqc::Severity::Warning, "MISSING_VALUES", "f3",
        "Column 'f3' has 12 missing values (0.001%).", "Auto-impute median.",
    });
    dq.findings.push_back({
        neurograph::dataqc::Severity::Warning, "CONSTANT_COLUMN", "f5",
        "Column 'f5' has only one unique value.", "Drop the column.",
    });
    s.dq_report = std::move(dq);
}

} // namespace

// ---------------------------------------------------------------------------
// Step 1 — Load. Renders the file path, browse button, target dropdown,
// preview table.
// ---------------------------------------------------------------------------
TEST(PerfWizardRender, Step1LoadFitsInFrameBudget) {
    HeadlessImGui imgui;
    WizardState state;
    fill_loaded_state(state);
    state.current_step = WizardStep::Load;
    auto rt = neurograph::workbench::make_wizard_runtime();

    // Warm — first frame builds the font atlas + lays out widgets.
    (void) imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    const double elapsed = imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    report_and_gate("wizard_step_load_1m_rows", elapsed,
                    /*default_gate_ms=*/16,
                    "NEUROGRAPH_PERF_WIZARD_GATE_MS");
}

// ---------------------------------------------------------------------------
// Step 2 — Data quality. Renders the summary line, findings list with
// fix buttons, post-transform shape preview.
// ---------------------------------------------------------------------------
TEST(PerfWizardRender, Step2QualityFitsInFrameBudget) {
    HeadlessImGui imgui;
    WizardState state;
    fill_loaded_state(state);
    state.current_step = WizardStep::Quality;
    auto rt = neurograph::workbench::make_wizard_runtime();

    (void) imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    const double elapsed = imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    report_and_gate("wizard_step_quality_1m_rows", elapsed,
                    /*default_gate_ms=*/16,
                    "NEUROGRAPH_PERF_WIZARD_GATE_MS");
}

// ---------------------------------------------------------------------------
// Step 3 — Train. Renders budget radio + task combo + Train button
// (Idle status — heaviest visible widget set).
// ---------------------------------------------------------------------------
TEST(PerfWizardRender, Step3TrainFitsInFrameBudget) {
    HeadlessImGui imgui;
    WizardState state;
    fill_loaded_state(state);
    state.current_step = WizardStep::Train;
    state.status.store(TrainingStatus::Idle);
    auto rt = neurograph::workbench::make_wizard_runtime();

    (void) imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    const double elapsed = imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    report_and_gate("wizard_step_train_1m_rows", elapsed,
                    /*default_gate_ms=*/16,
                    "NEUROGRAPH_PERF_WIZARD_GATE_MS");
}

// ---------------------------------------------------------------------------
// Step 4 — Results. Renders the hero metric + expandable per-class
// table + 3 export buttons. Heaviest layout in the wizard.
// ---------------------------------------------------------------------------
TEST(PerfWizardRender, Step4ResultsFitsInFrameBudget) {
    HeadlessImGui imgui;
    WizardState state;
    fill_loaded_state(state);
    state.current_step = WizardStep::Results;

    // Synthesize a result with classification metrics + baselines so
    // the per-class table + baseline comparison both render.
    neurograph::orchestration::AutoResult res;
    {
        neurograph::eval::ClassificationMetrics cm;
        cm.accuracy       = 0.93;
        cm.macro_f1       = 0.92;
        cm.weighted_f1    = 0.93;
        cm.roc_auc_macro  = 0.97;
        cm.brier_score    = 0.10;
        for (int c = 0; c < 3; ++c) {
            neurograph::eval::PerClassMetrics pm;
            pm.class_label = "class_" + std::to_string(c);
            pm.precision   = 0.9;
            pm.recall      = 0.9;
            pm.f1          = 0.9;
            pm.support     = 50;
            cm.per_class.push_back(std::move(pm));
        }
        cm.confusion.n_classes = 3;
        cm.confusion.cells = {{45, 3, 2}, {1, 47, 2}, {0, 5, 45}};
        res.metrics = std::move(cm);
    }
    res.baselines.lift_vs_best_baseline = 0.12;
    res.baselines.verdict = "✓ Model is learning. Beats best baseline by +0.12.";
    res.summary_one_liner = "Trained classifier: accuracy = 0.93 …";
    state.result = std::move(res);

    auto rt = neurograph::workbench::make_wizard_runtime();
    (void) imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    const double elapsed = imgui.time_one_frame([&] {
        (void) render_wizard(state, *rt);
    });
    report_and_gate("wizard_step_results_1m_rows", elapsed,
                    /*default_gate_ms=*/16,
                    "NEUROGRAPH_PERF_WIZARD_GATE_MS");
}
