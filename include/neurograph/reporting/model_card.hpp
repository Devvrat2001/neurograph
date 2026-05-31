#pragma once

// P1-12 — Model Card Generator.
//
// After every successful training run we emit a `card.md` next to the
// model bundle. It documents what the model is, how it was trained,
// what data went in, and how well it scored — in the format buyers in
// regulated industries (healthcare, finance) increasingly require for
// EU AI Act / NIST AI RMF compliance.
//
// Implementation note — no templating engine:
//
// The original spec mentioned an inja template file. We render directly
// from C++ instead. Reasons:
//   - Avoids adding a vcpkg dependency just to substitute strings.
//   - Markdown is structured enough that an `ostringstream` is clearer
//     than chasing template syntax.
//   - Determinism is easier to guarantee when there's no template
//     parser between the inputs and the output.
//
// Determinism contract: same `ModelCardInputs` → byte-identical markdown.
// All time-varying data (timestamps, version) lives on the input struct
// so callers control reproducibility.

#include "neurograph/dataqc/report.hpp"
#include "neurograph/eval/baselines.hpp"
#include "neurograph/eval/classification.hpp"
#include "neurograph/eval/learning_curve.hpp"
#include "neurograph/eval/regression.hpp"
#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/training/training_engine.hpp"

#include <filesystem>
#include <string>

namespace neurograph::reporting {

struct ModelCardInputs {
    /// Display name shown in the H1. Falls back to "Untrained model".
    std::string                          model_name;

    /// Free-form, but the renderer special-cases the strings
    /// "classification" and "regression" for the Evaluation section.
    std::string                          task_kind;

    dataqc::DataQualityReport            dq_report;
    preprocess::PipelineConfig           pipe_config;
    training::SupervisedConfig           train_config;

    /// Exactly one of these should be set (matching `task_kind`). The
    /// renderer skips the section if the matching pointer is null.
    const eval::ClassificationMetrics*   cls_metrics = nullptr;
    const eval::RegressionMetrics*       reg_metrics = nullptr;

    eval::BaselineReport                 baseline;
    eval::LearningCurveReport            curve;

    /// Free-form prose the wizard collects from the user. Empty → the
    /// section is rendered with a "(not specified)" placeholder so the
    /// auditor can see the omission.
    std::string                          user_intended_use;
    std::string                          user_out_of_scope;

    // --- Reproducibility metadata --------------------------------------

    /// e.g. "0.1.0". Empty → omitted from the footer.
    std::string                          neurograph_version;

    /// Caller's responsibility (typically `std::format("{:%FT%TZ}", utc_now())`).
    /// Empty → omitted from the footer. Kept on the input struct so the
    /// determinism contract holds — every byte is a function of inputs.
    std::string                          generated_at_iso8601;

    /// Original CSV path used for training. Shown in the data summary;
    /// empty → omitted.
    std::string                          dataset_path;
};

/// Render a markdown model card. Pure function: no I/O, no globals,
/// deterministic for a given input. Returns the markdown source.
std::string render_model_card(const ModelCardInputs& in);

/// Convenience: `render_model_card(in)` then write the result to `path`.
/// Throws `std::runtime_error` on filesystem error.
void save_model_card(const ModelCardInputs& in,
                     const std::filesystem::path& path);

} // namespace neurograph::reporting
