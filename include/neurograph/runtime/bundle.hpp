#pragma once

// P1-07 — Standalone-inference bundle loader.
//
// A "bundle" is the directory the wizard writes after a successful train
// (see `orchestration::AutoOptions::bundle_out_dir`). It contains:
//
//   pipeline.ngpipe   — preprocess::FittedPipeline serialized JSON
//   model.ngm         — model weights (current schema: linear-probe v1)
//   manifest.json     — reporting::Manifest (optional, used for version
//                       checks and to surface "this bundle was trained
//                       on a different machine" warnings)
//   card.md           — model card (not loaded by runtime — for humans)
//
// Two consumers read this format:
//
//   - `predict.exe` (this ticket) — shells in, predicts CSV → CSV.
//   - `neurograph_rt` shared library (P1-08) — same loader, different
//     entry surface (C ABI).
//
// Loader discipline: never throw across the API boundary. Bad bundles
// produce a `Result<ModelBundle>` carrying a human-readable message
// suitable for printing to stderr. The CLI maps these to exit code 3.
//
// Schema evolution note: `model.ngm` carries `schema_version`. When the
// real `TrainingEngine`-trained model lands, `kind` will gain new values
// ("dense_relu_v1", "graph_v1", ...) and the loader will branch. Old
// bundles with `kind=="classification"`/`"regression"` (the linear
// probe) will keep loading because we keep them as a supported `kind`.

#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/util/result.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>

namespace neurograph::runtime {

/// Linear probe — classification head. Mirrors the orchestrator's
/// `LinearModelCls` 1:1 (this is the wire format for schema_version 1
/// bundles). Predict by softmax(W·x + b).
struct LinearClassificationModel {
    int                              n_classes  = 0;
    int                              n_features = 0;
    std::vector<std::vector<double>> W;          // K × D
    std::vector<double>              b;          // K
    std::vector<std::string>         class_labels;
};

/// Linear probe — regression head. Predict y = w·x + bias.
struct LinearRegressionModel {
    int                  n_features = 0;
    std::vector<double>  w;                       // D
    double               bias = 0.0;
};

// ---------------------------------------------------------------------------
// Neural network models — schema_version 2, kind "neural_cls_v1" /
// "neural_reg_v1". Inference state (Network, IOManager, Backprop) is held
// behind a pImpl pointer so this header does not pull in the heavy core/
// training headers into every consumer.
// ---------------------------------------------------------------------------

/// Opaque inference state: Network + IOManager + Backprop. Defined only
/// in bundle.cpp where the core/training headers are available.
struct NeuralNetInferenceState;

struct NeuralNetClassificationModel {
    int n_classes  = 0;
    int n_features = 0;
    std::vector<std::string>                  class_labels;
    std::unique_ptr<NeuralNetInferenceState>  state;

    NeuralNetClassificationModel();
    ~NeuralNetClassificationModel();
    NeuralNetClassificationModel(NeuralNetClassificationModel&&) noexcept;
    NeuralNetClassificationModel& operator=(NeuralNetClassificationModel&&) noexcept;
};

struct NeuralNetRegressionModel {
    int n_features = 0;
    int n_outputs  = 1;
    std::unique_ptr<NeuralNetInferenceState>  state;

    NeuralNetRegressionModel();
    ~NeuralNetRegressionModel();
    NeuralNetRegressionModel(NeuralNetRegressionModel&&) noexcept;
    NeuralNetRegressionModel& operator=(NeuralNetRegressionModel&&) noexcept;
};

/// What `load_bundle` returns. The pipeline is held in a unique_ptr
/// because `FittedPipeline` is move-only and pImpl-shaped. Neural-net
/// inference state is similarly opaque.
struct ModelBundle {
    std::unique_ptr<preprocess::FittedPipeline>  pipeline;
    std::variant<LinearClassificationModel,
                 LinearRegressionModel,
                 NeuralNetClassificationModel,
                 NeuralNetRegressionModel>        model;
    nlohmann::json                               manifest;   // empty {} if missing
    std::filesystem::path                        source_dir;

    bool is_classification() const noexcept {
        return std::holds_alternative<LinearClassificationModel>(model)
            || std::holds_alternative<NeuralNetClassificationModel>(model);
    }

    /// Returns 0 when not a classifier.
    int n_classes() const noexcept {
        if (const auto* m = std::get_if<LinearClassificationModel>(&model))
            return m->n_classes;
        if (const auto* m = std::get_if<NeuralNetClassificationModel>(&model))
            return m->n_classes;
        return 0;
    }

    /// Pretty class label by id. Safe when `i` is out of range (returns
    /// empty string).
    std::string class_label(int i) const;
};

/// One prediction. For classification, `predicted_class` is the
/// argmax index and `probabilities[k]` is the per-class probability.
/// For regression, `value` is the point estimate and the other fields
/// stay at their defaults.
struct Prediction {
    int                 predicted_class = -1;
    std::vector<double> probabilities;
    double              value = 0.0;
};

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/// Read a bundle from disk. `dir` should contain `pipeline.ngpipe` and
/// `model.ngm` at minimum. The returned bundle owns its pipeline +
/// manifest copy; freeing it is just dropping the value.
///
/// On any IO/parse/schema error the returned Result carries an error
/// message that names the file that failed and (when possible) the
/// expected vs. actual schema version. Suitable for printing to stderr.
util::Result<ModelBundle>
load_bundle(const std::filesystem::path& dir);

/// Run inference on a single raw row (keys = original CSV column
/// names). Internally calls `preprocess::apply_pipeline` then the
/// linear model's forward pass. Never throws.
Prediction
predict_row(const ModelBundle& bundle, const nlohmann::json& row);

/// Run inference on a pre-transformed feature vector (caller has
/// already applied the preprocess pipeline, e.g. when batching many
/// predictions on the same input). Same dispatch as `predict_row`
/// but skips `apply_pipeline`. Used by the C ABI's `ng_predict_vec`
/// fast path. Never throws.
Prediction
predict_vec(const ModelBundle& bundle, const std::vector<double>& features);

// ---------------------------------------------------------------------------
// CSV streaming helpers — exposed so the unit tests can exercise the
// same parser the CLI uses.
// ---------------------------------------------------------------------------

/// Parse one CSV line into fields. Handles quoted fields with embedded
/// commas and "" → " escaping. Trailing CR is stripped. Empty input
/// returns an empty vector.
std::vector<std::string> split_csv_line(const std::string& line);

/// Quote `s` per RFC 4180 if it contains comma/quote/newline; otherwise
/// returns it unchanged. Used by the CLI's CSV writer.
std::string quote_csv_field(const std::string& s);

} // namespace neurograph::runtime
