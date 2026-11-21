#pragma once

// Native model families for tabular data.
//
// The automatic trainer (models/automl.hpp) fits several of these on the
// same train/validation split and keeps whichever generalises best:
//
//   linear_v1  — multinomial logistic regression / ridge regression.
//   mlp_v1     — dense ReLU network. With one hidden layer it can widen
//                itself during training by splitting its most stressed
//                units (the self-organizing idea, in a dense
//                representation with biases and a proper optimiser).
//   gbt_v1     — histogram gradient-boosted decision trees.
//   blend_v1   — weighted average of other models.
//
// Every model is immutable once trained. `predict` is const and keeps no
// per-call state, so one instance can serve many threads without a lock.
//
// Inputs are the feature vectors produced by preprocess::apply_pipeline
// (already imputed and scaled). Models never see raw CSV values.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace neurograph::models {

enum class Task { Classification, Regression };

class Model {
public:
    virtual ~Model() = default;

    /// Stable identifier written to model.ngm ("gbt_v1", "mlp_v1", ...).
    virtual const char* kind() const noexcept = 0;
    virtual Task        task() const noexcept = 0;
    virtual int         n_features() const noexcept = 0;

    /// Number of classes for classification, 1 for regression.
    virtual int         n_outputs() const noexcept = 0;

    /// Classification: writes `n_outputs()` probabilities that sum to 1.
    /// Regression: writes the prediction to `out[0]`.
    /// `x` must hold `n_features()` values.
    virtual void predict(const double* x, double* out) const = 0;

    virtual nlohmann::json to_json() const = 0;

    /// One line for the results screen, e.g.
    /// "Gradient-boosted trees (143 trees, up to 31 leaves each)".
    virtual std::string describe() const = 0;

    /// Rough size indicator (weights, or tree nodes).
    virtual std::size_t parameter_count() const noexcept = 0;
};

/// Rebuild a model from `Model::to_json()` output. Throws
/// std::runtime_error with a readable message on malformed input.
std::unique_ptr<Model> model_from_json(const nlohmann::json& j);

// ---------------------------------------------------------------------------
// Training data
// ---------------------------------------------------------------------------

/// Row-major dense matrix. Trainers work on this rather than on
/// vector<vector<double>> so inner loops run over contiguous memory.
struct Matrix {
    std::size_t         rows = 0;
    std::size_t         cols = 0;
    std::vector<double> data;

    const double* row(std::size_t r) const { return data.data() + r * cols; }
    double*       row(std::size_t r)       { return data.data() + r * cols; }

    /// Copies `rows_in`. Short rows are zero-padded to the widest row.
    static Matrix from_rows(const std::vector<std::vector<double>>& rows_in);
};

struct TrainSet {
    Task   task      = Task::Classification;
    int    n_classes = 0;                 // classification only

    Matrix X_train;
    Matrix X_val;

    std::vector<int>    y_train_cls, y_val_cls;   // class index per row
    std::vector<double> y_train_reg, y_val_reg;   // target per row
};

/// Shared knobs for every trainer.
struct FitControl {
    /// Wall-clock limit for this fit. 0 means no limit. Trainers check it
    /// between mini-batches / boosting rounds and return the best model
    /// found so far when it expires.
    double        time_limit_seconds = 0.0;
    std::uint64_t seed               = 42;

    /// Polled regularly; return true to stop early. May be empty.
    std::function<bool()> cancelled;

    /// Progress in [0, 1] plus a short status line. May be empty.
    std::function<void(double, const std::string&)> progress;
};

struct FitReport {
    std::unique_ptr<Model> model;      // null only if the fit was cancelled before any model existed
    std::string name;                  // "Linear model", "Neural network", "Gradient-boosted trees"
    std::string detail;                // e.g. "hidden layer grew 16 -> 40"
    double      val_loss   = std::numeric_limits<double>::infinity();
    int         iterations = 0;        // epochs or boosting rounds kept
    double      seconds    = 0.0;
    bool        stopped_by_time = false;
};

/// Mean log-loss (classification) or mean squared error (regression).
double evaluate_loss(const Model& model,
                     const Matrix& X,
                     const std::vector<int>&    y_cls,
                     const std::vector<double>& y_reg);

// ---------------------------------------------------------------------------
// Trainers
// ---------------------------------------------------------------------------

struct MlpOptions {
    /// Hidden layer widths. Empty trains a linear model.
    std::vector<int> hidden;

    /// When true and there is exactly one hidden layer, the layer starts
    /// at `hidden[0]` and widens itself while that keeps helping.
    bool   grow          = true;
    int    max_hidden    = 256;

    int    max_epochs    = 400;
    int    patience      = 20;       // epochs without improvement before stopping
    double learning_rate = 3e-3;
    double weight_decay  = 1e-4;
    int    batch_size    = 0;        // 0 picks from the dataset size
};

struct GbtOptions {
    int    max_rounds       = 600;
    int    patience         = 30;    // rounds without improvement before stopping
    double learning_rate    = 0.1;
    int    max_leaves       = 31;
    int    max_depth        = 10;
    int    min_samples_leaf = 0;     // 0 picks from the dataset size
    double l2               = 1.0;
    int    max_bins         = 255;
};

FitReport fit_mlp(const TrainSet& data, const MlpOptions& opts, const FitControl& ctl);
FitReport fit_gbt(const TrainSet& data, const GbtOptions& opts, const FitControl& ctl);

/// Weighted average of already-trained models (probabilities for
/// classification, values for regression). Weights are normalised.
std::unique_ptr<Model> make_blend(std::vector<std::shared_ptr<const Model>> members,
                                  std::vector<double> weights);

} // namespace neurograph::models
