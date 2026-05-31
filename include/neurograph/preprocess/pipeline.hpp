#pragma once

// P1-02 — Preprocessing pipeline public API.
//
// One pipeline object encapsulates the full "raw CSV → numeric matrix"
// transformation. The same object is reused at inference time (predict.exe,
// the C-ABI DLL, the wizard) so training-time and serving-time features
// stay byte-for-byte identical.
//
// Flow:
//   auto pipe = fit_pipeline(csv, config, &out_ds);
//   train(out_ds.X_train, out_ds.y_train);
//   save_pipeline(*pipe, "model.ngpipe");
//   // ... later ...
//   auto pipe2 = load_pipeline("model.ngpipe");
//   auto x     = apply_pipeline(*pipe2, incoming_json_row);
//   auto pred  = network.predict(x);
//
// FittedPipeline is opaque (pImpl) so the serialized schema and internal
// storage can evolve without breaking callers. Move-only; not copyable.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace neurograph::preprocess {

// ---------------------------------------------------------------------------
// Strategies — small enums so callers can override auto-spec defaults.
// ---------------------------------------------------------------------------

enum class ImputeStrategy { Mean, Median, Mode, Constant };
enum class ScaleStrategy  { None, Standard, MinMax, Robust };
enum class EncodeStrategy { OneHot, Ordinal, HashFixedBuckets };

// Per-column instructions. Leave `is_numeric` at its default when unsure —
// fit_pipeline will sniff the type from data if no ColumnSpec is provided
// (see auto-spec inference below).
struct ColumnSpec {
    std::string     name;
    bool            is_target        = false;
    bool            is_numeric       = true;
    bool            drop             = false;
    ImputeStrategy  impute           = ImputeStrategy::Mean;
    double          impute_constant  = 0.0;      // used only when impute == Constant
    ScaleStrategy   scale            = ScaleStrategy::Standard;
    EncodeStrategy  encode           = EncodeStrategy::OneHot;
    int             hash_buckets     = 32;       // used only when encode == HashFixedBuckets
};

// Top-level config. Leave `columns` empty to trigger auto-spec inference
// (the pipeline will run the P1-01 scanner internally and pick sensible
// defaults — Standard + Mean for numeric, OneHot + Mode for categorical,
// Hash for high-cardinality, drop for constant / all-NaN).
struct PipelineConfig {
    std::vector<ColumnSpec> columns;
    std::string             target_column;       // required — must match a CSV header
    double                  val_fraction  = 0.15;
    double                  test_fraction = 0.15;
    bool                    stratify      = true;   // classification only
    std::uint64_t           split_seed    = 42;
};

// Output of fit_pipeline. X rows are fixed-width feature vectors; y rows are
// one-hot for classification and one-element vectors for regression.
struct TransformedDataset {
    std::vector<std::vector<double>> X_train, X_val, X_test;
    std::vector<std::vector<double>> y_train, y_val, y_test;
    std::vector<std::string>         feature_names_out;
    std::vector<std::string>         class_labels;  // empty for regression
    bool                             is_classification = true;
};

// Opaque fitted pipeline. Use the free functions below to operate on it.
// pImpl so the JSON schema and stored transform state can evolve without
// recompiles in callers.
class FittedPipeline {
public:
    FittedPipeline();
    ~FittedPipeline();
    FittedPipeline(FittedPipeline&&) noexcept;
    FittedPipeline& operator=(FittedPipeline&&) noexcept;
    FittedPipeline(const FittedPipeline&)            = delete;
    FittedPipeline& operator=(const FittedPipeline&) = delete;

    // Accessors the UI / caller needs without cracking open the impl.
    const std::vector<std::string>& feature_names_out() const;
    const std::vector<std::string>& class_labels()      const;
    bool                            is_classification() const;
    int                             version()           const;  // schema version

    struct Impl;
    std::unique_ptr<Impl> impl;
};

// ---------------------------------------------------------------------------
// Fit — at training time. Reads the CSV, computes per-column stats, fits
// encoders, produces transformed train/val/test datasets, and returns a
// re-applicable FittedPipeline. Throws std::runtime_error on fatal errors
// (missing file, target column not in header, empty data).
// ---------------------------------------------------------------------------
std::unique_ptr<FittedPipeline>
fit_pipeline(const std::filesystem::path& csv,
             const PipelineConfig& config,
             TransformedDataset& out_ds);

// ---------------------------------------------------------------------------
// Apply — at inference time. Accepts a raw JSON object keyed by the
// original CSV column names and returns the transformed feature vector
// ready to feed to the network.
//
// Contract for malformed input (per ticket acceptance criteria):
//   - Missing key        → treated as missing value (imputed), warning logged
//   - Extra key          → ignored
//   - Wrong type         → treated as missing value, warning logged
//   - Unknown category   → routed to the reserved "unknown" bucket
// None of these throw.
// ---------------------------------------------------------------------------
std::vector<double>
apply_pipeline(const FittedPipeline& pipe, const nlohmann::json& raw_row);

// ---------------------------------------------------------------------------
// Persistence — JSON on disk, `.ngpipe` by convention. Load checks the
// schema version and returns nullptr on unsupported / corrupt files
// (never throws).
// ---------------------------------------------------------------------------
void save_pipeline(const FittedPipeline& pipe,
                   const std::filesystem::path& file);
std::unique_ptr<FittedPipeline>
load_pipeline(const std::filesystem::path& file);

} // namespace neurograph::preprocess
