#pragma once

// P1-02 — Individual preprocessing transforms.
//
// These are the building blocks used by the Pipeline. Exposed as public so
// tests can exercise them in isolation and so callers with unusual needs
// can compose them directly without going through fit_pipeline.
//
// Each transform follows the fit-then-apply pattern:
//   1. fit_*(values)  — compute the transform's internal state from a
//                       column of training data. No output.
//   2. apply(x)       — produce the transformed value(s) for a single raw
//                       input. Called at both train and inference time.
//
// Transforms are value types — copyable, serializable. The Pipeline holds
// a vector of them in column-index order.

#include "neurograph/preprocess/pipeline.hpp"   // for ImputeStrategy etc.

#include <string>
#include <unordered_map>
#include <vector>

namespace neurograph::preprocess {

// ---------------------------------------------------------------------------
// Imputer — fills missing values. Numeric and categorical paths are
// deliberately separate: numeric uses a double fill value, categorical
// uses a reserved string token.
// ---------------------------------------------------------------------------
class Imputer {
public:
    ImputeStrategy strategy  = ImputeStrategy::Mean;

    // Numeric fit/apply. `values` may contain NaN — they are skipped for
    // mean/median computation.
    void   fit_numeric(const std::vector<double>& values,
                       double constant_value = 0.0);
    double apply_numeric(double raw) const;   // NaN → fill_value_

    // Categorical fit/apply. Missing tokens in `values` (empty or NaN-like)
    // are skipped when computing the mode.
    void        fit_categorical(const std::vector<std::string>& values);
    std::string apply_categorical(const std::string& raw) const;

    // Inspection / serialization.
    double              numeric_fill()    const { return fill_value_; }
    const std::string&  categorical_fill() const { return fill_token_; }

    // Used by save/load — set the internal fill directly.
    void set_numeric_fill(double v)           { fill_value_ = v; }
    void set_categorical_fill(std::string s)  { fill_token_ = std::move(s); }

private:
    double      fill_value_ = 0.0;
    std::string fill_token_;
};

// ---------------------------------------------------------------------------
// Scaler — linear map (x - a) / b. The (a, b) depend on strategy:
//   None     → (0, 1)            — no-op
//   Standard → (mean, stddev)
//   MinMax   → (min, max-min)
//   Robust   → (median, IQR)     — where IQR = q75 - q25
//
// b is floored at 1e-12 to avoid divide-by-zero on constant columns.
// ---------------------------------------------------------------------------
class Scaler {
public:
    ScaleStrategy strategy = ScaleStrategy::Standard;

    void   fit(const std::vector<double>& values);
    double apply(double x) const;

    double offset() const { return a_; }
    double scale()  const { return b_; }
    void set_params(double a, double b) { a_ = a; b_ = b; }

private:
    double a_ = 0.0;
    double b_ = 1.0;
};

// ---------------------------------------------------------------------------
// Encoder — maps a categorical string to a fixed-width vector of doubles.
//
// Strategies:
//   OneHot            — N+1 dims. Index 0..N-1 for known categories, index
//                       N reserved for unknown values seen at inference.
//   Ordinal           — 1 dim (integer index cast to double). Unknown → -1.
//   HashFixedBuckets  — `hash_buckets` dims. Raw string hashed into a
//                       fixed number of buckets. No notion of "unknown" —
//                       any string maps to some bucket deterministically.
//
// Known categories are stored alphabetically sorted so the output layout
// is reproducible across runs (important for save/load).
// ---------------------------------------------------------------------------
class Encoder {
public:
    EncodeStrategy strategy     = EncodeStrategy::OneHot;
    int            hash_buckets = 32;

    // Fit. For OneHot / Ordinal the category list is capped at
    // `max_categories` (rarest dropped). The scanner's HIGH_CARDINALITY
    // finding is the heuristic that flips a column to Hash automatically
    // upstream — this cap is a safety net.
    void fit(const std::vector<std::string>& values,
             int max_categories = 10000);

    // Apply — returns `output_dim()` doubles for every input.
    std::vector<double> apply(const std::string& raw) const;

    // Width of the output vector. Depends on strategy + fitted state.
    std::size_t output_dim() const;

    const std::vector<std::string>& categories() const { return categories_; }

    // Set categories directly — used by load_pipeline.
    void set_categories(std::vector<std::string> cats);

private:
    std::vector<std::string>                 categories_;    // known, sorted
    std::unordered_map<std::string, int>     cat_to_idx_;    // built from categories_
};

} // namespace neurograph::preprocess
