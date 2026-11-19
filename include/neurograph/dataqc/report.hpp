#pragma once

// P1-01 — Data Quality Report types.
//
// These types are the value produced by the scanner (scanner.hpp). Split
// out so UI code can consume / display a DataQualityReport without having
// to pull in the scanner's implementation or its CSV-parsing guts.
//
// Stable finding codes (never rename — UI + docs reference them directly):
//
//   FILE_NOT_FOUND           — input CSV does not exist
//   EMPTY_FILE               — file has zero data rows
//   TOO_FEW_ROWS             — fewer rows than min_rows_for_train
//   MISSING_VALUES           — column has NaN/blank fraction beyond threshold
//   CONSTANT_COLUMN          — column has exactly one unique value
//   HIGH_CARDINALITY         — categorical with too many distinct values
//   CLASS_IMBALANCE          — max/min class frequency ratio exceeds threshold
//   TARGET_LEAKAGE_SUSPECT   — feature nearly perfectly correlated with target
//   DUPLICATE_ROWS           — duplicate rows exceed 1% of data
//   MIXED_TYPES              — numeric column contains un-parseable values
//   ALL_NAN_COLUMN           — column is entirely missing
//   OUTLIERS                 — >5% of rows outside mean±5σ
//   TARGET_CONSTANT          — target column has one distinct value
//
// Only FILE_NOT_FOUND, EMPTY_FILE, TOO_FEW_ROWS, TARGET_CONSTANT and an
// all-blank target are errors (they make training impossible). Everything
// else is a warning or a note: the pipeline handles it, and the user can
// leave a column out.

#include <string>
#include <utility>
#include <vector>

namespace neurograph::dataqc {

enum class Severity { Info, Warning, Error };

struct Finding {
    Severity    severity;
    std::string code;     // stable finding ID (see header comment)
    std::string column;   // empty string when dataset-level
    std::string message;  // human-readable detail for the UI
    std::string remedy;   // one-line suggested fix
};

struct ColumnStats {
    std::string name;
    bool   is_numeric = false;
    int    n_missing  = 0;
    int    n_unique   = 0;   // 0 when unique set overflowed (see high_cardinality)
    bool   cardinality_overflowed = false;  // true when we stopped tracking

    // Numeric columns only. Meaningless for categoricals.
    double mean   = 0.0;
    double stddev = 0.0;
    double min    = 0.0;
    double max    = 0.0;

    // Top-K category frequencies, descending. Empty for numeric columns.
    std::vector<std::pair<std::string, int>> top_categories;
};

struct DataQualityReport {
    int                        n_rows = 0;
    int                        n_cols = 0;
    std::vector<ColumnStats>   columns;
    std::vector<Finding>       findings;

    /// false iff any Error finding exists — the UI refuses to start training.
    bool                       trainable = true;

    /// Ratio of most-common to least-common class. 1.0 means balanced.
    /// NaN when target is not classified as categorical.
    double                     class_imbalance_ratio = 1.0;

    /// One-line description suitable for the "Data Health" header in the UI.
    std::string                summary_line;

    // --- Convenience helpers -------------------------------------------------

    /// Number of findings at a given severity.
    int count_of(Severity s) const {
        int n = 0;
        for (const auto& f : findings) if (f.severity == s) ++n;
        return n;
    }

    /// First finding with the given code, or nullptr.
    const Finding* find_code(std::string_view code) const {
        for (const auto& f : findings) if (f.code == code) return &f;
        return nullptr;
    }
};

} // namespace neurograph::dataqc
