#pragma once

// P1-01 — Data Quality Scanner public API.
//
// Two entry points:
//   scan_csv     — read a CSV off disk and emit a DataQualityReport. This
//                  is the path the wizard takes right after the user drops
//                  a file in. Streaming so we handle multi-GB inputs.
//   scan_samples — scan an already-loaded vector of training::Sample. Use
//                  this from code that has gone through data_loader.cpp
//                  (post-normalization) as a "post-transform" sanity check.
//
// scan_samples deliberately has no column name or class label knowledge —
// it can only report structural issues (mismatch sizes, constants,
// outliers, imbalance, duplicates). Text/categorical-specific findings
// (HIGH_CARDINALITY, MIXED_TYPES, MISSING_VALUES) require the raw CSV.

#include "neurograph/dataqc/report.hpp"
#include "neurograph/training/data_loader.hpp"   // for training::Sample

#include <filesystem>
#include <string>

namespace neurograph::dataqc {

struct ScanOptions {
    // Header name of the column being predicted. Empty means the last
    // column. Class-imbalance and leakage checks are made against it.
    std::string target_column;

    // Max categories to surface in ColumnStats::top_categories per column.
    int    top_k_categories      = 5;

    // Class imbalance thresholds (ratio = max_class_freq / min_class_freq).
    double imbalance_warn_at     = 10.0;
    double imbalance_error_at    = 50.0;

    // Missing-value fraction per column.
    double missing_warn_frac     = 0.05;
    double missing_error_frac    = 0.40;

    // Anything fewer rows than this blocks training entirely.
    int    min_rows_for_train    = 50;

    // Correlation magnitude above this is flagged as target leakage.
    double leakage_corr_threshold = 0.98;

    // Rows outside mean±(outlier_sigma × stddev) are counted as outliers.
    // Emit OUTLIERS finding when fraction exceeds outlier_warn_frac.
    double outlier_sigma         = 5.0;
    double outlier_warn_frac     = 0.05;

    // Duplicate-rows finding fires when duplicate fraction exceeds this.
    double duplicate_warn_frac   = 0.01;

    // Sampling behaviour for very large files. When the CSV exceeds this
    // size we switch to a reservoir sample of sample_cap rows. Heuristic
    // that keeps scanning bounded on 10 GB inputs without forcing users
    // to pre-sample.
    std::size_t max_file_bytes_full_scan = 500ull * 1024ull * 1024ull; // 500 MB
    int         sample_cap               = 200000;

    // Cap on unique values tracked per column before we flip the overflow
    // bit. Matters for memory bounds on high-cardinality text columns.
    int    max_uniques_tracked   = 1024;
};

/// Scan a CSV. The target is `opts.target_column`, or the last column when
/// that is empty. Passes the
/// file stream once if file size < max_file_bytes_full_scan, otherwise
/// reservoir-samples sample_cap rows.
///
/// Never throws on IO errors. A missing / unreadable file produces a
/// report containing only a FILE_NOT_FOUND Error finding.
DataQualityReport scan_csv(const std::filesystem::path& csv,
                           const ScanOptions& opts = {});

/// Scan already-loaded training samples. Produces a subset of the findings
/// that scan_csv can produce — structural only, no column-type findings.
DataQualityReport scan_samples(const std::vector<training::Sample>& samples,
                               const ScanOptions& opts = {});

} // namespace neurograph::dataqc
