// Unit tests for neurograph::dataqc::scan_csv / scan_samples (ticket P1-01).
//
// Coverage goal: one test per finding code, plus structural contracts (the
// report stays populated when errors fire, IO failure never throws, sampling
// path keeps row counts accurate, etc.). Tests write throwaway CSVs into the
// platform temp dir and clean up after themselves.
//
// Finding codes exercised:
//   FILE_NOT_FOUND, EMPTY_FILE, TOO_FEW_ROWS, MISSING_VALUES,
//   CONSTANT_COLUMN, HIGH_CARDINALITY, CLASS_IMBALANCE,
//   TARGET_LEAKAGE_SUSPECT, DUPLICATE_ROWS, MIXED_TYPES,
//   ALL_NAN_COLUMN, OUTLIERS

#include "neurograph/dataqc/scanner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using neurograph::dataqc::DataQualityReport;
using neurograph::dataqc::Finding;
using neurograph::dataqc::ScanOptions;
using neurograph::dataqc::scan_csv;
using neurograph::dataqc::scan_samples;
using neurograph::dataqc::Severity;
using neurograph::training::Sample;

namespace {

// Writes `contents` to a fresh temp CSV and returns its path. The path goes
// inside a unique subdir so parallel test runs don't stomp on each other.
class TempCsv {
public:
    TempCsv(const std::string& name, const std::string& contents) {
        namespace fs = std::filesystem;
        dir_ = fs::temp_directory_path() / ("ngraph_dataqc_" +
               std::to_string(reinterpret_cast<uintptr_t>(this)));
        fs::create_directories(dir_);
        path_ = dir_ / name;
        std::ofstream f(path_, std::ios::binary);
        f << contents;
    }
    ~TempCsv() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path dir_;
    std::filesystem::path path_;
};

// Returns true if `r` contains a finding with the given code at any severity.
bool has_code(const DataQualityReport& r, std::string_view code) {
    return r.find_code(code) != nullptr;
}

// Small loosened-threshold options so tests don't need thousands of rows.
ScanOptions test_opts() {
    ScanOptions o;
    o.min_rows_for_train = 5;       // tests run with tiny fixtures
    return o;
}

} // namespace

// ---------------------------------------------------------------------------
// scan_csv — file-level failure modes
// ---------------------------------------------------------------------------

TEST(DataQcScanner, MissingFileEmitsFileNotFoundAndIsNotTrainable) {
    const auto r = scan_csv("C:/definitely/not/a/real/path.csv", test_opts());
    EXPECT_TRUE(has_code(r, "FILE_NOT_FOUND"));
    EXPECT_FALSE(r.trainable);
    EXPECT_GT(r.count_of(Severity::Error), 0);
    // Summary line must still be populated — UI depends on it.
    EXPECT_FALSE(r.summary_line.empty());
}

TEST(DataQcScanner, EmptyFileEmitsEmptyFile) {
    TempCsv csv("empty.csv", "");
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_TRUE(has_code(r, "EMPTY_FILE"));
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScanner, HeaderOnlyFileEmitsEmptyFile) {
    TempCsv csv("header_only.csv", "a,b,c\n");
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_TRUE(has_code(r, "EMPTY_FILE"));
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScanner, TooFewRowsEmitsTooFewRows) {
    // 3 data rows vs. min_rows_for_train = 5.
    TempCsv csv("small.csv",
        "x,y,label\n"
        "1,2,A\n"
        "3,4,B\n"
        "5,6,A\n");
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_TRUE(has_code(r, "TOO_FEW_ROWS"));
    EXPECT_FALSE(r.trainable);
    EXPECT_EQ(r.n_rows, 3);
    EXPECT_EQ(r.n_cols, 3);
}

// ---------------------------------------------------------------------------
// Per-column findings
// ---------------------------------------------------------------------------

TEST(DataQcScanner, MissingValuesAtWarningThreshold) {
    // 10 rows, 1 missing value in `x` → 10% missing. Default warn = 5%, err = 40%.
    // Expect a WARNING-level MISSING_VALUES, not an error.
    std::string body = "x,y,label\n";
    for (int i = 0; i < 10; ++i) {
        const std::string xv = (i == 0) ? "" : std::to_string(i);
        body += xv + "," + std::to_string(i * 2) + (i % 2 ? ",A\n" : ",B\n");
    }
    TempCsv csv("missing.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("MISSING_VALUES");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "x");
    EXPECT_EQ(f->severity, Severity::Warning);
    EXPECT_TRUE(r.trainable);  // warnings don't block training
}

TEST(DataQcScanner, HeavilyBlankInputColumnWarnsButDoesNotBlock) {
    // 10 rows, 5 blank in `x` → 50% blank, above the 40% "consider leaving
    // it out" level. Gaps are filled automatically, so training goes ahead.
    std::string body = "x,y,label\n";
    for (int i = 0; i < 10; ++i) {
        const std::string xv = (i < 5) ? "" : std::to_string(i);
        body += xv + "," + std::to_string(i) + (i % 2 ? ",A\n" : ",B\n");
    }
    TempCsv csv("missing_err.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("MISSING_VALUES");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "x");
    EXPECT_EQ(f->severity, Severity::Warning);
    EXPECT_NE(f->message.find("50% blank"), std::string::npos) << f->message;
    EXPECT_NE(f->remedy.find("leaving it out"), std::string::npos) << f->remedy;
    EXPECT_TRUE(r.trainable);
}

TEST(DataQcScanner, ConstantTargetBlocksTraining) {
    // Nothing to predict when the target has one value on every row.
    std::string body = "x,label\n";
    for (int i = 0; i < 30; ++i) body += std::to_string(i) + ",A\n";
    TempCsv csv("constant_target.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("TARGET_CONSTANT");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Error);
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScanner, ChecksAreMadeAgainstTheChosenTarget) {
    // `label` is first, not last. With target_column set, the imbalance
    // check looks at it rather than at the last column.
    std::string body = "label,x\n";
    for (int i = 0; i < 60; ++i) body += "A," + std::to_string(i) + "\n";
    for (int i = 0; i < 4;  ++i) body += "B," + std::to_string(i) + "\n";
    TempCsv csv("target_first.csv", body);
    auto opts = test_opts();
    opts.target_column = "label";
    const auto r = scan_csv(csv.path(), opts);
    const Finding* f = r.find_code("CLASS_IMBALANCE");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "label");
    EXPECT_GE(r.class_imbalance_ratio, 10.0);
    EXPECT_TRUE(r.trainable);
}

TEST(DataQcScanner, AllNanColumnEmitsAllNanColumn) {
    std::string body = "x,y,label\n";
    for (int i = 0; i < 10; ++i) {
        body += "NaN," + std::to_string(i) + ",A\n";
    }
    TempCsv csv("all_nan.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("ALL_NAN_COLUMN");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "x");
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScanner, ConstantColumnEmitsWarning) {
    std::string body = "x,y,label\n";
    for (int i = 0; i < 10; ++i) {
        body += "7," + std::to_string(i) + ",A\n";
    }
    TempCsv csv("const.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("CONSTANT_COLUMN");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "x");
    EXPECT_EQ(f->severity, Severity::Warning);
}

TEST(DataQcScanner, HighCardinalityEmitsWarning) {
    // 200 rows, 200 distinct `user_id` strings → above the max(100, n/2)
    // threshold. `label` alternates A/B so target isn't also flagged.
    std::string body = "user_id,x,label\n";
    for (int i = 0; i < 200; ++i) {
        body += "U" + std::to_string(i) + ",1," + ((i % 2) ? "A" : "B") + "\n";
    }
    TempCsv csv("highcard.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("HIGH_CARDINALITY");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "user_id");
}

TEST(DataQcScanner, ClassImbalanceWarnsAtBothLevels) {
    // 50 A + 4 B → ratio ~12.5 → WARN threshold (10) hit, ERROR (50) not.
    {
        std::string body = "x,label\n";
        for (int i = 0; i < 50; ++i) body += "1,A\n";
        for (int i = 0; i < 4;  ++i) body += "1,B\n";
        TempCsv csv("imb_warn.csv", body);
        const auto r = scan_csv(csv.path(), test_opts());
        const Finding* f = r.find_code("CLASS_IMBALANCE");
        ASSERT_NE(f, nullptr);
        EXPECT_EQ(f->severity, Severity::Warning);
        EXPECT_GE(r.class_imbalance_ratio, 10.0);
    }
    // 100 A + 1 B → ratio 100 → past the severe level (50): a stronger
    // warning that explains why accuracy will mislead, but not a block.
    {
        std::string body = "x,label\n";
        for (int i = 0; i < 100; ++i) body += "1,A\n";
        body += "1,B\n";
        TempCsv csv("imb_err.csv", body);
        const auto r = scan_csv(csv.path(), test_opts());
        const Finding* f = r.find_code("CLASS_IMBALANCE");
        ASSERT_NE(f, nullptr);
        EXPECT_EQ(f->severity, Severity::Warning);
        EXPECT_NE(f->remedy.find("rare"), std::string::npos) << f->remedy;
        EXPECT_TRUE(r.trainable);
    }
}

TEST(DataQcScanner, TargetLeakageDetectsNearPerfectCorrelation) {
    // `leak` is y*2 + tiny noise so |corr| ~ 1 but not exactly 1 (otherwise
    // we hit CONSTANT_COLUMN territory for target stats). `x` is unrelated.
    std::string body = "x,leak,y\n";
    for (int i = 0; i < 60; ++i) {
        const double y    = static_cast<double>(i);
        const double leak = y * 2.0 + (i % 3) * 1e-6;  // tiny jitter
        const double x    = std::sin(i);               // uncorrelated
        body += std::to_string(x) + "," +
                std::to_string(leak) + "," +
                std::to_string(y) + "\n";
    }
    TempCsv csv("leak.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("TARGET_LEAKAGE_SUSPECT");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "leak");
    EXPECT_EQ(f->severity, Severity::Warning);
    EXPECT_TRUE(r.trainable);
}

TEST(DataQcScanner, DuplicateRowsEmitWarning) {
    // 20 distinct rows + 20 exact duplicates = 50% dup fraction.
    std::string body = "x,y,label\n";
    for (int i = 0; i < 20; ++i) {
        body += std::to_string(i) + "," + std::to_string(i * 2) + ",A\n";
    }
    for (int i = 0; i < 20; ++i) {
        body += std::to_string(i) + "," + std::to_string(i * 2) + ",A\n";
    }
    TempCsv csv("dup.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_TRUE(has_code(r, "DUPLICATE_ROWS"));
}

TEST(DataQcScanner, MixedTypesEmitsWarningOnNumericColumnWithStringOutlier) {
    // 19 numeric values + 1 "oops" string in column `x` → >90% numeric, so
    // the column is classified numeric, and the single offending string
    // triggers MIXED_TYPES.
    std::string body = "x,y,label\n";
    for (int i = 0; i < 19; ++i) {
        body += std::to_string(i) + ",2,A\n";
    }
    body += "oops,2,A\n";
    TempCsv csv("mixed.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    const Finding* f = r.find_code("MIXED_TYPES");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "x");
}

TEST(DataQcScanner, OutliersEmitInfoFindingBeyondSigmaThreshold) {
    // Two point masses at 0 and 100 with p=1.5% at 100 is detectable at 5σ
    // (see p < 1/(1+k²) bound). Default outlier_warn_frac is 5% which is
    // mathematically unreachable for point-mass distributions at k=5 — so
    // we lower it to 1% for this test.
    std::string body = "x,label\n";
    for (int i = 0; i < 200; ++i) {
        body += std::to_string((i % 3) - 1) + ",A\n";  // bulk near 0
    }
    for (int i = 0; i < 3; ++i) {
        body += "1000,A\n";  // clear 5σ+ outliers
    }

    ScanOptions opts = test_opts();
    opts.outlier_warn_frac = 0.01;  // 1% — matches our ~1.5% outlier fraction

    TempCsv csv("out.csv", body);
    const auto r = scan_csv(csv.path(), opts);
    const Finding* f = r.find_code("OUTLIERS");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "x");
    EXPECT_EQ(f->severity, Severity::Info);  // outliers are info, not warning
}

// ---------------------------------------------------------------------------
// CSV parser correctness
// ---------------------------------------------------------------------------

TEST(DataQcScanner, HandlesUtf8BomAndCrlfLineEndings) {
    // Leading BOM + CRLF — common Excel-export quirk.
    std::string body;
    body.push_back(static_cast<char>(0xEF));
    body.push_back(static_cast<char>(0xBB));
    body.push_back(static_cast<char>(0xBF));
    body += "a,b,label\r\n";
    for (int i = 0; i < 10; ++i) {
        body += std::to_string(i) + "," + std::to_string(i * 2) + ",A\r\n";
    }
    TempCsv csv("bom.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    // BOM must not leak into the first column name — if it did, the column
    // would be named "\xEF\xBB\xBFa" and `columns[0].name` would start with
    // non-ASCII bytes.
    ASSERT_GE(r.columns.size(), 1u);
    EXPECT_EQ(r.columns[0].name, "a");
    EXPECT_EQ(r.n_rows, 10);
    EXPECT_EQ(r.n_cols, 3);
}

TEST(DataQcScanner, HandlesQuotedFieldsWithEmbeddedCommas) {
    // Row 2 has a quoted field containing a comma.
    std::string body =
        "city,note,label\n"
        "NYC,\"hello, world\",A\n"
        "SFO,quiet,B\n"
        "LAX,sunny,A\n"
        "ATL,humid,B\n"
        "BOS,cold,A\n";
    TempCsv csv("quoted.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_EQ(r.n_cols, 3);
    EXPECT_EQ(r.n_rows, 5);
    // `note` is categorical — the "hello, world" row must have been preserved
    // intact (not split on the embedded comma into two fields).
    ASSERT_GE(r.columns.size(), 2u);
    bool found = false;
    for (const auto& [k, v] : r.columns[1].top_categories) {
        if (k == "hello, world") found = true;
    }
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// Clean CSV — no findings beyond Info
// ---------------------------------------------------------------------------

TEST(DataQcScanner, CleanCsvProducesNoErrorOrWarningFindings) {
    // Balanced, no missing, no duplicates, no leakage — the "happy path".
    std::string body = "x,y,label\n";
    for (int i = 0; i < 60; ++i) {
        // x and y unrelated to label; label balanced A/B.
        body += std::to_string(i) + "," +
                std::to_string((i * 7) % 11) + "," +
                ((i % 2) ? "A" : "B") + "\n";
    }
    TempCsv csv("clean.csv", body);
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_TRUE(r.trainable);
    EXPECT_EQ(r.count_of(Severity::Error),   0);
    EXPECT_EQ(r.count_of(Severity::Warning), 0);
    EXPECT_EQ(r.n_rows, 60);
    EXPECT_EQ(r.n_cols, 3);
}

// ---------------------------------------------------------------------------
// scan_samples
// ---------------------------------------------------------------------------

TEST(DataQcScannerSamples, EmptySamplesEmitsEmptyFile) {
    std::vector<Sample> samples;
    const auto r = scan_samples(samples, test_opts());
    EXPECT_TRUE(has_code(r, "EMPTY_FILE"));
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScannerSamples, TooFewSamplesEmitsTooFewRows) {
    std::vector<Sample> samples;
    for (int i = 0; i < 3; ++i) {
        Sample s;
        s.features = {double(i), double(i) * 2};
        s.target   = {1.0, 0.0};
        samples.push_back(std::move(s));
    }
    const auto r = scan_samples(samples, test_opts());
    EXPECT_TRUE(has_code(r, "TOO_FEW_ROWS"));
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScannerSamples, ConstantFeatureEmitsWarning) {
    std::vector<Sample> samples;
    for (int i = 0; i < 20; ++i) {
        Sample s;
        s.features = {1.0, double(i)};  // feature 0 is constant, feature 1 varies
        s.target   = {(i % 2) ? 1.0 : 0.0, (i % 2) ? 0.0 : 1.0};
        samples.push_back(std::move(s));
    }
    const auto r = scan_samples(samples, test_opts());
    const Finding* f = r.find_code("CONSTANT_COLUMN");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "feature_0");
}

TEST(DataQcScannerSamples, AllNanFeatureEmitsAllNanColumn) {
    std::vector<Sample> samples;
    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < 20; ++i) {
        Sample s;
        s.features = {nan_v, double(i)};  // feature 0 all NaN
        s.target   = {(i % 2) ? 1.0 : 0.0, (i % 2) ? 0.0 : 1.0};
        samples.push_back(std::move(s));
    }
    const auto r = scan_samples(samples, test_opts());
    const Finding* f = r.find_code("ALL_NAN_COLUMN");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->column, "feature_0");
    EXPECT_FALSE(r.trainable);
}

TEST(DataQcScannerSamples, ClassImbalanceDetectedOnOneHotTarget) {
    // 50 class-0 + 1 class-1 one-hot → ratio 50.
    std::vector<Sample> samples;
    for (int i = 0; i < 50; ++i) {
        Sample s;
        s.features = {1.0, 2.0};
        s.target   = {1.0, 0.0};
        samples.push_back(s);
    }
    Sample minority;
    minority.features = {1.0, 2.0};
    minority.target   = {0.0, 1.0};
    samples.push_back(minority);

    const auto r = scan_samples(samples, test_opts());
    const Finding* f = r.find_code("CLASS_IMBALANCE");
    ASSERT_NE(f, nullptr);
    EXPECT_GE(r.class_imbalance_ratio, 50.0);
}

TEST(DataQcScannerSamples, MismatchedShapesEmitMixedTypes) {
    std::vector<Sample> samples;
    Sample a; a.features = {1, 2, 3}; a.target = {1, 0};
    Sample b; b.features = {1, 2};    b.target = {1, 0};  // wrong feature size
    samples.push_back(a);
    samples.push_back(b);
    const auto r = scan_samples(samples, test_opts());
    EXPECT_TRUE(has_code(r, "MIXED_TYPES"));
    EXPECT_FALSE(r.trainable);
}

// ---------------------------------------------------------------------------
// Structural guarantees
// ---------------------------------------------------------------------------

TEST(DataQcScanner, ReportAlwaysHasSummaryLineAndColumnStats) {
    TempCsv csv("ok.csv",
        "x,y,label\n"
        "1,2,A\n"
        "3,4,B\n"
        "5,6,A\n"
        "7,8,B\n"
        "9,0,A\n"
        "2,4,B\n");
    const auto r = scan_csv(csv.path(), test_opts());
    EXPECT_FALSE(r.summary_line.empty());
    EXPECT_EQ(r.columns.size(), 3u);
    EXPECT_EQ(r.columns[0].name, "x");
    EXPECT_TRUE(r.columns[0].is_numeric);
    EXPECT_FALSE(r.columns[2].is_numeric);  // "label" is A/B
}
