#include "neurograph/dataqc/scanner.hpp"
#include "neurograph/util/text_parse.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace neurograph::dataqc {

// ---------------------------------------------------------------------------
// CSV parsing — minimal but correct for the cases that matter: quoted
// fields, embedded commas, doubled-quote escape, CRLF, leading UTF-8 BOM.
// We deliberately avoid a dep here so the scanner is self-contained.
// ---------------------------------------------------------------------------
namespace {

constexpr char kBom0 = static_cast<char>(0xEF);
constexpr char kBom1 = static_cast<char>(0xBB);
constexpr char kBom2 = static_cast<char>(0xBF);

// Strip a leading UTF-8 BOM if present; no-op otherwise.
void strip_bom(std::string& s) {
    if (s.size() >= 3 && s[0] == kBom0 && s[1] == kBom1 && s[2] == kBom2) {
        s.erase(0, 3);
    }
}

// Remove a single trailing \r (handles CRLF files read with std::getline).
void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

// Parse one CSV line into fields. Handles quoted fields and doubled quotes.
// Keeps behavior close to RFC 4180 — the 99% case that matters.
std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    // Escaped quote inside quoted field.
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else {
            if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else if (c == '"' && cur.empty()) {
                // Opening quote only allowed at the start of a field.
                in_quotes = true;
            } else {
                cur += c;
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

// Trim leading/trailing ASCII whitespace. We intentionally don't touch the
// interior of a field — customers sometimes have legitimate trailing-space
// values (e.g. " east " vs "east"). Trimming only the boundaries handles
// the common CSV-export quirk without corrupting intent.
std::string trim(std::string s) {
    auto notspace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

// Number parsing is shared with the preprocessing pipeline so the scanner
// and the model agree on what counts as a number or a blank.
using util::NumericParse;
using util::try_parse_number;

// Welford's online mean/variance — one pass, numerically stable. Tracks
// min/max alongside.
struct OnlineStats {
    int    n = 0;
    double mean = 0.0, m2 = 0.0;
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();

    void push(double x) {
        ++n;
        const double delta = x - mean;
        mean += delta / n;
        m2   += delta * (x - mean);
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
    }

    double stddev() const {
        if (n < 2) return 0.0;
        return std::sqrt(m2 / static_cast<double>(n - 1));
    }
};

// Two-pass Pearson correlation. Two passes avoids the subtle numeric issues
// a Welford-style covariance update has on highly-correlated data, and the
// memory cost is a non-issue — we already keep the columns in memory for
// categorical tracking.
double pearson_correlation(const std::vector<double>& x,
                           const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    const auto n = static_cast<double>(x.size());

    double mx = 0.0, my = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;

    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if (sxx < 1e-18 || syy < 1e-18) return std::numeric_limits<double>::quiet_NaN();
    return sxy / std::sqrt(sxx * syy);
}

// Row hash used for duplicate detection. We hash field-by-field with a
// separator that can't appear in a CSV field value we care about — '\x1f'
// (ASCII unit separator). Cheap and collision-resistant enough.
std::size_t hash_row(const std::vector<std::string>& row) {
    std::string joined;
    joined.reserve(row.size() * 8);
    for (const auto& f : row) {
        joined += f;
        joined.push_back('\x1f');
    }
    return std::hash<std::string>{}(joined);
}

// Per-column accumulator during CSV streaming. We don't know yet whether
// a column is numeric or categorical — treat everything as a string until
// we've seen enough rows to judge.
struct ColumnBuffer {
    std::string name;

    // Running numeric stats over "successfully parsed as number" values.
    OnlineStats numeric_stats;

    // Count of values that were numeric, missing, or non-numeric strings.
    int n_numeric     = 0;
    int n_missing     = 0;
    int n_categorical = 0;

    // Keep string values for numeric-column correlation + category frequency.
    // Capped via ScanOptions::max_uniques_tracked to bound memory.
    std::unordered_map<std::string, int> category_counts;
    bool cardinality_overflowed = false;

    // Numeric values (only those that parsed). Kept for correlation with
    // target. Bounded by max_numeric_cache to avoid O(file_size) memory.
    std::vector<double> numeric_cache;
    static constexpr std::size_t max_numeric_cache = 200000;

    void observe(std::string_view raw, int max_uniques_tracked) {
        const NumericParse p = try_parse_number(raw);
        if (p.is_missing) {
            ++n_missing;
            return;
        }
        if (p.parsed) {
            numeric_stats.push(p.value);
            ++n_numeric;
            if (numeric_cache.size() < max_numeric_cache) {
                numeric_cache.push_back(p.value);
            }
        } else {
            ++n_categorical;
        }

        // Always track categories even for "numeric" strings so we can
        // report them if the column ends up classified as categorical.
        if (!cardinality_overflowed) {
            auto it = category_counts.find(std::string(raw));
            if (it != category_counts.end()) {
                ++it->second;
            } else if (static_cast<int>(category_counts.size()) <
                       max_uniques_tracked) {
                category_counts.emplace(std::string(raw), 1);
            } else {
                cardinality_overflowed = true;
                category_counts.clear();  // stop wasting memory
            }
        }
    }

    int total_seen() const { return n_numeric + n_missing + n_categorical; }
};

// Decision rule: a column is "numeric" when the vast majority of non-missing
// values parsed as numbers. We don't flip on a single stray string because
// real CSVs have the odd typo.
bool column_is_numeric(const ColumnBuffer& b) {
    const int non_missing = b.n_numeric + b.n_categorical;
    if (non_missing == 0) return false;
    // 90%: strict enough to catch "mostly numeric with 3 typos" as mixed,
    // lenient enough to avoid downgrading real numeric columns for one bad row.
    return static_cast<double>(b.n_numeric) / non_missing >= 0.9;
}

// Finalize per-column stats into the public report format.
ColumnStats finalize_column(const ColumnBuffer& b, int top_k) {
    ColumnStats cs;
    cs.name      = b.name;
    cs.is_numeric = column_is_numeric(b);
    cs.n_missing = b.n_missing;
    cs.cardinality_overflowed = b.cardinality_overflowed;
    cs.n_unique  = b.cardinality_overflowed
                 ? 0
                 : static_cast<int>(b.category_counts.size());

    if (cs.is_numeric) {
        cs.mean   = b.numeric_stats.mean;
        cs.stddev = b.numeric_stats.stddev();
        cs.min    = (b.numeric_stats.n > 0) ? b.numeric_stats.min_v : 0.0;
        cs.max    = (b.numeric_stats.n > 0) ? b.numeric_stats.max_v : 0.0;
    } else if (!b.cardinality_overflowed) {
        std::vector<std::pair<std::string, int>> sorted(b.category_counts.begin(),
                                                        b.category_counts.end());
        std::partial_sort(
            sorted.begin(),
            sorted.begin() + std::min<std::size_t>(sorted.size(),
                                                   static_cast<std::size_t>(top_k)),
            sorted.end(),
            [](const auto& a, const auto& c) { return a.second > c.second; });
        sorted.resize(std::min<std::size_t>(sorted.size(),
                                            static_cast<std::size_t>(top_k)));
        cs.top_categories = std::move(sorted);
    }
    return cs;
}

// Build the one-line human summary that headers the Data Health screen.
std::string make_summary(const DataQualityReport& r) {
    const int n_err  = r.count_of(Severity::Error);
    const int n_warn = r.count_of(Severity::Warning);
    const int n_info = r.count_of(Severity::Info);
    const int total  = n_err + n_warn + n_info;

    std::ostringstream oss;
    oss << r.n_rows << " rows × " << r.n_cols << " cols, ";

    if (total == 0) {
        oss << "clean (no issues)";
    } else {
        oss << total << (total == 1 ? " issue" : " issues") << " found";
        if (n_err + n_warn > 0) {
            oss << " — ";
            bool comma = false;
            auto append = [&](int count, const char* label_s, const char* label_p) {
                if (count == 0) return;
                if (comma) oss << ", ";
                oss << count << " " << (count == 1 ? label_s : label_p);
                comma = true;
            };
            append(n_err,  "error",   "errors");
            append(n_warn, "warning", "warnings");
        }
    }
    return oss.str();
}

// All findings share these defaults; we factor the construction.
Finding make_finding(Severity s, std::string code, std::string column,
                     std::string message, std::string remedy) {
    Finding f;
    f.severity = s;
    f.code     = std::move(code);
    f.column   = std::move(column);
    f.message  = std::move(message);
    f.remedy   = std::move(remedy);
    return f;
}

// Finalize the trainable flag + summary_line and return the report.
DataQualityReport finalize_report(DataQualityReport r) {
    r.trainable    = (r.count_of(Severity::Error) == 0);
    r.summary_line = make_summary(r);
    return r;
}

// ---------------------------------------------------------------------------
// scan_csv helpers
// ---------------------------------------------------------------------------

struct CsvData {
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;   // kept in-memory when sampled
    std::vector<ColumnBuffer>             columns;
    // Raw row fingerprints for duplicate detection.
    std::unordered_map<std::size_t, int>  row_hash_counts;

    // When sampling: total rows in file (so we can report accurate n_rows
    // even when only a subset was scanned).
    int total_rows_in_file = 0;
    int scanned_rows       = 0;
};

std::optional<DataQualityReport>
read_csv_into(CsvData& out,
              const std::filesystem::path& csv,
              const ScanOptions& opts) {
    // File must exist and be readable. IO errors go into the report, never
    // become exceptions.
    std::error_code ec;
    if (!std::filesystem::exists(csv, ec) || ec) {
        DataQualityReport r;
        r.findings.push_back(make_finding(
            Severity::Error, "FILE_NOT_FOUND", "",
            "File '" + csv.string() + "' does not exist.",
            "Check the path, or re-export the dataset."));
        return finalize_report(r);
    }

    const auto file_size = std::filesystem::file_size(csv, ec);
    if (ec) {
        DataQualityReport r;
        r.findings.push_back(make_finding(
            Severity::Error, "FILE_NOT_FOUND", "",
            "Cannot read file size for '" + csv.string() + "'.",
            "Check permissions."));
        return finalize_report(r);
    }

    std::ifstream f(csv, std::ios::binary);
    if (!f) {
        DataQualityReport r;
        r.findings.push_back(make_finding(
            Severity::Error, "FILE_NOT_FOUND", "",
            "File '" + csv.string() + "' could not be opened.",
            "Check permissions."));
        return finalize_report(r);
    }

    // Read header.
    std::string line;
    if (!std::getline(f, line)) {
        DataQualityReport r;
        r.findings.push_back(make_finding(
            Severity::Error, "EMPTY_FILE", "",
            "File is empty.", "Re-export the dataset."));
        return finalize_report(r);
    }
    strip_bom(line);
    strip_cr(line);
    out.header = parse_csv_line(line);
    for (auto& h : out.header) h = trim(std::move(h));
    out.columns.resize(out.header.size());
    for (std::size_t i = 0; i < out.header.size(); ++i) {
        out.columns[i].name = out.header[i];
    }

    const bool sampling = file_size > opts.max_file_bytes_full_scan;
    std::mt19937_64 rng(0xC0FFEEu);  // fixed seed — deterministic sampling

    // Reservoir sample algorithm R. Keep sample_cap rows; each new row has
    // probability sample_cap/N of being included. Deterministic given seed.
    int rows_accepted = 0;
    while (std::getline(f, line)) {
        strip_cr(line);
        if (line.empty()) continue;

        ++out.total_rows_in_file;
        auto row = parse_csv_line(line);
        // Pad short rows / ignore extra fields — protects against stray
        // trailing commas that don't represent real data.
        if (row.size() < out.header.size()) row.resize(out.header.size());

        if (sampling) {
            if (rows_accepted < opts.sample_cap) {
                out.rows.push_back(std::move(row));
                ++rows_accepted;
            } else {
                std::uniform_int_distribution<int> dist(0, out.total_rows_in_file - 1);
                const int j = dist(rng);
                if (j < opts.sample_cap) {
                    out.rows[j] = std::move(row);
                }
            }
        } else {
            out.rows.push_back(std::move(row));
            ++rows_accepted;
        }
    }
    out.scanned_rows = static_cast<int>(out.rows.size());

    // Accumulate per-column stats from the kept rows.
    for (const auto& row : out.rows) {
        ++out.row_hash_counts[hash_row(row)];
        for (std::size_t c = 0; c < out.header.size(); ++c) {
            const std::string_view v = (c < row.size())
                                     ? std::string_view(row[c])
                                     : std::string_view{};
            out.columns[c].observe(v, opts.max_uniques_tracked);
        }
    }
    return std::nullopt;  // success path
}

// Emit the dataset-level and per-column findings into `r` from a `CsvData`.
void emit_findings(DataQualityReport& r, const CsvData& data,
                   const ScanOptions& opts) {
    const int n_rows = data.scanned_rows;
    const int n_cols = static_cast<int>(data.header.size());

    // TOO_FEW_ROWS / EMPTY_FILE.
    if (n_rows == 0) {
        r.findings.push_back(make_finding(
            Severity::Error, "EMPTY_FILE", "",
            "File has a header but no data rows.",
            "Check the export — you may have exported headers only."));
        return;
    }
    if (n_rows < opts.min_rows_for_train) {
        r.findings.push_back(make_finding(
            Severity::Error, "TOO_FEW_ROWS", "",
            std::to_string(n_rows) + " rows is below the minimum (" +
            std::to_string(opts.min_rows_for_train) + ") needed to train a "
            "stable model.",
            "Collect at least " +
            std::to_string(opts.min_rows_for_train - n_rows) +
            " more rows or reduce task complexity."));
    }

    // Which column is being predicted. Defaults to the last one.
    int target_idx = n_cols - 1;
    if (!opts.target_column.empty()) {
        for (int c = 0; c < n_cols; ++c) {
            if (data.header[c] == opts.target_column) { target_idx = c; break; }
        }
    }

    // Per-column findings.
    //
    // Severity policy: only problems that make training impossible are
    // errors (and those concern the file or the target). A troublesome
    // input column is a warning, because the pipeline deals with it —
    // blank columns are left out, gaps are filled — and the user can
    // always leave a column out.
    const auto n_rows_d = static_cast<double>(n_rows);
    for (int c = 0; c < n_cols; ++c) {
        const auto& col = data.columns[c];
        const std::string& name = data.header[c];
        const double missing_frac = col.n_missing / n_rows_d;
        const bool is_target = (c == target_idx);

        if (col.n_missing == n_rows) {
            if (is_target) {
                r.findings.push_back(make_finding(
                    Severity::Error, "ALL_NAN_COLUMN", name,
                    "Every value in the target column '" + name + "' is blank.",
                    "Pick a different column to predict."));
            } else {
                r.findings.push_back(make_finding(
                    Severity::Warning, "ALL_NAN_COLUMN", name,
                    "Every value in '" + name + "' is blank.",
                    "The column will be left out."));
            }
            continue;  // no other findings matter for an all-blank column
        }
        if (is_target && col.n_missing > 0) {
            r.findings.push_back(make_finding(
                Severity::Warning, "MISSING_VALUES", name,
                std::to_string(col.n_missing) + " rows have a blank target ('" +
                name + "').",
                "Those rows will be skipped; there is nothing to learn from them."));
        } else if (missing_frac >= opts.missing_error_frac) {
            r.findings.push_back(make_finding(
                Severity::Warning, "MISSING_VALUES", name,
                "Column '" + name + "' is " +
                std::to_string(int(missing_frac * 100)) + "% blank.",
                "Gaps are filled in automatically, but a column this empty adds "
                "little. Consider leaving it out."));
        } else if (missing_frac >= opts.missing_warn_frac) {
            r.findings.push_back(make_finding(
                Severity::Warning, "MISSING_VALUES", name,
                "Column '" + name + "' is " +
                std::to_string(int(missing_frac * 100)) + "% blank.",
                "Gaps are filled in automatically."));
        }

        if (is_target && !col.cardinality_overflowed
            && col.category_counts.size() == 1) {
            r.findings.push_back(make_finding(
                Severity::Error, "TARGET_CONSTANT", name,
                "The target column '" + name + "' has the same value on every row.",
                "There is nothing to predict. Pick a different column."));
            continue;
        }

        // CONSTANT_COLUMN — after excluding missing, is there only one value?
        const bool is_numeric = column_is_numeric(col);
        if (is_numeric) {
            if (col.numeric_stats.n > 0 && col.numeric_stats.stddev() == 0.0) {
                r.findings.push_back(make_finding(
                    Severity::Warning, "CONSTANT_COLUMN", name,
                    "Column '" + name + "' has the same value for every row.",
                    "Column carries no information; it will be dropped."));
            }
        } else if (!col.cardinality_overflowed &&
                   col.category_counts.size() == 1) {
            r.findings.push_back(make_finding(
                Severity::Warning, "CONSTANT_COLUMN", name,
                "Column '" + name + "' has the same value for every row.",
                "Column carries no information; it will be dropped."));
        }

        // HIGH_CARDINALITY — categorical with > max(100, n_rows*0.5) uniques.
        if (!is_numeric && !is_target) {
            const int threshold = std::max(100, n_rows / 2);
            if (col.cardinality_overflowed ||
                static_cast<int>(col.category_counts.size()) > threshold) {
                r.findings.push_back(make_finding(
                    Severity::Warning, "HIGH_CARDINALITY", name,
                    "Column '" + name + "' has a very large number of different "
                    "values.",
                    "If it is an identifier or free text it will be left out; "
                    "otherwise only the most common values are kept."));
            }
        }

        // MIXED_TYPES — column was judged numeric but some values failed to
        // parse (or judged categorical but has partial numeric content is
        // not worth flagging — that's normal).
        if (is_numeric && col.n_categorical > 0) {
            r.findings.push_back(make_finding(
                Severity::Warning, "MIXED_TYPES", name,
                "Column '" + name + "' is mostly numeric but contains " +
                std::to_string(col.n_categorical) + " non-numeric values.",
                "Coerce to string or clean the offending values."));
        }

        // OUTLIERS — >outlier_warn_frac of rows outside mean±k*sigma.
        if (is_numeric && col.numeric_stats.n > 10) {
            const double m = col.numeric_stats.mean;
            const double s = col.numeric_stats.stddev();
            if (s > 0.0) {
                int outside = 0;
                for (double v : col.numeric_cache) {
                    if (std::abs(v - m) > opts.outlier_sigma * s) ++outside;
                }
                const double frac = outside / static_cast<double>(col.numeric_cache.size());
                if (frac >= opts.outlier_warn_frac) {
                    r.findings.push_back(make_finding(
                        Severity::Info, "OUTLIERS", name,
                        std::to_string(int(frac * 100)) + "% of values in '" +
                        name + "' are beyond ±" +
                        std::to_string(static_cast<int>(opts.outlier_sigma)) +
                        "σ of the mean.",
                        "Consider robust scaling or clipping."));
                }
            }
        }
    }

    // DUPLICATE_ROWS.
    int dup_rows = 0;
    for (const auto& kv : data.row_hash_counts) {
        if (kv.second > 1) dup_rows += kv.second - 1;
    }
    if (dup_rows > 0 &&
        static_cast<double>(dup_rows) / n_rows > opts.duplicate_warn_frac) {
        r.findings.push_back(make_finding(
            Severity::Warning, "DUPLICATE_ROWS", "",
            std::to_string(dup_rows) + " duplicate rows detected (" +
            std::to_string(int(100.0 * dup_rows / n_rows)) + "% of data).",
            "Dedupe via the Preprocessing panel."));
    }

    // CLASS_IMBALANCE — on the target column when it holds classes.
    // Ratio = most_common / least_common. 1.0 means perfectly balanced.
    r.class_imbalance_ratio = 1.0;
    if (n_cols >= 1) {
        const auto& target = data.columns[target_idx];
        const bool target_numeric = column_is_numeric(target);
        // Same rule the pipeline uses to tell classes from quantities.
        const std::size_t class_limit = std::max<std::size_t>(
            20, static_cast<std::size_t>(std::sqrt(static_cast<double>(n_rows))));
        const bool holds_classes =
            !target.cardinality_overflowed && target.category_counts.size() >= 2
            && (!target_numeric || target.category_counts.size() <= class_limit);
        if (holds_classes) {
            int mn = std::numeric_limits<int>::max();
            int mx = 0;
            for (const auto& kv : target.category_counts) {
                mn = std::min(mn, kv.second);
                mx = std::max(mx, kv.second);
            }
            if (mn > 0) {
                r.class_imbalance_ratio =
                    static_cast<double>(mx) / static_cast<double>(mn);
            }
            if (r.class_imbalance_ratio >= opts.imbalance_error_at) {
                r.findings.push_back(make_finding(
                    Severity::Warning, "CLASS_IMBALANCE", target.name,
                    "The rarest class in '" + target.name + "' is about " +
                    std::to_string(static_cast<int>(r.class_imbalance_ratio)) +
                    " times rarer than the commonest.",
                    "Overall accuracy will look high even for a useless model. "
                    "Judge the result by how many of the rare cases it finds, "
                    "and collect more of them if you can."));
            } else if (r.class_imbalance_ratio >= opts.imbalance_warn_at) {
                r.findings.push_back(make_finding(
                    Severity::Warning, "CLASS_IMBALANCE", target.name,
                    "Class frequencies differ by a factor of " +
                    std::to_string(static_cast<int>(r.class_imbalance_ratio)) +
                    ":1 in the target column.",
                    "Check the per-class results, not just overall accuracy."));
            }
        }
    }

    // TARGET_LEAKAGE_SUSPECT — numeric feature almost perfectly correlated
    // with a numeric target. Cheap to compute; skip when either side is
    // non-numeric (categorical correlation needs a different method).
    if (n_cols >= 2) {
        const auto& target = data.columns[target_idx];
        if (column_is_numeric(target)) {
            for (int c = 0; c < n_cols; ++c) {
                if (c == target_idx) continue;
                const auto& col = data.columns[c];
                if (!column_is_numeric(col)) continue;
                if (col.numeric_cache.size() < 10 ||
                    target.numeric_cache.size() < 10) continue;

                // Align vectors by index — they correspond row-for-row only
                // when both values were numeric for the same rows. In the
                // common "all-numeric column" case that's every row; we
                // truncate to the shorter one as a safe approximation.
                const std::size_t n = std::min(col.numeric_cache.size(),
                                               target.numeric_cache.size());
                std::vector<double> x(col.numeric_cache.begin(),
                                      col.numeric_cache.begin() + n);
                std::vector<double> y(target.numeric_cache.begin(),
                                      target.numeric_cache.begin() + n);
                const double corr = pearson_correlation(x, y);
                if (!std::isnan(corr) &&
                    std::abs(corr) > opts.leakage_corr_threshold) {
                    r.findings.push_back(make_finding(
                        Severity::Warning, "TARGET_LEAKAGE_SUSPECT",
                        data.header[c],
                        "'" + data.header[c] + "' tracks the target almost "
                        "exactly. That usually means it is another form of the "
                        "answer rather than something known beforehand.",
                        "If '" + data.header[c] + "' is only known after the "
                        "outcome, leave it out; otherwise the model will look "
                        "accurate and be useless."));
                }
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

DataQualityReport scan_csv(const std::filesystem::path& csv,
                           const ScanOptions& opts) {
    CsvData data;
    if (auto early = read_csv_into(data, csv, opts); early.has_value()) {
        return *early;
    }

    DataQualityReport r;
    r.n_cols = static_cast<int>(data.header.size());
    r.n_rows = data.scanned_rows > 0 ? data.total_rows_in_file : 0;

    // Per-column finalized stats.
    r.columns.reserve(data.columns.size());
    for (const auto& col : data.columns) {
        r.columns.push_back(finalize_column(col, opts.top_k_categories));
    }

    emit_findings(r, data, opts);
    return finalize_report(std::move(r));
}

DataQualityReport scan_samples(const std::vector<training::Sample>& samples,
                               const ScanOptions& opts) {
    DataQualityReport r;
    r.n_rows = static_cast<int>(samples.size());

    if (samples.empty()) {
        r.findings.push_back(make_finding(
            Severity::Error, "EMPTY_FILE", "",
            "No samples loaded.", "Re-load the dataset."));
        return finalize_report(std::move(r));
    }

    const std::size_t n_features = samples.front().features.size();
    const std::size_t n_targets  = samples.front().target.size();
    r.n_cols = static_cast<int>(n_features + n_targets);

    // Feature-shape consistency check (programmer-level bug if this trips).
    for (std::size_t i = 1; i < samples.size(); ++i) {
        if (samples[i].features.size() != n_features ||
            samples[i].target.size()   != n_targets) {
            r.findings.push_back(make_finding(
                Severity::Error, "MIXED_TYPES", "",
                "Sample " + std::to_string(i) + " has a different feature "
                "or target size from sample 0.",
                "Re-check the dataset loader — all rows must have the same "
                "number of features."));
            return finalize_report(std::move(r));
        }
    }

    // TOO_FEW_ROWS.
    if (r.n_rows < opts.min_rows_for_train) {
        r.findings.push_back(make_finding(
            Severity::Error, "TOO_FEW_ROWS", "",
            std::to_string(r.n_rows) + " samples is below the minimum (" +
            std::to_string(opts.min_rows_for_train) + ") needed to train a "
            "stable model.",
            "Collect more samples."));
    }

    // Per-feature numeric stats and CONSTANT_COLUMN / OUTLIERS.
    r.columns.resize(n_features);
    for (std::size_t j = 0; j < n_features; ++j) {
        OnlineStats st;
        for (const auto& s : samples) {
            const double v = s.features[j];
            if (std::isnan(v)) continue;
            st.push(v);
        }
        ColumnStats cs;
        cs.name       = "feature_" + std::to_string(j);
        cs.is_numeric = true;
        cs.mean       = st.mean;
        cs.stddev     = st.stddev();
        cs.min        = (st.n > 0) ? st.min_v : 0.0;
        cs.max        = (st.n > 0) ? st.max_v : 0.0;
        cs.n_missing  = static_cast<int>(samples.size()) - st.n;
        r.columns[j]  = cs;

        if (st.n == 0) {
            r.findings.push_back(make_finding(
                Severity::Error, "ALL_NAN_COLUMN", cs.name,
                "Every value in '" + cs.name + "' is missing (NaN).",
                "Drop this feature from the dataset."));
        } else if (st.stddev() == 0.0) {
            r.findings.push_back(make_finding(
                Severity::Warning, "CONSTANT_COLUMN", cs.name,
                "Feature '" + cs.name + "' has the same value for every sample.",
                "Drop this feature — it carries no information."));
        }

        // OUTLIERS using the same criterion as scan_csv.
        if (st.n > 10 && st.stddev() > 0.0) {
            int outside = 0;
            for (const auto& s : samples) {
                const double v = s.features[j];
                if (std::isnan(v)) continue;
                if (std::abs(v - st.mean) > opts.outlier_sigma * st.stddev()) ++outside;
            }
            const double frac = static_cast<double>(outside) /
                                static_cast<double>(st.n);
            if (frac >= opts.outlier_warn_frac) {
                r.findings.push_back(make_finding(
                    Severity::Info, "OUTLIERS", cs.name,
                    std::to_string(int(frac * 100)) + "% of values in '" +
                    cs.name + "' are beyond ±" +
                    std::to_string(static_cast<int>(opts.outlier_sigma)) +
                    "σ of the mean.",
                    "Consider robust scaling or clipping."));
            }
        }
    }

    // CLASS_IMBALANCE — only if target looks like one-hot (values are
    // 0 or 1 and exactly one element is 1 per sample).
    if (n_targets >= 2) {
        bool looks_one_hot = true;
        std::vector<int> class_counts(n_targets, 0);
        for (const auto& s : samples) {
            int hot = -1;
            for (std::size_t k = 0; k < n_targets; ++k) {
                const double v = s.target[k];
                if (v != 0.0 && v != 1.0) { looks_one_hot = false; break; }
                if (v == 1.0) {
                    if (hot != -1) { looks_one_hot = false; break; }
                    hot = static_cast<int>(k);
                }
            }
            if (!looks_one_hot || hot == -1) { looks_one_hot = false; break; }
            ++class_counts[hot];
        }
        if (looks_one_hot) {
            int mn = std::numeric_limits<int>::max();
            int mx = 0;
            for (int c : class_counts) {
                mn = std::min(mn, c);
                mx = std::max(mx, c);
            }
            if (mn > 0) {
                r.class_imbalance_ratio = static_cast<double>(mx) / mn;
                if (r.class_imbalance_ratio >= opts.imbalance_error_at) {
                    r.findings.push_back(make_finding(
                        Severity::Error, "CLASS_IMBALANCE", "",
                        "Class frequencies differ by a factor of " +
                        std::to_string(static_cast<int>(r.class_imbalance_ratio)) +
                        ":1.",
                        "Use class weights or oversample the minority classes."));
                } else if (r.class_imbalance_ratio >= opts.imbalance_warn_at) {
                    r.findings.push_back(make_finding(
                        Severity::Warning, "CLASS_IMBALANCE", "",
                        "Class frequencies differ by a factor of " +
                        std::to_string(static_cast<int>(r.class_imbalance_ratio)) +
                        ":1.",
                        "Use class weights or collect more minority samples."));
                }
            }
        }
    }

    // DUPLICATE_ROWS across sample vectors (feature-only hash — post-load
    // samples don't preserve category strings anyway).
    std::unordered_map<std::size_t, int> hash_counts;
    hash_counts.reserve(samples.size() * 2);
    for (const auto& s : samples) {
        std::string key;
        key.reserve(s.features.size() * 8);
        for (double v : s.features) {
            // Round to 12 decimal places so tiny FP jitter doesn't make
            // "duplicates" look unique.
            const double rv = std::round(v * 1e12) / 1e12;
            key.append(reinterpret_cast<const char*>(&rv), sizeof(rv));
        }
        ++hash_counts[std::hash<std::string>{}(key)];
    }
    int dup = 0;
    for (const auto& kv : hash_counts) if (kv.second > 1) dup += kv.second - 1;
    if (dup > 0 &&
        static_cast<double>(dup) / r.n_rows > opts.duplicate_warn_frac) {
        r.findings.push_back(make_finding(
            Severity::Warning, "DUPLICATE_ROWS", "",
            std::to_string(dup) + " duplicate samples detected.",
            "Dedupe the dataset."));
    }

    return finalize_report(std::move(r));
}

} // namespace neurograph::dataqc
