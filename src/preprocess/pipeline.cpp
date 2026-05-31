#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/preprocess/transforms.hpp"
#include "neurograph/dataqc/scanner.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace neurograph::preprocess {

// ---------------------------------------------------------------------------
// CSV parsing — same semantics as dataqc/scanner.cpp (quoted fields,
// doubled-quote escape, UTF-8 BOM, CRLF). Pragmatic copy rather than
// extracting into a shared header yet — will consolidate once a third
// caller needs it.
// ---------------------------------------------------------------------------
namespace {

constexpr char kBom0 = static_cast<char>(0xEF);
constexpr char kBom1 = static_cast<char>(0xBB);
constexpr char kBom2 = static_cast<char>(0xBF);

void strip_bom(std::string& s) {
    if (s.size() >= 3 && s[0] == kBom0 && s[1] == kBom1 && s[2] == kBom2) {
        s.erase(0, 3);
    }
}

void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
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
                in_quotes = true;
            } else {
                cur += c;
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

std::string trim(std::string s) {
    auto notspace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

// Returns {value, parsed_ok, is_missing}. Mirrors the scanner's semantics.
struct NumericParse {
    double value      = 0.0;
    bool   parsed     = false;
    bool   is_missing = false;
};

NumericParse try_parse_number(std::string_view raw) {
    NumericParse out{};
    const std::string s = trim(std::string(raw));
    if (s.empty()) { out.is_missing = true; return out; }

    auto eq_ci = [](std::string_view a, std::string_view b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(a[i])) !=
                std::tolower(static_cast<unsigned char>(b[i]))) return false;
        }
        return true;
    };
    for (std::string_view tok : {"nan", "na", "n/a", "null", "none", "?", "-"}) {
        if (eq_ci(s, tok)) { out.is_missing = true; return out; }
    }

    double v = 0.0;
    const char* first = s.data();
    const char* last  = s.data() + s.size();
    auto result = std::from_chars(first, last, v);
    if (result.ec == std::errc{} && result.ptr == last) {
        out.value  = v;
        out.parsed = true;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Enum ↔ string — used by save/load.
// ---------------------------------------------------------------------------

const char* to_string(ImputeStrategy s) {
    switch (s) {
    case ImputeStrategy::Mean:     return "Mean";
    case ImputeStrategy::Median:   return "Median";
    case ImputeStrategy::Mode:     return "Mode";
    case ImputeStrategy::Constant: return "Constant";
    }
    return "Mean";
}
const char* to_string(ScaleStrategy s) {
    switch (s) {
    case ScaleStrategy::None:     return "None";
    case ScaleStrategy::Standard: return "Standard";
    case ScaleStrategy::MinMax:   return "MinMax";
    case ScaleStrategy::Robust:   return "Robust";
    }
    return "None";
}
const char* to_string(EncodeStrategy s) {
    switch (s) {
    case EncodeStrategy::OneHot:           return "OneHot";
    case EncodeStrategy::Ordinal:          return "Ordinal";
    case EncodeStrategy::HashFixedBuckets: return "HashFixedBuckets";
    }
    return "OneHot";
}
ImputeStrategy parse_impute(std::string_view s) {
    if (s == "Median")   return ImputeStrategy::Median;
    if (s == "Mode")     return ImputeStrategy::Mode;
    if (s == "Constant") return ImputeStrategy::Constant;
    return ImputeStrategy::Mean;
}
ScaleStrategy parse_scale(std::string_view s) {
    if (s == "Standard") return ScaleStrategy::Standard;
    if (s == "MinMax")   return ScaleStrategy::MinMax;
    if (s == "Robust")   return ScaleStrategy::Robust;
    return ScaleStrategy::None;
}
EncodeStrategy parse_encode(std::string_view s) {
    if (s == "Ordinal")          return EncodeStrategy::Ordinal;
    if (s == "HashFixedBuckets") return EncodeStrategy::HashFixedBuckets;
    return EncodeStrategy::OneHot;
}

} // namespace

// ---------------------------------------------------------------------------
// FittedPipeline::Impl — all internal state. Hidden from the header so
// the on-disk schema can evolve without ABI breaks.
// ---------------------------------------------------------------------------

struct FittedColumn {
    ColumnSpec  spec;
    bool        is_numeric        = true;
    bool        is_target         = false;
    bool        drop              = false;
    Imputer     imputer;
    Scaler      scaler;
    Encoder     encoder;
};

struct FittedPipeline::Impl {
    int                        version          = 1;
    std::vector<std::string>   header;
    std::vector<FittedColumn>  columns;
    std::string                target_column;
    bool                       is_classification = true;
    std::vector<std::string>   class_labels;
    std::vector<std::string>   feature_names_out;
};

// pImpl boilerplate — all defined here where Impl is complete.
FittedPipeline::FittedPipeline()
    : impl(std::make_unique<Impl>()) {}
FittedPipeline::~FittedPipeline() = default;
FittedPipeline::FittedPipeline(FittedPipeline&&) noexcept            = default;
FittedPipeline& FittedPipeline::operator=(FittedPipeline&&) noexcept = default;

const std::vector<std::string>& FittedPipeline::feature_names_out() const {
    return impl->feature_names_out;
}
const std::vector<std::string>& FittedPipeline::class_labels() const {
    return impl->class_labels;
}
bool FittedPipeline::is_classification() const { return impl->is_classification; }
int  FittedPipeline::version()           const { return impl->version; }

// ---------------------------------------------------------------------------
// Helpers used by fit_pipeline
// ---------------------------------------------------------------------------
namespace {

// Raw CSV shape we load into memory before fitting. For v1 we assume the
// dataset fits — the scanner's reservoir sampling strategy belongs on a
// later pass (P1-14 AutoPipeline with explicit sampling opt-in).
struct RawCsv {
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
};

RawCsv read_whole_csv(const std::filesystem::path& csv) {
    std::ifstream f(csv, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open CSV: " + csv.string());
    }
    RawCsv r;
    std::string line;
    if (!std::getline(f, line)) {
        throw std::runtime_error("CSV is empty: " + csv.string());
    }
    strip_bom(line);
    strip_cr(line);
    r.header = parse_csv_line(line);

    while (std::getline(f, line)) {
        strip_cr(line);
        if (line.empty()) continue;
        auto row = parse_csv_line(line);
        if (row.size() < r.header.size()) row.resize(r.header.size());
        r.rows.push_back(std::move(row));
    }
    if (r.rows.empty()) {
        throw std::runtime_error("CSV has header but no data rows: " +
                                 csv.string());
    }
    return r;
}

int find_column_index(const std::vector<std::string>& header,
                      std::string_view name) {
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[i] == name) return i;
    }
    return -1;
}

// Sniff whether a column parses mostly as numeric. Matches the scanner's
// 90% rule so auto-spec here agrees with the scanner's type decisions.
bool column_parses_numeric(const std::vector<std::vector<std::string>>& rows,
                           int col) {
    int n_numeric = 0, n_non_missing = 0;
    for (const auto& row : rows) {
        if (col >= static_cast<int>(row.size())) continue;
        const auto p = try_parse_number(row[col]);
        if (p.is_missing) continue;
        ++n_non_missing;
        if (p.parsed) ++n_numeric;
    }
    if (n_non_missing == 0) return false;
    return static_cast<double>(n_numeric) / n_non_missing >= 0.9;
}

// Build auto-specs when the caller passed an empty `config.columns`. Uses
// the P1-01 scanner as the source of truth so the wizard's Data Health
// screen and the pipeline agree on types and drops.
std::vector<ColumnSpec>
auto_spec_from_scanner(const std::filesystem::path& csv,
                       const std::vector<std::string>& header,
                       const std::string& target_column) {
    dataqc::ScanOptions sopts;
    sopts.min_rows_for_train = 1;   // we want the stats, not the gate
    const auto report = dataqc::scan_csv(csv, sopts);

    std::unordered_map<std::string, const dataqc::ColumnStats*> by_name;
    by_name.reserve(report.columns.size());
    for (const auto& c : report.columns) by_name[c.name] = &c;

    std::vector<ColumnSpec> specs;
    specs.reserve(header.size());
    for (const auto& name : header) {
        ColumnSpec s;
        s.name      = name;
        s.is_target = (name == target_column);

        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            // Shouldn't happen — report covers every header column.
            specs.push_back(std::move(s));
            continue;
        }
        const auto& cs = *it->second;

        s.is_numeric = cs.is_numeric;

        // Constant / all-NaN columns are dead weight. Never drop the target
        // though — that's a CLASS_IMBALANCE / ALL_NAN finding instead.
        if (!s.is_target) {
            if (cs.cardinality_overflowed) {
                // HIGH_CARDINALITY → switch to hashing so the matrix stays
                // bounded.
                s.is_numeric   = false;
                s.encode       = EncodeStrategy::HashFixedBuckets;
                s.hash_buckets = 32;
            }
            const bool all_nan  = (cs.n_missing > 0 && cs.n_unique == 0 &&
                                   !cs.cardinality_overflowed && !cs.is_numeric);
            const bool constant = (cs.n_unique == 1) ||
                                  (cs.is_numeric && cs.stddev == 0.0 && cs.n_missing == 0);
            if (all_nan || constant) s.drop = true;
        }

        if (s.is_numeric) {
            s.impute = ImputeStrategy::Mean;
            s.scale  = ScaleStrategy::Standard;
        } else {
            s.impute = ImputeStrategy::Mode;
            if (s.encode != EncodeStrategy::HashFixedBuckets) {
                s.encode = EncodeStrategy::OneHot;
            }
        }
        specs.push_back(std::move(s));
    }
    return specs;
}

// Merge user-provided specs with auto defaults — fills in missing columns
// and stamps `is_target` based on config.target_column.
std::vector<ColumnSpec>
reconcile_specs(std::vector<ColumnSpec> base,
                const std::vector<ColumnSpec>& user,
                const std::string& target_column) {
    std::unordered_map<std::string, const ColumnSpec*> overrides;
    for (const auto& s : user) overrides[s.name] = &s;
    for (auto& s : base) {
        auto it = overrides.find(s.name);
        if (it != overrides.end()) s = *it->second;
        s.is_target = (s.name == target_column);
    }
    return base;
}

// Decide classification vs. regression and produce class labels.
// Rules (applied in order):
//   1. target has non-numeric values      → classification, labels = sorted unique
//   2. numeric target with ≤ K distinct   → classification, labels = sorted unique as strings
//      (K = max(20, sqrt(n)))
//   3. otherwise                          → regression
struct TargetProfile {
    bool                     is_classification = true;
    std::vector<std::string> class_labels;     // classification only
    std::vector<std::string> raw_target_values; // for all rows, original strings
};

TargetProfile profile_target(
    const std::vector<std::vector<std::string>>& rows,
    int target_col) {

    TargetProfile p;
    p.raw_target_values.reserve(rows.size());
    bool all_numeric = true;
    std::set<std::string> distinct;
    for (const auto& row : rows) {
        const std::string& v = (target_col < static_cast<int>(row.size()))
                             ? row[target_col] : std::string{};
        p.raw_target_values.push_back(v);
        distinct.insert(v);
        const auto np = try_parse_number(v);
        if (!np.is_missing && !np.parsed) all_numeric = false;
    }

    if (!all_numeric) {
        p.is_classification = true;
        p.class_labels.assign(distinct.begin(), distinct.end());
        return p;
    }

    const std::size_t k_thresh = std::max<std::size_t>(
        20, static_cast<std::size_t>(std::sqrt(static_cast<double>(rows.size()))));
    if (distinct.size() <= k_thresh && distinct.size() >= 2) {
        p.is_classification = true;
        p.class_labels.assign(distinct.begin(), distinct.end());
        // Sort numerically when all entries parse as numbers so "2" < "10"
        // instead of lex "10" < "2".
        std::sort(p.class_labels.begin(), p.class_labels.end(),
                  [](const std::string& a, const std::string& b) {
                      double da = 0, db = 0;
                      std::from_chars(a.data(), a.data() + a.size(), da);
                      std::from_chars(b.data(), b.data() + b.size(), db);
                      return da < db;
                  });
        return p;
    }

    p.is_classification = false;
    return p;
}

// Split indices stratified by class. Round-robin within each class bucket
// so proportions stay matched across train/val/test.
struct SplitIndices {
    std::vector<std::size_t> train, val, test;
};

SplitIndices stratified_split(std::size_t n,
                              const std::vector<int>& class_of_row,
                              double val_frac, double test_frac,
                              std::uint64_t seed) {
    SplitIndices s;

    std::unordered_map<int, std::vector<std::size_t>> by_class;
    for (std::size_t i = 0; i < n; ++i) {
        by_class[class_of_row[i]].push_back(i);
    }
    std::mt19937_64 rng(seed);
    for (auto& kv : by_class) {
        std::shuffle(kv.second.begin(), kv.second.end(), rng);
    }
    for (auto& kv : by_class) {
        const auto& idxs = kv.second;
        const std::size_t n_test = static_cast<std::size_t>(
            std::floor(idxs.size() * test_frac));
        const std::size_t n_val  = static_cast<std::size_t>(
            std::floor(idxs.size() * val_frac));
        for (std::size_t i = 0;                        i < n_test; ++i) s.test.push_back(idxs[i]);
        for (std::size_t i = n_test;                   i < n_test + n_val; ++i) s.val.push_back(idxs[i]);
        for (std::size_t i = n_test + n_val;           i < idxs.size(); ++i) s.train.push_back(idxs[i]);
    }
    // Final shuffle across classes so training order isn't sorted by class.
    std::shuffle(s.train.begin(), s.train.end(), rng);
    std::shuffle(s.val.begin(),   s.val.end(),   rng);
    std::shuffle(s.test.begin(),  s.test.end(),  rng);
    return s;
}

SplitIndices random_split(std::size_t n,
                          double val_frac, double test_frac,
                          std::uint64_t seed) {
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i) all[i] = i;
    std::mt19937_64 rng(seed);
    std::shuffle(all.begin(), all.end(), rng);
    const std::size_t n_test  = static_cast<std::size_t>(std::floor(n * test_frac));
    const std::size_t n_val   = static_cast<std::size_t>(std::floor(n * val_frac));
    SplitIndices s;
    s.test .assign(all.begin(),              all.begin() + n_test);
    s.val  .assign(all.begin() + n_test,     all.begin() + n_test + n_val);
    s.train.assign(all.begin() + n_test + n_val, all.end());
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// fit_pipeline
// ---------------------------------------------------------------------------
std::unique_ptr<FittedPipeline>
fit_pipeline(const std::filesystem::path& csv,
             const PipelineConfig& config,
             TransformedDataset& out_ds) {
    // 1. Load CSV into memory.
    const RawCsv raw = read_whole_csv(csv);
    if (config.target_column.empty()) {
        throw std::runtime_error("PipelineConfig::target_column is required.");
    }
    const int target_idx = find_column_index(raw.header, config.target_column);
    if (target_idx < 0) {
        throw std::runtime_error("Target column '" + config.target_column +
                                 "' not found in CSV header.");
    }

    // 2. Resolve per-column specs. Auto-spec fills in defaults, then user
    //    overrides merge on top.
    std::vector<ColumnSpec> auto_specs =
        auto_spec_from_scanner(csv, raw.header, config.target_column);
    std::vector<ColumnSpec> specs =
        reconcile_specs(std::move(auto_specs), config.columns,
                        config.target_column);

    // 3. Target profile — decides classification vs regression and class
    //    labels. Must come before fitting because some per-column decisions
    //    depend on the target being excluded.
    TargetProfile target = profile_target(raw.rows, target_idx);

    // 4. Build FittedPipeline state and fit each non-target column.
    auto pipe = std::make_unique<FittedPipeline>();
    auto& impl = *pipe->impl;
    impl.header            = raw.header;
    impl.target_column     = config.target_column;
    impl.is_classification = target.is_classification;
    impl.class_labels      = target.class_labels;
    impl.columns.resize(raw.header.size());

    // Target column state — trivial (we don't transform the target further
    // here, but keep its spec so save/load round-trips).
    impl.columns[target_idx].spec      = specs[target_idx];
    impl.columns[target_idx].is_target = true;
    impl.columns[target_idx].drop      = false;
    impl.columns[target_idx].is_numeric = !target.is_classification;

    // Non-target columns — impute → scale / encode.
    for (int c = 0; c < static_cast<int>(raw.header.size()); ++c) {
        if (c == target_idx) continue;

        FittedColumn& fc = impl.columns[c];
        fc.spec      = specs[c];
        fc.is_target = false;
        fc.drop      = specs[c].drop;
        fc.is_numeric = specs[c].is_numeric;
        if (fc.drop) continue;

        if (fc.is_numeric) {
            std::vector<double> values;
            values.reserve(raw.rows.size());
            for (const auto& row : raw.rows) {
                const std::string& s = (c < static_cast<int>(row.size()))
                                      ? row[c] : std::string{};
                const auto p = try_parse_number(s);
                values.push_back(p.parsed
                                 ? p.value
                                 : std::numeric_limits<double>::quiet_NaN());
            }
            fc.imputer.strategy = fc.spec.impute;
            fc.imputer.fit_numeric(values, fc.spec.impute_constant);

            // Apply imputation before fitting the scaler so the scaler
            // sees the same distribution as inference time.
            std::vector<double> imputed;
            imputed.reserve(values.size());
            for (double v : values) imputed.push_back(fc.imputer.apply_numeric(v));

            fc.scaler.strategy = fc.spec.scale;
            fc.scaler.fit(imputed);
        } else {
            std::vector<std::string> values;
            values.reserve(raw.rows.size());
            for (const auto& row : raw.rows) {
                values.push_back(c < static_cast<int>(row.size())
                                 ? row[c] : std::string{});
            }
            fc.imputer.strategy = fc.spec.impute;
            fc.imputer.fit_categorical(values);

            std::vector<std::string> imputed;
            imputed.reserve(values.size());
            for (const auto& v : values) imputed.push_back(fc.imputer.apply_categorical(v));

            fc.encoder.strategy     = fc.spec.encode;
            fc.encoder.hash_buckets = fc.spec.hash_buckets;
            fc.encoder.fit(imputed);
        }
    }

    // 5. Build feature_names_out from the fitted columns — reproducible
    //    across runs because encoder categories are sorted alphabetically.
    for (const auto& fc : impl.columns) {
        if (fc.is_target || fc.drop) continue;
        if (fc.is_numeric) {
            impl.feature_names_out.push_back(fc.spec.name);
            continue;
        }
        switch (fc.encoder.strategy) {
        case EncodeStrategy::OneHot:
            for (const auto& cat : fc.encoder.categories()) {
                impl.feature_names_out.push_back(fc.spec.name + "_" + cat);
            }
            impl.feature_names_out.push_back(fc.spec.name + "_unknown");
            break;
        case EncodeStrategy::Ordinal:
            impl.feature_names_out.push_back(fc.spec.name);
            break;
        case EncodeStrategy::HashFixedBuckets:
            for (int i = 0; i < fc.encoder.hash_buckets; ++i) {
                impl.feature_names_out.push_back(
                    fc.spec.name + "_h" + std::to_string(i));
            }
            break;
        }
    }

    // 6. Transform all rows → X matrix, y vector.
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;
    std::vector<int>                 class_of_row;  // classification only
    X.reserve(raw.rows.size());
    y.reserve(raw.rows.size());
    class_of_row.reserve(raw.rows.size());

    std::unordered_map<std::string, int> label_to_idx;
    for (std::size_t i = 0; i < impl.class_labels.size(); ++i) {
        label_to_idx[impl.class_labels[i]] = static_cast<int>(i);
    }

    for (std::size_t r = 0; r < raw.rows.size(); ++r) {
        const auto& row = raw.rows[r];
        std::vector<double> feat;
        feat.reserve(impl.feature_names_out.size());

        for (int c = 0; c < static_cast<int>(raw.header.size()); ++c) {
            const FittedColumn& fc = impl.columns[c];
            if (fc.is_target || fc.drop) continue;
            const std::string& raw_val = (c < static_cast<int>(row.size()))
                                        ? row[c] : std::string{};
            if (fc.is_numeric) {
                const auto p = try_parse_number(raw_val);
                const double x = p.parsed
                    ? p.value : std::numeric_limits<double>::quiet_NaN();
                const double imp = fc.imputer.apply_numeric(x);
                feat.push_back(fc.scaler.apply(imp));
            } else {
                const std::string imp = fc.imputer.apply_categorical(raw_val);
                for (double d : fc.encoder.apply(imp)) feat.push_back(d);
            }
        }
        X.push_back(std::move(feat));

        // y: one-hot for classification, raw numeric for regression.
        const std::string& tgt_raw = target.raw_target_values[r];
        if (impl.is_classification) {
            std::vector<double> one_hot(impl.class_labels.size(), 0.0);
            auto it = label_to_idx.find(tgt_raw);
            int  cls = (it != label_to_idx.end()) ? it->second : 0;
            if (it != label_to_idx.end()) one_hot[it->second] = 1.0;
            y.push_back(std::move(one_hot));
            class_of_row.push_back(cls);
        } else {
            const auto p = try_parse_number(tgt_raw);
            y.push_back({ p.parsed ? p.value : 0.0 });
            class_of_row.push_back(0);
        }
    }

    // 7. Split train/val/test.
    SplitIndices split;
    if (impl.is_classification && config.stratify) {
        split = stratified_split(X.size(), class_of_row,
                                 config.val_fraction, config.test_fraction,
                                 config.split_seed);
    } else {
        split = random_split(X.size(),
                             config.val_fraction, config.test_fraction,
                             config.split_seed);
    }

    auto pick = [&](const std::vector<std::size_t>& idx) {
        std::vector<std::vector<double>> Xs, Ys;
        Xs.reserve(idx.size()); Ys.reserve(idx.size());
        for (std::size_t i : idx) { Xs.push_back(X[i]); Ys.push_back(y[i]); }
        return std::pair{std::move(Xs), std::move(Ys)};
    };
    auto [Xtr, Ytr] = pick(split.train);
    auto [Xva, Yva] = pick(split.val);
    auto [Xte, Yte] = pick(split.test);

    out_ds.X_train            = std::move(Xtr);
    out_ds.X_val              = std::move(Xva);
    out_ds.X_test             = std::move(Xte);
    out_ds.y_train            = std::move(Ytr);
    out_ds.y_val              = std::move(Yva);
    out_ds.y_test             = std::move(Yte);
    out_ds.feature_names_out  = impl.feature_names_out;
    out_ds.class_labels       = impl.class_labels;
    out_ds.is_classification  = impl.is_classification;

    return pipe;
}

// ---------------------------------------------------------------------------
// apply_pipeline
// ---------------------------------------------------------------------------
std::vector<double>
apply_pipeline(const FittedPipeline& pipe, const nlohmann::json& raw_row) {
    const auto& impl = *pipe.impl;

    auto extract_numeric = [&](const std::string& name) -> double {
        if (!raw_row.is_object() || !raw_row.contains(name)) {
            spdlog::warn("apply_pipeline: missing column '{}' — imputing", name);
            return std::numeric_limits<double>::quiet_NaN();
        }
        const auto& v = raw_row.at(name);
        if (v.is_number()) return v.get<double>();
        if (v.is_null())   return std::numeric_limits<double>::quiet_NaN();
        if (v.is_string()) {
            const auto p = try_parse_number(v.get<std::string>());
            if (p.parsed) return p.value;
        }
        spdlog::warn("apply_pipeline: non-numeric value for numeric column '{}'",
                     name);
        return std::numeric_limits<double>::quiet_NaN();
    };

    auto extract_string = [&](const std::string& name) -> std::string {
        if (!raw_row.is_object() || !raw_row.contains(name)) {
            spdlog::warn("apply_pipeline: missing column '{}' — imputing", name);
            return "";
        }
        const auto& v = raw_row.at(name);
        if (v.is_string()) return v.get<std::string>();
        if (v.is_number()) {
            std::ostringstream oss; oss << v.get<double>();
            return oss.str();
        }
        if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
        if (v.is_null())    return "";
        return "";
    };

    std::vector<double> out;
    out.reserve(impl.feature_names_out.size());
    for (const auto& fc : impl.columns) {
        if (fc.is_target || fc.drop) continue;
        if (fc.is_numeric) {
            const double x   = extract_numeric(fc.spec.name);
            const double imp = fc.imputer.apply_numeric(x);
            out.push_back(fc.scaler.apply(imp));
        } else {
            const std::string s   = extract_string(fc.spec.name);
            const std::string imp = fc.imputer.apply_categorical(s);
            for (double d : fc.encoder.apply(imp)) out.push_back(d);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// save_pipeline / load_pipeline — JSON on disk.
// ---------------------------------------------------------------------------

void save_pipeline(const FittedPipeline& pipe,
                   const std::filesystem::path& file) {
    const auto& impl = *pipe.impl;
    nlohmann::json j;
    j["version"]           = impl.version;
    j["target_column"]     = impl.target_column;
    j["is_classification"] = impl.is_classification;
    j["class_labels"]      = impl.class_labels;
    j["header"]            = impl.header;
    j["feature_names_out"] = impl.feature_names_out;

    j["columns"] = nlohmann::json::array();
    for (const auto& fc : impl.columns) {
        nlohmann::json c;
        c["name"]       = fc.spec.name;
        c["is_numeric"] = fc.is_numeric;
        c["is_target"]  = fc.is_target;
        c["drop"]       = fc.drop;

        nlohmann::json impJ;
        impJ["strategy"]   = to_string(fc.imputer.strategy);
        impJ["fill_value"] = fc.imputer.numeric_fill();
        impJ["fill_token"] = fc.imputer.categorical_fill();
        c["imputer"] = impJ;

        nlohmann::json scJ;
        scJ["strategy"] = to_string(fc.scaler.strategy);
        scJ["a"]        = fc.scaler.offset();
        scJ["b"]        = fc.scaler.scale();
        c["scaler"] = scJ;

        nlohmann::json enJ;
        enJ["strategy"]     = to_string(fc.encoder.strategy);
        enJ["hash_buckets"] = fc.encoder.hash_buckets;
        enJ["categories"]   = fc.encoder.categories();
        c["encoder"] = enJ;

        c["spec_hash_buckets"]    = fc.spec.hash_buckets;
        c["spec_impute_constant"] = fc.spec.impute_constant;

        j["columns"].push_back(std::move(c));
    }

    std::ofstream f(file);
    if (!f) {
        spdlog::error("save_pipeline: cannot open '{}' for write", file.string());
        return;
    }
    f << j.dump(2);
}

std::unique_ptr<FittedPipeline>
load_pipeline(const std::filesystem::path& file) {
    std::ifstream f(file);
    if (!f) {
        spdlog::error("load_pipeline: cannot open '{}'", file.string());
        return nullptr;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        spdlog::error("load_pipeline: JSON parse error: {}", e.what());
        return nullptr;
    }

    const int version = j.value("version", 0);
    if (version != 1) {
        spdlog::error("load_pipeline: unsupported schema version {}", version);
        return nullptr;
    }

    auto pipe = std::make_unique<FittedPipeline>();
    auto& impl = *pipe->impl;
    impl.version           = version;
    impl.target_column     = j.value("target_column", "");
    impl.is_classification = j.value("is_classification", true);
    impl.class_labels      = j.value("class_labels",
                                     std::vector<std::string>{});
    impl.header            = j.value("header",
                                     std::vector<std::string>{});
    impl.feature_names_out = j.value("feature_names_out",
                                     std::vector<std::string>{});

    const auto& cols = j.at("columns");
    impl.columns.resize(cols.size());
    for (std::size_t i = 0; i < cols.size(); ++i) {
        const auto& c = cols[i];
        FittedColumn& fc = impl.columns[i];

        fc.spec.name            = c.value("name", "");
        fc.is_numeric           = c.value("is_numeric", true);
        fc.is_target            = c.value("is_target",  false);
        fc.drop                 = c.value("drop",       false);
        fc.spec.is_numeric      = fc.is_numeric;
        fc.spec.is_target       = fc.is_target;
        fc.spec.drop            = fc.drop;
        fc.spec.hash_buckets    = c.value("spec_hash_buckets", 32);
        fc.spec.impute_constant = c.value("spec_impute_constant", 0.0);

        const auto& iJ = c.at("imputer");
        fc.imputer.strategy = parse_impute(iJ.value("strategy", "Mean"));
        fc.imputer.set_numeric_fill(iJ.value("fill_value", 0.0));
        fc.imputer.set_categorical_fill(iJ.value("fill_token", ""));
        fc.spec.impute = fc.imputer.strategy;

        const auto& sJ = c.at("scaler");
        fc.scaler.strategy = parse_scale(sJ.value("strategy", "None"));
        fc.scaler.set_params(sJ.value("a", 0.0), sJ.value("b", 1.0));
        fc.spec.scale = fc.scaler.strategy;

        const auto& eJ = c.at("encoder");
        fc.encoder.strategy     = parse_encode(eJ.value("strategy", "OneHot"));
        fc.encoder.hash_buckets = eJ.value("hash_buckets", 32);
        fc.encoder.set_categories(eJ.value("categories",
                                           std::vector<std::string>{}));
        fc.spec.encode = fc.encoder.strategy;
    }

    return pipe;
}

} // namespace neurograph::preprocess
