#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/preprocess/transforms.hpp"
#include "neurograph/util/text_parse.hpp"

#include <nlohmann/json.hpp>
#include "neurograph/util/logger.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace neurograph::preprocess {

namespace {

using util::try_parse_date;
using util::try_parse_number;

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

// Thresholds for the automatic column decisions. Named so the rules read
// the same here, in plan_columns and in the tests.
constexpr double kNumericShare        = 0.90;  // share of non-blank cells that must parse as numbers
constexpr double kDateShare           = 0.90;  // same, for dates
constexpr double kIdentifierShare     = 0.50;  // distinct / non-blank above this looks like an identifier
constexpr int    kIdentifierMinValues = 20;    // ...but only when there are at least this many distinct values
constexpr int    kManyCategories      = 50;    // above this, keep only the most common
constexpr int    kKeptCategories      = 30;
constexpr double kMissingFlagShare    = 0.01;  // add an "is blank" feature above this share

// Date parts are scaled to roughly [-1, 1] with fixed constants so the
// transform needs no fitted state beyond the day count's mean and spread.
constexpr double kMonthCentre = 6.5,  kMonthScale = 3.5;
constexpr double kWeekdayCentre = 3.0, kWeekdayScale = 2.0;

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
    bool        is_date           = false;
    bool        date_day_first    = true;
    bool        missing_indicator = false;   // numeric only: extra 0/1 feature for blank cells
    std::string note;                        // why the column is treated this way
    Imputer     imputer;
    Scaler      scaler;
    Encoder     encoder;
};

struct FittedPipeline::Impl {
    int                        version          = 2;
    std::vector<std::string>   header;
    std::vector<FittedColumn>  columns;
    std::string                target_column;
    bool                       is_classification = true;
    std::vector<std::string>   class_labels;
    std::vector<std::string>   feature_names_out;
    std::vector<std::string>   feature_sources;
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

const std::vector<std::string>& FittedPipeline::feature_sources() const {
    return impl->feature_sources;
}
const std::string& FittedPipeline::target_column() const {
    return impl->target_column;
}
std::vector<std::string> FittedPipeline::input_columns() const {
    std::vector<std::string> out;
    for (const auto& fc : impl->columns) {
        if (!fc.is_target && !fc.drop) out.push_back(fc.spec.name);
    }
    return out;
}

namespace {

// ---------------------------------------------------------------------------
// Reading the CSV
// ---------------------------------------------------------------------------

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
    util::strip_bom(line);
    util::strip_cr(line);
    r.header = util::parse_csv_line(line);
    for (auto& h : r.header) h = util::trim(std::move(h));

    while (std::getline(f, line)) {
        util::strip_cr(line);
        if (line.empty()) continue;
        auto row = util::parse_csv_line(line);
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

const std::string& cell(const std::vector<std::string>& row, int col) {
    static const std::string empty;
    return (col < static_cast<int>(row.size())) ? row[static_cast<std::size_t>(col)] : empty;
}

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------

// Decide classification vs. regression and produce class labels.
// Rules (applied in order, unless the caller forces the task):
//   1. target has non-numeric values      → classification, labels = sorted unique
//   2. numeric target with ≤ K distinct   → classification, labels sorted numerically
//      (K = max(20, sqrt(n)))
//   3. otherwise                          → regression
struct TargetProfile {
    bool                     is_classification = true;
    std::vector<std::string> class_labels;      // classification only
};

TargetProfile profile_target(const std::vector<std::vector<std::string>>& rows,
                             int target_col, TaskChoice task) {
    TargetProfile p;
    bool all_numeric = true;
    std::set<std::string> distinct;
    for (const auto& row : rows) {
        const std::string v = util::trim(cell(row, target_col));
        distinct.insert(v);
        if (!try_parse_number(v).parsed) all_numeric = false;
    }

    auto as_classes = [&]() {
        p.is_classification = true;
        p.class_labels.assign(distinct.begin(), distinct.end());
        if (all_numeric) {
            // "2" before "10", not the other way round.
            std::stable_sort(p.class_labels.begin(), p.class_labels.end(),
                             [](const std::string& a, const std::string& b) {
                                 return try_parse_number(a).value < try_parse_number(b).value;
                             });
        }
    };

    if (task == TaskChoice::Regression) {
        if (!all_numeric) {
            throw std::runtime_error(
                "The target column contains text, so it cannot be predicted as a number. "
                "Choose 'Classification' or let the task be detected automatically.");
        }
        p.is_classification = false;
        return p;
    }
    if (task == TaskChoice::Classification) {
        as_classes();
        return p;
    }

    if (!all_numeric) { as_classes(); return p; }
    const std::size_t k_thresh = std::max<std::size_t>(
        20, static_cast<std::size_t>(std::sqrt(static_cast<double>(rows.size()))));
    if (distinct.size() <= k_thresh && distinct.size() >= 2) { as_classes(); return p; }
    p.is_classification = false;
    return p;
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

struct SplitIndices {
    std::vector<std::size_t> train, val, test;
};

// Stratified by class so each split keeps the class proportions.
SplitIndices stratified_split(std::size_t n,
                              const std::vector<int>& class_of_row,
                              double val_frac, double test_frac,
                              std::uint64_t seed) {
    SplitIndices s;
    // std::map (not unordered) so the split is identical across compilers.
    std::map<int, std::vector<std::size_t>> by_class;
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
            std::floor(static_cast<double>(idxs.size()) * test_frac));
        const std::size_t n_val  = static_cast<std::size_t>(
            std::floor(static_cast<double>(idxs.size()) * val_frac));
        for (std::size_t i = 0;              i < n_test;         ++i) s.test.push_back(idxs[i]);
        for (std::size_t i = n_test;         i < n_test + n_val; ++i) s.val.push_back(idxs[i]);
        for (std::size_t i = n_test + n_val; i < idxs.size();    ++i) s.train.push_back(idxs[i]);
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
    const std::size_t n_test = static_cast<std::size_t>(std::floor(static_cast<double>(n) * test_frac));
    const std::size_t n_val  = static_cast<std::size_t>(std::floor(static_cast<double>(n) * val_frac));
    SplitIndices s;
    s.test .assign(all.begin(),                  all.begin() + static_cast<std::ptrdiff_t>(n_test));
    s.val  .assign(all.begin() + static_cast<std::ptrdiff_t>(n_test),
                   all.begin() + static_cast<std::ptrdiff_t>(n_test + n_val));
    s.train.assign(all.begin() + static_cast<std::ptrdiff_t>(n_test + n_val), all.end());
    return s;
}

// ---------------------------------------------------------------------------
// Deciding what to do with a column
// ---------------------------------------------------------------------------

struct ColumnPlan {
    ColumnSpec  spec;
    bool        is_date           = false;
    bool        date_day_first    = true;
    bool        missing_indicator = false;
    std::string note;
};

bool name_suggests_identifier(const std::string& raw_name) {
    std::string n;
    for (char c : raw_name) n.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    n = util::trim(std::move(n));
    if (n == "id" || n == "index" || n == "idx" || n == "row" || n == "rownum"
        || n == "row_number" || n == "#" || n == "sr no" || n == "sr. no."
        || n == "s.no" || n == "sno" || n == "serial" || n == "unnamed: 0") {
        return true;
    }
    auto ends_with = [&](std::string_view suffix) {
        return n.size() > suffix.size()
            && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    return ends_with("_id") || ends_with(" id") || ends_with("-id")
        || ends_with("uuid") || ends_with("guid");
}

// Look at one column's cells (the rows listed in `rows_to_use`) and decide
// how to treat it. Pure function of the data, so plan_columns and
// fit_pipeline always agree.
ColumnPlan decide_column(const std::string& name,
                         const std::vector<std::vector<std::string>>& rows,
                         const std::vector<std::size_t>& rows_to_use,
                         int col) {
    ColumnPlan plan;
    plan.spec.name = name;

    std::size_t n_missing = 0, n_numeric = 0;
    std::size_t n_date_dmy = 0, n_date_mdy = 0;
    bool all_integers = true;
    double num_min = std::numeric_limits<double>::infinity();
    double num_max = -std::numeric_limits<double>::infinity();
    std::unordered_set<std::string> distinct;

    for (std::size_t r : rows_to_use) {
        const std::string& raw = cell(rows[r], col);
        const auto np = try_parse_number(raw);
        if (np.is_missing) { ++n_missing; continue; }
        distinct.insert(util::trim(raw));
        if (np.parsed) {
            ++n_numeric;
            num_min = std::min(num_min, np.value);
            num_max = std::max(num_max, np.value);
            if (np.value != std::floor(np.value)) all_integers = false;
        } else {
            if (try_parse_date(raw, true).parsed)  ++n_date_dmy;
            if (try_parse_date(raw, false).parsed) ++n_date_mdy;
        }
    }

    const std::size_t n_rows      = rows_to_use.size();
    const std::size_t non_missing = n_rows - n_missing;

    if (non_missing == 0) {
        plan.spec.drop = true;
        plan.note = "every value is blank";
        return plan;
    }
    if (distinct.size() == 1) {
        plan.spec.drop = true;
        plan.note = "the same value on every row";
        return plan;
    }

    const double numeric_share = static_cast<double>(n_numeric) / static_cast<double>(non_missing);
    if (numeric_share >= kNumericShare) {
        plan.spec.is_numeric = true;
        plan.spec.impute     = ImputeStrategy::Mean;
        plan.spec.scale      = ScaleStrategy::Standard;

        // A run of distinct whole numbers, or a distinct whole-number column
        // called something like "customer_id", is a row label, not a measurement.
        const bool all_distinct = (distinct.size() == non_missing) && n_missing == 0
                               && n_numeric == non_missing;
        if (all_distinct && all_integers && n_rows >= 20) {
            const bool sequential =
                (num_max - num_min + 1.0) == static_cast<double>(n_rows);
            if (sequential || name_suggests_identifier(name)) {
                plan.spec.drop = true;
                plan.note = sequential ? "looks like a row number"
                                       : "looks like an identifier";
                return plan;
            }
        }
        const double missing_share = static_cast<double>(n_missing) / static_cast<double>(n_rows);
        if (missing_share >= kMissingFlagShare) {
            plan.missing_indicator = true;
            plan.note = "blank cells are filled with the average and flagged";
        }
        return plan;
    }

    // Text column.
    plan.spec.is_numeric = false;
    plan.spec.impute     = ImputeStrategy::Mode;
    plan.spec.encode     = EncodeStrategy::OneHot;

    const std::size_t n_date = std::max(n_date_dmy, n_date_mdy);
    if (static_cast<double>(n_date) / static_cast<double>(non_missing) >= kDateShare) {
        plan.is_date        = true;
        // Day-first unless only the month-first reading fits the data.
        plan.date_day_first = (n_date_dmy >= n_date_mdy);
        plan.note = "read as a date: day count, month and weekday";
        return plan;
    }

    const double distinct_share =
        static_cast<double>(distinct.size()) / static_cast<double>(non_missing);
    if (distinct_share > kIdentifierShare
        && static_cast<int>(distinct.size()) > kIdentifierMinValues) {
        plan.spec.drop = true;
        plan.note = "a different value on almost every row (identifier or free text)";
        return plan;
    }
    if (static_cast<int>(distinct.size()) > kManyCategories) {
        plan.spec.max_categories = kKeptCategories;
        plan.note = "kept the " + std::to_string(kKeptCategories)
                  + " most common values; the rest are grouped together";
    }
    return plan;
}

ColumnPlan plan_from_user_spec(const ColumnSpec& user) {
    ColumnPlan p;
    p.spec = user;
    if (user.drop) p.note = "left out by request";
    return p;
}

// Plans for every non-target column, honouring per-column overrides and
// the exclusion list in `config`.
std::vector<ColumnPlan> plan_all(const RawCsv& raw,
                                 const std::vector<std::size_t>& rows_to_use,
                                 int target_idx,
                                 const PipelineConfig& config) {
    std::unordered_map<std::string, const ColumnSpec*> overrides;
    for (const auto& s : config.columns) overrides[s.name] = &s;
    const std::unordered_set<std::string> excluded(config.exclude_columns.begin(),
                                                   config.exclude_columns.end());

    std::vector<ColumnPlan> plans(raw.header.size());
    for (int c = 0; c < static_cast<int>(raw.header.size()); ++c) {
        const std::string& name = raw.header[static_cast<std::size_t>(c)];
        ColumnPlan& p = plans[static_cast<std::size_t>(c)];
        if (c == target_idx) {
            p.spec.name      = name;
            p.spec.is_target = true;
            continue;
        }
        if (auto it = overrides.find(name); it != overrides.end()) {
            p = plan_from_user_spec(*it->second);
            p.spec.name      = name;
            p.spec.is_target = false;
        } else {
            p = decide_column(name, raw.rows, rows_to_use, c);
        }
        if (excluded.count(name)) {
            p.spec.drop = true;
            p.note      = "left out by request";
        }
    }
    return plans;
}

ColumnDecision describe(const ColumnPlan& p) {
    ColumnDecision d;
    d.name = p.spec.name;
    d.note = p.note;
    if (p.spec.is_target)       d.role = "target";
    else if (p.spec.drop)       d.role = "left out";
    else if (p.is_date)         d.role = "date";
    else if (p.spec.is_numeric) d.role = "number";
    else                        d.role = "category";
    return d;
}

// Rows whose target cell is blank cannot be learned from.
std::vector<std::size_t> rows_with_target(const RawCsv& raw, int target_idx) {
    std::vector<std::size_t> keep;
    keep.reserve(raw.rows.size());
    for (std::size_t r = 0; r < raw.rows.size(); ++r) {
        if (!util::is_missing_token(cell(raw.rows[r], target_idx))) keep.push_back(r);
    }
    return keep;
}

int require_target(const RawCsv& raw, const PipelineConfig& config) {
    if (config.target_column.empty()) {
        throw std::runtime_error("PipelineConfig::target_column is required.");
    }
    const int target_idx = find_column_index(raw.header, config.target_column);
    if (target_idx < 0) {
        throw std::runtime_error("Target column '" + config.target_column +
                                 "' not found in CSV header.");
    }
    return target_idx;
}

// ---------------------------------------------------------------------------
// Turning one cell into features
// ---------------------------------------------------------------------------

void emit_numeric(const FittedColumn& fc, double x, std::vector<double>& out) {
    out.push_back(fc.scaler.apply(fc.imputer.apply_numeric(x)));
    if (fc.missing_indicator) out.push_back(std::isnan(x) ? 1.0 : 0.0);
}

void emit_date(const FittedColumn& fc, const std::string& raw, std::vector<double>& out) {
    const auto dp = try_parse_date(raw, fc.date_day_first);
    const double days = dp.parsed ? static_cast<double>(dp.days)
                                  : std::numeric_limits<double>::quiet_NaN();
    out.push_back(fc.scaler.apply(fc.imputer.apply_numeric(days)));
    out.push_back(dp.parsed ? (dp.month   - kMonthCentre)   / kMonthScale   : 0.0);
    out.push_back(dp.parsed ? (dp.weekday - kWeekdayCentre) / kWeekdayScale : 0.0);
}

void emit_category(const FittedColumn& fc, const std::string& raw, std::vector<double>& out) {
    // Blank markers ("NA", "null", ...) count as blank here exactly as they did when fitting.
    const std::string imp = fc.imputer.apply_categorical(
        util::is_missing_token(raw) ? std::string{} : util::trim(raw));
    for (double d : fc.encoder.apply(imp)) out.push_back(d);
}

void emit_from_text(const FittedColumn& fc, const std::string& raw, std::vector<double>& out) {
    if (fc.is_date) {
        emit_date(fc, raw, out);
    } else if (fc.is_numeric) {
        const auto p = try_parse_number(raw);
        emit_numeric(fc, p.parsed ? p.value : std::numeric_limits<double>::quiet_NaN(), out);
    } else {
        emit_category(fc, raw, out);
    }
}

void build_feature_names(FittedPipeline::Impl& impl) {
    impl.feature_names_out.clear();
    impl.feature_sources.clear();
    auto add = [&](const std::string& feature, const std::string& source) {
        impl.feature_names_out.push_back(feature);
        impl.feature_sources.push_back(source);
    };
    for (const auto& fc : impl.columns) {
        if (fc.is_target || fc.drop) continue;
        const std::string& name = fc.spec.name;
        if (fc.is_date) {
            add(name + "_days", name);
            add(name + "_month", name);
            add(name + "_weekday", name);
        } else if (fc.is_numeric) {
            add(name, name);
            if (fc.missing_indicator) add(name + "_missing", name);
        } else {
            switch (fc.encoder.strategy) {
            case EncodeStrategy::OneHot:
                for (const auto& cat : fc.encoder.categories()) add(name + "_" + cat, name);
                add(name + "_unknown", name);
                break;
            case EncodeStrategy::Ordinal:
                add(name, name);
                break;
            case EncodeStrategy::HashFixedBuckets:
                for (int i = 0; i < fc.encoder.hash_buckets; ++i) {
                    add(name + "_h" + std::to_string(i), name);
                }
                break;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// plan_columns
// ---------------------------------------------------------------------------

std::vector<ColumnDecision>
plan_columns(const std::filesystem::path& csv, const PipelineConfig& config) {
    const RawCsv raw = read_whole_csv(csv);
    const int target_idx = require_target(raw, config);
    const auto usable = rows_with_target(raw, target_idx);
    const auto plans = plan_all(raw, usable, target_idx, config);
    std::vector<ColumnDecision> out;
    out.reserve(plans.size());
    for (const auto& p : plans) out.push_back(describe(p));
    return out;
}

// ---------------------------------------------------------------------------
// fit_pipeline
// ---------------------------------------------------------------------------

std::unique_ptr<FittedPipeline>
fit_pipeline(const std::filesystem::path& csv,
             const PipelineConfig& config,
             TransformedDataset& out_ds) {
    // 1. Load the CSV and set aside rows with no answer to learn from.
    RawCsv raw = read_whole_csv(csv);
    const int target_idx = require_target(raw, config);
    {
        const auto usable = rows_with_target(raw, target_idx);
        out_ds.rows_dropped_missing_target =
            static_cast<int>(raw.rows.size() - usable.size());
        if (usable.empty()) {
            throw std::runtime_error("Every value in the target column '" +
                                     config.target_column + "' is blank.");
        }
        if (usable.size() != raw.rows.size()) {
            std::vector<std::vector<std::string>> kept;
            kept.reserve(usable.size());
            for (std::size_t r : usable) kept.push_back(std::move(raw.rows[r]));
            raw.rows = std::move(kept);
        }
    }
    const std::size_t n = raw.rows.size();

    // 2. Classification or regression, and the class labels.
    const TargetProfile target = profile_target(raw.rows, target_idx, config.task);
    if (target.is_classification && target.class_labels.size() < 2) {
        throw std::runtime_error("The target column '" + config.target_column +
                                 "' has only one distinct value, so there is nothing to predict.");
    }

    auto pipe = std::make_unique<FittedPipeline>();
    auto& impl = *pipe->impl;
    impl.header            = raw.header;
    impl.target_column     = config.target_column;
    impl.is_classification = target.is_classification;
    impl.class_labels      = target.class_labels;
    impl.columns.resize(raw.header.size());

    std::unordered_map<std::string, int> label_to_idx;
    for (std::size_t i = 0; i < impl.class_labels.size(); ++i) {
        label_to_idx[impl.class_labels[i]] = static_cast<int>(i);
    }
    std::vector<int> class_of_row(n, 0);
    if (impl.is_classification) {
        for (std::size_t r = 0; r < n; ++r) {
            const auto it = label_to_idx.find(util::trim(cell(raw.rows[r], target_idx)));
            class_of_row[r] = (it != label_to_idx.end()) ? it->second : 0;
        }
    }

    // 3. Split first. Everything fitted below sees training rows only, so
    //    validation and test rows stay genuinely unseen.
    SplitIndices split;
    if (impl.is_classification && config.stratify) {
        split = stratified_split(n, class_of_row,
                                 config.val_fraction, config.test_fraction,
                                 config.split_seed);
    } else {
        split = random_split(n, config.val_fraction, config.test_fraction,
                             config.split_seed);
    }
    if (split.train.empty()) {
        throw std::runtime_error("No rows left for training after splitting.");
    }

    // 4. Decide and fit each column on the training rows.
    const std::vector<ColumnPlan> plans = plan_all(raw, split.train, target_idx, config);
    out_ds.columns.clear();
    for (const auto& p : plans) out_ds.columns.push_back(describe(p));

    for (int c = 0; c < static_cast<int>(raw.header.size()); ++c) {
        const ColumnPlan& plan = plans[static_cast<std::size_t>(c)];
        FittedColumn& fc = impl.columns[static_cast<std::size_t>(c)];
        fc.spec              = plan.spec;
        fc.is_numeric        = plan.spec.is_numeric;
        fc.drop              = plan.spec.drop;
        fc.is_date           = plan.is_date;
        fc.date_day_first    = plan.date_day_first;
        fc.missing_indicator = plan.missing_indicator;
        fc.note              = plan.note;

        if (c == target_idx) {
            fc.is_target  = true;
            fc.drop       = false;
            fc.is_numeric = !target.is_classification;
            fc.spec.is_target = true;
            continue;
        }
        if (fc.drop) continue;

        if (fc.is_date) {
            std::vector<double> days;
            days.reserve(split.train.size());
            for (std::size_t r : split.train) {
                const auto dp = try_parse_date(cell(raw.rows[r], c), fc.date_day_first);
                days.push_back(dp.parsed ? static_cast<double>(dp.days)
                                         : std::numeric_limits<double>::quiet_NaN());
            }
            fc.imputer.strategy = ImputeStrategy::Mean;
            fc.imputer.fit_numeric(days, 0.0);
            std::vector<double> imputed;
            imputed.reserve(days.size());
            for (double v : days) imputed.push_back(fc.imputer.apply_numeric(v));
            fc.scaler.strategy = ScaleStrategy::Standard;
            fc.scaler.fit(imputed);
        } else if (fc.is_numeric) {
            std::vector<double> values;
            values.reserve(split.train.size());
            for (std::size_t r : split.train) {
                const auto p = try_parse_number(cell(raw.rows[r], c));
                values.push_back(p.parsed ? p.value
                                          : std::numeric_limits<double>::quiet_NaN());
            }
            fc.imputer.strategy = fc.spec.impute;
            fc.imputer.fit_numeric(values, fc.spec.impute_constant);

            // Impute before fitting the scaler so it sees the same
            // distribution it will see at prediction time.
            std::vector<double> imputed;
            imputed.reserve(values.size());
            for (double v : values) imputed.push_back(fc.imputer.apply_numeric(v));
            fc.scaler.strategy = fc.spec.scale;
            fc.scaler.fit(imputed);
        } else {
            std::vector<std::string> values;
            values.reserve(split.train.size());
            for (std::size_t r : split.train) {
                const std::string& raw_cell = cell(raw.rows[r], c);
                values.push_back(util::is_missing_token(raw_cell) ? std::string{}
                                                                  : util::trim(raw_cell));
            }
            fc.imputer.strategy = fc.spec.impute;
            fc.imputer.fit_categorical(values);

            std::vector<std::string> imputed;
            imputed.reserve(values.size());
            for (const auto& v : values) imputed.push_back(fc.imputer.apply_categorical(v));

            fc.encoder.strategy     = fc.spec.encode;
            fc.encoder.hash_buckets = fc.spec.hash_buckets;
            fc.encoder.fit(imputed, std::max(1, fc.spec.max_categories));
        }
    }

    // 5. Output feature names, in a fixed order.
    build_feature_names(impl);
    if (impl.feature_names_out.empty()) {
        throw std::runtime_error(
            "No usable input columns: every column other than the target is blank, "
            "constant, an identifier, or was left out.");
    }

    // 6. Transform a set of rows.
    auto transform = [&](const std::vector<std::size_t>& idx,
                         std::vector<std::vector<double>>& X,
                         std::vector<std::vector<double>>& Y) {
        X.clear(); Y.clear();
        X.reserve(idx.size()); Y.reserve(idx.size());
        for (std::size_t r : idx) {
            const auto& row = raw.rows[r];
            std::vector<double> feat;
            feat.reserve(impl.feature_names_out.size());
            for (int c = 0; c < static_cast<int>(raw.header.size()); ++c) {
                const FittedColumn& fc = impl.columns[static_cast<std::size_t>(c)];
                if (fc.is_target || fc.drop) continue;
                emit_from_text(fc, cell(row, c), feat);
            }
            X.push_back(std::move(feat));

            if (impl.is_classification) {
                std::vector<double> one_hot(impl.class_labels.size(), 0.0);
                one_hot[static_cast<std::size_t>(class_of_row[r])] = 1.0;
                Y.push_back(std::move(one_hot));
            } else {
                Y.push_back({ try_parse_number(cell(row, target_idx)).value });
            }
        }
    };
    transform(split.train, out_ds.X_train, out_ds.y_train);
    transform(split.val,   out_ds.X_val,   out_ds.y_val);
    transform(split.test,  out_ds.X_test,  out_ds.y_test);

    out_ds.feature_names_out  = impl.feature_names_out;
    out_ds.feature_sources    = impl.feature_sources;
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
    const bool is_object = raw_row.is_object();

    std::vector<double> out;
    out.reserve(impl.feature_names_out.size());
    for (const auto& fc : impl.columns) {
        if (fc.is_target || fc.drop) continue;
        const std::string& name = fc.spec.name;

        const nlohmann::json* v = nullptr;
        if (is_object) {
            const auto it = raw_row.find(name);
            if (it != raw_row.end()) v = &*it;
        }
        if (!v) {
            NG_LOG_WARN("apply_pipeline: missing column '{}' — imputing", name);
        }

        // Numbers given as JSON numbers skip text parsing for numeric columns.
        if (v && v->is_number() && fc.is_numeric && !fc.is_date) {
            emit_numeric(fc, v->get<double>(), out);
            continue;
        }

        std::string text;
        if (v) {
            if (v->is_string())       text = v->get<std::string>();
            else if (v->is_number()) {
                std::ostringstream oss;
                oss << v->get<double>();
                text = oss.str();
            }
            else if (v->is_boolean()) text = v->get<bool>() ? "true" : "false";
            else if (!v->is_null()) {
                NG_LOG_WARN("apply_pipeline: unsupported value type for column '{}'", name);
            }
        }
        emit_from_text(fc, text, out);
    }
    return out;
}

// ---------------------------------------------------------------------------
// save_pipeline / load_pipeline — JSON on disk.
//
// version 1: original schema.
// version 2: adds per-column is_date / date_day_first / missing_indicator /
//            max_categories / note, and top-level feature_sources. Version 1
//            files still load; the new fields take their defaults.
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
    j["feature_sources"]   = impl.feature_sources;

    j["columns"] = nlohmann::json::array();
    for (const auto& fc : impl.columns) {
        nlohmann::json c;
        c["name"]       = fc.spec.name;
        c["is_numeric"] = fc.is_numeric;
        c["is_target"]  = fc.is_target;
        c["drop"]       = fc.drop;
        c["is_date"]           = fc.is_date;
        c["date_day_first"]    = fc.date_day_first;
        c["missing_indicator"] = fc.missing_indicator;
        c["note"]              = fc.note;

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
        c["spec_max_categories"]  = fc.spec.max_categories;

        j["columns"].push_back(std::move(c));
    }

    std::ofstream f(file);
    if (!f) {
        NG_LOG_ERROR("save_pipeline: cannot open '{}' for write", file.string());
        return;
    }
    f << j.dump(2);
}

std::unique_ptr<FittedPipeline>
load_pipeline(const std::filesystem::path& file) {
    std::ifstream f(file);
    if (!f) {
        NG_LOG_ERROR("load_pipeline: cannot open '{}'", file.string());
        return nullptr;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        NG_LOG_ERROR("load_pipeline: JSON parse error: {}", e.what());
        return nullptr;
    }

    try {
        const int version = j.value("version", 0);
        if (version != 1 && version != 2) {
            NG_LOG_ERROR("load_pipeline: unsupported schema version {}", version);
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

        const auto& cols = j.at("columns");
        impl.columns.resize(cols.size());
        for (std::size_t i = 0; i < cols.size(); ++i) {
            const auto& c = cols[i];
            FittedColumn& fc = impl.columns[i];

            fc.spec.name            = c.value("name", "");
            fc.is_numeric           = c.value("is_numeric", true);
            fc.is_target            = c.value("is_target",  false);
            fc.drop                 = c.value("drop",       false);
            fc.is_date              = c.value("is_date",    false);
            fc.date_day_first       = c.value("date_day_first", true);
            fc.missing_indicator    = c.value("missing_indicator", false);
            fc.note                 = c.value("note", "");
            fc.spec.is_numeric      = fc.is_numeric;
            fc.spec.is_target       = fc.is_target;
            fc.spec.drop            = fc.drop;
            fc.spec.hash_buckets    = c.value("spec_hash_buckets", 32);
            fc.spec.impute_constant = c.value("spec_impute_constant", 0.0);
            fc.spec.max_categories  = c.value("spec_max_categories", 10000);

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

        // Recompute names from the fitted state rather than trusting the
        // file, so names, sources and feature count can never disagree.
        build_feature_names(impl);
        return pipe;
    } catch (const std::exception& e) {
        NG_LOG_ERROR("load_pipeline: malformed pipeline file: {}", e.what());
        return nullptr;
    }
}

} // namespace neurograph::preprocess
