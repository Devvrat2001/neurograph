#include "neurograph/preprocess/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neurograph::preprocess {

namespace {

// Standard quantile estimator (R-7 / Excel-style). Works on a copy because
// we sort in place. For the sizes we care about here (a single column's
// non-missing numeric values), O(n log n) is fine.
double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double h = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const auto hi = static_cast<std::size_t>(std::ceil(h));
    if (lo == hi) return v[lo];
    return v[lo] + (h - lo) * (v[hi] - v[lo]);
}

// Floor the scale divisor so constant columns don't divide by zero. We
// still produce a valid (but trivial) transform when a column has no
// variance — the scaled value is just (x - a).
constexpr double kScaleFloor = 1e-12;

} // namespace

// ---------------------------------------------------------------------------
// Imputer
// ---------------------------------------------------------------------------

void Imputer::fit_numeric(const std::vector<double>& values,
                          double constant_value) {
    switch (strategy) {
    case ImputeStrategy::Constant:
        fill_value_ = constant_value;
        return;

    case ImputeStrategy::Mean: {
        double sum = 0.0;
        int    n   = 0;
        for (double v : values) {
            if (std::isnan(v)) continue;
            sum += v;
            ++n;
        }
        fill_value_ = (n > 0) ? (sum / n) : 0.0;
        return;
    }

    case ImputeStrategy::Median: {
        std::vector<double> clean;
        clean.reserve(values.size());
        for (double v : values) {
            if (!std::isnan(v)) clean.push_back(v);
        }
        fill_value_ = clean.empty() ? 0.0 : quantile(std::move(clean), 0.5);
        return;
    }

    case ImputeStrategy::Mode: {
        // Mode on numeric: most-frequent value. Falls back to mean when
        // every value is unique.
        std::unordered_map<double, int> freq;
        for (double v : values) {
            if (std::isnan(v)) continue;
            ++freq[v];
        }
        int    best_count = 0;
        double best_value = 0.0;
        for (const auto& kv : freq) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_value = kv.first;
            }
        }
        if (best_count <= 1 && !freq.empty()) {
            // Unique values — fall back to mean.
            double sum = 0.0;
            for (const auto& kv : freq) sum += kv.first;
            best_value = sum / freq.size();
        }
        fill_value_ = best_value;
        return;
    }
    }
}

double Imputer::apply_numeric(double raw) const {
    return std::isnan(raw) ? fill_value_ : raw;
}

void Imputer::fit_categorical(const std::vector<std::string>& values) {
    // For categoricals we only support Mode and Constant semantically.
    // Mean/Median fall back to Mode because they don't make sense for
    // strings.
    if (strategy == ImputeStrategy::Constant) {
        // fill_token_ must be set via set_categorical_fill() — leave as-is.
        return;
    }
    std::unordered_map<std::string, int> freq;
    for (const auto& v : values) {
        if (v.empty()) continue;
        ++freq[v];
    }
    int         best_count = 0;
    std::string best_token;
    for (const auto& kv : freq) {
        if (kv.second > best_count) {
            best_count = kv.second;
            best_token = kv.first;
        }
    }
    fill_token_ = std::move(best_token);
}

std::string Imputer::apply_categorical(const std::string& raw) const {
    return raw.empty() ? fill_token_ : raw;
}

// ---------------------------------------------------------------------------
// Scaler
// ---------------------------------------------------------------------------

void Scaler::fit(const std::vector<double>& values) {
    switch (strategy) {
    case ScaleStrategy::None:
        a_ = 0.0;
        b_ = 1.0;
        return;

    case ScaleStrategy::Standard: {
        // Welford would be fine here, but the vector is already materialized
        // and we only fit once per column, so two passes for clarity.
        double sum = 0.0;
        int    n   = 0;
        for (double v : values) {
            if (std::isnan(v)) continue;
            sum += v;
            ++n;
        }
        const double mean = (n > 0) ? sum / n : 0.0;
        double ssq = 0.0;
        for (double v : values) {
            if (std::isnan(v)) continue;
            const double d = v - mean;
            ssq += d * d;
        }
        const double var = (n > 1) ? ssq / (n - 1) : 0.0;
        a_ = mean;
        b_ = std::max(std::sqrt(var), kScaleFloor);
        return;
    }

    case ScaleStrategy::MinMax: {
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : values) {
            if (std::isnan(v)) continue;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        if (!std::isfinite(mn) || !std::isfinite(mx)) { mn = 0.0; mx = 1.0; }
        a_ = mn;
        b_ = std::max(mx - mn, kScaleFloor);
        return;
    }

    case ScaleStrategy::Robust: {
        std::vector<double> clean;
        clean.reserve(values.size());
        for (double v : values) {
            if (!std::isnan(v)) clean.push_back(v);
        }
        if (clean.empty()) { a_ = 0.0; b_ = 1.0; return; }
        const double q25 = quantile(clean, 0.25);
        const double q50 = quantile(clean, 0.50);
        const double q75 = quantile(clean, 0.75);
        a_ = q50;
        b_ = std::max(q75 - q25, kScaleFloor);
        return;
    }
    }
}

double Scaler::apply(double x) const {
    if (std::isnan(x)) return x;  // caller should impute first; pass-through for safety
    return (x - a_) / b_;
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

void Encoder::fit(const std::vector<std::string>& values, int max_categories) {
    categories_.clear();
    cat_to_idx_.clear();

    if (strategy == EncodeStrategy::HashFixedBuckets) {
        return;  // no vocabulary to learn — hash handles every string
    }

    // Count frequency; we'll keep the top `max_categories` most common.
    std::unordered_map<std::string, int> freq;
    for (const auto& v : values) {
        if (v.empty()) continue;
        ++freq[v];
    }

    std::vector<std::pair<std::string, int>> pairs(freq.begin(), freq.end());
    // Sort by (descending count, ascending label) so the cut is deterministic
    // when two labels have the same count.
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first  < b.first;
              });
    if (static_cast<int>(pairs.size()) > max_categories) {
        pairs.resize(max_categories);
    }

    categories_.reserve(pairs.size());
    for (auto& p : pairs) categories_.push_back(std::move(p.first));
    // Re-sort alphabetically so on-disk and output-column order is stable
    // across fits (the frequency sort above was only for the top-K cut).
    std::sort(categories_.begin(), categories_.end());

    for (int i = 0; i < static_cast<int>(categories_.size()); ++i) {
        cat_to_idx_[categories_[i]] = i;
    }
}

std::vector<double> Encoder::apply(const std::string& raw) const {
    switch (strategy) {
    case EncodeStrategy::OneHot: {
        // N known + 1 "unknown" slot.
        std::vector<double> out(categories_.size() + 1, 0.0);
        auto it = cat_to_idx_.find(raw);
        if (it != cat_to_idx_.end()) {
            out[it->second] = 1.0;
        } else {
            out.back() = 1.0;
        }
        return out;
    }
    case EncodeStrategy::Ordinal: {
        auto it = cat_to_idx_.find(raw);
        if (it != cat_to_idx_.end()) {
            return { static_cast<double>(it->second) };
        }
        return { -1.0 };  // reserved "unknown" index
    }
    case EncodeStrategy::HashFixedBuckets: {
        const int n = std::max(1, hash_buckets);
        std::vector<double> out(static_cast<std::size_t>(n), 0.0);
        const std::size_t h = std::hash<std::string>{}(raw);
        out[h % static_cast<std::size_t>(n)] = 1.0;
        return out;
    }
    }
    return {};  // unreachable
}

std::size_t Encoder::output_dim() const {
    switch (strategy) {
    case EncodeStrategy::OneHot:           return categories_.size() + 1;
    case EncodeStrategy::Ordinal:          return 1;
    case EncodeStrategy::HashFixedBuckets: return static_cast<std::size_t>(std::max(1, hash_buckets));
    }
    return 0;
}

void Encoder::set_categories(std::vector<std::string> cats) {
    categories_ = std::move(cats);
    cat_to_idx_.clear();
    for (int i = 0; i < static_cast<int>(categories_.size()); ++i) {
        cat_to_idx_[categories_[i]] = i;
    }
}

} // namespace neurograph::preprocess
