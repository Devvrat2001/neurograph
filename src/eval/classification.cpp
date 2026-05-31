#include "neurograph/eval/classification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace neurograph::eval {

namespace {

constexpr double kQuietNaN = std::numeric_limits<double>::quiet_NaN();

// argmax over a probability row.
int argmax_row(const std::vector<double>& row) {
    if (row.empty()) return -1;
    return static_cast<int>(
        std::distance(row.begin(), std::max_element(row.begin(), row.end())));
}

// Top-K accuracy: 1 if y_true is among the K highest-probability classes.
double top_k_accuracy(const std::vector<int>& y_true,
                      const std::vector<std::vector<double>>& probs,
                      int k) {
    if (y_true.empty()) return kQuietNaN;
    const int n_classes = static_cast<int>(probs.front().size());
    if (k >= n_classes) return 1.0;

    int correct = 0;
    std::vector<int> idx(n_classes);
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        std::iota(idx.begin(), idx.end(), 0);
        // Partial sort is plenty — only need the top-K entries.
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return probs[i][a] > probs[i][b]; });
        if (std::find(idx.begin(), idx.begin() + k, y_true[i]) != idx.begin() + k) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

// ROC AUC for a single class via rank statistic. Returns NaN if either
// class (positive / negative) is empty.
double roc_auc_one_vs_rest(const std::vector<int>& y_true,
                           const std::vector<std::vector<double>>& probs,
                           int cls) {
    const std::size_t n = y_true.size();
    std::vector<std::pair<double, int>> scored;  // (prob_cls, is_positive)
    scored.reserve(n);
    int n_pos = 0, n_neg = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int is_pos = (y_true[i] == cls) ? 1 : 0;
        scored.emplace_back(probs[i][cls], is_pos);
        (is_pos ? n_pos : n_neg)++;
    }
    if (n_pos == 0 || n_neg == 0) return kQuietNaN;

    // Sort ascending by score so smallest rank = 1 goes to lowest score.
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Assign mean ranks for ties — essential for correct AUC under tied scores
    // (otherwise a constant classifier would get AUC = 0.5 only by luck).
    double sum_ranks_pos = 0.0;
    std::size_t i = 0;
    while (i < scored.size()) {
        std::size_t j = i;
        while (j < scored.size() && scored[j].first == scored[i].first) ++j;
        // Positions [i, j) share the same score. Average rank = (i+1 + j) / 2.
        const double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            if (scored[k].second) sum_ranks_pos += avg_rank;
        }
        i = j;
    }

    // Mann-Whitney U formula: AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    const double u = sum_ranks_pos
                   - static_cast<double>(n_pos) * (n_pos + 1) / 2.0;
    return u / (static_cast<double>(n_pos) * static_cast<double>(n_neg));
}

// PR AUC via trapezoidal integration over unique thresholds.
double pr_auc_one_vs_rest(const std::vector<int>& y_true,
                          const std::vector<std::vector<double>>& probs,
                          int cls) {
    std::vector<std::pair<double, int>> scored;
    scored.reserve(y_true.size());
    int n_pos = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const int is_pos = (y_true[i] == cls) ? 1 : 0;
        scored.emplace_back(probs[i][cls], is_pos);
        if (is_pos) ++n_pos;
    }
    if (n_pos == 0) return kQuietNaN;

    // Sort descending — walk from highest-score to lowest, accumulating tp/fp.
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    double tp = 0.0, fp = 0.0;
    double prev_recall = 0.0, prev_precision = 1.0;
    double auc = 0.0;

    std::size_t i = 0;
    while (i < scored.size()) {
        std::size_t j = i;
        // Walk through all samples sharing the current score — threshold
        // decisions happen at unique score boundaries, not per-sample.
        while (j < scored.size() && scored[j].first == scored[i].first) {
            if (scored[j].second) tp += 1.0;
            else                  fp += 1.0;
            ++j;
        }
        const double recall    = tp / static_cast<double>(n_pos);
        const double precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 1.0;
        // Trapezoidal rule on the P-R curve.
        auc += (recall - prev_recall) * (precision + prev_precision) / 2.0;
        prev_recall    = recall;
        prev_precision = precision;
        i = j;
    }
    return auc;
}

} // namespace

ClassificationMetrics evaluate_classification(
    const std::vector<int>&                 y_true,
    const std::vector<std::vector<double>>& probs,
    const std::vector<std::string>&         class_labels) {

    const int n_classes = static_cast<int>(class_labels.size());
    if (n_classes < 2) {
        throw std::invalid_argument(
            "evaluate_classification: need >=2 class labels");
    }
    if (y_true.size() != probs.size()) {
        throw std::invalid_argument(
            "evaluate_classification: y_true/probs size mismatch");
    }
    for (const auto& row : probs) {
        if (static_cast<int>(row.size()) != n_classes) {
            throw std::invalid_argument(
                "evaluate_classification: probs row length != n_classes");
        }
    }
    for (int y : y_true) {
        if (y < 0 || y >= n_classes) {
            throw std::invalid_argument(
                "evaluate_classification: y_true label out of range");
        }
    }

    ClassificationMetrics m;
    m.per_class.resize(n_classes);
    m.confusion.n_classes = n_classes;
    m.confusion.cells.assign(n_classes, std::vector<int>(n_classes, 0));

    if (y_true.empty()) {
        m.accuracy = m.macro_f1 = m.weighted_f1 = kQuietNaN;
        m.top_5_accuracy = kQuietNaN;
        m.roc_auc_macro = m.pr_auc_macro = m.brier_score = kQuietNaN;
        for (int c = 0; c < n_classes; ++c) {
            m.per_class[c].class_label = class_labels[c];
        }
        return m;
    }

    // --- Pass 1: confusion matrix + Brier score + accuracy.
    int correct = 0;
    double brier_sum = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const int pred = argmax_row(probs[i]);
        m.confusion.cells[y_true[i]][pred] += 1;
        if (pred == y_true[i]) ++correct;

        // Brier: sum_c (p_c - I[c==y])^2, averaged over samples.
        for (int c = 0; c < n_classes; ++c) {
            const double target = (c == y_true[i]) ? 1.0 : 0.0;
            const double d = probs[i][c] - target;
            brier_sum += d * d;
        }
    }
    const double n_d = static_cast<double>(y_true.size());
    m.accuracy    = static_cast<double>(correct) / n_d;
    m.brier_score = brier_sum / n_d;

    // --- Per-class precision / recall / F1.
    double macro_f1_sum = 0.0;
    double weighted_f1_sum = 0.0;
    int total_support = 0;
    for (int c = 0; c < n_classes; ++c) {
        int tp = m.confusion.cells[c][c];
        int fn = 0, fp = 0, support = 0;
        for (int j = 0; j < n_classes; ++j) {
            support += m.confusion.cells[c][j];
            if (j != c) {
                fn += m.confusion.cells[c][j];
                fp += m.confusion.cells[j][c];
            }
        }
        auto& pc = m.per_class[c];
        pc.class_label = class_labels[c];
        pc.support     = support;
        pc.precision   = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        pc.recall      = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        pc.f1 = (pc.precision + pc.recall) > 0.0
              ? 2.0 * pc.precision * pc.recall / (pc.precision + pc.recall)
              : 0.0;
        macro_f1_sum    += pc.f1;
        weighted_f1_sum += pc.f1 * support;
        total_support   += support;
    }
    m.macro_f1    = macro_f1_sum / static_cast<double>(n_classes);
    m.weighted_f1 = total_support > 0
                  ? weighted_f1_sum / static_cast<double>(total_support)
                  : 0.0;

    m.top_5_accuracy = n_classes >= 5 ? top_k_accuracy(y_true, probs, 5) : kQuietNaN;

    // --- Probability-based macro metrics.
    double roc_sum = 0.0;
    double pr_sum  = 0.0;
    int roc_valid = 0, pr_valid = 0;
    for (int c = 0; c < n_classes; ++c) {
        const double roc = roc_auc_one_vs_rest(y_true, probs, c);
        const double pr  = pr_auc_one_vs_rest (y_true, probs, c);
        if (!std::isnan(roc)) { roc_sum += roc; ++roc_valid; }
        if (!std::isnan(pr))  { pr_sum  += pr;  ++pr_valid;  }
    }
    m.roc_auc_macro = roc_valid > 0 ? roc_sum / roc_valid : kQuietNaN;
    m.pr_auc_macro  = pr_valid  > 0 ? pr_sum  / pr_valid  : kQuietNaN;

    // --- 10-bin reliability diagram on predicted-class confidence.
    constexpr int kBins = 10;
    m.reliability.assign(kBins, {});
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const int    pred = argmax_row(probs[i]);
        const double conf = probs[i][pred];
        // Map [0, 1] → bin index, clamp upper bound to kBins-1 so conf==1.0
        // doesn't land in a non-existent bucket kBins.
        int bin = static_cast<int>(conf * kBins);
        if (bin >= kBins) bin = kBins - 1;
        if (bin < 0)      bin = 0;
        auto& b = m.reliability[bin];
        b.mean_prob     += conf;
        b.frac_positive += (pred == y_true[i]) ? 1.0 : 0.0;
        b.count         += 1;
    }
    for (auto& b : m.reliability) {
        if (b.count > 0) {
            b.mean_prob     /= b.count;
            b.frac_positive /= b.count;
        }
    }

    return m;
}

} // namespace neurograph::eval
