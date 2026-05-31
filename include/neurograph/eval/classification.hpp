#pragma once

// P1-04 — Classification metrics.
//
// Everything needed to report on a supervised classifier: accuracy, per-class
// precision/recall/F1, macro+weighted F1, top-K accuracy, confusion matrix,
// plus probability-based metrics (ROC AUC, PR AUC, Brier score, reliability
// bins for calibration).
//
// We compute all of it from the same inputs so the caller never has to make
// two passes: the label vector and the per-class probability matrix.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurograph::eval {

struct ConfusionMatrix {
    int n_classes = 0;
    /// cells[true][pred] — row = truth, column = prediction.
    std::vector<std::vector<int>> cells;
};

struct PerClassMetrics {
    std::string class_label;
    double      precision = 0.0;
    double      recall    = 0.0;
    double      f1        = 0.0;
    int         support   = 0;
};

struct ReliabilityBin {
    double mean_prob     = 0.0;
    double frac_positive = 0.0;
    int    count         = 0;
};

struct ClassificationMetrics {
    double accuracy       = 0.0;
    double macro_f1       = 0.0;
    double weighted_f1    = 0.0;
    /// NaN when n_classes < 5.
    double top_5_accuracy = 0.0;
    std::vector<PerClassMetrics> per_class;
    ConfusionMatrix              confusion;

    /// Probability-based metrics. For multi-class problems we report macro
    /// averages of one-vs-rest AUCs. NaN when a metric is undefined (e.g.
    /// a class has zero positives in the eval set).
    double roc_auc_macro = 0.0;
    double pr_auc_macro  = 0.0;
    double brier_score   = 0.0;

    /// 10 equal-width bins on the confidence of the predicted class.
    std::vector<ReliabilityBin> reliability;
};

/// Evaluate a classifier.
///   - `y_true`     : integer class indices, one per sample.
///   - `probs`      : probs[i][c] = P(class = c | sample i). Rows should sum
///                    to ~1 (we don't enforce but poorly-normalized input
///                    makes Brier/calibration meaningless).
///   - `class_labels`: display names, indexed by class id. Size defines the
///                    number of classes.
///
/// Throws std::invalid_argument on size mismatch or out-of-range labels.
ClassificationMetrics evaluate_classification(
    const std::vector<int>&                 y_true,
    const std::vector<std::vector<double>>& probs,
    const std::vector<std::string>&         class_labels);

} // namespace neurograph::eval
