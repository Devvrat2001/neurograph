#include "neurograph/reporting/model_card.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurograph::reporting {

namespace {

// All numeric formatting funnels through these helpers so the determinism
// contract holds regardless of locale or default-precision changes.

std::string fmt_pct(double v, int decimals = 2) {
    if (std::isnan(v)) return "n/a";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f%%", decimals, v * 100.0);
    return buf;
}

std::string fmt_num(double v, int decimals = 4) {
    if (std::isnan(v)) return "n/a";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string fmt_int(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    return buf;
}

std::string fmt_seconds(double s) {
    if (!(s > 0.0)) return "—";
    char buf[48];
    if (s < 10.0)       std::snprintf(buf, sizeof(buf), "%.1f s", s);
    else if (s < 120.0) std::snprintf(buf, sizeof(buf), "%.0f s", s);
    else                std::snprintf(buf, sizeof(buf), "%d min %d s",
                                      static_cast<int>(s) / 60, static_cast<int>(s) % 60);
    return buf;
}

// Last path component, split by hand: std::filesystem's narrow-string
// conversion can throw on Windows for names outside the code page.
std::string file_name_only(const std::string& p) {
    const auto cut = p.find_last_of("/\\");
    return cut == std::string::npos ? p : p.substr(cut + 1);
}

const char* impute_name(preprocess::ImputeStrategy s) {
    using S = preprocess::ImputeStrategy;
    switch (s) {
        case S::Mean:     return "mean";
        case S::Median:   return "median";
        case S::Mode:     return "mode";
        case S::Constant: return "constant";
    }
    return "?";
}

const char* scale_name(preprocess::ScaleStrategy s) {
    using S = preprocess::ScaleStrategy;
    switch (s) {
        case S::None:     return "none";
        case S::Standard: return "standard";
        case S::MinMax:   return "minmax";
        case S::Robust:   return "robust";
    }
    return "?";
}

const char* encode_name(preprocess::EncodeStrategy s) {
    using S = preprocess::EncodeStrategy;
    switch (s) {
        case S::OneHot:           return "one-hot";
        case S::Ordinal:          return "ordinal";
        case S::HashFixedBuckets: return "hash";
    }
    return "?";
}

const char* severity_name(dataqc::Severity s) {
    switch (s) {
        case dataqc::Severity::Info:    return "Info";
        case dataqc::Severity::Warning: return "Warning";
        case dataqc::Severity::Error:   return "Error";
    }
    return "?";
}

const char* verdict_name(eval::LearningCurveReport::Verdict v) {
    using V = eval::LearningCurveReport::Verdict;
    switch (v) {
        case V::Plateau:      return "Plateau";
        case V::Improving:    return "Improving";
        case V::Inconclusive: return "Inconclusive";
    }
    return "?";
}

// Render a value or a "(not specified)" marker. Markdown auditors use
// the marker to spot fields the wizard didn't capture.
std::string or_default(const std::string& s, const char* placeholder = "(not specified)") {
    return s.empty() ? placeholder : s;
}

// Section helpers — collapse the verbose `<<` chains that show up
// repeatedly in the renderer.
void h1(std::ostringstream& out, const std::string& title) {
    out << "# " << title << "\n\n";
}

void h2(std::ostringstream& out, const std::string& title) {
    out << "## " << title << "\n\n";
}

void h3(std::ostringstream& out, const std::string& title) {
    out << "### " << title << "\n\n";
}

// --- Section renderers -----------------------------------------------------

void render_intended_use(std::ostringstream& out, const ModelCardInputs& in) {
    h2(out, "Intended Use");
    out << or_default(in.user_intended_use,
                      "(not specified — fill in via the wizard before deployment)")
        << "\n\n";

    h2(out, "Out-of-Scope Use");
    out << or_default(in.user_out_of_scope,
                      "(not specified — fill in via the wizard before deployment)")
        << "\n\n";
}

void render_training_data(std::ostringstream& out, const ModelCardInputs& in) {
    h2(out, "Training Data");
    out << "- **Source:** " << or_default(in.dq_report.summary_line, "n/a") << "\n";
    out << "- **Rows:** " << fmt_int(in.dq_report.n_rows) << "\n";
    out << "- **Columns:** " << fmt_int(in.dq_report.n_cols) << "\n";
    if (!in.dataset_path.empty()) {
        out << "- **File:** `" << file_name_only(in.dataset_path) << "`\n";
    }
    if (in.n_train > 0) {
        out << "- **Rows used:** " << fmt_int(in.n_train) << " to train, "
            << fmt_int(in.n_val) << " to compare models and decide when to stop, "
            << fmt_int(in.n_test) << " held back to measure the result\n";
    }
    if (in.rows_dropped_missing_target > 0) {
        out << "- **Rows skipped (blank target):** "
            << fmt_int(in.rows_dropped_missing_target) << "\n";
    }
    if (!std::isnan(in.dq_report.class_imbalance_ratio)) {
        out << "- **Class imbalance ratio:** "
            << fmt_num(in.dq_report.class_imbalance_ratio, 2) << "\n";
    }
    out << "- **Trainable:** " << (in.dq_report.trainable ? "yes" : "no") << "\n\n";

    if (!in.dq_report.findings.empty()) {
        h3(out, "Data Quality Findings");
        out << "| Severity | Code | Column | Message |\n";
        out << "|----------|------|--------|---------|\n";
        for (const auto& f : in.dq_report.findings) {
            out << "| " << severity_name(f.severity)
                << " | " << f.code
                << " | " << (f.column.empty() ? "—" : f.column)
                << " | " << f.message
                << " |\n";
        }
        out << "\n";
    }
}

void render_preprocessing(std::ostringstream& out, const ModelCardInputs& in) {
    h2(out, "Preprocessing");
    out << "- **Target column:** `" << or_default(in.pipe_config.target_column) << "`\n";
    out << "- **Validation fraction:** " << fmt_num(in.pipe_config.val_fraction, 3) << "\n";
    out << "- **Test fraction:** " << fmt_num(in.pipe_config.test_fraction, 3) << "\n";
    out << "- **Stratified split:** " << (in.pipe_config.stratify ? "yes" : "no") << "\n";
    out << "- **Split seed:** " << fmt_int(static_cast<long long>(in.pipe_config.split_seed))
        << "\n\n";

    if (!in.columns.empty()) {
        h3(out, "Columns");
        out << "| Column | Used as | Note |\n";
        out << "|--------|---------|------|\n";
        for (const auto& c : in.columns) {
            out << "| `" << c.name << "` | " << c.role << " | "
                << (c.note.empty() ? "—" : c.note) << " |\n";
        }
        out << "\n";
    } else if (in.pipe_config.columns.empty()) {
        out << "_Per-column transforms were inferred automatically by the preprocessing scanner._\n\n";
    } else {
        h3(out, "Per-Column Transforms");
        out << "| Column | Role | Type | Impute | Scale | Encode |\n";
        out << "|--------|------|------|--------|-------|--------|\n";
        for (const auto& c : in.pipe_config.columns) {
            const char* role = c.is_target ? "target" : (c.drop ? "dropped" : "feature");
            out << "| `" << c.name << "`"
                << " | " << role
                << " | " << (c.is_numeric ? "numeric" : "categorical")
                << " | " << impute_name(c.impute)
                << " | " << scale_name(c.scale)
                << " | " << encode_name(c.encode)
                << " |\n";
        }
        out << "\n";
    }
}

void render_model_selection(std::ostringstream& out, const ModelCardInputs& in) {
    h2(out, "Training Procedure");
    out << "Several kinds of model were trained on the same rows. Each was "
           "checked on rows it had not learned from, and the one with the "
           "lowest error there was kept.\n\n";
    if (!in.model_description.empty()) {
        out << "- **Chosen model:** " << in.model_description << "\n";
        if (!in.model_choice_reason.empty()) {
            out << "- **Why:** " << in.model_choice_reason << "\n";
        }
        out << "\n";
    }
    const bool cls = (in.task_kind == "classification");
    out << "| Model | " << (cls ? "Accuracy" : "R²")
        << " on check rows | Training time | Detail |\n";
    out << "|-------|---------------------|---------------|--------|\n";
    for (const auto& c : in.candidates) {
        out << "| " << (c.chosen ? "**" + c.name + "** (chosen)" : c.name)
            << " | " << (cls ? fmt_pct(c.val_score) : fmt_num(c.val_score))
            << " | " << fmt_seconds(c.seconds)
            << (c.stopped_by_time ? " (hit time limit)" : "")
            << " | " << (c.detail.empty() ? "—" : c.detail) << " |\n";
    }
    out << "\n";
}

void render_importance(std::ostringstream& out, const ModelCardInputs& in) {
    if (in.importance.empty()) return;
    h2(out, "What the Model Relies On");
    out << "Each column's values were shuffled between rows, one column at a "
           "time, to see how much worse the model does without it. Columns "
           "near the top matter most.\n\n";
    const bool cls = (in.task_kind == "classification");
    out << "| Column | Relative importance | " << (cls ? "Accuracy lost" : "R² lost")
        << " |\n";
    out << "|--------|---------------------|------|\n";
    constexpr std::size_t kMaxRows = 20;
    for (std::size_t i = 0; i < in.importance.size() && i < kMaxRows; ++i) {
        const auto& m = in.importance[i];
        out << "| `" << m.column << "` | " << fmt_num(m.relative, 2) << " | "
            << (cls ? fmt_pct(m.score_drop, 1) : fmt_num(m.score_drop, 3)) << " |\n";
    }
    if (in.importance.size() > kMaxRows) {
        out << "\n_" << fmt_int(static_cast<long long>(in.importance.size() - kMaxRows))
            << " more columns have less influence._\n";
    }
    out << "\n";
}

void render_training_procedure(std::ostringstream& out, const ModelCardInputs& in) {
    if (!in.candidates.empty()) {
        render_model_selection(out, in);
        return;
    }
    const auto& tc = in.train_config;
    const auto& oc = tc.optimizer_config;

    h2(out, "Training Procedure");
    out << "- **Epochs:** " << fmt_int(tc.epochs) << "\n";
    out << "- **Batch size:** " << fmt_int(static_cast<long long>(tc.batch_size)) << "\n";
    out << "- **Optimizer:** " << or_default(tc.optimizer_name) << "\n";
    out << "- **Learning rate:** " << fmt_num(oc.learning_rate, 6) << "\n";
    out << "- **LR schedule:** " << or_default(tc.schedule_name) << "\n";
    out << "- **Weight decay:** " << fmt_num(oc.weight_decay, 6) << "\n";
    out << "- **Gradient clip:** " << fmt_num(oc.gradient_clip, 4) << "\n";
    out << "- **Dropout rate:** " << fmt_num(tc.dropout_rate, 3) << "\n";
    out << "- **Batch norm:** " << (tc.batch_norm ? "yes" : "no") << "\n";
    out << "- **Early stopping patience:** " << fmt_int(tc.patience) << "\n";
    out << "- **Self-org interleave:** "
        << (tc.interleave_self_org ? "yes" : "no")
        << " (every " << fmt_int(tc.self_org_interval) << " epochs, "
        << fmt_int(tc.self_org_steps) << " steps)\n";
    out << "\n";
}

void render_classification_metrics(std::ostringstream& out,
                                    const eval::ClassificationMetrics& m) {
    h3(out, "Headline Numbers");
    out << "- **Accuracy:** " << fmt_pct(m.accuracy) << "\n";
    out << "- **Macro F1:** " << fmt_num(m.macro_f1) << "\n";
    out << "- **Weighted F1:** " << fmt_num(m.weighted_f1) << "\n";
    if (!std::isnan(m.top_5_accuracy)) {
        out << "- **Top-5 accuracy:** " << fmt_pct(m.top_5_accuracy) << "\n";
    }
    out << "- **ROC AUC (macro):** " << fmt_num(m.roc_auc_macro) << "\n";
    out << "- **PR AUC (macro):** " << fmt_num(m.pr_auc_macro) << "\n";
    out << "- **Brier score:** " << fmt_num(m.brier_score) << "\n\n";

    if (!m.per_class.empty()) {
        h3(out, "Per-Class Breakdown");
        out << "| Class | Precision | Recall | F1 | Support |\n";
        out << "|-------|-----------|--------|----|---------|\n";
        for (const auto& pc : m.per_class) {
            out << "| `" << pc.class_label << "`"
                << " | " << fmt_num(pc.precision)
                << " | " << fmt_num(pc.recall)
                << " | " << fmt_num(pc.f1)
                << " | " << fmt_int(pc.support)
                << " |\n";
        }
        out << "\n";
    }

    if (m.confusion.n_classes > 0 && !m.confusion.cells.empty()) {
        h3(out, "Confusion Matrix");
        out << "Rows = ground truth, columns = prediction.\n\n";
        // Header row.
        out << "| true \\\\ pred |";
        for (const auto& pc : m.per_class) out << " `" << pc.class_label << "` |";
        out << "\n|---|";
        for (std::size_t i = 0; i < m.per_class.size(); ++i) out << "---|";
        out << "\n";
        // Body.
        for (int i = 0; i < m.confusion.n_classes; ++i) {
            const std::string row_label =
                (i < static_cast<int>(m.per_class.size())) ? m.per_class[i].class_label
                                                           : ("class_" + std::to_string(i));
            out << "| `" << row_label << "` |";
            for (int j = 0; j < m.confusion.n_classes; ++j) {
                out << " " << fmt_int(m.confusion.cells[i][j]) << " |";
            }
            out << "\n";
        }
        out << "\n";
    }
}

void render_regression_metrics(std::ostringstream& out,
                                const eval::RegressionMetrics& m) {
    h3(out, "Headline Numbers");
    out << "- **R²:** " << fmt_num(m.r2) << "\n";
    out << "- **RMSE:** " << fmt_num(m.rmse) << "\n";
    out << "- **MAE:** " << fmt_num(m.mae) << "\n";
    out << "- **MAPE:** " << fmt_pct(m.mape) << "\n";
    out << "- **MSE:** " << fmt_num(m.mse) << "\n";
    out << "- **Explained variance:** " << fmt_num(m.explained_variance) << "\n";
    out << "- **Residual mean:** " << fmt_num(m.residual_mean) << "\n";
    out << "- **Residual stddev:** " << fmt_num(m.residual_stddev) << "\n";
    out << "- **n:** " << fmt_int(static_cast<long long>(m.n)) << "\n\n";
    if (!std::isnan(m.q50_loss)) {
        h3(out, "Quantile Losses");
        out << "- **q10 (pinball):** " << fmt_num(m.q10_loss) << "\n";
        out << "- **q50 (pinball):** " << fmt_num(m.q50_loss) << "\n";
        out << "- **q90 (pinball):** " << fmt_num(m.q90_loss) << "\n\n";
    }
}

void render_baseline(std::ostringstream& out, const eval::BaselineReport& b) {
    h2(out, "Baseline Comparison");
    if (b.baselines.empty()) {
        out << b.verdict << "\n\n";
        return;
    }
    out << "| Baseline | Primary | Secondary |\n";
    out << "|----------|---------|-----------|\n";
    for (const auto& r : b.baselines) {
        out << "| " << r.name
            << " | " << fmt_num(r.primary_metric)
            << " | " << fmt_num(r.secondary_metric)
            << " |\n";
    }
    out << "\n";
    out << "- **Model primary:** " << fmt_num(b.model_primary) << "\n";
    out << "- **Model secondary:** " << fmt_num(b.model_secondary) << "\n";
    out << "- **Lift over best baseline:** " << fmt_num(b.lift_vs_best_baseline) << "\n\n";
    out << "**Verdict:** " << b.verdict << "\n\n";
}

void render_curve(std::ostringstream& out, const eval::LearningCurveReport& c) {
    h2(out, "Learning Curve");
    if (c.points.empty()) {
        out << c.message << "\n\n";
        return;
    }
    out << "| Fraction | n_samples | Train metric | Val metric |\n";
    out << "|----------|-----------|--------------|------------|\n";
    for (const auto& p : c.points) {
        out << "| " << fmt_num(p.fraction, 2)
            << " | " << fmt_int(p.n_samples)
            << " | " << fmt_num(p.train_metric)
            << " | " << fmt_num(p.val_metric)
            << " |\n";
    }
    out << "\n";
    out << "- **Slope at 100% (per sample):** " << fmt_num(c.slope_at_100pct, 6) << "\n";
    out << "- **Projected at 2× data:** " << fmt_num(c.projected_at_2x_data) << "\n";
    out << "- **Verdict:** " << verdict_name(c.verdict) << "\n";
    out << "- **Message:** " << c.message << "\n\n";
}

void render_limitations(std::ostringstream& out, const ModelCardInputs& in) {
    h2(out, "Known Limitations");
    out << "- This model was trained on a single static dataset; performance "
           "on data drawn from a different distribution is not characterised.\n";
    if (in.dq_report.count_of(dataqc::Severity::Warning) > 0) {
        out << "- Training data raised "
            << fmt_int(in.dq_report.count_of(dataqc::Severity::Warning))
            << " quality warning(s) — review the Data Quality Findings section above.\n";
    }
    if (in.curve.verdict == eval::LearningCurveReport::Verdict::Improving) {
        out << "- The learning-curve verdict is **Improving**: collecting more "
               "training data is likely to raise the headline metric further.\n";
    }
    const bool cls = (in.task_kind == "classification");
    if (in.baseline.lift_vs_trivial <= 0.0 && !in.baseline.baselines.empty()) {
        out << "- The model does not currently outperform a trivial baseline on "
            << (cls ? "accuracy" : "R²")
            << ". Re-evaluate the columns before relying on its output.\n";
    }
    const int n_scored = (in.n_test > 0) ? in.n_test : in.n_val;
    if (in.n_train > 0 && n_scored > 0 && n_scored < 100) {
        out << "- The scores come from only " << fmt_int(n_scored)
            << " held-back rows, so they are rough: on new data they could be "
               "several points higher or lower.\n";
    }
    out << "\n";
}

void render_footer(std::ostringstream& out, const ModelCardInputs& in) {
    if (in.neurograph_version.empty() && in.generated_at_iso8601.empty()) return;
    out << "---\n\n";
    if (!in.neurograph_version.empty()) {
        out << "*NeuroGraph version:* `" << in.neurograph_version << "`  \n";
    }
    if (!in.generated_at_iso8601.empty()) {
        out << "*Generated at:* `" << in.generated_at_iso8601 << "`\n";
    }
}

} // namespace

std::string render_model_card(const ModelCardInputs& in) {
    std::ostringstream out;

    h1(out, or_default(in.model_name, "Untrained model"));
    out << "**Task:** " << or_default(in.task_kind) << "\n\n";

    render_intended_use(out, in);
    render_training_data(out, in);
    render_preprocessing(out, in);
    render_training_procedure(out, in);

    h2(out, "Evaluation");
    if (!in.evaluated_on.empty()) {
        out << "Measured on " << in.evaluated_on << ".\n\n";
    }
    if (in.task_kind == "classification" && in.cls_metrics) {
        render_classification_metrics(out, *in.cls_metrics);
    } else if (in.task_kind == "regression" && in.reg_metrics) {
        render_regression_metrics(out, *in.reg_metrics);
    } else {
        out << "_No evaluation metrics were attached to this report._\n\n";
    }

    render_importance(out, in);
    render_baseline(out, in.baseline);
    render_curve(out, in.curve);
    render_limitations(out, in);
    render_footer(out, in);

    return out.str();
}

void save_model_card(const ModelCardInputs& in,
                     const std::filesystem::path& path) {
    const std::string body = render_model_card(in);
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("save_model_card: cannot open " + path.string());
    }
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (!f.good()) {
        throw std::runtime_error("save_model_card: write failed for " + path.string());
    }
}

} // namespace neurograph::reporting
