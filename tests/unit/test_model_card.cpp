// Unit tests for neurograph::reporting::render_model_card (P1-12).
//
// We deliberately avoid a literal "golden file" — pinning a 200-line
// markdown blob makes every cosmetic tweak a test churn. Instead the
// suite checks:
//   - Required H2 sections exist for both classification and regression.
//   - Tables render with the expected columns.
//   - Determinism: same input → byte-identical output across renders.
//   - Optional fields (intended-use, version, timestamp) appear or are
//     omitted as documented.
//   - save_model_card writes the same bytes render_model_card returns.

#include "neurograph/reporting/model_card.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using neurograph::reporting::ModelCardInputs;
using neurograph::reporting::render_model_card;
using neurograph::reporting::save_model_card;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Build a fully-populated classification card so each test can mutate
// just the slice it cares about.
ModelCardInputs make_cls_inputs() {
    ModelCardInputs in;
    in.model_name = "test_classifier_v1";
    in.task_kind  = "classification";

    in.dq_report.n_rows       = 1000;
    in.dq_report.n_cols       = 8;
    in.dq_report.summary_line = "1000 rows, 8 columns, 0 errors, 1 warning";
    in.dq_report.trainable    = true;
    in.dq_report.class_imbalance_ratio = 1.4;
    in.dq_report.findings.push_back({
        neurograph::dataqc::Severity::Warning,
        "MISSING_VALUES", "age", "5.2% missing in age", "Median impute"
    });

    in.pipe_config.target_column = "survived";
    in.pipe_config.val_fraction  = 0.15;
    in.pipe_config.test_fraction = 0.15;
    in.pipe_config.stratify      = true;
    in.pipe_config.split_seed    = 42;
    in.pipe_config.columns.push_back({
        "survived", true, true, false,
        neurograph::preprocess::ImputeStrategy::Mean, 0.0,
        neurograph::preprocess::ScaleStrategy::None,
        neurograph::preprocess::EncodeStrategy::OneHot, 32,
    });
    in.pipe_config.columns.push_back({
        "age", false, true, false,
        neurograph::preprocess::ImputeStrategy::Median, 0.0,
        neurograph::preprocess::ScaleStrategy::Standard,
        neurograph::preprocess::EncodeStrategy::OneHot, 32,
    });
    in.pipe_config.columns.push_back({
        "embarked", false, false, false,
        neurograph::preprocess::ImputeStrategy::Mode, 0.0,
        neurograph::preprocess::ScaleStrategy::None,
        neurograph::preprocess::EncodeStrategy::OneHot, 32,
    });

    in.train_config.epochs = 50;
    in.train_config.batch_size = 32;
    in.train_config.optimizer_name = "adam";
    in.train_config.optimizer_config.learning_rate = 0.001;

    static neurograph::eval::ClassificationMetrics cls;
    cls.accuracy       = 0.87;
    cls.macro_f1       = 0.82;
    cls.weighted_f1    = 0.86;
    cls.top_5_accuracy = std::nan("");  // 2-class problem
    cls.roc_auc_macro  = 0.91;
    cls.pr_auc_macro   = 0.85;
    cls.brier_score    = 0.13;
    cls.per_class.push_back({"died",     0.88, 0.91, 0.89, 549});
    cls.per_class.push_back({"survived", 0.85, 0.81, 0.83, 342});
    cls.confusion.n_classes = 2;
    cls.confusion.cells = {{500, 49}, {65, 277}};
    in.cls_metrics = &cls;

    in.baseline.baselines.push_back({"majority",     0.62, 0.38});
    in.baseline.baselines.push_back({"random",       0.50, 0.50});
    in.baseline.baselines.push_back({"linear_probe", 0.79, 0.78});
    in.baseline.model_primary         = 0.87;
    in.baseline.model_secondary       = 0.82;
    in.baseline.lift_vs_best_baseline = 0.08;
    in.baseline.verdict =
        "Good result — model beats baselines by a meaningful margin.";

    in.curve.points.push_back({0.25, 250, 0.78, 0.74});
    in.curve.points.push_back({0.50, 500, 0.83, 0.80});
    in.curve.points.push_back({0.75, 750, 0.86, 0.84});
    in.curve.points.push_back({1.00, 1000, 0.88, 0.87});
    in.curve.slope_at_100pct      = 1.2e-4;
    in.curve.projected_at_2x_data = 0.91;
    in.curve.verdict = neurograph::eval::LearningCurveReport::Verdict::Improving;
    in.curve.message =
        "Collecting more data should meaningfully improve accuracy "
        "(projected +2% or more at 2× data).";

    in.user_intended_use = "Predict passenger survival on historical voyages.";
    in.user_out_of_scope = "Live insurance underwriting.";
    in.neurograph_version    = "0.1.0";
    in.generated_at_iso8601  = "2026-04-18T12:00:00Z";
    in.dataset_path          = "data/titanic.csv";
    return in;
}

ModelCardInputs make_reg_inputs() {
    ModelCardInputs in;
    in.model_name  = "house_price_predictor";
    in.task_kind   = "regression";
    in.dq_report.n_rows       = 5000;
    in.dq_report.n_cols       = 12;
    in.dq_report.summary_line = "5000 rows, 12 columns, 0 errors, 0 warnings";
    in.pipe_config.target_column = "price";
    in.pipe_config.val_fraction  = 0.2;

    static neurograph::eval::RegressionMetrics reg;
    reg.mse                = 12500.5;
    reg.rmse               = 111.8;
    reg.mae                = 78.2;
    reg.r2                 = 0.91;
    reg.explained_variance = 0.92;
    reg.mape               = 0.05;
    reg.q10_loss           = std::nan("");
    reg.q50_loss           = std::nan("");
    reg.q90_loss           = std::nan("");
    reg.residual_mean      = 0.3;
    reg.residual_stddev    = 109.7;
    reg.n                  = 750;
    in.reg_metrics = &reg;

    in.baseline.baselines.push_back({"mean",         0.0, 200.0});
    in.baseline.baselines.push_back({"random",       -0.5, 250.0});
    in.baseline.baselines.push_back({"linear_probe", 0.7, 130.0});
    in.baseline.model_primary         = 0.91;
    in.baseline.model_secondary       = 111.8;
    in.baseline.lift_vs_best_baseline = 0.21;
    in.baseline.verdict =
        "Strong result — model is substantially better than simple baselines.";

    return in;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Determinism
// ---------------------------------------------------------------------------

TEST(ModelCard, DeterministicForSameInputs) {
    auto in = make_cls_inputs();
    const std::string a = render_model_card(in);
    const std::string b = render_model_card(in);
    EXPECT_EQ(a, b);
}

// ---------------------------------------------------------------------------
// 2. Required sections — classification
// ---------------------------------------------------------------------------

TEST(ModelCard, ClassificationContainsAllRequiredSections) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "# test_classifier_v1"));
    EXPECT_TRUE(contains(md, "**Task:** classification"));
    EXPECT_TRUE(contains(md, "## Intended Use"));
    EXPECT_TRUE(contains(md, "## Out-of-Scope Use"));
    EXPECT_TRUE(contains(md, "## Training Data"));
    EXPECT_TRUE(contains(md, "## Preprocessing"));
    EXPECT_TRUE(contains(md, "## Training Procedure"));
    EXPECT_TRUE(contains(md, "## Evaluation"));
    EXPECT_TRUE(contains(md, "## Baseline Comparison"));
    EXPECT_TRUE(contains(md, "## Learning Curve"));
    EXPECT_TRUE(contains(md, "## Known Limitations"));
}

// ---------------------------------------------------------------------------
// 3. Confusion matrix renders as a markdown table
// ---------------------------------------------------------------------------

TEST(ModelCard, ConfusionMatrixIsMarkdownTable) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "### Confusion Matrix"));
    EXPECT_TRUE(contains(md, "true \\\\ pred"));
    EXPECT_TRUE(contains(md, "`died`"));
    EXPECT_TRUE(contains(md, "`survived`"));
    EXPECT_TRUE(contains(md, "500"));   // diag entry from fixture
    EXPECT_TRUE(contains(md, "277"));   // diag entry from fixture
}

// ---------------------------------------------------------------------------
// 4. Per-class table renders
// ---------------------------------------------------------------------------

TEST(ModelCard, PerClassBreakdownTableRenders) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "### Per-Class Breakdown"));
    EXPECT_TRUE(contains(md, "| Class | Precision | Recall | F1 | Support |"));
}

// ---------------------------------------------------------------------------
// 5. Headline metrics formatted with units
// ---------------------------------------------------------------------------

TEST(ModelCard, ClassificationHeadlineMetricsAreFormatted) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "**Accuracy:** 87.00%"));
    EXPECT_TRUE(contains(md, "**Macro F1:** 0.8200"));
    EXPECT_TRUE(contains(md, "**ROC AUC (macro):** 0.9100"));
}

// ---------------------------------------------------------------------------
// 6. Regression path
// ---------------------------------------------------------------------------

TEST(ModelCard, RegressionRendersHeadlineNumbers) {
    const auto md = render_model_card(make_reg_inputs());
    EXPECT_TRUE(contains(md, "**Task:** regression"));
    EXPECT_TRUE(contains(md, "**R²:** 0.9100"));
    EXPECT_TRUE(contains(md, "**RMSE:** 111.8000"));
    EXPECT_TRUE(contains(md, "**MAE:** 78.2000"));
    EXPECT_FALSE(contains(md, "Confusion Matrix"));   // regression-only
    EXPECT_FALSE(contains(md, "Quantile Losses"));    // q-losses are NaN
}

// ---------------------------------------------------------------------------
// 7. Baseline section reflects all three rows
// ---------------------------------------------------------------------------

TEST(ModelCard, BaselineTableListsAllBaselines) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "| majority |"));
    EXPECT_TRUE(contains(md, "| random |"));
    EXPECT_TRUE(contains(md, "| linear_probe |"));
    EXPECT_TRUE(contains(md,
        "**Verdict:** Good result — model beats baselines by a meaningful margin."));
}

// ---------------------------------------------------------------------------
// 8. Learning-curve section reflects all four points
// ---------------------------------------------------------------------------

TEST(ModelCard, LearningCurveTableListsAllPoints) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "| 0.25 | 250 |"));
    EXPECT_TRUE(contains(md, "| 0.50 | 500 |"));
    EXPECT_TRUE(contains(md, "| 0.75 | 750 |"));
    EXPECT_TRUE(contains(md, "| 1.00 | 1000 |"));
    EXPECT_TRUE(contains(md, "**Verdict:** Improving"));
}

// ---------------------------------------------------------------------------
// 9. Optional fields — intended use placeholder
// ---------------------------------------------------------------------------

TEST(ModelCard, IntendedUsePlaceholderShownWhenEmpty) {
    auto in = make_cls_inputs();
    in.user_intended_use.clear();
    in.user_out_of_scope.clear();
    const auto md = render_model_card(in);
    EXPECT_TRUE(contains(md, "(not specified — fill in via the wizard"));
}

// ---------------------------------------------------------------------------
// 10. Optional fields — footer omitted when version + timestamp empty
// ---------------------------------------------------------------------------

TEST(ModelCard, FooterOmittedWhenNoMetadata) {
    auto in = make_cls_inputs();
    in.neurograph_version.clear();
    in.generated_at_iso8601.clear();
    const auto md = render_model_card(in);
    EXPECT_FALSE(contains(md, "NeuroGraph version"));
    EXPECT_FALSE(contains(md, "Generated at"));
}

TEST(ModelCard, FooterRenderedWhenMetadataPresent) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "*NeuroGraph version:* `0.1.0`"));
    EXPECT_TRUE(contains(md, "*Generated at:* `2026-04-18T12:00:00Z`"));
}

// ---------------------------------------------------------------------------
// 11. Limitations appear conditionally
// ---------------------------------------------------------------------------

TEST(ModelCard, LimitationsMentionImprovingCurve) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "verdict is **Improving**"));
}

TEST(ModelCard, LimitationsMentionBaselineLossWhenNotLearning) {
    auto in = make_cls_inputs();
    in.baseline.lift_vs_best_baseline = -0.05;
    const auto md = render_model_card(in);
    EXPECT_TRUE(contains(md, "does not currently outperform a trivial baseline"));
}

// ---------------------------------------------------------------------------
// 12. Data quality findings table
// ---------------------------------------------------------------------------

TEST(ModelCard, DataQualityFindingsTableIncludesEachFinding) {
    const auto md = render_model_card(make_cls_inputs());
    EXPECT_TRUE(contains(md, "### Data Quality Findings"));
    EXPECT_TRUE(contains(md, "MISSING_VALUES"));
    EXPECT_TRUE(contains(md, "5.2% missing in age"));
}

// ---------------------------------------------------------------------------
// 13. save_model_card writes byte-identical content
// ---------------------------------------------------------------------------

TEST(ModelCard, SaveWritesIdenticalBytes) {
    auto in = make_cls_inputs();
    const std::string body = render_model_card(in);

    const auto tmp = std::filesystem::temp_directory_path() /
                     "neurograph_card_test.md";
    save_model_card(in, tmp);

    std::ifstream f(tmp, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    std::ostringstream ss;
    ss << f.rdbuf();
    EXPECT_EQ(body, ss.str());
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
}

// ---------------------------------------------------------------------------
// 14. Missing metrics fallback
// ---------------------------------------------------------------------------

TEST(ModelCard, MissingMetricsRendersFallbackText) {
    auto in = make_cls_inputs();
    in.cls_metrics = nullptr;
    const auto md = render_model_card(in);
    EXPECT_TRUE(contains(md, "_No evaluation metrics were attached"));
}
