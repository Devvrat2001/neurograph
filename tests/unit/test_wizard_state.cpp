// P1-09 — WizardState + csv_preview unit tests.
//
// The ImGui-free half of the wizard is exercised end-to-end here
// (that's the "refactor wizard state into a testable WizardState
// struct and unit-test the state transitions" bullet in the spec's
// test plan). The rendering layer is covered by the manual smoke
// checklist and by build-time gating (it won't compile if these
// types drift).

#include "neurograph/workbench/csv_preview.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace neurograph::workbench;

namespace {

// Build a CsvMetadata fixture without touching the filesystem —
// tests that don't care about file I/O should use this so they
// don't slow down on temp-file churn.
CsvMetadata make_csv_meta(std::vector<std::string> cols = {"a","b","c","label"},
                          std::size_t              rows = 100) {
    CsvMetadata m;
    m.path = "synthetic.csv";
    m.columns = std::move(cols);
    m.row_count = rows;
    // Two preview rows is enough for tests; some tests assert the
    // preview is propagated through set_csv unchanged.
    m.preview_rows = {
        {"1","2","3","A"},
        {"4","5","6","B"},
    };
    return m;
}

neurograph::dataqc::DataQualityReport make_report(bool with_error = false,
                                                  bool with_warning = false) {
    neurograph::dataqc::DataQualityReport r;
    r.n_rows = 100;
    r.n_cols = 4;
    r.trainable = !with_error;
    r.summary_line = "ok";
    if (with_warning) {
        r.findings.push_back({
            neurograph::dataqc::Severity::Warning,
            "MISSING_VALUES", "a",
            "12% of values missing", "impute with mean"
        });
    }
    if (with_error) {
        r.findings.push_back({
            neurograph::dataqc::Severity::Error,
            "TOO_FEW_ROWS", "",
            "only 3 rows", "add more data"
        });
    }
    return r;
}

class WizardStateTest : public ::testing::Test {
protected:
    WizardState state;
};

} // namespace

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

TEST_F(WizardStateTest, StartsOnLoadStepWithNothingLoaded) {
    EXPECT_EQ(state.current_step, WizardStep::Load);
    EXPECT_FALSE(state.csv.has_value());
    EXPECT_TRUE(state.target_column.empty());
    EXPECT_EQ(state.status.load(), TrainingStatus::Idle);
    EXPECT_FALSE(state.cancel_requested.load());
    EXPECT_FALSE(state.result.has_value());
}

TEST_F(WizardStateTest, CannotAdvanceFromLoadWithoutCsv) {
    EXPECT_FALSE(can_advance(state).empty());
    EXPECT_FALSE(advance(state));
    EXPECT_EQ(state.current_step, WizardStep::Load);
}

TEST_F(WizardStateTest, CannotGoBackFromLoad) {
    EXPECT_FALSE(back(state));
    EXPECT_EQ(state.current_step, WizardStep::Load);
}

// ---------------------------------------------------------------------------
// Step 1 — Load
// ---------------------------------------------------------------------------

TEST_F(WizardStateTest, SetCsvAutoPicksLastColumnAsTarget) {
    set_csv(state, make_csv_meta());
    ASSERT_TRUE(state.csv.has_value());
    EXPECT_EQ(state.target_column, "label");    // last column
}

TEST_F(WizardStateTest, DefaultTargetColumnIsLast) {
    auto meta = make_csv_meta({"a","b","c","target_y"});
    EXPECT_EQ(default_target_column(meta), "target_y");
}

TEST_F(WizardStateTest, DefaultTargetColumnIsEmptyForNoColumns) {
    CsvMetadata m;
    EXPECT_EQ(default_target_column(m), "");
}

TEST_F(WizardStateTest, SetTargetColumnRejectsUnknownName) {
    set_csv(state, make_csv_meta());
    EXPECT_FALSE(set_target_column(state, "not_a_column"));
    EXPECT_EQ(state.target_column, "label");    // unchanged
}

TEST_F(WizardStateTest, SetTargetColumnAcceptsKnownName) {
    set_csv(state, make_csv_meta());
    EXPECT_TRUE(set_target_column(state, "a"));
    EXPECT_EQ(state.target_column, "a");
}

TEST_F(WizardStateTest, ChangingTargetInvalidatesDqReport) {
    set_csv(state, make_csv_meta());
    set_dq_report(state, make_report(/*error=*/false, /*warning=*/true));
    ASSERT_TRUE(state.dq_report.has_value());

    EXPECT_TRUE(set_target_column(state, "a"));
    // The DQ scan's class-imbalance / leakage numbers depend on the
    // target; invalidating is safer than showing stale output.
    EXPECT_FALSE(state.dq_report.has_value());
}

TEST_F(WizardStateTest, SettingNewCsvClearsDownstreamState) {
    set_csv(state, make_csv_meta());
    set_dq_report(state, make_report(false, true));
    toggle_fix(state, "MISSING_VALUES");

    set_csv(state, make_csv_meta({"x","y","z","newtarget"}));
    EXPECT_FALSE(state.dq_report.has_value());
    EXPECT_TRUE(state.auto_fixes_applied.empty());
    EXPECT_EQ(state.target_column, "newtarget");
}

TEST_F(WizardStateTest, CanAdvanceOnceCsvAndTargetAreSet) {
    set_csv(state, make_csv_meta());
    EXPECT_TRUE(can_advance(state).empty());
}

TEST_F(WizardStateTest, AdvanceMovesLoadToQuality) {
    set_csv(state, make_csv_meta());
    EXPECT_TRUE(advance(state));
    EXPECT_EQ(state.current_step, WizardStep::Quality);
}

// ---------------------------------------------------------------------------
// Step 2 — Data Quality
// ---------------------------------------------------------------------------

TEST_F(WizardStateTest, CannotAdvanceFromQualityWithoutReport) {
    set_csv(state, make_csv_meta());
    advance(state);   // now on Quality
    EXPECT_FALSE(can_advance(state).empty());
    EXPECT_FALSE(advance(state));
    EXPECT_EQ(state.current_step, WizardStep::Quality);
}

TEST_F(WizardStateTest, CannotAdvanceFromQualityWithBlockingError) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(/*error=*/true, /*warning=*/false));

    EXPECT_TRUE(has_blocking_error(state));
    EXPECT_FALSE(can_advance(state).empty());
    EXPECT_FALSE(advance(state));
}

TEST_F(WizardStateTest, AdvanceFromQualityWhenWarningsOnly) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(/*error=*/false, /*warning=*/true));

    EXPECT_FALSE(has_blocking_error(state));
    EXPECT_TRUE(advance(state));
    EXPECT_EQ(state.current_step, WizardStep::Train);
}

TEST_F(WizardStateTest, ToggleFixAddsThenRemovesCode) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(false, true));

    EXPECT_TRUE(toggle_fix(state, "MISSING_VALUES"));
    EXPECT_EQ(state.auto_fixes_applied.count("MISSING_VALUES"), 1u);

    EXPECT_FALSE(toggle_fix(state, "MISSING_VALUES"));
    EXPECT_EQ(state.auto_fixes_applied.count("MISSING_VALUES"), 0u);
}

TEST_F(WizardStateTest, ToggleFixIgnoresUnfixableCode) {
    // The scanner emits codes (like TOO_FEW_ROWS) that can't be
    // auto-fixed. Toggling those must be a no-op — the wizard UI
    // won't render the checkbox, but test the guard anyway.
    set_csv(state, make_csv_meta());
    set_dq_report(state, make_report(true, false));
    EXPECT_FALSE(toggle_fix(state, "TOO_FEW_ROWS"));
    EXPECT_TRUE(state.auto_fixes_applied.empty());
}

TEST_F(WizardStateTest, FixIsApplicableMatchesKnownCodes) {
    neurograph::dataqc::Finding mv{};
    mv.code = "MISSING_VALUES";
    EXPECT_TRUE(fix_is_applicable(mv));

    neurograph::dataqc::Finding tfr{};
    tfr.code = "TOO_FEW_ROWS";
    EXPECT_FALSE(fix_is_applicable(tfr));
}

TEST_F(WizardStateTest, SettingDqReportClearsPriorFixes) {
    set_csv(state, make_csv_meta());
    set_dq_report(state, make_report(false, true));
    toggle_fix(state, "MISSING_VALUES");
    ASSERT_FALSE(state.auto_fixes_applied.empty());

    // Re-scanning produces a new report — fixes from the old one
    // are stale and shouldn't carry over.
    set_dq_report(state, make_report(false, true));
    EXPECT_TRUE(state.auto_fixes_applied.empty());
}

// ---------------------------------------------------------------------------
// Step 3 — Train
// ---------------------------------------------------------------------------

TEST_F(WizardStateTest, TimeBudgetSecondsMap) {
    EXPECT_EQ(time_budget_seconds(TimeBudget::Quick),     30);
    EXPECT_EQ(time_budget_seconds(TimeBudget::Balanced), 300);
    EXPECT_EQ(time_budget_seconds(TimeBudget::Best),    1800);
}

TEST_F(WizardStateTest, BuildOptionsReflectsPicks) {
    state.time_budget = TimeBudget::Quick;
    const auto opts = build_options(state, "/tmp/bundle");
    EXPECT_EQ(opts.time_budget_seconds, 30);
    EXPECT_FALSE(opts.do_learning_curve);
    EXPECT_TRUE(opts.do_export_bundle);
    EXPECT_EQ(opts.bundle_out_dir, fs::path("/tmp/bundle"));
}

TEST_F(WizardStateTest, LearningCurveOnlyOnBestQuality) {
    state.time_budget = TimeBudget::Balanced;
    EXPECT_FALSE(build_options(state, "/tmp").do_learning_curve);
    state.time_budget = TimeBudget::Best;
    EXPECT_TRUE(build_options(state, "/tmp").do_learning_curve);
}

TEST_F(WizardStateTest, OnProgressStoresMostRecent) {
    neurograph::orchestration::Progress p;
    p.stage = neurograph::orchestration::Stage::Training;
    p.stage_fraction = 0.3;
    p.overall_fraction = 0.5;
    p.message = "training epoch 3/10";
    on_progress(state, p);

    std::lock_guard<std::mutex> lk(state.progress_mutex);
    EXPECT_EQ(state.last_progress.stage, neurograph::orchestration::Stage::Training);
    EXPECT_DOUBLE_EQ(state.last_progress.overall_fraction, 0.5);
    EXPECT_EQ(state.last_progress.message, "training epoch 3/10");
}

TEST_F(WizardStateTest, OnTrainingFinishedOkFlipsStatusAndMovesToResults) {
    set_csv(state, make_csv_meta());
    advance(state); advance(state);   // step = Train
    state.status.store(TrainingStatus::Running);
    neurograph::orchestration::AutoResult r;
    r.summary_one_liner = "model trained";

    on_training_finished(state, neurograph::util::Result<neurograph::orchestration::AutoResult>(std::move(r)));

    EXPECT_EQ(state.status.load(), TrainingStatus::Succeeded);
    EXPECT_EQ(state.current_step, WizardStep::Results);
    ASSERT_TRUE(state.result.has_value());
    EXPECT_EQ(state.result->summary_one_liner, "model trained");
}

TEST_F(WizardStateTest, OnTrainingFinishedErrorFlipsStatusToFailed) {
    state.status.store(TrainingStatus::Running);
    on_training_finished(state,
        neurograph::util::Result<neurograph::orchestration::AutoResult>(neurograph::util::Error("something broke")));

    EXPECT_EQ(state.status.load(), TrainingStatus::Failed);
    EXPECT_EQ(state.error_message, "something broke");
    EXPECT_FALSE(state.result.has_value());
}

TEST_F(WizardStateTest, OnTrainingFinishedAfterCancelReportsCancelled) {
    state.status.store(TrainingStatus::Running);
    state.cancel_requested.store(true);
    on_training_finished(state,
        neurograph::util::Result<neurograph::orchestration::AutoResult>(neurograph::util::Error("Cancelled by user")));

    EXPECT_EQ(state.status.load(), TrainingStatus::Cancelled);
    // Cancellation flag self-clears so a subsequent Train click
    // doesn't come in pre-cancelled.
    EXPECT_FALSE(state.cancel_requested.load());
}

TEST_F(WizardStateTest, RequestCancelSetsFlag) {
    EXPECT_FALSE(state.cancel_requested.load());
    request_cancel(state);
    EXPECT_TRUE(state.cancel_requested.load());
}

TEST_F(WizardStateTest, CannotAdvanceFromTrainUntilSuccess) {
    set_csv(state, make_csv_meta());
    advance(state);                                    // Quality
    set_dq_report(state, make_report(false, false));   // no blocking error
    advance(state);                                    // Train
    ASSERT_EQ(state.current_step, WizardStep::Train);

    state.status.store(TrainingStatus::Idle);
    EXPECT_FALSE(can_advance(state).empty());
    state.status.store(TrainingStatus::Running);
    EXPECT_FALSE(can_advance(state).empty());
    state.status.store(TrainingStatus::Failed);
    EXPECT_FALSE(can_advance(state).empty());
    state.status.store(TrainingStatus::Cancelled);
    EXPECT_FALSE(can_advance(state).empty());
    state.status.store(TrainingStatus::Succeeded);
    EXPECT_TRUE(can_advance(state).empty());
}

// ---------------------------------------------------------------------------
// Step 4 — Results + reset flows
// ---------------------------------------------------------------------------

TEST_F(WizardStateTest, BackFromResultsReturnsToTrainAndClearsResult) {
    set_csv(state, make_csv_meta());
    advance(state); advance(state);   // Train
    state.status.store(TrainingStatus::Running);
    neurograph::orchestration::AutoResult r;
    r.summary_one_liner = "ok";
    on_training_finished(state,
        neurograph::util::Result<neurograph::orchestration::AutoResult>(std::move(r)));
    ASSERT_EQ(state.current_step, WizardStep::Results);

    EXPECT_TRUE(back(state));
    EXPECT_EQ(state.current_step, WizardStep::Train);
    EXPECT_FALSE(state.result.has_value());
    EXPECT_EQ(state.status.load(), TrainingStatus::Idle);
}

TEST_F(WizardStateTest, BackFromTrainBlockedWhileRunning) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(false, false));
    advance(state);
    ASSERT_EQ(state.current_step, WizardStep::Train);

    state.status.store(TrainingStatus::Running);
    EXPECT_FALSE(back(state));
    EXPECT_EQ(state.current_step, WizardStep::Train);
}

TEST_F(WizardStateTest, BackFromTrainWhenIdleGoesToQuality) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(false, false));
    advance(state);
    ASSERT_EQ(state.current_step, WizardStep::Train);

    EXPECT_TRUE(back(state));
    EXPECT_EQ(state.current_step, WizardStep::Quality);
}

TEST_F(WizardStateTest, ResetToLoadClearsEverything) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(false, true));
    toggle_fix(state, "MISSING_VALUES");

    reset_to_load(state);

    EXPECT_EQ(state.current_step, WizardStep::Load);
    EXPECT_FALSE(state.csv.has_value());
    EXPECT_FALSE(state.dq_report.has_value());
    EXPECT_TRUE(state.auto_fixes_applied.empty());
    EXPECT_TRUE(state.target_column.empty());
}

TEST_F(WizardStateTest, ResetToTrainKeepsCsvButClearsResult) {
    set_csv(state, make_csv_meta());
    advance(state);
    set_dq_report(state, make_report(false, true));
    toggle_fix(state, "MISSING_VALUES");
    advance(state);   // Train
    neurograph::orchestration::AutoResult r;
    r.summary_one_liner = "done";
    on_training_finished(state,
        neurograph::util::Result<neurograph::orchestration::AutoResult>(std::move(r)));

    reset_to_train(state);

    EXPECT_EQ(state.current_step, WizardStep::Train);
    EXPECT_TRUE(state.csv.has_value());
    EXPECT_TRUE(state.dq_report.has_value());
    EXPECT_EQ(state.auto_fixes_applied.count("MISSING_VALUES"), 1u);
    EXPECT_FALSE(state.result.has_value());
    EXPECT_EQ(state.status.load(), TrainingStatus::Idle);
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

TEST_F(WizardStateTest, StepLabelCoversAllValues) {
    EXPECT_STREQ(step_label(WizardStep::Load),    "Load data");
    EXPECT_STREQ(step_label(WizardStep::Quality), "Data quality");
    EXPECT_STREQ(step_label(WizardStep::Train),   "Train");
    EXPECT_STREQ(step_label(WizardStep::Results), "Results");
}

TEST_F(WizardStateTest, TimeBudgetLabelCoversAllValues) {
    EXPECT_NE(std::string(time_budget_label(TimeBudget::Quick)).find("30"),  std::string::npos);
    EXPECT_NE(std::string(time_budget_label(TimeBudget::Balanced)).find("min"),
              std::string::npos);
    EXPECT_NE(std::string(time_budget_label(TimeBudget::Best)).find("30"),   std::string::npos);
}

// ---------------------------------------------------------------------------
// CSV preview
// ---------------------------------------------------------------------------

class CsvPreviewTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& info = *::testing::UnitTest::GetInstance()->current_test_info();
        tmp_ = fs::temp_directory_path() / "neurograph_csv_preview_test"
             / info.test_suite_name() / info.name();
        std::error_code ec;
        fs::remove_all(tmp_, ec);
        fs::create_directories(tmp_, ec);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_, ec);
    }
    fs::path tmp_;
};

TEST_F(CsvPreviewTest, ReadsHeaderAndPreview) {
    const auto p = tmp_ / "tiny.csv";
    {
        std::ofstream f(p);
        f << "a,b,c,label\n";
        for (int i = 0; i < 20; ++i) f << i << ',' << i*2 << ',' << i*3 << ",X\n";
    }
    auto r = peek_csv_metadata(p, 5);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& meta = r.value();
    EXPECT_EQ(meta.columns.size(), 4u);
    EXPECT_EQ(meta.columns[0], "a");
    EXPECT_EQ(meta.columns[3], "label");
    EXPECT_EQ(meta.row_count, 20u);
    EXPECT_EQ(meta.preview_rows.size(), 5u);
    EXPECT_EQ(meta.preview_rows[0].size(), 4u);
}

TEST_F(CsvPreviewTest, HandlesQuotedCommaField) {
    const auto p = tmp_ / "quoted.csv";
    {
        std::ofstream f(p);
        f << "name,price\n";
        f << "\"widget, small\",9.99\n";
        f << "\"gadget\",19.99\n";
    }
    auto r = peek_csv_metadata(p, 5);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const auto& meta = r.value();
    EXPECT_EQ(meta.row_count, 2u);
    ASSERT_EQ(meta.preview_rows.size(), 2u);
    EXPECT_EQ(meta.preview_rows[0][0], "widget, small");
    EXPECT_EQ(meta.preview_rows[0][1], "9.99");
}

TEST_F(CsvPreviewTest, HandlesDoubledQuoteEscape) {
    const auto p = tmp_ / "escape.csv";
    {
        std::ofstream f(p);
        f << "quote\n";
        f << "\"he said \"\"hi\"\"\"\n";
    }
    auto r = peek_csv_metadata(p, 5);
    ASSERT_TRUE(r.ok()) << r.error().message;
    EXPECT_EQ(r.value().preview_rows[0][0], "he said \"hi\"");
}

TEST_F(CsvPreviewTest, StripsTrailingCr) {
    const auto p = tmp_ / "crlf.csv";
    {
        std::ofstream f(p, std::ios::binary);
        f << "a,b\r\n" << "1,2\r\n";
    }
    auto r = peek_csv_metadata(p, 5);
    ASSERT_TRUE(r.ok()) << r.error().message;
    EXPECT_EQ(r.value().columns.back(), "b");
    EXPECT_EQ(r.value().preview_rows[0].back(), "2");
}

TEST_F(CsvPreviewTest, ErrorsOnMissingFile) {
    auto r = peek_csv_metadata(tmp_ / "nope.csv", 5);
    EXPECT_FALSE(r.ok());
}

TEST_F(CsvPreviewTest, ErrorsOnEmptyFile) {
    const auto p = tmp_ / "empty.csv";
    {
        std::ofstream f(p);   // touch — scoped so the file flushes on close
    }
    auto r = peek_csv_metadata(p, 5);
    EXPECT_FALSE(r.ok());
}

TEST_F(CsvPreviewTest, CapsPreviewAtRequestedSize) {
    const auto p = tmp_ / "big.csv";
    {
        std::ofstream f(p);
        f << "a\n";
        for (int i = 0; i < 100; ++i) f << i << '\n';
    }
    auto r = peek_csv_metadata(p, 3);
    ASSERT_TRUE(r.ok()) << r.error().message;
    EXPECT_EQ(r.value().preview_rows.size(), 3u);
    EXPECT_EQ(r.value().row_count, 100u);
}
