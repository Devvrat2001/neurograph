// What the preprocessing does with each kind of column, and the shared
// number / date parsing it relies on.
//
// Several of these guard against defects found in review: numeric columns
// with more than 1,024 distinct values used to be hashed as text, and
// scaling statistics used to be computed over every row, test rows
// included.

#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/util/text_parse.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using neurograph::preprocess::ColumnDecision;
using neurograph::preprocess::PipelineConfig;
using neurograph::preprocess::TaskChoice;
using neurograph::preprocess::TransformedDataset;
using neurograph::preprocess::fit_pipeline;
using neurograph::preprocess::plan_columns;

namespace {

class Csv {
public:
    Csv(const std::string& name, const std::string& body) {
        path_ = fs::temp_directory_path() / ("ng_rules_" + name);
        std::ofstream f(path_, std::ios::binary);
        f << body;
    }
    ~Csv() {
        std::error_code ec;
        fs::remove(path_, ec);
    }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

const ColumnDecision* find_col(const std::vector<ColumnDecision>& cols,
                               const std::string& name) {
    for (const auto& c : cols) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

PipelineConfig config_for(const std::string& target) {
    PipelineConfig cfg;
    cfg.target_column = target;
    cfg.split_seed    = 7;
    return cfg;
}

std::size_t count_from(const TransformedDataset& ds, const std::string& source) {
    return static_cast<std::size_t>(
        std::count(ds.feature_sources.begin(), ds.feature_sources.end(), source));
}

} // namespace

// ---------------------------------------------------------------------------
// Number, date and hash parsing
// ---------------------------------------------------------------------------

TEST(TextParse, ReadsPlainAndFormattedNumbers) {
    using neurograph::util::try_parse_number;
    EXPECT_DOUBLE_EQ(try_parse_number("42").value, 42.0);
    EXPECT_DOUBLE_EQ(try_parse_number(" -3.5e2 ").value, -350.0);
    EXPECT_DOUBLE_EQ(try_parse_number("+7").value, 7.0);
    EXPECT_DOUBLE_EQ(try_parse_number("$1,200").value, 1200.0);
    EXPECT_DOUBLE_EQ(try_parse_number("Rs. 12,34,567").value, 1234567.0);
    EXPECT_DOUBLE_EQ(try_parse_number("45%").value, 45.0);
    EXPECT_DOUBLE_EQ(try_parse_number("(1,234.50)").value, -1234.5);
    EXPECT_DOUBLE_EQ(try_parse_number("\xE2\x82\xAC" "9.50").value, 9.5);
    EXPECT_TRUE(try_parse_number("1,234,567.89").parsed);
}

TEST(TextParse, RefusesAmbiguousTextAndRecognisesBlanks) {
    using neurograph::util::try_parse_number;
    EXPECT_FALSE(try_parse_number("12,34").parsed);   // 12.34 in much of Europe
    EXPECT_FALSE(try_parse_number("abc").parsed);
    EXPECT_FALSE(try_parse_number("12abc").parsed);
    EXPECT_FALSE(try_parse_number("inf").parsed);
    EXPECT_TRUE(try_parse_number("").is_missing);
    EXPECT_TRUE(try_parse_number(" NA ").is_missing);
    EXPECT_TRUE(try_parse_number("#DIV/0!").is_missing);
}

TEST(TextParse, ReadsDates) {
    using neurograph::util::try_parse_date;
    const auto d = try_parse_date("2024-03-31");
    ASSERT_TRUE(d.parsed);
    EXPECT_EQ(d.year, 2024);
    EXPECT_EQ(d.month, 3);
    EXPECT_EQ(d.day, 31);
    EXPECT_EQ(d.weekday, 6);                       // a Sunday
    EXPECT_EQ(try_parse_date("1970-01-01").days, 0);
    EXPECT_EQ(try_parse_date("1970-01-01").weekday, 3);   // a Thursday

    const auto dmy = try_parse_date("05/04/2023", /*day_first=*/true);
    const auto mdy = try_parse_date("05/04/2023", /*day_first=*/false);
    ASSERT_TRUE(dmy.parsed);
    ASSERT_TRUE(mdy.parsed);
    EXPECT_EQ(dmy.day, 5);
    EXPECT_EQ(dmy.month, 4);
    EXPECT_EQ(mdy.day, 4);
    EXPECT_EQ(mdy.month, 5);

    EXPECT_TRUE(try_parse_date("2024-03-31T14:05:00").parsed);
    EXPECT_FALSE(try_parse_date("2023-02-29").parsed);   // not a leap year
    EXPECT_FALSE(try_parse_date("31/31/2023").parsed);
    EXPECT_FALSE(try_parse_date("12345").parsed);
}

TEST(TextParse, HashMatchesTheFnv1aReference) {
    // Text hashed into buckets must land in the same bucket on every
    // platform, so the hash is pinned to published FNV-1a values.
    EXPECT_EQ(neurograph::util::fnv1a64(""),  14695981039346656037ull);
    EXPECT_EQ(neurograph::util::fnv1a64("a"), 0xaf63dc4c8601ec8cull);
}

// ---------------------------------------------------------------------------
// Column decisions
// ---------------------------------------------------------------------------

TEST(PreprocessRules, NumericColumnWithManyDistinctValuesStaysNumeric) {
    // 5,000 distinct amounts. Before 0.2 anything past 1,024 distinct
    // values was hashed as if it were text.
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(0.0, 100.0);
    std::string body = "amount,label\n";
    for (int i = 0; i < 5000; ++i) {
        const double a = std::round(u(rng) * 100.0) / 100.0;
        char line[64];
        std::snprintf(line, sizeof(line), "%.2f,%s\n", a, a > 50.0 ? "hi" : "lo");
        body += line;
    }
    Csv csv("many_numbers.csv", body);

    const auto plan = plan_columns(csv.path(), config_for("label"));
    const auto* amount = find_col(plan, "amount");
    ASSERT_NE(amount, nullptr);
    EXPECT_EQ(amount->role, "number");

    TransformedDataset ds;
    auto pipe = fit_pipeline(csv.path(), config_for("label"), ds);
    ASSERT_NE(pipe, nullptr);
    EXPECT_EQ(count_from(ds, "amount"), 1u) << "one numeric feature, not hash buckets";
}

TEST(PreprocessRules, RowNumbersAndIdentifiersAreLeftOut) {
    std::string body = "id,customer_code,x,label\n";
    for (int i = 0; i < 100; ++i) {
        body += std::to_string(i + 1) + ",CUST-" + std::to_string(7919 * i + 13) + ","
              + std::to_string((i * 37) % 11) + "," + (i % 3 ? "yes" : "no") + "\n";
    }
    Csv csv("identifiers.csv", body);
    const auto plan = plan_columns(csv.path(), config_for("label"));

    const auto* id = find_col(plan, "id");
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->role, "left out");
    EXPECT_TRUE(id->note.find("row number") != std::string::npos ||
                id->note.find("identifier") != std::string::npos) << id->note;

    const auto* code = find_col(plan, "customer_code");
    ASSERT_NE(code, nullptr);
    EXPECT_EQ(code->role, "left out");
    EXPECT_NE(code->note.find("identifier"), std::string::npos) << code->note;

    const auto* x = find_col(plan, "x");
    ASSERT_NE(x, nullptr);
    EXPECT_EQ(x->role, "number");
}

TEST(PreprocessRules, DatesBecomeDayCountMonthAndWeekday) {
    std::string body = "when,label\n";
    for (int i = 0; i < 120; ++i) {
        const int month = 1 + (i % 12);
        const int day   = 1 + (i % 28);
        char line[64];
        std::snprintf(line, sizeof(line), "2023-%02d-%02d,%s\n", month, day,
                      month < 7 ? "early" : "late");
        body += line;
    }
    Csv csv("dates.csv", body);

    TransformedDataset ds;
    auto pipe = fit_pipeline(csv.path(), config_for("label"), ds);
    ASSERT_NE(pipe, nullptr);
    const auto* when = find_col(ds.columns, "when");
    ASSERT_NE(when, nullptr);
    EXPECT_EQ(when->role, "date");
    for (const char* f : {"when_days", "when_month", "when_weekday"}) {
        EXPECT_NE(std::find(ds.feature_names_out.begin(), ds.feature_names_out.end(), f),
                  ds.feature_names_out.end()) << f;
    }
}

TEST(PreprocessRules, FormattedNumbersAreNumbers) {
    std::string body = "price,discount,label\n";
    for (int i = 0; i < 60; ++i) {
        body += "\"$" + std::to_string(1 + i % 9) + "," + std::to_string(100 + i) + "\","
              + std::to_string(i % 40) + "%," + (i % 2 ? "A" : "B") + "\n";
    }
    Csv csv("formatted.csv", body);
    const auto plan = plan_columns(csv.path(), config_for("label"));
    ASSERT_NE(find_col(plan, "price"), nullptr);
    ASSERT_NE(find_col(plan, "discount"), nullptr);
    EXPECT_EQ(find_col(plan, "price")->role, "number");
    EXPECT_EQ(find_col(plan, "discount")->role, "number");
}

TEST(PreprocessRules, BlankCellsGetAFlagColumn) {
    std::string body = "x,label\n";
    for (int i = 0; i < 200; ++i) {
        body += (i % 20 == 0 ? std::string() : std::to_string(i % 17)) + ","
              + (i % 2 ? "A" : "B") + "\n";
    }
    Csv csv("blanks.csv", body);
    TransformedDataset ds;
    auto pipe = fit_pipeline(csv.path(), config_for("label"), ds);
    ASSERT_NE(pipe, nullptr);
    EXPECT_NE(std::find(ds.feature_names_out.begin(), ds.feature_names_out.end(), "x_missing"),
              ds.feature_names_out.end());
}

TEST(PreprocessRules, RowsWithABlankTargetAreSkipped) {
    std::string body = "x,label\n";
    for (int i = 0; i < 60; ++i) {
        body += std::to_string(i) + "," + (i % 12 == 0 ? "" : (i % 2 ? "A" : "B")) + "\n";
    }
    Csv csv("blank_target.csv", body);
    TransformedDataset ds;
    auto pipe = fit_pipeline(csv.path(), config_for("label"), ds);
    ASSERT_NE(pipe, nullptr);
    EXPECT_EQ(ds.rows_dropped_missing_target, 5);
    EXPECT_EQ(ds.X_train.size() + ds.X_val.size() + ds.X_test.size(), 55u);
}

TEST(PreprocessRules, ExcludedColumnsAreLeftOut) {
    std::string body = "x1,x2,label\n";
    for (int i = 0; i < 60; ++i) {
        body += std::to_string(i % 7) + "," + std::to_string(i % 5) + ","
              + (i % 2 ? "A" : "B") + "\n";
    }
    Csv csv("exclude.csv", body);
    auto cfg = config_for("label");
    cfg.exclude_columns = {"x2", "no_such_column"};
    TransformedDataset ds;
    auto pipe = fit_pipeline(csv.path(), cfg, ds);
    ASSERT_NE(pipe, nullptr);
    const auto* x2 = find_col(ds.columns, "x2");
    ASSERT_NE(x2, nullptr);
    EXPECT_EQ(x2->role, "left out");
    EXPECT_EQ(count_from(ds, "x2"), 0u);
    const auto inputs = pipe->input_columns();
    EXPECT_EQ(std::find(inputs.begin(), inputs.end(), "x2"), inputs.end());
}

TEST(PreprocessRules, ScalingIsFittedOnTrainingRowsOnly) {
    // Standard scaling fitted on the training rows centres those rows
    // exactly. Fitted on every row (the old behaviour) it would not.
    std::mt19937_64 rng(5);
    std::normal_distribution<double> nd(10.0, 3.0);
    std::string body = "x,label\n";
    for (int i = 0; i < 400; ++i) {
        body += std::to_string(nd(rng)) + "," + (i % 2 ? "A" : "B") + "\n";
    }
    Csv csv("scaling.csv", body);
    TransformedDataset ds;
    auto pipe = fit_pipeline(csv.path(), config_for("label"), ds);
    ASSERT_NE(pipe, nullptr);
    ASSERT_FALSE(ds.X_train.empty());
    double sum = 0.0;
    for (const auto& row : ds.X_train) sum += row[0];
    EXPECT_NEAR(sum / static_cast<double>(ds.X_train.size()), 0.0, 1e-9);
}

TEST(PreprocessRules, TaskCanBeForced) {
    std::string body = "x,flag\n";
    for (int i = 0; i < 80; ++i) {
        body += std::to_string(i % 13) + "," + std::to_string(i % 2) + "\n";
    }
    Csv csv("task.csv", body);

    TransformedDataset auto_ds;
    fit_pipeline(csv.path(), config_for("flag"), auto_ds);
    EXPECT_TRUE(auto_ds.is_classification) << "two whole-number values read as classes";

    auto cfg = config_for("flag");
    cfg.task = TaskChoice::Regression;
    TransformedDataset reg_ds;
    fit_pipeline(csv.path(), cfg, reg_ds);
    EXPECT_FALSE(reg_ds.is_classification);
}

TEST(PreprocessRules, TextTargetCannotBeForcedToRegression) {
    std::string body = "x,label\n";
    for (int i = 0; i < 40; ++i) body += std::to_string(i) + "," + (i % 2 ? "A" : "B") + "\n";
    Csv csv("text_target.csv", body);
    auto cfg = config_for("label");
    cfg.task = TaskChoice::Regression;
    TransformedDataset ds;
    EXPECT_THROW(fit_pipeline(csv.path(), cfg, ds), std::runtime_error);
}
