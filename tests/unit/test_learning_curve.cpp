// Unit tests for neurograph::eval::compute_learning_curve (ticket P1-06).
//
// These tests inject a mock trainer that returns a programmed (train, val)
// metric per fraction. That keeps the slope/verdict math under test
// without relying on a real GD loop — the integration test will exercise
// the full path end-to-end later.

#include "neurograph/eval/learning_curve.hpp"
#include "neurograph/preprocess/pipeline.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <vector>

using neurograph::eval::compute_learning_curve;
using neurograph::eval::LearningCurveConfig;
using neurograph::eval::LearningCurvePoint;
using neurograph::eval::LearningCurveReport;
using neurograph::eval::TrainerFn;
using neurograph::preprocess::TransformedDataset;

namespace {

// Bare-bones dataset factory: n_train rows of a single dummy feature,
// n_val rows on the val side. Content of X/y doesn't matter — the
// trainer mock ignores it.
TransformedDataset make_dummy(int n_train = 200, int n_val = 50) {
    TransformedDataset ds;
    ds.is_classification = true;
    ds.class_labels      = {"a", "b"};
    for (int i = 0; i < n_train; ++i) {
        ds.X_train.push_back({double(i)});
        ds.y_train.push_back({i % 2 == 0 ? 1.0 : 0.0,
                              i % 2 == 0 ? 0.0 : 1.0});
    }
    for (int i = 0; i < n_val; ++i) {
        ds.X_val.push_back({double(i)});
        ds.y_val.push_back({i % 2 == 0 ? 1.0 : 0.0,
                            i % 2 == 0 ? 0.0 : 1.0});
    }
    return ds;
}

// Trainer mock that returns a metric scripted by fraction. The
// implementation indexes by n_samples (the actual subset size) so we
// can verify the implementation passes the expected subset size.
struct ScriptedTrainer {
    std::map<int, std::pair<double, double>> by_n;  // n_samples → (train, val)
    int                                      calls = 0;

    TrainerFn fn() {
        return [this](const std::vector<std::vector<double>>& X_sub,
                      const std::vector<std::vector<double>>& /*y_sub*/,
                      const std::vector<std::vector<double>>& /*X_val*/,
                      const std::vector<std::vector<double>>& /*y_val*/,
                      int /*epochs*/, std::uint64_t /*seed*/)
            -> std::pair<double, double> {
            ++calls;
            const int n = static_cast<int>(X_sub.size());
            auto it = by_n.find(n);
            if (it == by_n.end()) return {0.5, 0.5};
            return it->second;
        };
    }
};

} // namespace

// ---------------------------------------------------------------------------
// 1. Determinism
// ---------------------------------------------------------------------------

TEST(LearningCurve, DeterministicForSameSeed) {
    auto ds = make_dummy(200);
    ScriptedTrainer t1, t2;
    // Same scripted curve for both runs.
    for (int n : {50, 100, 150, 200}) t1.by_n[n] = t2.by_n[n] = {0.7, 0.6};

    LearningCurveConfig cfg;
    const auto a = compute_learning_curve(ds, true, cfg, t1.fn());
    const auto b = compute_learning_curve(ds, true, cfg, t2.fn());

    ASSERT_EQ(a.points.size(), b.points.size());
    for (std::size_t i = 0; i < a.points.size(); ++i) {
        EXPECT_EQ(a.points[i].n_samples, b.points[i].n_samples);
        EXPECT_DOUBLE_EQ(a.points[i].val_metric, b.points[i].val_metric);
    }
    EXPECT_DOUBLE_EQ(a.slope_at_100pct, b.slope_at_100pct);
    EXPECT_EQ(a.message, b.message);
}

// ---------------------------------------------------------------------------
// 2. Slope math — flat curve
// ---------------------------------------------------------------------------

TEST(LearningCurve, FlatCurveYieldsZeroSlopeAndPlateau) {
    auto ds = make_dummy(200);
    ScriptedTrainer t;
    for (int n : {50, 100, 150, 200}) t.by_n[n] = {0.8, 0.8};

    const auto r = compute_learning_curve(ds, true, {}, t.fn());

    ASSERT_EQ(r.points.size(), 4u);
    EXPECT_NEAR(r.slope_at_100pct, 0.0, 1e-12);
    EXPECT_NEAR(r.projected_at_2x_data, 0.8, 1e-12);
    EXPECT_EQ(r.verdict, LearningCurveReport::Verdict::Plateau);
}

// ---------------------------------------------------------------------------
// 3. Slope math — climbing curve, Improving verdict
// ---------------------------------------------------------------------------
//
// Construct val = a + b*log(n) with b chosen so b*log(2) > 0.02. With
// b = 0.05, b*log(2) ≈ 0.0347 → Improving.

TEST(LearningCurve, MonotonicClimbYieldsImproving) {
    auto ds = make_dummy(400);
    ScriptedTrainer t;
    const double a = 0.4, b = 0.05;
    for (int n : {100, 200, 300, 400}) {
        const double v = a + b * std::log(double(n));
        t.by_n[n] = {v, v};
    }
    const auto r = compute_learning_curve(ds, true, {}, t.fn());

    EXPECT_EQ(r.verdict, LearningCurveReport::Verdict::Improving);
    // Recovered slope should match the planted b within fp noise.
    const double n_full = 400.0;
    EXPECT_NEAR(r.slope_at_100pct, b / n_full, 1e-9);
    EXPECT_NEAR(r.projected_at_2x_data,
                t.by_n[400].second + b * std::log(2.0), 1e-9);
}

// ---------------------------------------------------------------------------
// 4. Slope math — gentle climb, Inconclusive verdict
// ---------------------------------------------------------------------------
//
// 0.005 ≤ b*log(2) ≤ 0.02. Pick b such that b*log(2) ≈ 0.01.

TEST(LearningCurve, GentleClimbYieldsInconclusive) {
    auto ds = make_dummy(400);
    ScriptedTrainer t;
    const double a = 0.6;
    const double b = 0.01 / std::log(2.0);  // → b*log(2) = 0.01
    for (int n : {100, 200, 300, 400}) {
        const double v = a + b * std::log(double(n));
        t.by_n[n] = {v, v};
    }
    const auto r = compute_learning_curve(ds, true, {}, t.fn());

    EXPECT_EQ(r.verdict, LearningCurveReport::Verdict::Inconclusive);
}

// ---------------------------------------------------------------------------
// 5. Edge case — too few training samples
// ---------------------------------------------------------------------------

TEST(LearningCurve, ReturnsInconclusiveWhenTrainSetTooSmall) {
    TransformedDataset ds;
    ds.is_classification = true;
    for (int i = 0; i < 3; ++i) {
        ds.X_train.push_back({double(i)});
        ds.y_train.push_back({1.0, 0.0});
    }
    ds.X_val.push_back({0.0});
    ds.y_val.push_back({1.0, 0.0});

    ScriptedTrainer t;
    const auto r = compute_learning_curve(ds, true, {}, t.fn());

    EXPECT_EQ(r.verdict, LearningCurveReport::Verdict::Inconclusive);
    EXPECT_TRUE(r.points.empty());
    EXPECT_EQ(t.calls, 0) << "trainer should not be called for tiny datasets";
}

// ---------------------------------------------------------------------------
// 6. Edge case — empty val set
// ---------------------------------------------------------------------------

TEST(LearningCurve, ReturnsInconclusiveWhenValEmpty) {
    auto ds = make_dummy(200, 0);
    ScriptedTrainer t;
    const auto r = compute_learning_curve(ds, true, {}, t.fn());

    EXPECT_EQ(r.verdict, LearningCurveReport::Verdict::Inconclusive);
    EXPECT_TRUE(r.points.empty());
    EXPECT_EQ(t.calls, 0);
}

// ---------------------------------------------------------------------------
// 7. Subset sizes match the configured fractions
// ---------------------------------------------------------------------------

TEST(LearningCurve, ProducesPointsAtConfiguredFractions) {
    auto ds = make_dummy(200);
    ScriptedTrainer t;
    for (int n : {50, 100, 150, 200}) t.by_n[n] = {0.7, 0.7};

    const auto r = compute_learning_curve(ds, true, {}, t.fn());

    ASSERT_EQ(r.points.size(), 4u);
    EXPECT_EQ(r.points[0].n_samples, 50);
    EXPECT_EQ(r.points[1].n_samples, 100);
    EXPECT_EQ(r.points[2].n_samples, 150);
    EXPECT_EQ(r.points[3].n_samples, 200);
    EXPECT_EQ(t.calls, 4);
}

// ---------------------------------------------------------------------------
// 8. Custom fractions are honoured (and 1.0 always present)
// ---------------------------------------------------------------------------

TEST(LearningCurve, AppendsFullSetIfFractionsOmitOne) {
    auto ds = make_dummy(200);
    ScriptedTrainer t;
    for (int n : {20, 40, 200}) t.by_n[n] = {0.7, 0.7};

    LearningCurveConfig cfg;
    cfg.fractions = {0.1, 0.2};   // user forgot 1.0 — we add it
    const auto r = compute_learning_curve(ds, true, cfg, t.fn());

    ASSERT_EQ(r.points.size(), 3u);
    EXPECT_EQ(r.points[0].n_samples, 20);
    EXPECT_EQ(r.points[1].n_samples, 40);
    EXPECT_EQ(r.points[2].n_samples, 200);
}

// ---------------------------------------------------------------------------
// 9. Subsets are nested (incremental data)
// ---------------------------------------------------------------------------
//
// The 50% subset should be a superset of the 25% subset, because the
// learning curve narrative is "add more data, watch it improve" — not
// "draw four independent random samples." We verify by tagging each row
// with a unique value and checking the trainer sees nested prefixes.

TEST(LearningCurve, SubsetsAreNestedAcrossFractions) {
    TransformedDataset ds;
    ds.is_classification = true;
    for (int i = 0; i < 100; ++i) {
        ds.X_train.push_back({double(i)});   // unique tag per row
        ds.y_train.push_back({1.0, 0.0});
    }
    ds.X_val.push_back({0.0});
    ds.y_val.push_back({1.0, 0.0});

    std::vector<std::vector<double>> seen_25, seen_50;
    auto trainer = [&](const std::vector<std::vector<double>>& X_sub,
                       const std::vector<std::vector<double>>& /*y*/,
                       const std::vector<std::vector<double>>& /*Xv*/,
                       const std::vector<std::vector<double>>& /*yv*/,
                       int, std::uint64_t) -> std::pair<double, double> {
        if (X_sub.size() == 25) seen_25 = X_sub;
        if (X_sub.size() == 50) seen_50 = X_sub;
        return {0.7, 0.7};
    };

    LearningCurveConfig cfg;
    cfg.fractions = {0.25, 0.5, 1.0};
    (void)compute_learning_curve(ds, true, cfg, trainer);

    ASSERT_EQ(seen_25.size(), 25u);
    ASSERT_EQ(seen_50.size(), 50u);
    // Every 25%-row must reappear in the 50% subset.
    for (const auto& row : seen_25) {
        bool found = false;
        for (const auto& other : seen_50) {
            if (other == row) { found = true; break; }
        }
        EXPECT_TRUE(found) << "25% subset is not contained in 50% subset";
    }
}

// ---------------------------------------------------------------------------
// 10. Per-point seeds differ from the master seed
// ---------------------------------------------------------------------------

TEST(LearningCurve, PerPointSeedsAreDistinct) {
    auto ds = make_dummy(200);
    std::vector<std::uint64_t> seeds_seen;
    auto trainer = [&](const std::vector<std::vector<double>>&,
                       const std::vector<std::vector<double>>&,
                       const std::vector<std::vector<double>>&,
                       const std::vector<std::vector<double>>&,
                       int, std::uint64_t s) -> std::pair<double, double> {
        seeds_seen.push_back(s);
        return {0.7, 0.7};
    };
    LearningCurveConfig cfg;
    cfg.seed = 12345;
    (void)compute_learning_curve(ds, true, cfg, trainer);

    ASSERT_EQ(seeds_seen.size(), 4u);
    // All distinct.
    for (std::size_t i = 0; i < seeds_seen.size(); ++i) {
        EXPECT_NE(seeds_seen[i], cfg.seed);
        for (std::size_t j = i + 1; j < seeds_seen.size(); ++j) {
            EXPECT_NE(seeds_seen[i], seeds_seen[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// 11. Spec-string verdict messages are stable
// ---------------------------------------------------------------------------
//
// The wizard renders these strings verbatim, so changes need to be
// intentional (and ideally accompanied by a copy review).

TEST(LearningCurve, PlateauMessageMatchesSpec) {
    auto ds = make_dummy(200);
    ScriptedTrainer t;
    for (int n : {50, 100, 150, 200}) t.by_n[n] = {0.8, 0.8};
    const auto r = compute_learning_curve(ds, true, {}, t.fn());
    EXPECT_EQ(r.message,
              "More data will likely not help much (projected +<0.5% at 2× data).");
}

TEST(LearningCurve, ImprovingMessageMatchesSpec) {
    auto ds = make_dummy(400);
    ScriptedTrainer t;
    const double a = 0.4, b = 0.05;
    for (int n : {100, 200, 300, 400}) {
        const double v = a + b * std::log(double(n));
        t.by_n[n] = {v, v};
    }
    const auto r = compute_learning_curve(ds, true, {}, t.fn());
    EXPECT_EQ(r.message,
              "Collecting more data should meaningfully improve accuracy "
              "(projected +2% or more at 2× data).");
}

// ---------------------------------------------------------------------------
// 12. Null trainer guard
// ---------------------------------------------------------------------------

TEST(LearningCurve, NullTrainerReturnsInconclusive) {
    auto ds = make_dummy(200);
    const auto r = compute_learning_curve(ds, true, {}, nullptr);
    EXPECT_EQ(r.verdict, LearningCurveReport::Verdict::Inconclusive);
    EXPECT_TRUE(r.points.empty());
}
