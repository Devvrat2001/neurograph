// Native model families and automatic selection.
//
// Each test builds a small synthetic problem with a known structure and
// checks that the family which should be able to learn it does, that
// models survive a JSON round trip bit-for-bit, and that malformed model
// files are rejected with an error rather than a crash.

#include "neurograph/models/automl.hpp"
#include "neurograph/models/model.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <random>

using namespace neurograph::models;

namespace {

// y = 1 when x0 * x1 > 0. Not separable by any straight line.
TrainSet make_xor(std::size_t n_train, std::size_t n_val, std::uint64_t seed) {
    TrainSet d;
    d.task = Task::Classification;
    d.n_classes = 2;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto fill = [&](std::size_t n, Matrix& X, std::vector<int>& y) {
        X.rows = n; X.cols = 3; X.data.resize(n * 3);
        y.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double a = nd(rng), b = nd(rng);
            X.row(i)[0] = a; X.row(i)[1] = b; X.row(i)[2] = nd(rng);   // third column is noise
            y[i] = (a * b > 0.0) ? 1 : 0;
        }
    };
    fill(n_train, d.X_train, d.y_train_cls);
    fill(n_val,   d.X_val,   d.y_val_cls);
    return d;
}

// Three classes separated by straight lines.
TrainSet make_linear_cls(std::size_t n_train, std::size_t n_val, std::uint64_t seed) {
    TrainSet d;
    d.task = Task::Classification;
    d.n_classes = 3;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto fill = [&](std::size_t n, Matrix& X, std::vector<int>& y) {
        X.rows = n; X.cols = 2; X.data.resize(n * 2);
        y.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const int c = static_cast<int>(i % 3);
            X.row(i)[0] = (c == 0 ? -3.0 : c == 1 ? 0.0 : 3.0) + 0.5 * nd(rng);
            X.row(i)[1] = (c == 1 ? 3.0 : 0.0) + 0.5 * nd(rng);
            y[i] = c;
        }
    };
    fill(n_train, d.X_train, d.y_train_cls);
    fill(n_val,   d.X_val,   d.y_val_cls);
    return d;
}

// y = 100 + 3*x0 - 2*x1 + noise. The offset is what a bias-free network cannot fit.
TrainSet make_linear_reg(std::size_t n_train, std::size_t n_val, std::uint64_t seed) {
    TrainSet d;
    d.task = Task::Regression;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto fill = [&](std::size_t n, Matrix& X, std::vector<double>& y) {
        X.rows = n; X.cols = 2; X.data.resize(n * 2);
        y.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double a = nd(rng), b = nd(rng);
            X.row(i)[0] = a; X.row(i)[1] = b;
            y[i] = 100.0 + 3.0 * a - 2.0 * b + 0.1 * nd(rng);
        }
    };
    fill(n_train, d.X_train, d.y_train_reg);
    fill(n_val,   d.X_val,   d.y_val_reg);
    return d;
}

// y is a step function of x0, plus noise: natural for trees, awkward for lines.
TrainSet make_step_reg(std::size_t n_train, std::size_t n_val, std::uint64_t seed) {
    TrainSet d;
    d.task = Task::Regression;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto fill = [&](std::size_t n, Matrix& X, std::vector<double>& y) {
        X.rows = n; X.cols = 2; X.data.resize(n * 2);
        y.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double a = ud(rng);
            X.row(i)[0] = a; X.row(i)[1] = ud(rng);
            y[i] = (a < -0.3 ? 10.0 : a < 0.4 ? -5.0 : 20.0) + 0.2 * nd(rng);
        }
    };
    fill(n_train, d.X_train, d.y_train_reg);
    fill(n_val,   d.X_val,   d.y_val_reg);
    return d;
}

double val_score(const Model& m, const TrainSet& d) {
    return evaluate_score(m, d.X_val, d.y_val_cls, d.y_val_reg);
}

// Every prediction on the validation rows must be identical after to_json → from_json.
void expect_round_trip_identical(const Model& m, const TrainSet& d) {
    const nlohmann::json j = m.to_json();
    // Go through text, as the bundle on disk does.
    const auto reloaded = model_from_json(nlohmann::json::parse(j.dump()));
    ASSERT_NE(reloaded, nullptr);
    EXPECT_STREQ(reloaded->kind(), m.kind());
    EXPECT_EQ(reloaded->n_features(), m.n_features());
    EXPECT_EQ(reloaded->n_outputs(), m.n_outputs());
    const int k = std::max(1, m.n_outputs());
    std::vector<double> a(k), b(k);
    for (std::size_t i = 0; i < d.X_val.rows; ++i) {
        m.predict(d.X_val.row(i), a.data());
        reloaded->predict(d.X_val.row(i), b.data());
        for (int c = 0; c < k; ++c) {
            ASSERT_EQ(a[c], b[c]) << "row " << i << " output " << c;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Linear model
// ---------------------------------------------------------------------------

TEST(ModelsLinear, SeparatesLinearClasses) {
    const auto d = make_linear_cls(300, 150, 1);
    MlpOptions o;
    o.hidden = {};
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_STREQ(rep.model->kind(), "linear_v1");
    EXPECT_GT(val_score(*rep.model, d), 0.97);
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsLinear, CannotLearnXor) {
    // Guards the XOR dataset itself: if a linear model could solve it, the
    // non-linear tests below would prove nothing.
    const auto d = make_xor(600, 300, 2);
    MlpOptions o;
    o.hidden = {};
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_LT(val_score(*rep.model, d), 0.65);
}

TEST(ModelsLinear, RegressionRecoversOffset) {
    const auto d = make_linear_reg(400, 200, 3);
    MlpOptions o;
    o.hidden = {};
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), 0.99);
    // The prediction at the origin should be the offset, about 100.
    const double origin[2] = {0.0, 0.0};
    double out = 0.0;
    rep.model->predict(origin, &out);
    EXPECT_NEAR(out, 100.0, 1.0);
}

// ---------------------------------------------------------------------------
// Neural network
// ---------------------------------------------------------------------------

TEST(ModelsMlp, LearnsXor) {
    const auto d = make_xor(800, 300, 4);
    MlpOptions o;
    o.hidden = {16};
    o.grow   = false;
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_STREQ(rep.model->kind(), "mlp_v1");
    EXPECT_GT(val_score(*rep.model, d), 0.93);
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsMlp, GrowsWhenStartedTooSmall) {
    // Two hidden units cannot carve out the XOR regions; the layer has to widen.
    const auto d = make_xor(800, 300, 5);
    MlpOptions o;
    o.hidden = {2};
    o.grow   = true;
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    const nlohmann::json j = rep.model->to_json();
    const auto sizes = j.at("sizes").get<std::vector<int>>();
    ASSERT_EQ(sizes.size(), 3u);
    EXPECT_GT(sizes[1], 2) << "hidden layer did not grow";
    EXPECT_EQ(j.at("grown_from").get<int>(), 2);
    EXPECT_GT(val_score(*rep.model, d), 0.90);
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsMlp, GrowthNeverExceedsMaxHidden) {
    const auto d = make_xor(400, 200, 6);
    MlpOptions o;
    o.hidden     = {2};
    o.grow       = true;
    o.max_hidden = 6;
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    const auto sizes = rep.model->to_json().at("sizes").get<std::vector<int>>();
    EXPECT_LE(sizes[1], 6);
}

TEST(ModelsMlp, RegressionWithLargeOffsetTarget) {
    // Target far from zero and in large units: needs biases and target scaling.
    auto d = make_linear_reg(400, 200, 7);
    for (double& y : d.y_train_reg) y = y * 1000.0;
    for (double& y : d.y_val_reg)   y = y * 1000.0;
    MlpOptions o;
    o.hidden = {8};
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), 0.98);
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsMlp, SameSeedGivesIdenticalModel) {
    const auto d = make_xor(300, 100, 8);
    MlpOptions o;
    o.hidden = {8};
    FitControl c;
    c.seed = 99;
    const auto a = fit_mlp(d, o, c);
    const auto b = fit_mlp(d, o, c);
    ASSERT_NE(a.model, nullptr);
    ASSERT_NE(b.model, nullptr);
    EXPECT_EQ(a.model->to_json().dump(), b.model->to_json().dump());
}

TEST(ModelsMlp, HonoursTimeLimit) {
    const auto d = make_xor(20000, 2000, 9);
    MlpOptions o;
    o.hidden     = {64};
    o.max_epochs = 100000;
    o.patience   = 100000;
    FitControl c;
    c.time_limit_seconds = 0.3;
    const auto t0 = std::chrono::steady_clock::now();
    const auto rep = fit_mlp(d, o, c);
    const double took = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    ASSERT_NE(rep.model, nullptr);
    EXPECT_TRUE(rep.stopped_by_time);
    EXPECT_LT(took, 5.0);
}

TEST(ModelsMlp, CancelBeforeFirstEpochReturnsNoModel) {
    const auto d = make_xor(2000, 200, 10);
    MlpOptions o;
    o.hidden = {16};
    FitControl c;
    c.cancelled = [] { return true; };
    const auto rep = fit_mlp(d, o, c);
    EXPECT_EQ(rep.model, nullptr);
}

// ---------------------------------------------------------------------------
// Gradient-boosted trees
// ---------------------------------------------------------------------------

TEST(ModelsGbt, LearnsXor) {
    const auto d = make_xor(800, 300, 11);
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_STREQ(rep.model->kind(), "gbt_v1");
    EXPECT_GT(val_score(*rep.model, d), 0.93);
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsGbt, MulticlassProbabilitiesSumToOne) {
    const auto d = make_linear_cls(300, 150, 12);
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), 0.95);
    double p[3];
    for (std::size_t i = 0; i < d.X_val.rows; ++i) {
        rep.model->predict(d.X_val.row(i), p);
        EXPECT_NEAR(p[0] + p[1] + p[2], 1.0, 1e-9);
        for (double v : p) { EXPECT_GE(v, 0.0); EXPECT_LE(v, 1.0); }
    }
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsGbt, RegressionFitsStepFunction) {
    // Not 0.99+: a step this sharp costs dearly for each validation row
    // that falls between the true edge and the nearest bin boundary.
    const auto d = make_step_reg(600, 300, 13);
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), 0.95);
    expect_round_trip_identical(*rep.model, d);
}

TEST(ModelsGbt, ContinuousColumnWithThousandsOfDistinctValues) {
    // 5,000 rows of continuous values: every value distinct. The binning
    // must cope and the model must still use the numbers' order.
    const auto d = make_step_reg(5000, 1000, 14);
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), 0.95);
    // A straight line through a three-level step explains far less.
    MlpOptions lin;
    lin.hidden = {};
    const auto linear = fit_mlp(d, lin, FitControl{});
    ASSERT_NE(linear.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), val_score(*linear.model, d) + 0.2);
}

TEST(ModelsGbt, ConstantFeaturesAreIgnored) {
    auto d = make_xor(400, 200, 15);
    for (std::size_t i = 0; i < d.X_train.rows; ++i) d.X_train.row(i)[2] = 7.0;
    for (std::size_t i = 0; i < d.X_val.rows; ++i)   d.X_val.row(i)[2]   = 7.0;
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    EXPECT_GT(val_score(*rep.model, d), 0.90);
}

TEST(ModelsGbt, SameInputGivesIdenticalModel) {
    const auto d = make_xor(300, 100, 16);
    const auto a = fit_gbt(d, GbtOptions{}, FitControl{});
    const auto b = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(a.model, nullptr);
    ASSERT_NE(b.model, nullptr);
    EXPECT_EQ(a.model->to_json().dump(), b.model->to_json().dump());
}

TEST(ModelsGbt, TinyDatasetDoesNotCrash) {
    auto d = make_xor(6, 3, 17);
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    double p[2];
    rep.model->predict(d.X_val.row(0), p);
    EXPECT_NEAR(p[0] + p[1], 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// JSON loading
// ---------------------------------------------------------------------------

TEST(ModelsJson, RejectsUnknownKind) {
    EXPECT_THROW(model_from_json(nlohmann::json{{"kind", "mystery_v9"}}), std::runtime_error);
    EXPECT_THROW(model_from_json(nlohmann::json::object()), std::runtime_error);
    EXPECT_THROW(model_from_json(nlohmann::json::array()), std::runtime_error);
}

TEST(ModelsJson, RejectsTreeWithOutOfRangeChild) {
    const auto d = make_xor(300, 100, 18);
    const auto rep = fit_gbt(d, GbtOptions{}, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    nlohmann::json j = rep.model->to_json();
    ASSERT_FALSE(j.at("trees").empty());
    // Point a split node's child at itself: would loop forever in predict().
    auto& t = j["trees"][0];
    for (std::size_t i = 0; i < t["f"].size(); ++i) {
        if (t["f"][i].get<int>() >= 0) { t["l"][i] = static_cast<int>(i); break; }
    }
    EXPECT_THROW(model_from_json(j), std::runtime_error);
}

TEST(ModelsJson, RejectsMlpWithWrongWeightShape) {
    const auto d = make_xor(200, 100, 19);
    MlpOptions o;
    o.hidden = {4};
    o.grow   = false;
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    nlohmann::json j = rep.model->to_json();
    j["W"][0].erase(j["W"][0].begin());      // drop one weight
    EXPECT_THROW(model_from_json(j), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Blend
// ---------------------------------------------------------------------------

TEST(ModelsBlend, AveragesMembersAndRoundTrips) {
    const auto d = make_xor(500, 200, 20);
    MlpOptions o;
    o.hidden = {16};
    o.grow   = false;
    std::shared_ptr<const Model> a(fit_mlp(d, o, FitControl{}).model.release());
    std::shared_ptr<const Model> b(fit_gbt(d, GbtOptions{}, FitControl{}).model.release());
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    const auto blend = make_blend({a, b}, {3.0, 1.0});      // normalised to 0.75 / 0.25
    double pa[2], pb[2], pm[2];
    a->predict(d.X_val.row(0), pa);
    b->predict(d.X_val.row(0), pb);
    blend->predict(d.X_val.row(0), pm);
    EXPECT_NEAR(pm[1], 0.75 * pa[1] + 0.25 * pb[1], 1e-12);
    expect_round_trip_identical(*blend, d);
}

TEST(ModelsBlend, RejectsMismatchedMembers) {
    const auto dc = make_xor(200, 100, 21);
    const auto dr = make_linear_reg(200, 100, 22);
    std::shared_ptr<const Model> a(fit_gbt(dc, GbtOptions{}, FitControl{}).model.release());
    std::shared_ptr<const Model> b(fit_gbt(dr, GbtOptions{}, FitControl{}).model.release());
    EXPECT_THROW(make_blend({a, b}, {0.5, 0.5}), std::runtime_error);
    EXPECT_THROW(make_blend({}, {}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Automatic selection
// ---------------------------------------------------------------------------

TEST(ModelsAutoFit, PicksANonLinearModelForXor) {
    const auto d = make_xor(800, 300, 23);
    AutoFitOptions o;
    o.time_budget_seconds = 0.0;      // unlimited: keeps the test deterministic
    const auto r = auto_fit(d, o);
    ASSERT_NE(r.model, nullptr);
    ASSERT_GE(r.leaderboard.size(), 3u);
    EXPECT_GT(val_score(*r.model, d), 0.93);
    int chosen = 0;
    for (const auto& c : r.leaderboard) {
        if (c.chosen) { ++chosen; EXPECT_NE(c.kind, "linear_v1"); }
    }
    EXPECT_EQ(chosen, 1);
}

TEST(ModelsAutoFit, ChosenModelIsNeverWorseThanAnyCandidate) {
    const auto d = make_step_reg(600, 300, 24);
    AutoFitOptions o;
    o.time_budget_seconds = 0.0;
    const auto r = auto_fit(d, o);
    ASSERT_NE(r.model, nullptr);
    const double chosen_loss = evaluate_loss(*r.model, d.X_val, d.y_val_cls, d.y_val_reg);
    for (const auto& c : r.leaderboard) {
        EXPECT_LE(chosen_loss, c.val_loss * (1.0 + 1e-9)) << c.name;
    }
}

TEST(ModelsAutoFit, SameSeedGivesIdenticalModel) {
    const auto d = make_linear_cls(300, 150, 25);
    AutoFitOptions o;
    o.time_budget_seconds = 0.0;
    const auto a = auto_fit(d, o);
    const auto b = auto_fit(d, o);
    ASSERT_NE(a.model, nullptr);
    ASSERT_NE(b.model, nullptr);
    EXPECT_EQ(a.model->to_json().dump(), b.model->to_json().dump());
}

TEST(ModelsAutoFit, StaysInsideTimeBudget) {
    const auto d = make_xor(30000, 3000, 26);
    AutoFitOptions o;
    o.time_budget_seconds = 1.0;
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = auto_fit(d, o);
    const double took = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    ASSERT_NE(r.model, nullptr);
    // Generous ceiling: each family finishes the batch or round it is in.
    EXPECT_LT(took, 6.0);
}

TEST(ModelsAutoFit, CancelledBeforeStartReturnsNothing) {
    const auto d = make_xor(500, 100, 27);
    const auto r = auto_fit(d, AutoFitOptions{}, [] { return true; });
    EXPECT_EQ(r.model, nullptr);
    EXPECT_TRUE(r.cancelled);
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

TEST(ModelsScore, EvaluateLossMatchesHandComputation) {
    const auto d = make_linear_reg(100, 50, 28);
    MlpOptions o;
    o.hidden = {};
    const auto rep = fit_mlp(d, o, FitControl{});
    ASSERT_NE(rep.model, nullptr);
    double sse = 0.0, out = 0.0;
    for (std::size_t i = 0; i < d.X_val.rows; ++i) {
        rep.model->predict(d.X_val.row(i), &out);
        sse += (out - d.y_val_reg[i]) * (out - d.y_val_reg[i]);
    }
    EXPECT_NEAR(evaluate_loss(*rep.model, d.X_val, d.y_val_cls, d.y_val_reg),
                sse / d.X_val.rows, 1e-9);
    // The trainer's reported loss is in the target's own units and agrees.
    EXPECT_NEAR(rep.val_loss, sse / d.X_val.rows, 1e-6 * (1.0 + sse / d.X_val.rows));
}
