#include "neurograph/orchestration/auto_pipeline.hpp"

#include "neurograph/dataqc/scanner.hpp"
#include "neurograph/models/automl.hpp"
#include "neurograph/models/model.hpp"
#include "neurograph/preprocess/pipeline.hpp"
#include "neurograph/reporting/manifest.hpp"
#include "neurograph/reporting/model_card.hpp"
#include "neurograph/version.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neurograph::orchestration {

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// File names inside a bundle. runtime/bundle.cpp and reporting/manifest.cpp
// use the same names.
constexpr const char* kBundlePipeline = "pipeline.ngpipe";
constexpr const char* kBundleModel    = "model.ngm";
constexpr const char* kBundleManifest = "manifest.json";
constexpr const char* kBundleCard     = "card.md";

// model.ngm schema 3: the chosen models::Model, serialised by the model
// itself. Schemas 1 (linear probe) and 2 (SON network) are still loaded
// by the runtime for bundles written by older versions.
constexpr int         kModelSchemaVersion = 3;
constexpr const char* kTrainerName        = "neurograph_automl_v1";

// Share of the progress bar for each stage, in Stage order. Training is
// where the time goes.
constexpr double kStageWeights[] = {
    0.03,  // Scanning
    0.07,  // Preprocessing
    0.01,  // AutoConfig
    0.70,  // Training
    0.05,  // Evaluating (scores and column importance)
    0.03,  // Baselining
    0.08,  // LearningCurve
    0.03,  // Packaging
};
constexpr int kNumStages = sizeof(kStageWeights) / sizeof(kStageWeights[0]);

// Share of the time budget given to training the candidate models. The
// rest covers evaluation and, when requested, the learning curve.
constexpr double kTrainShareWithCurve    = 0.80;
constexpr double kTrainShareWithoutCurve = 0.92;

double overall_at(Stage s, double stage_fraction) {
    const int idx = static_cast<int>(s);
    if (idx >= kNumStages) return 1.0;
    double before = 0.0;
    for (int i = 0; i < idx; ++i) before += kStageWeights[i];
    return before + kStageWeights[idx] * std::clamp(stage_fraction, 0.0, 1.0);
}

// Progress sink. The overall bar never moves backwards, whatever a
// sub-stage reports.
class Reporter {
public:
    explicit Reporter(ProgressCb cb) : cb_(std::move(cb)) {}

    void operator()(Stage s, double stage_fraction, std::string msg) {
        if (!cb_) return;
        Progress p;
        p.stage            = s;
        p.stage_fraction   = std::clamp(stage_fraction, 0.0, 1.0);
        p.overall_fraction = std::max(last_, overall_at(s, stage_fraction));
        p.message          = std::move(msg);
        last_ = p.overall_fraction;
        cb_(p);
    }

private:
    ProgressCb cb_;
    double     last_ = 0.0;
};

bool is_cancelled(const CancelCb& cb) { return cb && cb(); }

util::Error fail(std::string msg) { return util::Error(std::move(msg), 1); }

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// UTF-8 text of a path, for messages. path::string() throws on Windows
// for names outside the ANSI code page.
std::string to_utf8(const fs::path& p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::string fmt(const char* format, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), format, v);
    return buf;
}

// ---------------------------------------------------------------------------
// Data shaping
// ---------------------------------------------------------------------------

int argmax_row(const std::vector<double>& v) {
    if (v.empty()) return -1;
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

std::vector<int> argmax_rows(const std::vector<std::vector<double>>& rows) {
    std::vector<int> out;
    out.reserve(rows.size());
    for (const auto& r : rows) out.push_back(argmax_row(r));
    return out;
}

std::vector<double> flatten_scalar(const std::vector<std::vector<double>>& rows) {
    std::vector<double> out;
    out.reserve(rows.size());
    for (const auto& r : rows) out.push_back(r.empty() ? 0.0 : r[0]);
    return out;
}

models::TrainSet make_train_set(bool cls, int n_classes,
                                const std::vector<std::vector<double>>& X_train,
                                const std::vector<std::vector<double>>& y_train,
                                const std::vector<std::vector<double>>& X_val,
                                const std::vector<std::vector<double>>& y_val) {
    models::TrainSet t;
    t.task      = cls ? models::Task::Classification : models::Task::Regression;
    t.n_classes = cls ? n_classes : 0;
    t.X_train   = models::Matrix::from_rows(X_train);
    t.X_val     = models::Matrix::from_rows(X_val);
    if (cls) {
        t.y_train_cls = argmax_rows(y_train);
        t.y_val_cls   = argmax_rows(y_val);
    } else {
        t.y_train_reg = flatten_scalar(y_train);
        t.y_val_reg   = flatten_scalar(y_val);
    }
    return t;
}

// Probabilities (classification) or a one-value row (regression) per row.
std::vector<std::vector<double>> predict_rows(const models::Model& m,
                                              const models::Matrix& X) {
    const std::size_t k = static_cast<std::size_t>(std::max(1, m.n_outputs()));
    std::vector<std::vector<double>> out(X.rows, std::vector<double>(k, 0.0));
    for (std::size_t i = 0; i < X.rows; ++i) m.predict(X.row(i), out[i].data());
    return out;
}

// ---------------------------------------------------------------------------
// Column importance
//
// Shuffle one source column's values between rows (all the features made
// from it move together: one-hot slots, date parts, the blank flag) and
// measure how much worse the model gets. Deterministic for a given seed:
// the amount of work depends on the shape of the data, never on timing.
// ---------------------------------------------------------------------------

struct LossScore {
    double loss  = 0.0;   // log-loss, or squared error / variance
    double score = 0.0;   // accuracy, or R²
};

LossScore loss_and_score(const models::Model& m, const models::Matrix& X,
                         const std::vector<int>& y_cls,
                         const std::vector<double>& y_reg) {
    LossScore r;
    if (X.rows == 0) return r;
    const std::size_t k = static_cast<std::size_t>(std::max(1, m.n_outputs()));
    std::vector<double> out(k);
    if (m.task() == models::Task::Classification) {
        double ll = 0.0;
        std::size_t correct = 0;
        for (std::size_t i = 0; i < X.rows; ++i) {
            m.predict(X.row(i), out.data());
            const int y = y_cls[i];
            ll += -std::log(std::max(out[static_cast<std::size_t>(y)], 1e-15));
            if (argmax_row(out) == y) ++correct;
        }
        r.loss  = ll / static_cast<double>(X.rows);
        r.score = static_cast<double>(correct) / static_cast<double>(X.rows);
        return r;
    }
    double mean = 0.0;
    for (std::size_t i = 0; i < X.rows; ++i) mean += y_reg[i];
    mean /= static_cast<double>(X.rows);
    double sse = 0.0, sst = 0.0;
    for (std::size_t i = 0; i < X.rows; ++i) {
        m.predict(X.row(i), out.data());
        sse += (out[0] - y_reg[i]) * (out[0] - y_reg[i]);
        sst += (y_reg[i] - mean) * (y_reg[i] - mean);
    }
    r.score = (sst > 0.0) ? 1.0 - sse / sst : 0.0;
    r.loss  = (sst > 0.0) ? sse / sst : sse / static_cast<double>(X.rows);
    return r;
}

std::vector<ColumnImportance> permutation_importance(
    const models::Model&            model,
    const models::Matrix&           X_all,
    const std::vector<int>&         y_cls_all,
    const std::vector<double>&      y_reg_all,
    const std::vector<std::string>& feature_sources,
    std::uint64_t                   seed,
    const CancelCb&                 cancel) {

    std::vector<ColumnImportance> out;
    const bool cls = (model.task() == models::Task::Classification);

    // Group feature indices by the CSV column they came from.
    std::vector<std::string>              names;
    std::vector<std::vector<std::size_t>> groups;
    const std::size_t width = std::min(feature_sources.size(), X_all.cols);
    for (std::size_t f = 0; f < width; ++f) {
        const auto it = std::find(names.begin(), names.end(), feature_sources[f]);
        if (it == names.end()) {
            names.push_back(feature_sources[f]);
            groups.push_back({f});
        } else {
            groups[static_cast<std::size_t>(it - names.begin())].push_back(f);
        }
    }
    if (names.empty() || X_all.rows < 10) return out;

    // Work grows with columns x repeats x rows; keep it to a few hundred
    // thousand predictions.
    const std::size_t n_groups = names.size();
    const int repeats = (n_groups <= 30) ? 3 : (n_groups <= 100 ? 2 : 1);
    const std::size_t row_cap = std::clamp<std::size_t>(
        600000 / (n_groups * static_cast<std::size_t>(repeats)), 300, 3000);
    const std::size_t rows = std::min(X_all.rows, row_cap);

    std::mt19937_64 rng(seed);
    std::vector<std::size_t> pick(X_all.rows);
    std::iota(pick.begin(), pick.end(), std::size_t{0});
    if (rows < X_all.rows) {
        std::shuffle(pick.begin(), pick.end(), rng);
        pick.resize(rows);
        std::sort(pick.begin(), pick.end());
    }

    // Working copy; shuffled columns are written into it and restored.
    models::Matrix X;
    X.rows = rows;
    X.cols = X_all.cols;
    X.data.resize(rows * X.cols);
    std::vector<int>    y_cls;
    std::vector<double> y_reg;
    for (std::size_t i = 0; i < rows; ++i) {
        std::copy_n(X_all.row(pick[i]), X.cols, X.row(i));
        if (cls) y_cls.push_back(y_cls_all[pick[i]]);
        else     y_reg.push_back(y_reg_all[pick[i]]);
    }
    const LossScore base = loss_and_score(model, X, y_cls, y_reg);

    std::vector<std::size_t> perm(rows);
    std::vector<double>      saved;
    for (std::size_t g = 0; g < n_groups; ++g) {
        if (is_cancelled(cancel)) return {};
        const auto& feats = groups[g];
        const std::size_t nf = feats.size();
        saved.resize(rows * nf);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < nf; ++j)
                saved[i * nf + j] = X.row(i)[feats[j]];

        double loss_up = 0.0, score_down = 0.0;
        for (int r = 0; r < repeats; ++r) {
            std::iota(perm.begin(), perm.end(), std::size_t{0});
            std::shuffle(perm.begin(), perm.end(), rng);
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < nf; ++j)
                    X.row(i)[feats[j]] = saved[perm[i] * nf + j];
            const LossScore p = loss_and_score(model, X, y_cls, y_reg);
            loss_up    += p.loss - base.loss;
            score_down += base.score - p.score;
        }
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < nf; ++j)
                X.row(i)[feats[j]] = saved[i * nf + j];

        ColumnImportance ci;
        ci.column        = names[g];
        ci.loss_increase = loss_up / repeats;
        ci.score_drop    = score_down / repeats;
        out.push_back(std::move(ci));
    }

    double top = 0.0;
    for (const auto& c : out) top = std::max(top, c.loss_increase);
    for (auto& c : out) c.relative = (top > 0.0) ? std::max(0.0, c.loss_increase) / top : 0.0;
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.loss_increase > b.loss_increase;
    });
    return out;
}

// ---------------------------------------------------------------------------
// Bundle files
// ---------------------------------------------------------------------------

void save_model_v3(const models::Model& m, bool is_cls,
                   const std::vector<std::string>& class_labels,
                   const std::string& target_column, const fs::path& path) {
    nlohmann::json j;
    j["schema_version"] = kModelSchemaVersion;
    j["kind"]           = m.kind();
    j["task"]           = is_cls ? "classification" : "regression";
    j["n_features"]     = m.n_features();
    j["class_labels"]   = is_cls ? class_labels : std::vector<std::string>{};
    j["target_column"]  = target_column;
    j["trainer"]        = kTrainerName;
    j["description"]    = m.describe();
    j["model"]          = m.to_json();
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + to_utf8(path));
    f << j.dump();
    if (!f) throw std::runtime_error("write failed for " + to_utf8(path));
}

const char* task_name(preprocess::TaskChoice t) {
    switch (t) {
        case preprocess::TaskChoice::Classification: return "classification";
        case preprocess::TaskChoice::Regression:     return "regression";
        case preprocess::TaskChoice::Auto:           break;
    }
    return "auto";
}

nlohmann::json pipeline_config_to_json(const preprocess::PipelineConfig& c,
                                       const std::vector<preprocess::ColumnDecision>& cols) {
    nlohmann::json j;
    j["target_column"]   = c.target_column;
    j["val_fraction"]    = c.val_fraction;
    j["test_fraction"]   = c.test_fraction;
    j["stratify"]        = c.stratify;
    j["split_seed"]      = c.split_seed;
    j["exclude_columns"] = c.exclude_columns;
    j["task"]            = task_name(c.task);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& d : cols) {
        arr.push_back({{"name", d.name}, {"role", d.role}, {"note", d.note}});
    }
    j["columns"] = std::move(arr);
    return j;
}

nlohmann::json number_or_null(double v) {
    return std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(nullptr);
}

nlohmann::json training_config_to_json(const AutoOptions& opts, const AutoResult& res,
                                       const models::Model& model) {
    nlohmann::json j;
    j["trainer"]             = kTrainerName;
    j["time_budget_seconds"] = opts.time_budget_seconds;
    j["thorough"]            = opts.thorough;
    j["seed"]                = opts.seed;
    j["chosen"] = {
        {"name",        res.model_name},
        {"kind",        model.kind()},
        {"description", res.model_description},
        {"reason",      res.model_choice_reason},
        {"parameters",  model.parameter_count()},
    };
    nlohmann::json cands = nlohmann::json::array();
    for (const auto& c : res.leaderboard) {
        cands.push_back({
            {"name",            c.name},
            {"kind",            c.kind},
            {"detail",          c.detail},
            {"check_loss",      number_or_null(c.val_loss)},
            {"check_score",     number_or_null(c.val_score)},
            {"seconds",         c.seconds},
            {"iterations",      c.iterations},
            {"chosen",          c.chosen},
            {"stopped_by_time", c.stopped_by_time},
        });
    }
    j["candidates"] = std::move(cands);
    j["rows"] = {
        {"train",                res.n_train},
        {"check",                res.n_val},
        {"test",                 res.n_test},
        {"skipped_blank_target", res.rows_dropped_missing_target},
    };
    j["evaluated_on"] = res.evaluated_on;
    return j;
}

// ---------------------------------------------------------------------------
// Wording
// ---------------------------------------------------------------------------

std::string make_summary(const AutoResult& r, bool is_cls) {
    const std::string n = std::to_string(r.n_evaluated);
    const std::string where = (r.evaluated_on == "test")
        ? " on " + n + " rows held back from training"
        : " on " + n + " check rows (too few rows to hold back a separate test set)";
    std::string s = r.model_name + ": ";
    if (is_cls) {
        const auto& cm = std::get<eval::ClassificationMetrics>(r.metrics);
        s += "accuracy " + fmt("%.1f%%", cm.accuracy * 100.0) + where;
        for (const auto& b : r.baselines.baselines) {
            if (b.name == "majority") {
                s += " (always guessing the most common class: "
                   + fmt("%.1f%%", b.primary_metric * 100.0) + ")";
                break;
            }
        }
    } else {
        const auto& rm = std::get<eval::RegressionMetrics>(r.metrics);
        s += "R\xc2\xb2 " + fmt("%.3f", rm.r2) + where
           + " (typical error " + fmt("%.4g", rm.rmse) + ")";
    }
    s += ". " + r.baselines.verdict;
    return s;
}

std::string evaluated_on_phrase(const AutoResult& r) {
    if (r.evaluated_on == "test") {
        return std::to_string(r.n_evaluated)
             + " rows held back from training and from choosing the model";
    }
    return "the " + std::to_string(r.n_evaluated)
         + " check rows, which were also used to choose the model (the file was "
           "too small to hold back a separate test set), so these numbers are "
           "slightly optimistic";
}

} // namespace

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

util::Result<AutoResult>
run(const std::filesystem::path& csv,
    const std::string&           target_column,
    const AutoOptions&           opts,
    ProgressCb                   progress,
    CancelCb                     cancel) {

    const auto t_start = Clock::now();
    Reporter report(std::move(progress));
    AutoResult res;

    if (opts.do_export_bundle && opts.bundle_out_dir.empty()) {
        return fail("Bundle write failed: bundle_out_dir is empty.");
    }

    // --- 1. Scanning -------------------------------------------------------
    report(Stage::Scanning, 0.0, "Scanning the file");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");
    {
        dataqc::ScanOptions so;
        so.target_column = target_column;
        res.dq = dataqc::scan_csv(csv, so);
    }
    report(Stage::Scanning, 1.0, res.dq.summary_line);
    if (!res.dq.trainable) {
        std::string detail;
        for (const auto& f : res.dq.findings) {
            if (f.severity == dataqc::Severity::Error) { detail = f.message; break; }
        }
        return fail("Cannot train — " + (detail.empty() ? std::string("the data quality scan failed.")
                                                        : detail));
    }
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- 2. Preprocessing ----------------------------------------------------
    report(Stage::Preprocessing, 0.0, "Preparing the columns");
    preprocess::PipelineConfig pcfg;
    pcfg.target_column   = target_column;
    pcfg.split_seed      = opts.seed;
    pcfg.exclude_columns = opts.exclude_columns;
    pcfg.task            = opts.task;

    preprocess::TransformedDataset ds;
    std::unique_ptr<preprocess::FittedPipeline> pipe;
    try {
        pipe = preprocess::fit_pipeline(csv, pcfg, ds);
    } catch (const std::exception& e) {
        return fail(std::string("Cannot train — ") + e.what());
    }
    if (!pipe) return fail("Cannot train — the columns could not be prepared.");
    if (ds.X_train.empty()) return fail("Cannot train — no rows left to learn from.");
    if (ds.X_val.empty()) {
        return fail("Cannot train — too few rows to hold some back for checking "
                    "the model.");
    }
    res.columns                     = ds.columns;
    res.rows_dropped_missing_target = ds.rows_dropped_missing_target;
    res.n_train = static_cast<int>(ds.X_train.size());
    res.n_val   = static_cast<int>(ds.X_val.size());
    res.n_test  = static_cast<int>(ds.X_test.size());
    report(Stage::Preprocessing, 1.0,
           std::to_string(ds.feature_names_out.size()) + " model inputs from "
           + std::to_string(res.columns.size()) + " columns");
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- 3. Set up -----------------------------------------------------------
    const bool is_cls    = ds.is_classification;
    const int  n_classes = is_cls ? static_cast<int>(ds.class_labels.size()) : 0;
    report(Stage::AutoConfig, 0.0, "Setting up");
    models::TrainSet train_set =
        make_train_set(is_cls, n_classes, ds.X_train, ds.y_train, ds.X_val, ds.y_val);
    report(Stage::AutoConfig, 1.0,
           is_cls ? "Predicting one of " + std::to_string(n_classes) + " classes"
                  : std::string("Predicting a number"));
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- 4. Training ---------------------------------------------------------
    report(Stage::Training, 0.0, "Training models");
    const double budget = static_cast<double>(opts.time_budget_seconds);
    models::AutoFitResult fit;
    try {
        models::AutoFitOptions afo;
        afo.time_budget_seconds = (budget > 0.0)
            ? budget * (opts.do_learning_curve ? kTrainShareWithCurve : kTrainShareWithoutCurve)
            : 0.0;
        afo.seed     = opts.seed;
        afo.thorough = opts.thorough;
        fit = models::auto_fit(train_set, afo, cancel,
                               [&](double f, const std::string& msg) {
                                   report(Stage::Training, f, msg);
                               });
    } catch (const std::exception& e) {
        return fail(std::string("Training failed: ") + e.what());
    }
    if (fit.cancelled || is_cancelled(cancel)) return fail("Cancelled by user.");
    if (!fit.model) {
        return fail("Training failed: none of the model types could be trained "
                    "on this data.");
    }
    const models::Model& model = *fit.model;
    res.leaderboard       = fit.leaderboard;
    res.model_description = model.describe();
    res.model_choice_reason = fit.choice_reason;
    for (const auto& c : res.leaderboard) {
        if (c.chosen) { res.model_name = c.name; res.check_score = c.val_score; }
    }
    report(Stage::Training, 1.0, "Chose " + res.model_name + ": " + res.model_description);
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- 5. Evaluating -------------------------------------------------------
    report(Stage::Evaluating, 0.0, "Measuring accuracy on held-back rows");
    const bool have_test = !ds.X_test.empty();
    const auto& X_eval_rows = have_test ? ds.X_test : ds.X_val;
    const auto& y_eval_rows = have_test ? ds.y_test : ds.y_val;
    res.evaluated_on = have_test ? "test" : "check";
    res.n_evaluated  = static_cast<int>(X_eval_rows.size());

    const models::Matrix X_eval = models::Matrix::from_rows(X_eval_rows);
    const auto eval_preds = predict_rows(model, X_eval);
    try {
        if (is_cls) {
            res.metrics = eval::evaluate_classification(argmax_rows(y_eval_rows),
                                                        eval_preds, ds.class_labels);
        } else {
            res.metrics = eval::evaluate_regression(flatten_scalar(y_eval_rows),
                                                    flatten_scalar(eval_preds));
        }
    } catch (const std::exception& e) {
        return fail(std::string("Evaluation failed: ") + e.what());
    }
    report(Stage::Evaluating, 0.4, "Measuring which columns matter");

    // Importance on every held-back row (test and check together): more
    // rows give a steadier ranking, and it describes the model rather
    // than grading it.
    {
        std::vector<std::vector<double>> imp_rows = ds.X_test;
        imp_rows.insert(imp_rows.end(), ds.X_val.begin(), ds.X_val.end());
        std::vector<std::vector<double>> imp_y = ds.y_test;
        imp_y.insert(imp_y.end(), ds.y_val.begin(), ds.y_val.end());
        const models::Matrix X_imp = models::Matrix::from_rows(imp_rows);
        res.importance = permutation_importance(
            model, X_imp,
            is_cls ? argmax_rows(imp_y) : std::vector<int>{},
            is_cls ? std::vector<double>{} : flatten_scalar(imp_y),
            ds.feature_sources, opts.seed ^ 0x1234567ull, cancel);
    }
    if (is_cancelled(cancel)) return fail("Cancelled by user.");
    report(Stage::Evaluating, 1.0, "Evaluation complete");

    // --- 6. Baselines (same rows as the scores) ------------------------------
    report(Stage::Baselining, 0.0, "Comparing with simple guesses");
    res.baselines = eval::compare_to_baselines(ds.X_train, ds.y_train,
                                               X_eval_rows, y_eval_rows,
                                               eval_preds, is_cls, opts.seed);
    report(Stage::Baselining, 1.0, res.baselines.verdict);
    if (is_cancelled(cancel)) return fail("Cancelled by user.");

    // --- 7. Learning curve ---------------------------------------------------
    // Retrains the chosen recipe on 25%, 50% and 75% of the training rows;
    // the 100% point is the model already trained. Scored on the check rows.
    if (opts.do_learning_curve) {
        report(Stage::LearningCurve, 0.0, "Checking whether more data would help");
        const double elapsed = seconds_since(t_start);
        // 25% + 50% + 75% of one full fit, with some margin.
        const double need = fit.refit_seconds * 1.5 * 1.2;
        std::string skip;
        if (!fit.refit) {
            skip = "Skipped: the chosen model cannot be retrained on smaller samples.";
        } else if (fit.refit_stopped_by_time) {
            skip = "Skipped: training used its whole time allowance, so smaller "
                   "samples cannot be compared fairly. Choose a longer training "
                   "time to see this.";
        } else if (budget > 0.0 && need > 1.0 && elapsed + need > budget) {
            skip = "Skipped: not enough of the time allowance was left to retrain "
                   "the model on smaller samples. Choose a longer training time "
                   "to see this.";
        }

        if (!skip.empty()) {
            res.curve.message = skip;
            report(Stage::LearningCurve, 1.0, skip);
        } else {
            const std::size_t n_full = ds.X_train.size();
            const double full_train_score = models::evaluate_score(
                model, train_set.X_train, train_set.y_train_cls, train_set.y_train_reg);
            int points_done = 0;
            eval::TrainerFn trainer =
                [&](const std::vector<std::vector<double>>& X_sub,
                    const std::vector<std::vector<double>>& y_sub,
                    const std::vector<std::vector<double>>& X_v,
                    const std::vector<std::vector<double>>& y_v,
                    int /*epochs*/, std::uint64_t point_seed) -> std::pair<double, double> {
                std::pair<double, double> scores{0.0, 0.0};
                if (X_sub.size() == n_full) {
                    // The chosen model was trained on exactly these rows.
                    scores = {full_train_score, res.check_score};
                } else if (!is_cancelled(cancel)) {
                    const models::TrainSet t =
                        make_train_set(is_cls, n_classes, X_sub, y_sub, X_v, y_v);
                    models::FitControl c;
                    c.seed      = point_seed;
                    c.cancelled = cancel;
                    if (budget > 0.0) {
                        const double frac = static_cast<double>(X_sub.size())
                                          / static_cast<double>(n_full);
                        c.time_limit_seconds = std::max(1.0, fit.refit_seconds * 2.0 * frac + 0.5);
                    }
                    models::FitReport rep = fit.refit(t, c);
                    if (rep.model) {
                        scores = {models::evaluate_score(*rep.model, t.X_train,
                                                         t.y_train_cls, t.y_train_reg),
                                  models::evaluate_score(*rep.model, t.X_val,
                                                         t.y_val_cls, t.y_val_reg)};
                    }
                }
                ++points_done;
                report(Stage::LearningCurve, points_done / 4.0,
                       "Learning curve: " + std::to_string(X_sub.size()) + " rows");
                return scores;
            };
            eval::LearningCurveConfig lcc;
            lcc.seed = opts.seed;
            try {
                res.curve = eval::compute_learning_curve(ds, is_cls, lcc, trainer);
            } catch (const std::exception& e) {
                return fail(std::string("Learning curve failed: ") + e.what());
            }
            if (is_cancelled(cancel)) return fail("Cancelled by user.");
            report(Stage::LearningCurve, 1.0, res.curve.message);
        }
    } else {
        report(Stage::LearningCurve, 1.0, "Learning curve skipped");
    }

    res.summary_one_liner = make_summary(res, is_cls);

    // --- 8. Packaging --------------------------------------------------------
    report(Stage::Packaging, 0.0, "Writing the model files");
    if (opts.do_export_bundle) {
        std::error_code ec;
        fs::create_directories(opts.bundle_out_dir, ec);
        if (ec) {
            return fail("Bundle write failed: " + to_utf8(opts.bundle_out_dir)
                        + ": " + ec.message());
        }
        const auto pipe_path     = opts.bundle_out_dir / kBundlePipeline;
        const auto model_path    = opts.bundle_out_dir / kBundleModel;
        const auto manifest_path = opts.bundle_out_dir / kBundleManifest;
        const auto card_path     = opts.bundle_out_dir / kBundleCard;

        reporting::Manifest mf;
        try {
            preprocess::save_pipeline(*pipe, pipe_path);
            save_model_v3(model, is_cls, ds.class_labels, target_column, model_path);
            report(Stage::Packaging, 0.5, "Model written");

            // The training CSV is not copied into the bundle (privacy), so
            // its hash is left empty and verify_against skips it.
            mf.neurograph_version = kVersion;
            mf.build_git_sha      = "";
            mf.dataset_sha256     = "";
            mf.pipeline_sha256    = reporting::sha256_of_file(pipe_path);
            mf.model_sha256       = reporting::sha256_of_file(model_path);
            mf.rng_seed           = opts.seed;
            mf.timestamp_iso8601  = reporting::current_iso8601_utc();
            reporting::populate_host_info(mf);
            mf.pipeline_config = pipeline_config_to_json(pcfg, res.columns);
            mf.training_config = training_config_to_json(opts, res, model);
            reporting::save_manifest(mf, manifest_path);

            reporting::ModelCardInputs mc;
            mc.model_name           = "Model predicting '" + target_column + "'";
            mc.task_kind            = is_cls ? "classification" : "regression";
            mc.dq_report            = res.dq;
            mc.pipe_config          = pcfg;
            if (is_cls) mc.cls_metrics = std::get_if<eval::ClassificationMetrics>(&res.metrics);
            else        mc.reg_metrics = std::get_if<eval::RegressionMetrics>(&res.metrics);
            mc.baseline             = res.baselines;
            mc.curve                = res.curve;
            mc.neurograph_version   = mf.neurograph_version;
            mc.generated_at_iso8601 = mf.timestamp_iso8601;
            mc.dataset_path         = to_utf8(csv.filename());
            mc.model_description    = res.model_description;
            mc.model_choice_reason  = res.model_choice_reason;
            mc.candidates           = res.leaderboard;
            mc.columns              = res.columns;
            for (const auto& ci : res.importance) {
                mc.importance.push_back({ci.column, ci.relative, ci.score_drop});
            }
            mc.evaluated_on = evaluated_on_phrase(res);
            mc.n_train = res.n_train;
            mc.n_val   = res.n_val;
            mc.n_test  = res.n_test;
            mc.rows_dropped_missing_target = res.rows_dropped_missing_target;
            reporting::save_model_card(mc, card_path);
        } catch (const std::exception& e) {
            return fail(std::string("Bundle write failed: ") + e.what());
        }
        res.bundle_path = opts.bundle_out_dir;
        report(Stage::Packaging, 1.0, "Saved to " + to_utf8(opts.bundle_out_dir));
    } else {
        report(Stage::Packaging, 1.0, "Bundle export skipped");
    }

    report(Stage::Done, 1.0, res.summary_one_liner);
    return res;
}

} // namespace neurograph::orchestration
