// neurograph_train — train a model from a CSV file without the GUI.
//
//   neurograph_train --data <csv> --target <column> --out <bundle_dir>
//                    [--time <seconds>] [--seed <n>] [--exclude a,b,c]
//                    [--task auto|classification|regression]
//                    [--thorough] [--no-curve] [--json] [--quiet]
//
// Runs the same pipeline as the wizard's Train button and writes the same
// bundle, which neurograph_predict, the C API (neurograph_rt) and the
// wizard can all load.
//
// Output: a readable report on stdout (or one JSON object with --json);
// progress on stderr unless --quiet. Ctrl+C stops cleanly.
//
// Exit codes:
//   0    trained and saved
//   2    bad arguments
//   3    the data cannot be trained on (the message says why)
//   4    training failed
//   5    the bundle could not be written, or another error
//   130  stopped with Ctrl+C

#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/version.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace orch = neurograph::orchestration;
namespace eval = neurograph::eval;
using Clock = std::chrono::steady_clock;

constexpr int kExitOk         = 0;
constexpr int kExitBadArgs    = 2;
constexpr int kExitCannotTrain = 3;
constexpr int kExitTrainFail  = 4;
constexpr int kExitOther      = 5;
constexpr int kExitCancelled  = 130;

std::atomic<bool> g_stop{false};

extern "C" void on_sigint(int) { g_stop.store(true); }

struct Args {
    fs::path    data;
    std::string target;
    fs::path    out;
    int         time_seconds = 300;
    std::uint64_t seed       = 42;
    std::vector<std::string> exclude;
    neurograph::preprocess::TaskChoice task = neurograph::preprocess::TaskChoice::Auto;
    bool thorough = false;
    bool curve    = true;
    bool json     = false;
    bool quiet    = false;
};

void print_usage(std::ostream& os) {
    os <<
"Usage: neurograph_train --data <file.csv> --target <column> --out <folder>\n"
"                        [--time <seconds>] [--seed <n>] [--exclude a,b,c]\n"
"                        [--task auto|classification|regression]\n"
"                        [--thorough] [--no-curve] [--json] [--quiet]\n"
"\n"
"Trains several kinds of model on the file, keeps the best one, measures it\n"
"on rows held back from training, and saves it to --out for\n"
"neurograph_predict.\n"
"\n"
"  --time       Time allowance in seconds (default 300). Small files finish\n"
"               much sooner. 0 means no limit.\n"
"  --exclude    Columns to leave out, comma-separated.\n"
"  --task       Predict a category or a number (default: decided from the\n"
"               target column).\n"
"  --thorough   Also try larger models. Worth it with several minutes.\n"
"  --no-curve   Skip the 'would more data help?' check.\n"
"  --json       Print the result as one JSON object.\n"
"\n"
"Exit codes: 0 ok, 2 bad arguments, 3 the data cannot be trained on,\n"
"            4 training failed, 5 could not save, 130 stopped (Ctrl+C).\n";
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto b = item.find_first_not_of(" \t");
        const auto e = item.find_last_not_of(" \t");
        if (b != std::string::npos) out.push_back(item.substr(b, e - b + 1));
    }
    return out;
}

bool parse_args(int argc, char** argv, Args& a, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view k = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { err = std::string(name) + " needs a value"; return nullptr; }
            return argv[++i];
        };
        if (k == "--data") {
            const char* v = value("--data"); if (!v) return false;
            a.data = fs::path(v);
        } else if (k == "--target") {
            const char* v = value("--target"); if (!v) return false;
            a.target = v;
        } else if (k == "--out") {
            const char* v = value("--out"); if (!v) return false;
            a.out = fs::path(v);
        } else if (k == "--time") {
            const char* v = value("--time"); if (!v) return false;
            char* end = nullptr;
            const long t = std::strtol(v, &end, 10);
            if (!end || *end != '\0' || t < 0 || t > 7 * 24 * 3600) {
                err = "--time must be a whole number of seconds"; return false;
            }
            a.time_seconds = static_cast<int>(t);
        } else if (k == "--seed") {
            const char* v = value("--seed"); if (!v) return false;
            char* end = nullptr;
            a.seed = std::strtoull(v, &end, 10);
            if (!end || *end != '\0') { err = "--seed must be a whole number"; return false; }
        } else if (k == "--exclude") {
            const char* v = value("--exclude"); if (!v) return false;
            for (auto& c : split_commas(v)) a.exclude.push_back(std::move(c));
        } else if (k == "--task") {
            const char* v = value("--task"); if (!v) return false;
            const std::string_view t = v;
            if (t == "auto")                a.task = neurograph::preprocess::TaskChoice::Auto;
            else if (t == "classification") a.task = neurograph::preprocess::TaskChoice::Classification;
            else if (t == "regression")     a.task = neurograph::preprocess::TaskChoice::Regression;
            else { err = "--task must be auto, classification or regression"; return false; }
        } else if (k == "--thorough") {
            a.thorough = true;
        } else if (k == "--no-curve") {
            a.curve = false;
        } else if (k == "--json") {
            a.json = true;
        } else if (k == "--quiet") {
            a.quiet = true;
        } else if (k == "--version") {
            std::cout << "neurograph_train " << neurograph::kVersion << '\n';
            std::exit(kExitOk);
        } else if (k == "-h" || k == "--help") {
            print_usage(std::cout);
            std::exit(kExitOk);
        } else {
            err = "unknown argument: " + std::string(k);
            return false;
        }
    }
    if (a.data.empty())   { err = "--data is required";   return false; }
    if (a.target.empty()) { err = "--target is required"; return false; }
    if (a.out.empty())    { err = "--out is required";    return false; }
    return true;
}

std::string pct(double v)          { char b[32]; std::snprintf(b, sizeof b, "%.1f%%", v * 100.0); return b; }
std::string num(double v, int d)   { char b[48]; std::snprintf(b, sizeof b, "%.*f", d, v); return b; }
std::string secs(double v) {
    char b[32];
    if (v < 10.0) std::snprintf(b, sizeof b, "%.1f s", v);
    else          std::snprintf(b, sizeof b, "%.0f s", v);
    return b;
}
std::string utf8(const fs::path& p) { const auto u = p.u8string(); return std::string(u.begin(), u.end()); }

std::string pad(std::string s, std::size_t w) {
    if (s.size() < w) s.append(w - s.size(), ' ');
    return s;
}

const char* stage_name(orch::Stage s) {
    switch (s) {
        case orch::Stage::Scanning:      return "Scanning";
        case orch::Stage::Preprocessing: return "Preparing columns";
        case orch::Stage::AutoConfig:    return "Setting up";
        case orch::Stage::Training:      return "Training";
        case orch::Stage::Evaluating:    return "Evaluating";
        case orch::Stage::Baselining:    return "Comparing with baselines";
        case orch::Stage::LearningCurve: return "Learning curve";
        case orch::Stage::Packaging:     return "Saving";
        case orch::Stage::Done:          return "Done";
    }
    return "";
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

void print_report(const Args& a, const orch::AutoResult& r, double seconds) {
    const bool cls = std::holds_alternative<eval::ClassificationMetrics>(r.metrics);
    std::ostream& o = std::cout;

    o << "NeuroGraph " << neurograph::kVersion << ": " << utf8(a.data.filename())
      << ", predicting '" << a.target << "' ("
      << (cls ? "a category" : "a number") << ")\n\n";

    int used = 0, left_out = 0;
    for (const auto& c : r.columns) {
        if (c.role == "target") continue;
        (c.role == "left out" ? left_out : used) += 1;
    }
    o << "Columns: " << used << " used, " << left_out << " left out\n";
    for (const auto& c : r.columns) {
        if (c.role == "target") continue;
        o << "  " << pad(c.name, 24) << pad(c.role, 10);
        if (!c.note.empty()) o << c.note;
        o << '\n';
    }
    if (r.rows_dropped_missing_target > 0) {
        o << "  (" << r.rows_dropped_missing_target << " rows skipped: blank target)\n";
    }
    o << "Rows: " << r.n_train << " to train, " << r.n_val << " to compare models, "
      << r.n_test << " held back to measure the result\n\n";

    o << "Models tried (" << (cls ? "accuracy" : "R\xc2\xb2") << " on the "
      << r.n_val << " check rows):\n";
    for (const auto& c : r.leaderboard) {
        o << (c.chosen ? "* " : "  ") << pad(c.name, 32)
          << pad(cls ? pct(c.val_score) : num(c.val_score, 3), 9)
          << pad(c.seconds > 0.0 ? secs(c.seconds) : std::string("-"), 9)
          << c.detail << (c.stopped_by_time ? " (hit time limit)" : "") << '\n';
    }
    o << "Chosen: " << r.model_description << "\n";
    if (!r.model_choice_reason.empty()) o << "  " << r.model_choice_reason << '\n';
    o << '\n';

    o << "Result on " << r.n_evaluated
      << (r.evaluated_on == "test" ? " rows held back from training:\n"
                                   : " check rows (too few rows for a separate test set):\n");
    if (cls) {
        const auto& m = std::get<eval::ClassificationMetrics>(r.metrics);
        o << "  accuracy " << pct(m.accuracy) << ", macro F1 " << num(m.macro_f1, 3);
        if (std::isfinite(m.roc_auc_macro)) o << ", ROC AUC " << num(m.roc_auc_macro, 3);
        o << '\n';
        for (const auto& b : r.baselines.baselines) {
            if (b.name == "majority") {
                o << "  always guessing the most common class: " << pct(b.primary_metric) << '\n';
            }
        }
    } else {
        const auto& m = std::get<eval::RegressionMetrics>(r.metrics);
        o << "  R\xc2\xb2 " << num(m.r2, 3) << ", typical error (RMSE) " << num(m.rmse, 4)
          << ", average error (MAE) " << num(m.mae, 4) << '\n';
    }
    o << "  " << r.baselines.verdict << "\n\n";

    if (!r.importance.empty()) {
        o << "What the model relies on most:\n";
        for (std::size_t i = 0; i < r.importance.size() && i < 8; ++i) {
            const auto& c = r.importance[i];
            const int bars = static_cast<int>(c.relative * 20.0 + 0.5);
            o << "  " << pad(c.column, 24) << std::string(static_cast<std::size_t>(bars), '#')
              << '\n';
        }
        o << '\n';
    }
    if (!r.curve.message.empty()) o << "More data? " << r.curve.message << "\n\n";

    if (!r.bundle_path.empty()) {
        o << "Saved to " << utf8(r.bundle_path) << " (" << secs(seconds) << ")\n"
          << "Predict with: neurograph_predict --model \"" << utf8(r.bundle_path)
          << "\" --in new.csv --out predictions.csv\n";
    }
}

nlohmann::json finite_or_null(double v) {
    return std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(nullptr);
}

void print_json(const orch::AutoResult& r, double seconds) {
    const bool cls = std::holds_alternative<eval::ClassificationMetrics>(r.metrics);
    nlohmann::json j;
    j["ok"]   = true;
    j["task"] = cls ? "classification" : "regression";
    j["model"] = {{"name", r.model_name}, {"description", r.model_description},
                  {"reason", r.model_choice_reason}};
    nlohmann::json lb = nlohmann::json::array();
    for (const auto& c : r.leaderboard) {
        lb.push_back({{"name", c.name}, {"kind", c.kind}, {"detail", c.detail},
                      {"check_score", finite_or_null(c.val_score)},
                      {"check_loss", finite_or_null(c.val_loss)},
                      {"seconds", c.seconds}, {"chosen", c.chosen},
                      {"stopped_by_time", c.stopped_by_time}});
    }
    j["leaderboard"]  = std::move(lb);
    j["evaluated_on"] = r.evaluated_on;
    j["n_evaluated"]  = r.n_evaluated;
    j["rows"] = {{"train", r.n_train}, {"check", r.n_val}, {"test", r.n_test},
                 {"skipped_blank_target", r.rows_dropped_missing_target}};
    if (cls) {
        const auto& m = std::get<eval::ClassificationMetrics>(r.metrics);
        j["metrics"] = {{"accuracy", m.accuracy}, {"macro_f1", finite_or_null(m.macro_f1)},
                        {"roc_auc_macro", finite_or_null(m.roc_auc_macro)},
                        {"brier", finite_or_null(m.brier_score)}};
    } else {
        const auto& m = std::get<eval::RegressionMetrics>(r.metrics);
        j["metrics"] = {{"r2", finite_or_null(m.r2)}, {"rmse", finite_or_null(m.rmse)},
                        {"mae", finite_or_null(m.mae)}};
    }
    nlohmann::json bl = nlohmann::json::array();
    for (const auto& b : r.baselines.baselines) {
        bl.push_back({{"name", b.name}, {"primary", finite_or_null(b.primary_metric)}});
    }
    j["baselines"] = std::move(bl);
    j["verdict"]   = r.baselines.verdict;
    nlohmann::json imp = nlohmann::json::array();
    for (const auto& c : r.importance) {
        imp.push_back({{"column", c.column}, {"relative", c.relative},
                       {"score_drop", c.score_drop}});
    }
    j["importance"] = std::move(imp);
    nlohmann::json pts = nlohmann::json::array();
    for (const auto& p : r.curve.points) {
        pts.push_back({{"rows", p.n_samples}, {"train", p.train_metric}, {"check", p.val_metric}});
    }
    j["learning_curve"] = {{"message", r.curve.message}, {"points", std::move(pts)}};
    j["seconds"] = seconds;
    j["bundle"]  = utf8(r.bundle_path);
    std::cout << j.dump(2) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    std::string err;
    if (!parse_args(argc, argv, args, err)) {
        std::cerr << "neurograph_train: " << err << "\n\n";
        print_usage(std::cerr);
        return kExitBadArgs;
    }
    std::signal(SIGINT, on_sigint);

    orch::AutoOptions opts;
    opts.time_budget_seconds = args.time_seconds;
    opts.do_learning_curve   = args.curve;
    opts.do_export_bundle    = true;
    opts.bundle_out_dir      = args.out;
    opts.seed                = args.seed;
    opts.exclude_columns     = args.exclude;
    opts.task                = args.task;
    opts.thorough            = args.thorough;

    // Progress: a line when the stage changes, and training news at most
    // once a second.
    orch::Stage last_stage = orch::Stage::Done;
    auto last_line = Clock::now() - std::chrono::seconds(10);
    orch::ProgressCb progress;
    if (!args.quiet) {
        progress = [&](const orch::Progress& p) {
            const auto now = Clock::now();
            const bool new_stage = (p.stage != last_stage);
            if (!new_stage && now - last_line < std::chrono::seconds(1)) return;
            last_stage = p.stage;
            last_line  = now;
            std::fprintf(stderr, "[%3d%%] %s: %s\n",
                         static_cast<int>(p.overall_fraction * 100.0 + 0.5),
                         stage_name(p.stage), p.message.c_str());
        };
    }

    const auto t0 = Clock::now();
    auto r = orch::run(args.data, args.target, opts, progress,
                       [] { return g_stop.load(); });
    const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();

    if (!r.ok()) {
        const std::string& msg = r.error().message;
        if (args.json) {
            std::cout << nlohmann::json{{"ok", false}, {"error", msg}}.dump(2) << '\n';
        } else {
            std::cerr << "neurograph_train: " << msg << '\n';
        }
        if (g_stop.load() || msg.rfind("Cancelled", 0) == 0) return kExitCancelled;
        if (msg.rfind("Cannot train", 0) == 0)                return kExitCannotTrain;
        if (msg.rfind("Training failed", 0) == 0)             return kExitTrainFail;
        return kExitOther;
    }

    if (args.json) print_json(r.value(), seconds);
    else           print_report(args, r.value(), seconds);
    return kExitOk;
}
