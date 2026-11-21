// P1-07 — `predict.exe` entry point.
//
// Self-contained CLI that loads a model bundle and emits predictions.
// Wired so end users can drop the exe + their bundle dir on a clean
// Windows machine and get inference without Python, vcredist install,
// or any other runtime hoop.
//
// Usage:
//
//   predict --model <bundle_dir> [--in <csv|->] [--out <csv|->]
//           [--format csv|json] [--top-k N] [--allow-missing]
//
//     --model     Required. Directory containing pipeline.ngpipe + model.ngm.
//     --in        Default "-" (stdin). CSV file with a header row naming
//                 every column the model uses (extra columns, including
//                 the target, are ignored). A missing or misspelt column
//                 stops the run with exit code 4 and the names, because
//                 filling a whole column with the training average gives
//                 confident nonsense.
//     --allow-missing
//                 Fill missing columns with the training average / most
//                 common value instead of stopping. A warning is printed.
//     --out       Default "-" (stdout). CSV file where predictions are
//                 written. Directory is created on demand.
//     --format    Output format. csv (default) emits a CSV with
//                 "predicted_class" + per-class prob columns for
//                 classification; "predicted_value" for regression.
//                 json emits one JSON object per line (JSONL).
//     --top-k     Classification only. When N>1, an additional
//                 "top_k_classes" / "top_k_probs" pair is included.
//                 Default 1.
//
// Exit codes (per spec):
//   0  success
//   2  bad args
//   3  model load failed
//   4  input parse failed
//   5  internal error
//
// Hand-rolled arg parsing intentionally — pulling cxxopts/CLI11 from
// vcpkg for five flags is overkill, and it keeps `predict.exe`
// trivially statically-linkable.

#include "neurograph/runtime/bundle.hpp"
#include "neurograph/runtime/score.hpp"
#include "neurograph/util/text_parse.hpp"
#include "neurograph/version.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using neurograph::runtime::load_bundle;
using neurograph::runtime::ModelBundle;

namespace {

namespace fs = std::filesystem;

constexpr int kExitOk            = 0;
constexpr int kExitBadArgs       = 2;
constexpr int kExitModelLoadFail = 3;
constexpr int kExitInputParseErr = 4;
constexpr int kExitInternal      = 5;

struct CliArgs {
    fs::path     model_dir;
    std::string  in_path  = "-";
    std::string  out_path = "-";
    std::string  format   = "csv";
    int          top_k    = 1;
    bool         allow_missing = false;
};

void print_usage(std::ostream& os) {
    os <<
"Usage: predict --model <dir> [--in <csv|-> ] [--out <csv|-> ]\n"
"               [--format csv|json] [--top-k N] [--allow-missing]\n"
"\n"
"Read prediction inputs from --in (CSV with header), write predictions\n"
"to --out. Use '-' for stdin/stdout (pipeline-friendly).\n"
"\n"
"Every column the model uses must be in the header. --allow-missing fills\n"
"absent columns with the training average or most common value instead\n"
"of stopping.\n"
"\n"
"Exit codes: 0 ok, 2 bad args, 3 model load failed,\n"
"            4 input parse failed, 5 internal error.\n";
}

bool parse_int(const char* s, int& out) {
    try {
        std::size_t end = 0;
        out = std::stoi(s, &end);
        return end == std::strlen(s);
    } catch (...) { return false; }
}

bool parse_args(int argc, char** argv, CliArgs& a, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view k = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                err = std::string(name) + " requires a value";
                return nullptr;
            }
            return argv[++i];
        };
        if (k == "--model") {
            const auto* v = need_val("--model"); if (!v) return false;
            a.model_dir = v;
        } else if (k == "--in") {
            const auto* v = need_val("--in"); if (!v) return false;
            a.in_path = v;
        } else if (k == "--out") {
            const auto* v = need_val("--out"); if (!v) return false;
            a.out_path = v;
        } else if (k == "--format") {
            const auto* v = need_val("--format"); if (!v) return false;
            a.format = v;
            if (a.format != "csv" && a.format != "json") {
                err = "--format must be csv or json"; return false;
            }
        } else if (k == "--top-k") {
            const auto* v = need_val("--top-k"); if (!v) return false;
            if (!parse_int(v, a.top_k) || a.top_k < 1) {
                err = "--top-k must be a positive integer"; return false;
            }
        } else if (k == "--allow-missing") {
            a.allow_missing = true;
        } else if (k == "--version") {
            std::cout << "neurograph_predict " << neurograph::kVersion << '\n';
            std::exit(kExitOk);
        } else if (k == "-h" || k == "--help") {
            print_usage(std::cout);
            std::exit(kExitOk);
        } else {
            err = "unknown argument: " + std::string(k); return false;
        }
    }
    if (a.model_dir.empty()) { err = "--model is required"; return false; }
    return true;
}

// Open `path` for input. "-" means stdin. Returns nullptr if open fails.
std::unique_ptr<std::istream> open_in(const std::string& path,
                                       std::string& err) {
    if (path == "-") {
        // Wrap stdin in an istream that doesn't own it.
        struct CinHolder : std::istream {
            CinHolder() : std::istream(std::cin.rdbuf()) {}
        };
        return std::make_unique<CinHolder>();
    }
    auto f = std::make_unique<std::ifstream>(path);
    if (!f->good()) {
        err = "cannot open input '" + path + "'";
        return nullptr;
    }
    return f;
}

// Open `path` for output. "-" means stdout. Creates parent dirs if needed.
std::unique_ptr<std::ostream> open_out(const std::string& path,
                                       std::string& err) {
    if (path == "-") {
        struct CoutHolder : std::ostream {
            CoutHolder() : std::ostream(std::cout.rdbuf()) {}
        };
        return std::make_unique<CoutHolder>();
    }
    const fs::path p = path;
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec) {
            err = "cannot create dir " + p.parent_path().string()
                + ": " + ec.message();
            return nullptr;
        }
    }
    auto f = std::make_unique<std::ofstream>(path);
    if (!f->good()) {
        err = "cannot open output '" + path + "'";
        return nullptr;
    }
    return f;
}

} // namespace

int main(int argc, char** argv) {
    CliArgs args;
    std::string err;
    if (!parse_args(argc, argv, args, err)) {
        std::cerr << "predict: " << err << "\n\n";
        print_usage(std::cerr);
        return kExitBadArgs;
    }

    auto br = load_bundle(args.model_dir);
    if (!br.ok()) {
        std::cerr << "predict: " << br.error().message << '\n';
        return kExitModelLoadFail;
    }
    const auto bundle = std::move(br).value();

    auto in = open_in(args.in_path, err);
    if (!in) { std::cerr << "predict: " << err << '\n'; return kExitInputParseErr; }
    auto out = open_out(args.out_path, err);
    if (!out) { std::cerr << "predict: " << err << '\n'; return kExitInternal; }

    neurograph::runtime::ScoreOptions so;
    so.format        = args.format;
    so.top_k         = args.top_k;
    so.allow_missing = args.allow_missing;
    auto sr = neurograph::runtime::score_stream(bundle, *in, *out, so);
    if (!sr.ok()) {
        std::cerr << "predict: " << sr.error().message << '\n';
        return sr.error().code == neurograph::runtime::kScoreInputError
             ? kExitInputParseErr : kExitInternal;
    }
    const auto& filled = sr.value().filled_columns;
    if (!filled.empty()) {
        std::string names;
        for (const auto& m : filled) names += (names.empty() ? "'" : ", '") + m + "'";
        std::cerr << "predict: warning: no column " << names
                  << " in the input; filled with the training average or most "
                     "common value, so these predictions are less accurate.\n";
    }
    return kExitOk;
}
