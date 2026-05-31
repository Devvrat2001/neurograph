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
//           [--format csv|json] [--top-k N]
//
//     --model     Required. Directory containing pipeline.ngpipe + model.ngm.
//     --in        Default "-" (stdin). CSV file with header row that
//                 names every column the original training CSV had
//                 (target column is allowed but ignored).
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
using neurograph::runtime::Prediction;
using neurograph::runtime::predict_row;
using neurograph::runtime::quote_csv_field;
using neurograph::runtime::split_csv_line;

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
};

void print_usage(std::ostream& os) {
    os <<
"Usage: predict --model <dir> [--in <csv|-> ] [--out <csv|-> ]\n"
"               [--format csv|json] [--top-k N]\n"
"\n"
"Read prediction inputs from --in (CSV with header), write predictions\n"
"to --out. Use '-' for stdin/stdout (pipeline-friendly).\n"
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

// Convert a CSV row (header + values) into a JSON object. Numeric
// strings become numbers, everything else stays a string. Matches
// what `apply_pipeline` expects.
nlohmann::json row_to_json(const std::vector<std::string>& header,
                           const std::vector<std::string>& fields) {
    nlohmann::json j = nlohmann::json::object();
    const std::size_t n = std::min(header.size(), fields.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& s = fields[i];
        if (s.empty()) {
            j[header[i]] = nullptr;
            continue;
        }
        try {
            std::size_t end = 0;
            const double d = std::stod(s, &end);
            if (end == s.size()) { j[header[i]] = d; continue; }
        } catch (...) { /* fall through to string */ }
        j[header[i]] = s;
    }
    return j;
}

// Top-K (class index, prob) sorted descending.
std::vector<std::pair<int, double>> topk(const std::vector<double>& p, int k) {
    std::vector<int> idx(p.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(),
                      idx.begin() + std::min<int>(k, static_cast<int>(p.size())),
                      idx.end(),
                      [&](int a, int b) { return p[a] > p[b]; });
    std::vector<std::pair<int, double>> out;
    out.reserve(std::min<int>(k, static_cast<int>(p.size())));
    for (int i = 0; i < k && i < static_cast<int>(p.size()); ++i) {
        out.emplace_back(idx[i], p[idx[i]]);
    }
    return out;
}

void write_csv_header(std::ostream& os, const ModelBundle& b, int top_k) {
    if (b.is_classification()) {
        os << "predicted_class";
        const int K = b.n_classes();
        for (int k = 0; k < K; ++k) {
            const auto lab = b.class_label(k);
            os << ',' << quote_csv_field(
                "p_" + (lab.empty() ? std::to_string(k) : lab));
        }
        if (top_k > 1) {
            for (int i = 0; i < top_k; ++i) {
                os << ",top" << (i + 1) << "_class,top" << (i + 1) << "_prob";
            }
        }
    } else {
        os << "predicted_value";
    }
    os << '\n';
}

void write_csv_row(std::ostream& os, const ModelBundle& b,
                   const Prediction& pr, int top_k) {
    if (b.is_classification()) {
        const auto lab = b.class_label(pr.predicted_class);
        os << quote_csv_field(lab.empty()
                                ? std::to_string(pr.predicted_class)
                                : lab);
        for (double p : pr.probabilities) os << ',' << p;
        if (top_k > 1) {
            const auto tops = topk(pr.probabilities, top_k);
            for (int i = 0; i < top_k; ++i) {
                if (i < static_cast<int>(tops.size())) {
                    const auto cl = b.class_label(tops[i].first);
                    os << ',' << quote_csv_field(
                        cl.empty() ? std::to_string(tops[i].first) : cl)
                       << ',' << tops[i].second;
                } else {
                    os << ",,";
                }
            }
        }
    } else {
        os << pr.value;
    }
    os << '\n';
}

void write_json_row(std::ostream& os, const ModelBundle& b,
                    const Prediction& pr, int top_k) {
    nlohmann::json out = nlohmann::json::object();
    if (b.is_classification()) {
        out["predicted_class_id"] = pr.predicted_class;
        out["predicted_class"]    = b.class_label(pr.predicted_class);
        out["probabilities"]      = pr.probabilities;
        if (top_k > 1) {
            const auto tops = topk(pr.probabilities, top_k);
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& [i, p] : tops) {
                arr.push_back({{"class", b.class_label(i)},
                               {"class_id", i},
                               {"prob", p}});
            }
            out["top_k"] = std::move(arr);
        }
    } else {
        out["predicted_value"] = pr.value;
    }
    os << out.dump() << '\n';
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

    // Read header.
    std::string line;
    if (!std::getline(*in, line)) {
        std::cerr << "predict: input is empty (need a CSV header)\n";
        return kExitInputParseErr;
    }
    const auto header = split_csv_line(line);
    if (header.empty()) {
        std::cerr << "predict: input header is empty\n";
        return kExitInputParseErr;
    }

    if (args.format == "csv") write_csv_header(*out, bundle, args.top_k);

    std::size_t row_no = 1;   // 1-based for error messages
    while (std::getline(*in, line)) {
        ++row_no;
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        const auto j      = row_to_json(header, fields);
        Prediction pr;
        try {
            pr = predict_row(bundle, j);
        } catch (const std::exception& e) {
            std::cerr << "predict: row " << row_no << ": " << e.what() << '\n';
            return kExitInternal;
        }
        if (args.format == "csv") write_csv_row(*out, bundle, pr, args.top_k);
        else                       write_json_row(*out, bundle, pr, args.top_k);
    }
    out->flush();
    return kExitOk;
}
