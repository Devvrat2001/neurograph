#include "neurograph/runtime/score.hpp"

#include "neurograph/util/text_parse.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <exception>
#include <fstream>
#include <numeric>
#include <ostream>
#include <istream>
#include <utility>

namespace neurograph::runtime {

namespace {

namespace fs = std::filesystem;

// Cells stay text: apply_pipeline reads numbers with the same parser
// training used, and a category such as "1.50" or "007" is not turned
// into 1.5 or 7 on the way.
nlohmann::json row_to_json(const std::vector<std::string>& header,
                           const std::vector<std::string>& fields) {
    nlohmann::json j = nlohmann::json::object();
    const std::size_t n = std::min(header.size(), fields.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (fields[i].empty()) j[header[i]] = nullptr;
        else                   j[header[i]] = fields[i];
    }
    return j;
}

// Top-K (class index, probability), highest first.
std::vector<std::pair<int, double>> topk(const std::vector<double>& p, int k) {
    std::vector<int> idx(p.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int kk = std::min<int>(k, static_cast<int>(p.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return p[a] > p[b]; });
    std::vector<std::pair<int, double>> out;
    out.reserve(static_cast<std::size_t>(kk));
    for (int i = 0; i < kk; ++i) out.emplace_back(idx[i], p[idx[i]]);
    return out;
}

std::string label_or_index(const ModelBundle& b, int i) {
    const auto lab = b.class_label(i);
    return lab.empty() ? std::to_string(i) : lab;
}

void write_csv_header(std::ostream& os, const ModelBundle& b, int top_k) {
    if (b.is_classification()) {
        os << "predicted_class";
        for (int k = 0; k < b.n_classes(); ++k) {
            os << ',' << quote_csv_field("p_" + label_or_index(b, k));
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
        os << quote_csv_field(label_or_index(b, pr.predicted_class));
        for (double p : pr.probabilities) os << ',' << p;
        if (top_k > 1) {
            const auto tops = topk(pr.probabilities, top_k);
            for (int i = 0; i < top_k; ++i) {
                if (i < static_cast<int>(tops.size())) {
                    os << ',' << quote_csv_field(label_or_index(b, tops[i].first))
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
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& [i, p] : topk(pr.probabilities, top_k)) {
                arr.push_back({{"class", b.class_label(i)}, {"class_id", i}, {"prob", p}});
            }
            out["top_k"] = std::move(arr);
        }
    } else {
        out["predicted_value"] = pr.value;
    }
    os << out.dump() << '\n';
}

util::Error input_error(std::string msg)    { return util::Error(std::move(msg), kScoreInputError); }
util::Error internal_error(std::string msg) { return util::Error(std::move(msg), kScoreInternalError); }

} // namespace

util::Result<ScoreSummary> score_stream(const ModelBundle& bundle,
                                        std::istream& in, std::ostream& out,
                                        const ScoreOptions& opts) {
    ScoreSummary summary;
    const bool json = (opts.format == "json");
    const int  top_k = std::max(1, opts.top_k);

    std::string line;
    if (!std::getline(in, line)) {
        return input_error("the input is empty (it needs a header row)");
    }
    util::strip_bom(line);   // Excel's "CSV UTF-8" writes one
    util::strip_cr(line);
    auto header = split_csv_line(line);
    for (auto& h : header) h = util::trim(std::move(h));
    if (header.empty() || (header.size() == 1 && header[0].empty())) {
        return input_error("the input's header row is empty");
    }

    const auto missing = missing_input_columns(bundle, header);
    if (!missing.empty()) {
        std::string names;
        for (const auto& m : missing) names += (names.empty() ? "'" : ", '") + m + "'";
        if (!opts.allow_missing) {
            return input_error(
                "the input has no column " + names + ", which the model needs. "
                "Check the spelling against the training file, or allow missing "
                "columns to fill " + std::string(missing.size() == 1 ? "it" : "them") +
                " with the training average or most common value");
        }
        summary.filled_columns = missing;
    }

    if (!json) write_csv_header(out, bundle, top_k);
    std::size_t row_no = 1;
    while (std::getline(in, line)) {
        ++row_no;
        util::strip_cr(line);
        if (line.empty()) continue;
        Prediction pr;
        try {
            pr = predict_row(bundle, row_to_json(header, split_csv_line(line)));
        } catch (const std::exception& e) {
            return internal_error("row " + std::to_string(row_no) + ": " + e.what());
        }
        if (json) write_json_row(out, bundle, pr, top_k);
        else      write_csv_row(out, bundle, pr, top_k);
        ++summary.rows;
    }
    out.flush();
    if (!out) return internal_error("could not write the output");
    return summary;
}

util::Result<ScoreSummary> score_file(const ModelBundle& bundle,
                                      const fs::path& in_csv,
                                      const fs::path& out_file,
                                      const ScoreOptions& opts) {
    std::ifstream in(in_csv, std::ios::binary);
    if (!in) return input_error("cannot open the input file");
    if (out_file.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(out_file.parent_path(), ec);
        if (ec) return internal_error("cannot create the output folder: " + ec.message());
    }
    std::ofstream out(out_file, std::ios::binary);
    if (!out) return internal_error("cannot open the output file for writing");
    return score_stream(bundle, in, out, opts);
}

} // namespace neurograph::runtime
