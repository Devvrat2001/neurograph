#pragma once

// Score a CSV file with a loaded bundle: read rows, predict, write one
// output row per input row. Shared by neurograph_predict and the wizard's
// "Score a file" button, so both behave identically.
//
// Input rules:
//   - The first line is a header. A UTF-8 byte-order mark and spaces
//     around names are ignored.
//   - Every column the model uses must be present (extra columns,
//     including the target, are ignored). Missing columns are refused
//     unless `allow_missing` is set, in which case they are filled with
//     the training average / most common value.
//   - Cells are passed on as text, so numbers are read by the same parser
//     training used ("$1,200", "45%", ...).
//
// Output columns:
//   classification: predicted_class, p_<class> for each class, and with
//                   top_k > 1, top<i>_class / top<i>_prob pairs
//   regression:     predicted_value
// "json" writes one JSON object per line instead.

#include "neurograph/runtime/bundle.hpp"
#include "neurograph/util/result.hpp"

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace neurograph::runtime {

struct ScoreOptions {
    std::string format        = "csv";   // "csv" or "json"
    int         top_k         = 1;       // classification only
    bool        allow_missing = false;
};

struct ScoreSummary {
    std::size_t              rows = 0;              // predictions written
    std::vector<std::string> filled_columns;        // missing, filled (allow_missing)
};

/// Error codes carried in util::Error::code.
constexpr int kScoreInputError    = 4;   // empty input, missing columns, bad header
constexpr int kScoreInternalError = 5;   // prediction threw, output failed

util::Result<ScoreSummary> score_stream(const ModelBundle& bundle,
                                        std::istream& in, std::ostream& out,
                                        const ScoreOptions& opts = {});

/// File-to-file convenience. Creates the output folder if needed.
util::Result<ScoreSummary> score_file(const ModelBundle& bundle,
                                      const std::filesystem::path& in_csv,
                                      const std::filesystem::path& out_file,
                                      const ScoreOptions& opts = {});

} // namespace neurograph::runtime
