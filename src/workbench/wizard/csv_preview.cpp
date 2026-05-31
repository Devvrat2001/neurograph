// P1-09 — CSV peek implementation.
//
// The "real" CSV parser (with quoted-comma + RFC-4180 edge cases)
// lives in runtime/bundle.cpp. This is a simpler variant — we only
// need it for the preview table, so "split on commas, strip matched
// double-quotes on a field, handle doubled quotes inside, strip
// trailing \r" is enough.
//
// Errors go through util::Result with messages suitable for display.
// Failures beyond a certain point in the file degrade gracefully —
// e.g. a malformed row after the preview is counted, not rejected.

#include "neurograph/workbench/csv_preview.hpp"

#include <fstream>
#include <sstream>

namespace neurograph::workbench {

namespace {

// Split one CSV line into fields. Deliberately minimal — doesn't
// handle the escape-newline-inside-quotes case. If the real dataset
// uses embedded newlines, the full parser downstream will flag it
// via the DQ scanner; we just want a viable preview.
std::vector<std::string> split_csv_line(std::string_view line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                // RFC-4180 doubled-quote escape.
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == ',') {
                out.push_back(std::move(cur));
                cur.clear();
            } else if (c == '"' && cur.empty()) {
                // Only enter quote mode at the start of a field;
                // a naked `"` in the middle is just a literal char.
                in_quotes = true;
            } else {
                cur.push_back(c);
            }
        }
    }
    out.push_back(std::move(cur));
    return out;
}

void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

} // namespace

util::Result<CsvMetadata>
peek_csv_metadata(const std::filesystem::path& path,
                  std::size_t                  preview_rows) {
    std::ifstream f(path);
    if (!f) {
        return util::Error("cannot open file: " + path.string());
    }

    CsvMetadata meta;
    meta.path = path;

    std::string header_line;
    if (!std::getline(f, header_line)) {
        return util::Error("file is empty: " + path.string());
    }
    strip_cr(header_line);
    meta.columns = split_csv_line(header_line);
    if (meta.columns.empty()) {
        return util::Error("header has no columns");
    }

    std::string line;
    while (std::getline(f, line)) {
        strip_cr(line);
        if (line.empty()) continue;   // tolerate trailing blank lines
        if (meta.preview_rows.size() < preview_rows) {
            meta.preview_rows.push_back(split_csv_line(line));
        }
        ++meta.row_count;
    }

    return meta;
}

} // namespace neurograph::workbench
