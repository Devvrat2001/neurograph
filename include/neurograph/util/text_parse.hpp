#pragma once

// Text → value parsing shared by the data scanner, the preprocessing
// pipeline and the prediction path. One implementation, so a value that
// counted as a number while training is read the same way when scoring.
//
// Header-only; no dependencies beyond the standard library.

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace neurograph::util {

// ---------------------------------------------------------------------------
// CSV lines
// ---------------------------------------------------------------------------

/// Remove a UTF-8 byte-order mark from the start of the first line.
inline void strip_bom(std::string& s) {
    if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF
        && static_cast<unsigned char>(s[1]) == 0xBB
        && static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }
}

/// Remove one trailing '\r' (CRLF files read with std::getline).
inline void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

/// Split one CSV line into fields: quoted fields, embedded commas and
/// doubled-quote escapes are handled; a field may not contain a newline.
inline std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
                else in_quotes = false;
            } else {
                cur += c;
            }
        } else if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else if (c == '"' && cur.empty()) {
            in_quotes = true;
        } else {
            cur += c;
        }
    }
    fields.push_back(cur);
    return fields;
}

inline std::string_view trim_view(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
    return s;
}

inline std::string trim(std::string s) {
    return std::string(trim_view(s));
}

inline bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i]))
            != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

struct NumericParse {
    double value      = 0.0;
    bool   parsed     = false;   // a number was read
    bool   is_missing = false;   // blank or a "no value" marker
};

/// True for blanks and the usual "no value" markers, including the error
/// cells Excel writes into exported files.
inline bool is_missing_token(std::string_view s) {
    s = trim_view(s);
    if (s.empty()) return true;
    for (std::string_view tok : {"nan", "na", "n/a", "null", "none", "?", "-",
                                 "#n/a", "#div/0!", "#value!", "#ref!", "#num!",
                                 "#null!", "#name?"}) {
        if (iequals(s, tok)) return true;
    }
    return false;
}

namespace detail {

inline bool from_chars_full(std::string_view s, double& out) {
    if (s.empty()) return false;
    const char* first = s.data();
    const char* last  = s.data() + s.size();
    const auto r = std::from_chars(first, last, out);
    return r.ec == std::errc{} && r.ptr == last;
}

/// Strip one leading currency marker. Returns true if one was removed.
inline bool strip_currency(std::string_view& s) {
    static constexpr std::string_view symbols[] = {
        "$", "\xE2\x82\xAC" /* euro */, "\xC2\xA3" /* pound */,
        "\xE2\x82\xB9" /* rupee */, "\xC2\xA5" /* yen */,
    };
    for (std::string_view sym : symbols) {
        if (s.size() > sym.size() && s.substr(0, sym.size()) == sym) {
            s.remove_prefix(sym.size());
            s = trim_view(s);
            return true;
        }
    }
    static constexpr std::string_view words[] = {"rs.", "rs", "inr", "usd", "eur", "gbp"};
    for (std::string_view w : words) {
        if (s.size() > w.size() && iequals(s.substr(0, w.size()), w)) {
            const char next = s[w.size()];
            if (next == ' ' || std::isdigit(static_cast<unsigned char>(next))
                || next == '-' || next == '.') {
                s.remove_prefix(w.size());
                s = trim_view(s);
                return true;
            }
        }
    }
    return false;
}

/// Remove thousands separators from "1,234,567.89" or "12,34,567" (Indian
/// grouping). Refuses anything that is not unambiguous grouping, such as
/// "12,34", which in much of Europe means 12.34.
inline bool remove_grouping(std::string_view s, std::string& out) {
    const std::size_t dot = s.find('.');
    const std::string_view int_part = s.substr(0, dot);
    if (int_part.find(',') == std::string_view::npos) return false;
    if (dot != std::string_view::npos && s.find(',', dot) != std::string_view::npos) return false;

    // Split the integer part on commas and check the group sizes.
    std::vector<std::string_view> groups;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = int_part.find(',', start);
        groups.push_back(int_part.substr(start, comma == std::string_view::npos
                                                    ? std::string_view::npos : comma - start));
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    for (std::string_view g : groups) {
        if (g.empty()) return false;
        for (char c : g) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    if (groups.front().size() > 3) return false;
    if (groups.back().size() != 3) return false;
    for (std::size_t i = 1; i + 1 < groups.size(); ++i) {
        if (groups[i].size() != 2 && groups[i].size() != 3) return false;
    }
    out.clear();
    for (std::string_view g : groups) out.append(g);
    if (dot != std::string_view::npos) out.append(s.substr(dot));
    return true;
}

} // namespace detail

/// Read a cell as a number.
///
/// Accepts plain and scientific notation, an optional leading '+', a
/// currency marker ("$1,200", "Rs. 450", "€9.50"), thousands separators
/// in Western or Indian grouping, a trailing percent sign ("45%" reads as
/// 45), and accountancy negatives in brackets ("(1,234.50)").
///
/// Anything else, including "12,34" and free text, is not a number.
inline NumericParse try_parse_number(std::string_view raw) {
    NumericParse out;
    std::string_view s = trim_view(raw);
    if (is_missing_token(s)) { out.is_missing = true; return out; }

    double v = 0.0;
    if (detail::from_chars_full(s, v)) {
        // "inf" parses, but an infinite cell is not a usable number.
        if (std::isfinite(v)) { out.value = v; out.parsed = true; }
        return out;
    }

    bool negative = false;
    if (s.size() >= 3 && s.front() == '(' && s.back() == ')') {
        negative = true;
        s = trim_view(s.substr(1, s.size() - 2));
    }
    if (!s.empty() && (s.front() == '+' || s.front() == '-')) {
        if (s.front() == '-') negative = !negative;
        s.remove_prefix(1);
        s = trim_view(s);
    }
    detail::strip_currency(s);
    if (!s.empty() && (s.front() == '+' || s.front() == '-')) {
        if (s.front() == '-') negative = !negative;
        s.remove_prefix(1);
        s = trim_view(s);
    }
    if (!s.empty() && s.back() == '%') {
        s.remove_suffix(1);
        s = trim_view(s);
    }
    if (s.empty() || !(std::isdigit(static_cast<unsigned char>(s.front())) || s.front() == '.')) {
        return out;
    }

    std::string cleaned;
    const bool ok = detail::remove_grouping(s, cleaned)
                        ? detail::from_chars_full(cleaned, v)
                        : detail::from_chars_full(s, v);
    if (ok && std::isfinite(v)) {
        out.value  = negative ? -v : v;
        out.parsed = true;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dates
// ---------------------------------------------------------------------------

struct DateParse {
    bool          parsed  = false;
    int           year    = 0;
    int           month   = 0;     // 1..12
    int           day     = 0;     // 1..31
    std::int64_t  days    = 0;     // days since 1970-01-01
    int           weekday = 0;     // 0 = Monday .. 6 = Sunday
};

/// Days since 1970-01-01 for a proleptic Gregorian date.
inline std::int64_t days_from_civil(int y, int m, int d) {
    y -= (m <= 2) ? 1 : 0;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153u * static_cast<unsigned>(m + (m > 2 ? -3 : 9)) + 2u) / 5u
                       + static_cast<unsigned>(d) - 1u;
    const unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

/// Parse a calendar date. Year-first forms ("2024-03-31", "2024/03/31",
/// optionally followed by a time) are unambiguous. For "a-b-yyyy" forms
/// `day_first` decides whether `a` is the day or the month.
inline DateParse try_parse_date(std::string_view raw, bool day_first = true) {
    DateParse out;
    std::string_view s = trim_view(raw);
    // Drop a time-of-day suffix: "2024-03-31 14:05:00" or "2024-03-31T14:05".
    const std::size_t cut = s.find_first_of("T ");
    if (cut != std::string_view::npos) s = s.substr(0, cut);
    if (s.size() < 8 || s.size() > 10) return out;

    int  parts[3] = {0, 0, 0};
    int  lens[3]  = {0, 0, 0};
    int  n = 0;
    char sep = 0;
    for (char c : s) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            if (lens[n] >= 4) return out;
            parts[n] = parts[n] * 10 + (c - '0');
            ++lens[n];
        } else if (c == '-' || c == '/' || c == '.') {
            if (sep == 0) sep = c;
            if (c != sep || lens[n] == 0 || n == 2) return out;
            ++n;
        } else {
            return out;
        }
    }
    if (n != 2 || lens[2] == 0) return out;

    int y = 0, m = 0, d = 0;
    if (lens[0] == 4) {                      // yyyy-mm-dd
        if (lens[1] > 2 || lens[2] > 2) return out;
        y = parts[0]; m = parts[1]; d = parts[2];
    } else if (lens[2] == 4) {               // a-b-yyyy
        if (lens[0] > 2 || lens[1] > 2) return out;
        y = parts[2];
        if (day_first) { d = parts[0]; m = parts[1]; }
        else           { m = parts[0]; d = parts[1]; }
    } else {
        return out;
    }
    if (y < 1900 || y > 2200 || m < 1 || m > 12 || d < 1) return out;
    static constexpr int month_len[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    const int max_d = (m == 2 && leap) ? 29 : month_len[m - 1];
    if (d > max_d) return out;

    out.parsed = true;
    out.year   = y;
    out.month  = m;
    out.day    = d;
    out.days   = days_from_civil(y, m, d);
    // 1970-01-01 was a Thursday (index 3 with Monday = 0).
    out.weekday = static_cast<int>(((out.days % 7) + 7 + 3) % 7);
    return out;
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// FNV-1a, 64-bit. Fixed here because std::hash differs between standard
/// libraries, and a model trained on one platform must hash text the same
/// way when it is used on another.
inline std::uint64_t fnv1a64(std::string_view s) {
    std::uint64_t h = 14695981039346656037ull;
    for (char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace neurograph::util
