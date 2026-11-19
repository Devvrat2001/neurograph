#include "neurograph/util/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace neurograph::util {

namespace {

// NEUROGRAPH_LOG_LEVEL=trace|debug|info|warn|error|off overrides the default.
spdlog::level::level_enum level_from_env() {
    const char* raw = std::getenv("NEUROGRAPH_LOG_LEVEL");
    if (!raw) return spdlog::level::warn;
    std::string s(raw);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "trace")                   return spdlog::level::trace;
    if (s == "debug")                   return spdlog::level::debug;
    if (s == "info")                    return spdlog::level::info;
    if (s == "warn" || s == "warning")  return spdlog::level::warn;
    if (s == "error" || s == "err")     return spdlog::level::err;
    if (s == "off" || s == "none")      return spdlog::level::off;
    return spdlog::level::warn;
}

} // namespace

std::shared_ptr<spdlog::logger>& get_logger() {
    static auto logger = [] {
        // Standard error, never standard output: predict writes its results
        // to stdout, and the runtime library runs inside other people's
        // programs. Quiet by default for the same reason.
        auto l = spdlog::stderr_color_mt("neurograph");
        l->set_level(level_from_env());
        l->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
        // The preprocessing code logs through spdlog's default logger; send
        // that to the same place at the same level.
        spdlog::set_default_logger(l);
        return l;
    }();
    return logger;
}

} // namespace neurograph::util
