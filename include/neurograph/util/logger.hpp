#pragma once

#include <spdlog/spdlog.h>
#include <memory>

namespace neurograph::util {

/// Returns the shared application logger. Initializes on first call.
std::shared_ptr<spdlog::logger>& get_logger();

/// Convenience macros
#define NG_LOG_TRACE(...) neurograph::util::get_logger()->trace(__VA_ARGS__)
#define NG_LOG_DEBUG(...) neurograph::util::get_logger()->debug(__VA_ARGS__)
#define NG_LOG_INFO(...)  neurograph::util::get_logger()->info(__VA_ARGS__)
#define NG_LOG_WARN(...)  neurograph::util::get_logger()->warn(__VA_ARGS__)
#define NG_LOG_ERROR(...) neurograph::util::get_logger()->error(__VA_ARGS__)

} // namespace neurograph::util
