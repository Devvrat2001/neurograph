#include "neurograph/util/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

namespace neurograph::util {

std::shared_ptr<spdlog::logger>& get_logger() {
    static auto logger = [] {
        auto l = spdlog::stdout_color_mt("neurograph");
        l->set_level(spdlog::level::debug);
        l->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
        return l;
    }();
    return logger;
}

} // namespace neurograph::util
