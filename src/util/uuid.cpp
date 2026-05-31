#include "neurograph/util/uuid.hpp"

#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace neurograph::util {

namespace {

// Thread-local stack of seeded RNGs pushed by DeterministicUuidScope.
// Top of stack (back of vector) wins. Empty stack ⇒ fall back to the
// random_device-seeded global thread-local RNG below.
thread_local std::vector<std::unique_ptr<std::mt19937>> g_uuid_rng_stack;

uint32_t next_uuid_word() {
    static thread_local std::uniform_int_distribution<uint32_t>
        dist(0, 0xFFFFFFFF);
    if (!g_uuid_rng_stack.empty()) {
        return dist(*g_uuid_rng_stack.back());
    }
    static thread_local std::mt19937 gen{std::random_device{}()};
    return dist(gen);
}

} // namespace

DeterministicUuidScope::DeterministicUuidScope(std::uint64_t seed) {
    // Each scope gets its own RNG so nested scopes are independent.
    g_uuid_rng_stack.push_back(
        std::make_unique<std::mt19937>(static_cast<std::mt19937::result_type>(seed)));
}

DeterministicUuidScope::~DeterministicUuidScope() {
    if (!g_uuid_rng_stack.empty()) g_uuid_rng_stack.pop_back();
}

std::string generate_uuid() {
    uint32_t parts[4];
    for (auto& p : parts) p = next_uuid_word();

    // Set version 4 (bits 12-15 of time_hi_and_version)
    parts[1] = (parts[1] & 0xFFFF0FFF) | 0x00004000;
    // Set variant (bits 6-7 of clock_seq_hi_and_reserved)
    parts[2] = (parts[2] & 0x3FFFFFFF) | 0x80000000;

    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    ss << std::setw(8) << parts[0] << '-';
    ss << std::setw(4) << (parts[1] >> 16) << '-';
    ss << std::setw(4) << (parts[1] & 0xFFFF) << '-';
    ss << std::setw(4) << (parts[2] >> 16) << '-';
    ss << std::setw(4) << (parts[2] & 0xFFFF);
    ss << std::setw(8) << parts[3];
    return ss.str();
}

} // namespace neurograph::util
