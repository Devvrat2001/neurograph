#pragma once

#include <cstdint>
#include <string>

namespace neurograph::util {

/// Generates a UUID v4 string (e.g., "550e8400-e29b-41d4-a716-446655440000").
/// Default behavior: thread-local RNG seeded from std::random_device, so
/// each call returns a globally-unique value.
///
/// While a `DeterministicUuidScope` is active on the current thread,
/// `generate_uuid()` instead draws from a seeded RNG owned by the
/// scope — so identical seeds across runs produce byte-identical UUID
/// sequences. Used by the orchestrator (P1-15) to keep `model.ngm`
/// reproducible when the SON network's neurons get auto-generated IDs.
std::string generate_uuid();

/// RAII guard that swaps `generate_uuid()` from random_device-seeded
/// to a deterministic seeded RNG for the current thread. Pushable —
/// nested scopes restore the previous override on destruction. Use
/// only on threads that own the work being seeded; cross-thread UUID
/// generation while a scope is active will not see the override
/// (the override is stored thread-local).
class DeterministicUuidScope {
public:
    explicit DeterministicUuidScope(std::uint64_t seed);
    ~DeterministicUuidScope();

    DeterministicUuidScope(const DeterministicUuidScope&)            = delete;
    DeterministicUuidScope& operator=(const DeterministicUuidScope&) = delete;
    DeterministicUuidScope(DeterministicUuidScope&&)                 = delete;
    DeterministicUuidScope& operator=(DeterministicUuidScope&&)      = delete;
};

} // namespace neurograph::util
