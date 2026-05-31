#pragma once

#include "neurograph/util/uuid.hpp"
#include <string>
#include <functional>

namespace neurograph::core {

/// Strong-typed identifier for synapses. Wraps a UUID string.
class SynapseId {
public:
    SynapseId() = default;
    explicit SynapseId(std::string id) : id_(std::move(id)) {}

    static SynapseId generate() {
        return SynapseId(util::generate_uuid());
    }

    const std::string& value() const noexcept { return id_; }
    bool empty() const noexcept { return id_.empty(); }

#if __cplusplus >= 202002L
    bool operator==(const SynapseId&) const = default;
    auto operator<=>(const SynapseId&) const = default;
#else
    bool operator==(const SynapseId& other) const { return id_ == other.id_; }
    bool operator!=(const SynapseId& other) const { return id_ != other.id_; }
    bool operator<(const SynapseId& other) const { return id_ < other.id_; }
#endif

private:
    std::string id_;
};

} // namespace neurograph::core

template <>
struct std::hash<neurograph::core::SynapseId> {
    size_t operator()(const neurograph::core::SynapseId& id) const noexcept {
        return std::hash<std::string>{}(id.value());
    }
};
