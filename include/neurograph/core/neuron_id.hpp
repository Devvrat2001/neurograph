#pragma once

#include "neurograph/util/uuid.hpp"
#include <string>
#include <functional>

namespace neurograph::core {

/// Strong-typed identifier for neurons. Wraps a UUID string.
class NeuronId {
public:
    NeuronId() = default;
    explicit NeuronId(std::string id) : id_(std::move(id)) {}

    /// Generates a new unique NeuronId.
    static NeuronId generate() {
        return NeuronId(util::generate_uuid());
    }

    const std::string& value() const noexcept { return id_; }
    bool empty() const noexcept { return id_.empty(); }

#if __cplusplus >= 202002L
    bool operator==(const NeuronId&) const = default;
    auto operator<=>(const NeuronId&) const = default;
#else
    bool operator==(const NeuronId& other) const { return id_ == other.id_; }
    bool operator!=(const NeuronId& other) const { return id_ != other.id_; }
    bool operator<(const NeuronId& other) const { return id_ < other.id_; }
#endif

private:
    std::string id_;
};

} // namespace neurograph::core

template <>
struct std::hash<neurograph::core::NeuronId> {
    size_t operator()(const neurograph::core::NeuronId& id) const noexcept {
        return std::hash<std::string>{}(id.value());
    }
};
