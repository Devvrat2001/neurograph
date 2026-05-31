#pragma once

#include "neurograph/core/neuron_id.hpp"
#include "neurograph/core/synapse_id.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <algorithm>

namespace neurograph::core {

enum class SynapseType : uint8_t {
    EXCITATORY,   // positive influence (+1 sign)
    INHIBITORY    // negative influence (-1 sign)
};

inline std::string_view synapse_type_str(SynapseType t) {
    return t == SynapseType::EXCITATORY ? "excitatory" : "inhibitory";
}

inline SynapseType parse_synapse_type(std::string_view s) {
    return (s == "inhibitory") ? SynapseType::INHIBITORY : SynapseType::EXCITATORY;
}

class Synapse {
public:
    Synapse(SynapseId id, NeuronId source, NeuronId target,
            double weight, SynapseType type)
        : id_(std::move(id))
        , source_id_(std::move(source))
        , target_id_(std::move(target))
        , weight_(weight)
        , type_(type) {}

    // --- Identity ---
    const SynapseId& id() const noexcept { return id_; }
    const NeuronId& source_id() const noexcept { return source_id_; }
    const NeuronId& target_id() const noexcept { return target_id_; }

    // --- Properties ---
    double weight() const noexcept { return weight_; }
    SynapseType type() const noexcept { return type_; }
    double plasticity() const noexcept { return plasticity_; }
    uint32_t delay_ms() const noexcept { return delay_ms_; }

    // --- Modification ---
    void set_weight(double w) { weight_ = w; }
    void set_plasticity(double p) { plasticity_ = std::clamp(p, 0.0, 10.0); }
    void set_delay(uint32_t ms) { delay_ms_ = ms; }
    void set_type(SynapseType t) { type_ = t; }

    /// Hebbian strengthening: increase weight magnitude.
    void strengthen(double delta) {
        weight_ += delta * plasticity_;
    }

    /// Weaken / decay: decrease weight magnitude toward zero.
    void weaken(double delta) {
        if (weight_ > 0)
            weight_ = std::max(0.0, weight_ - delta * plasticity_);
        else
            weight_ = std::min(0.0, weight_ + delta * plasticity_);
    }

    /// Signal propagation: applies weight and type sign to incoming signal.
    double propagate(double signal) const {
        double sign = (type_ == SynapseType::EXCITATORY) ? 1.0 : -1.0;
        return signal * weight_ * sign;
    }

    // --- Metadata ---
    const nlohmann::json& metadata() const noexcept { return metadata_; }
    void set_metadata(nlohmann::json meta) { metadata_ = std::move(meta); }

    // --- Serialization ---
    nlohmann::json to_json() const {
        return {
            {"id", id_.value()},
            {"source_id", source_id_.value()},
            {"target_id", target_id_.value()},
            {"weight", weight_},
            {"type", std::string(synapse_type_str(type_))},
            {"plasticity", plasticity_},
            {"delay_ms", delay_ms_},
            {"metadata", metadata_}
        };
    }

    static Synapse from_json(const nlohmann::json& j) {
        auto s = Synapse(
            SynapseId(j.at("id").get<std::string>()),
            NeuronId(j.at("source_id").get<std::string>()),
            NeuronId(j.at("target_id").get<std::string>()),
            j.at("weight").get<double>(),
            parse_synapse_type(j.at("type").get<std::string>())
        );
        if (j.contains("plasticity")) s.plasticity_ = j["plasticity"].get<double>();
        if (j.contains("delay_ms"))   s.delay_ms_ = j["delay_ms"].get<uint32_t>();
        if (j.contains("metadata"))   s.metadata_ = j["metadata"];
        return s;
    }

private:
    SynapseId id_;
    NeuronId source_id_;
    NeuronId target_id_;
    double weight_;
    SynapseType type_;
    double plasticity_ = 1.0;
    uint32_t delay_ms_ = 0;
    nlohmann::json metadata_ = nlohmann::json::object();
};

} // namespace neurograph::core
