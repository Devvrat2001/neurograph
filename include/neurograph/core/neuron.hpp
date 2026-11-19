#pragma once

#include "neurograph/core/neuron_id.hpp"
#include "neurograph/core/synapse_id.hpp"
#include "neurograph/core/activation.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include <optional>
#include <string>
#include <deque>

namespace neurograph::core {

enum class NeuronState : uint8_t {
    IDLE,
    ACTIVE,
    REFRACTORY,
    DISABLED,
    REPLICATING
};

inline std::string_view neuron_state_str(NeuronState s) {
    switch (s) {
        case NeuronState::IDLE:        return "idle";
        case NeuronState::ACTIVE:      return "active";
        case NeuronState::REFRACTORY:  return "refractory";
        case NeuronState::DISABLED:    return "disabled";
        case NeuronState::REPLICATING: return "replicating";
        default:                       return "unknown";
    }
}

inline NeuronState parse_neuron_state(std::string_view s) {
    if (s == "active")      return NeuronState::ACTIVE;
    if (s == "refractory")  return NeuronState::REFRACTORY;
    if (s == "disabled")    return NeuronState::DISABLED;
    if (s == "replicating") return NeuronState::REPLICATING;
    return NeuronState::IDLE;
}

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    double distance_to(const Vec3& other) const {
        double dx = x - other.x, dy = y - other.y, dz = z - other.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

inline void to_json(nlohmann::json& j, const Vec3& v) {
    j = {{"x", v.x}, {"y", v.y}, {"z", v.z}};
}

inline void from_json(const nlohmann::json& j, Vec3& v) {
    j.at("x").get_to(v.x);
    j.at("y").get_to(v.y);
    j.at("z").get_to(v.z);
}

class Neuron {
public:
    static constexpr size_t ACTIVATION_HISTORY_SIZE = 64;

    Neuron(NeuronId id, std::string label);

    // --- Identity ---
    const NeuronId& id() const noexcept { return id_; }
    const std::string& label() const noexcept { return label_; }
    void set_label(std::string label) { label_ = std::move(label); }

    // --- State ---
    NeuronState state() const noexcept { return state_; }
    void set_state(NeuronState s) { state_ = s; }
    double activation_level() const noexcept { return activation_level_; }
    void set_activation_level(double level) { activation_level_ = level; }
    double threshold() const noexcept { return threshold_; }
    void set_threshold(double t) { threshold_ = t; }

    // --- Activation ---
    ActivationType activation_type() const noexcept { return activation_type_; }
    void set_activation_type(ActivationType t) { activation_type_ = t; }

    /// Run activation function on input_sum. Updates activation_level and history.
    double activate(double input_sum);

    /// Access the recent activation history (for Hebbian correlation).
    const std::deque<double>& activation_history() const noexcept {
        return activation_history_;
    }

    // --- Self-replication ---
    /// Creates a copy of this neuron with a new ID and optional weight perturbation.
    Neuron replicate(double mutation_rate = 0.01) const;

    // --- Topology management ---
    /// Forget every synapse reference. A neuron that is not yet part of a
    /// network cannot have valid ones: Network::add_synapse is what creates
    /// them.
    void clear_synapse_links() {
        outgoing_synapses_.clear();
        incoming_synapses_.clear();
    }
    void add_outgoing(SynapseId sid);
    void add_incoming(SynapseId sid);
    void remove_outgoing(const SynapseId& sid);
    void remove_incoming(const SynapseId& sid);
    const std::vector<SynapseId>& outgoing_synapses() const noexcept {
        return outgoing_synapses_;
    }
    const std::vector<SynapseId>& incoming_synapses() const noexcept {
        return incoming_synapses_;
    }

    // --- Self-organization ---
    /// Evaluate how desirable a connection to `candidate` is.
    /// Returns affinity score [0.0, 1.0].
    double evaluate_connection_affinity(const Neuron& candidate) const;

    // --- Spatial ---
    const std::optional<Vec3>& position() const noexcept { return position_; }
    void set_position(Vec3 pos) { position_ = pos; }

    // --- Metadata ---
    const nlohmann::json& metadata() const noexcept { return metadata_; }
    void set_metadata(nlohmann::json meta) { metadata_ = std::move(meta); }

    // --- Serialization ---
    nlohmann::json to_json() const;
    static Neuron from_json(const nlohmann::json& j);

private:
    NeuronId id_;
    std::string label_;
    double activation_level_ = 0.0;
    double threshold_ = 0.5;
    NeuronState state_ = NeuronState::IDLE;
    ActivationType activation_type_ = ActivationType::SIGMOID;
    nlohmann::json metadata_ = nlohmann::json::object();
    std::vector<SynapseId> outgoing_synapses_;
    std::vector<SynapseId> incoming_synapses_;
    std::optional<Vec3> position_;
    std::deque<double> activation_history_;
};

} // namespace neurograph::core
