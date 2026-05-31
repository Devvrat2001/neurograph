#include "neurograph/core/neuron.hpp"

#include "neurograph/core/topology.hpp"   // P1-19: self_org_rng()

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace neurograph::core {

Neuron::Neuron(NeuronId id, std::string label)
    : id_(std::move(id)), label_(std::move(label)) {}

double Neuron::activate(double input_sum) {
    if (state_ == NeuronState::DISABLED) return 0.0;

    // Apply activation function (CPU path)
    auto fn = make_activation(activation_type_);
    activation_level_ = fn->compute(input_sum);

    // Update state based on threshold
    state_ = (activation_level_ >= threshold_)
        ? NeuronState::ACTIVE
        : NeuronState::IDLE;

    // Record in history for Hebbian correlation
    activation_history_.push_back(activation_level_);
    if (activation_history_.size() > ACTIVATION_HISTORY_SIZE) {
        activation_history_.pop_front();
    }

    return activation_level_;
}

Neuron Neuron::replicate(double mutation_rate) const {
    auto clone = Neuron(NeuronId::generate(), label_ + "_clone");

    // Copy core properties
    clone.threshold_ = threshold_;
    clone.activation_type_ = activation_type_;
    clone.state_ = NeuronState::IDLE;
    clone.activation_level_ = 0.0;

    // Apply mutation: perturb threshold slightly. P1-19: route through
    // self_org_rng() so the deterministic scope can pin the sequence
    // (replicate is invoked from grow_neurons during self-org).
    auto& rng = self_org_rng();
    if (mutation_rate > 0.0) {
        std::normal_distribution<double> noise(0.0, mutation_rate);
        clone.threshold_ = std::clamp(threshold_ + noise(rng), 0.01, 0.99);
    }

    // Copy metadata but not connections (clone must form its own)
    clone.metadata_ = metadata_;

    // Copy position with slight offset
    if (position_) {
        std::normal_distribution<double> offset(0.0, 0.1);
        clone.position_ = Vec3{
            position_->x + offset(rng),
            position_->y + offset(rng),
            position_->z + offset(rng)
        };
    }

    return clone;
}

void Neuron::add_outgoing(SynapseId sid) {
    outgoing_synapses_.push_back(std::move(sid));
}

void Neuron::add_incoming(SynapseId sid) {
    incoming_synapses_.push_back(std::move(sid));
}

void Neuron::remove_outgoing(const SynapseId& sid) {
    auto it = std::find(outgoing_synapses_.begin(), outgoing_synapses_.end(), sid);
    if (it != outgoing_synapses_.end()) outgoing_synapses_.erase(it);
}

void Neuron::remove_incoming(const SynapseId& sid) {
    auto it = std::find(incoming_synapses_.begin(), incoming_synapses_.end(), sid);
    if (it != incoming_synapses_.end()) incoming_synapses_.erase(it);
}

double Neuron::evaluate_connection_affinity(const Neuron& candidate) const {
    double score = 0.0;
    double weights_sum = 0.0;

    // --- Spatial proximity score [0,1] ---
    constexpr double SPATIAL_WEIGHT = 0.3;
    if (position_ && candidate.position_) {
        double dist = position_->distance_to(*candidate.position_);
        // Closer neurons get higher scores; assume max relevant distance ~10.0
        double spatial = std::max(0.0, 1.0 - dist / 10.0);
        score += SPATIAL_WEIGHT * spatial;
        weights_sum += SPATIAL_WEIGHT;
    }

    // --- Activity correlation score [0,1] (Hebbian) ---
    constexpr double ACTIVITY_WEIGHT = 0.5;
    if (!activation_history_.empty() && !candidate.activation_history_.empty()) {
        // Pearson correlation of activation histories
        size_t n = std::min(activation_history_.size(),
                           candidate.activation_history_.size());
        if (n >= 2) {
            double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
            for (size_t i = 0; i < n; ++i) {
                double a = activation_history_[activation_history_.size() - n + i];
                double b = candidate.activation_history_[candidate.activation_history_.size() - n + i];
                sum_a += a; sum_b += b;
                sum_ab += a * b;
                sum_a2 += a * a;
                sum_b2 += b * b;
            }
            double dn = static_cast<double>(n);
            double num = dn * sum_ab - sum_a * sum_b;
            double den = std::sqrt((dn * sum_a2 - sum_a * sum_a) *
                                   (dn * sum_b2 - sum_b * sum_b));
            double correlation = (den > 1e-12) ? (num / den) : 0.0;
            // Normalize from [-1,1] to [0,1]
            double activity = (correlation + 1.0) / 2.0;
            score += ACTIVITY_WEIGHT * activity;
            weights_sum += ACTIVITY_WEIGHT;
        }
    }

    // --- Diversity score [0,1] ---
    constexpr double DIVERSITY_WEIGHT = 0.2;
    // Prefer connecting to neurons with different labels/types
    double diversity = (label_ != candidate.label_) ? 0.7 : 0.3;
    if (activation_type_ != candidate.activation_type_) diversity += 0.3;
    diversity = std::clamp(diversity, 0.0, 1.0);
    score += DIVERSITY_WEIGHT * diversity;
    weights_sum += DIVERSITY_WEIGHT;

    return (weights_sum > 0) ? score / weights_sum : 0.0;
}

nlohmann::json Neuron::to_json() const {
    nlohmann::json j = {
        {"id", id_.value()},
        {"label", label_},
        {"activation_level", activation_level_},
        {"threshold", threshold_},
        {"state", std::string(neuron_state_str(state_))},
        {"activation_type", static_cast<int>(activation_type_)},
        {"metadata", metadata_}
    };

    // Synapse IDs
    nlohmann::json out_ids = nlohmann::json::array();
    for (const auto& s : outgoing_synapses_) out_ids.push_back(s.value());
    j["outgoing_synapses"] = out_ids;

    nlohmann::json in_ids = nlohmann::json::array();
    for (const auto& s : incoming_synapses_) in_ids.push_back(s.value());
    j["incoming_synapses"] = in_ids;

    if (position_) {
        j["position"] = *position_;
    }

    return j;
}

Neuron Neuron::from_json(const nlohmann::json& j) {
    auto neuron = Neuron(
        NeuronId(j.at("id").get<std::string>()),
        j.at("label").get<std::string>()
    );
    neuron.activation_level_ = j.value("activation_level", 0.0);
    neuron.threshold_ = j.value("threshold", 0.5);
    neuron.state_ = parse_neuron_state(j.value("state", "idle"));
    neuron.activation_type_ = static_cast<ActivationType>(j.value("activation_type", 0));

    if (j.contains("metadata")) neuron.metadata_ = j["metadata"];
    if (j.contains("position")) neuron.position_ = j["position"].get<Vec3>();

    if (j.contains("outgoing_synapses")) {
        for (const auto& sid : j["outgoing_synapses"]) {
            neuron.outgoing_synapses_.emplace_back(sid.get<std::string>());
        }
    }
    if (j.contains("incoming_synapses")) {
        for (const auto& sid : j["incoming_synapses"]) {
            neuron.incoming_synapses_.emplace_back(sid.get<std::string>());
        }
    }

    return neuron;
}

} // namespace neurograph::core
