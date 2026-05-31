#pragma once

#include "neurograph/core/neuron.hpp"
#include "neurograph/core/synapse.hpp"
#include "neurograph/core/network.hpp"
#include "neurograph/util/result.hpp"
#include <vector>
#include <string>

namespace neurograph::persistence {

/// Abstract interface for persisting neurons and synapses to a graph database.
class GraphRepository {
public:
    virtual ~GraphRepository() = default;

    // --- Neuron persistence ---
    virtual util::Result<void> save_neuron(const core::Neuron& neuron) = 0;
    virtual util::Result<core::Neuron> load_neuron(const core::NeuronId& id) = 0;
    virtual util::Result<void> delete_neuron(const core::NeuronId& id) = 0;
    virtual util::Result<void> save_neurons_batch(
        const std::vector<const core::Neuron*>& neurons) = 0;

    // --- Synapse persistence ---
    virtual util::Result<void> save_synapse(const core::Synapse& synapse) = 0;
    virtual util::Result<core::Synapse> load_synapse(const core::SynapseId& id) = 0;
    virtual util::Result<void> delete_synapse(const core::SynapseId& id) = 0;
    virtual util::Result<void> save_synapses_batch(
        const std::vector<const core::Synapse*>& synapses) = 0;

    // --- Network-level operations ---
    virtual util::Result<void> save_network(const core::Network& network) = 0;
    virtual util::Result<core::Network> load_network() = 0;
    virtual util::Result<void> clear_all() = 0;

    // --- Schema setup ---
    virtual util::Result<void> ensure_schema() = 0;

    // --- Connectivity ---
    virtual bool is_connected() const = 0;
};

} // namespace neurograph::persistence
