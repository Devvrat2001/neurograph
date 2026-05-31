#pragma once

#include "neurograph/persistence/graph_repository.hpp"
#include "neurograph/persistence/neo4j_client.hpp"

namespace neurograph::persistence {

/// Neo4j implementation of GraphRepository.
/// Maps neurons to (:Neuron) nodes and synapses to [:SYNAPSE] relationships.
class Neo4jGraphRepository : public GraphRepository {
public:
    explicit Neo4jGraphRepository(Neo4jClient& client);

    // --- Neuron persistence ---
    util::Result<void> save_neuron(const core::Neuron& neuron) override;
    util::Result<core::Neuron> load_neuron(const core::NeuronId& id) override;
    util::Result<void> delete_neuron(const core::NeuronId& id) override;
    util::Result<void> save_neurons_batch(
        const std::vector<const core::Neuron*>& neurons) override;

    // --- Synapse persistence ---
    util::Result<void> save_synapse(const core::Synapse& synapse) override;
    util::Result<core::Synapse> load_synapse(const core::SynapseId& id) override;
    util::Result<void> delete_synapse(const core::SynapseId& id) override;
    util::Result<void> save_synapses_batch(
        const std::vector<const core::Synapse*>& synapses) override;

    // --- Network-level ---
    util::Result<void> save_network(const core::Network& network) override;
    util::Result<core::Network> load_network() override;
    util::Result<void> clear_all() override;

    // --- Schema ---
    util::Result<void> ensure_schema() override;

    // --- Connectivity ---
    bool is_connected() const override;

private:
    Neo4jClient& client_;

    // Cypher query builders
    static nlohmann::json neuron_to_params(const core::Neuron& neuron);
    static nlohmann::json synapse_to_params(const core::Synapse& synapse);
    static core::Neuron neuron_from_record(const nlohmann::json& record);
    static core::Synapse synapse_from_record(const nlohmann::json& record);
};

} // namespace neurograph::persistence
