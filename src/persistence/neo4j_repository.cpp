#include "neurograph/persistence/neo4j_repository.hpp"
#include "neurograph/util/logger.hpp"

namespace neurograph::persistence {

Neo4jGraphRepository::Neo4jGraphRepository(Neo4jClient& client)
    : client_(client) {}

// ============================================================
// Schema
// ============================================================

util::Result<void> Neo4jGraphRepository::ensure_schema() {
    // Create uniqueness constraint on Neuron.id
    auto r1 = client_.execute_cypher(
        "CREATE CONSTRAINT neuron_id_unique IF NOT EXISTS "
        "FOR (n:Neuron) REQUIRE n.id IS UNIQUE");
    if (r1.is_error()) {
        NG_LOG_WARN("Schema constraint (neuron): {}", r1.error().message);
    }

    // Create index on Neuron.state
    auto r2 = client_.execute_cypher(
        "CREATE INDEX neuron_state_idx IF NOT EXISTS "
        "FOR (n:Neuron) ON (n.state)");
    if (r2.is_error()) {
        NG_LOG_WARN("Schema index (state): {}", r2.error().message);
    }

    // Create index on synapse id property
    auto r3 = client_.execute_cypher(
        "CREATE INDEX synapse_id_idx IF NOT EXISTS "
        "FOR ()-[r:SYNAPSE]-() ON (r.id)");
    if (r3.is_error()) {
        NG_LOG_WARN("Schema index (synapse): {}", r3.error().message);
    }

    NG_LOG_INFO("Neo4j schema ensured");
    return util::Result<void>::success();
}

// ============================================================
// Neuron Persistence
// ============================================================

nlohmann::json Neo4jGraphRepository::neuron_to_params(const core::Neuron& neuron) {
    nlohmann::json p = {
        {"id", neuron.id().value()},
        {"label", neuron.label()},
        {"activation_level", neuron.activation_level()},
        {"threshold", neuron.threshold()},
        {"state", std::string(core::neuron_state_str(neuron.state()))},
        {"activation_type", static_cast<int>(neuron.activation_type())},
        {"metadata", neuron.metadata().dump()}
    };

    if (neuron.position()) {
        p["pos_x"] = neuron.position()->x;
        p["pos_y"] = neuron.position()->y;
        p["pos_z"] = neuron.position()->z;
    } else {
        p["pos_x"] = nullptr;
        p["pos_y"] = nullptr;
        p["pos_z"] = nullptr;
    }

    return p;
}

util::Result<void> Neo4jGraphRepository::save_neuron(const core::Neuron& neuron) {
    auto params = neuron_to_params(neuron);
    auto result = client_.execute_cypher(
        "MERGE (n:Neuron {id: $id}) "
        "SET n.label = $label, "
        "    n.activation_level = $activation_level, "
        "    n.threshold = $threshold, "
        "    n.state = $state, "
        "    n.activation_type = $activation_type, "
        "    n.metadata = $metadata, "
        "    n.pos_x = $pos_x, "
        "    n.pos_y = $pos_y, "
        "    n.pos_z = $pos_z, "
        "    n.updated_at = datetime() "
        "RETURN n.id",
        params);

    if (result.is_error()) return util::Error(result.error().message);
    return util::Result<void>::success();
}

util::Result<core::Neuron> Neo4jGraphRepository::load_neuron(const core::NeuronId& id) {
    auto result = client_.execute_cypher(
        "MATCH (n:Neuron {id: $id}) RETURN n",
        {{"id", id.value()}});

    if (result.is_error()) return util::Error(result.error().message);

    const auto& response = result.value();

    // Parse response - Query API v2 returns data.values
    if (!response.contains("data") || !response["data"].contains("values") ||
        response["data"]["values"].empty()) {
        return util::Error("Neuron not found: " + id.value());
    }

    try {
        // The first row, first column contains the node properties
        const auto& row = response["data"]["values"][0];
        const auto& node = row[0]; // node object
        return neuron_from_record(node);
    } catch (const std::exception& e) {
        return util::Error("Failed to parse neuron: " + std::string(e.what()));
    }
}

core::Neuron Neo4jGraphRepository::neuron_from_record(const nlohmann::json& record) {
    // Neo4j node properties may be nested under "properties" or flat
    const auto& props = record.contains("properties") ? record["properties"] : record;

    core::Neuron neuron(
        core::NeuronId(props.at("id").get<std::string>()),
        props.at("label").get<std::string>());

    neuron.set_activation_level(props.value("activation_level", 0.0));
    neuron.set_threshold(props.value("threshold", 0.5));
    neuron.set_state(core::parse_neuron_state(props.value("state", "idle")));
    neuron.set_activation_type(
        static_cast<core::ActivationType>(props.value("activation_type", 0)));

    if (props.contains("metadata") && props["metadata"].is_string()) {
        try {
            neuron.set_metadata(nlohmann::json::parse(props["metadata"].get<std::string>()));
        } catch (...) {}
    }

    if (props.contains("pos_x") && !props["pos_x"].is_null()) {
        neuron.set_position(core::Vec3{
            props["pos_x"].get<double>(),
            props["pos_y"].get<double>(),
            props["pos_z"].get<double>()
        });
    }

    return neuron;
}

util::Result<void> Neo4jGraphRepository::delete_neuron(const core::NeuronId& id) {
    // Delete neuron and all its relationships
    auto result = client_.execute_cypher(
        "MATCH (n:Neuron {id: $id}) DETACH DELETE n",
        {{"id", id.value()}});

    if (result.is_error()) return util::Error(result.error().message);
    return util::Result<void>::success();
}

util::Result<void> Neo4jGraphRepository::save_neurons_batch(
    const std::vector<const core::Neuron*>& neurons)
{
    if (neurons.empty()) return util::Result<void>::success();

    // Use UNWIND for batch upsert
    nlohmann::json neuron_data = nlohmann::json::array();
    for (const auto* neuron : neurons) {
        neuron_data.push_back(neuron_to_params(*neuron));
    }

    auto result = client_.execute_cypher(
        "UNWIND $neurons AS nd "
        "MERGE (n:Neuron {id: nd.id}) "
        "SET n.label = nd.label, "
        "    n.activation_level = nd.activation_level, "
        "    n.threshold = nd.threshold, "
        "    n.state = nd.state, "
        "    n.activation_type = nd.activation_type, "
        "    n.metadata = nd.metadata, "
        "    n.pos_x = nd.pos_x, "
        "    n.pos_y = nd.pos_y, "
        "    n.pos_z = nd.pos_z, "
        "    n.updated_at = datetime() "
        "RETURN count(n)",
        {{"neurons", neuron_data}});

    if (result.is_error()) return util::Error(result.error().message);

    NG_LOG_DEBUG("Batch saved {} neurons to Neo4j", neurons.size());
    return util::Result<void>::success();
}

// ============================================================
// Synapse Persistence
// ============================================================

nlohmann::json Neo4jGraphRepository::synapse_to_params(const core::Synapse& synapse) {
    return {
        {"id", synapse.id().value()},
        {"source_id", synapse.source_id().value()},
        {"target_id", synapse.target_id().value()},
        {"weight", synapse.weight()},
        {"type", std::string(core::synapse_type_str(synapse.type()))},
        {"plasticity", synapse.plasticity()},
        {"delay_ms", synapse.delay_ms()},
        {"metadata", synapse.metadata().dump()}
    };
}

util::Result<void> Neo4jGraphRepository::save_synapse(const core::Synapse& synapse) {
    auto params = synapse_to_params(synapse);
    auto result = client_.execute_cypher(
        "MATCH (src:Neuron {id: $source_id}), (tgt:Neuron {id: $target_id}) "
        "MERGE (src)-[r:SYNAPSE {id: $id}]->(tgt) "
        "SET r.weight = $weight, "
        "    r.type = $type, "
        "    r.plasticity = $plasticity, "
        "    r.delay_ms = $delay_ms, "
        "    r.metadata = $metadata, "
        "    r.updated_at = datetime() "
        "RETURN r.id",
        params);

    if (result.is_error()) return util::Error(result.error().message);
    return util::Result<void>::success();
}

util::Result<core::Synapse> Neo4jGraphRepository::load_synapse(const core::SynapseId& id) {
    auto result = client_.execute_cypher(
        "MATCH (src:Neuron)-[r:SYNAPSE {id: $id}]->(tgt:Neuron) "
        "RETURN r, src.id AS source_id, tgt.id AS target_id",
        {{"id", id.value()}});

    if (result.is_error()) return util::Error(result.error().message);

    const auto& response = result.value();
    if (!response.contains("data") || !response["data"].contains("values") ||
        response["data"]["values"].empty()) {
        return util::Error("Synapse not found: " + id.value());
    }

    try {
        const auto& row = response["data"]["values"][0];
        return synapse_from_record(row);
    } catch (const std::exception& e) {
        return util::Error("Failed to parse synapse: " + std::string(e.what()));
    }
}

core::Synapse Neo4jGraphRepository::synapse_from_record(const nlohmann::json& record) {
    // record is [relationship, source_id, target_id]
    const auto& rel = record[0];
    const auto& props = rel.contains("properties") ? rel["properties"] : rel;

    std::string source_id = record[1].get<std::string>();
    std::string target_id = record[2].get<std::string>();

    core::Synapse synapse(
        core::SynapseId(props.at("id").get<std::string>()),
        core::NeuronId(source_id),
        core::NeuronId(target_id),
        props.at("weight").get<double>(),
        core::parse_synapse_type(props.value("type", "excitatory")));

    synapse.set_plasticity(props.value("plasticity", 1.0));
    synapse.set_delay(props.value("delay_ms", 0u));

    if (props.contains("metadata") && props["metadata"].is_string()) {
        try {
            synapse.set_metadata(nlohmann::json::parse(props["metadata"].get<std::string>()));
        } catch (...) {}
    }

    return synapse;
}

util::Result<void> Neo4jGraphRepository::delete_synapse(const core::SynapseId& id) {
    auto result = client_.execute_cypher(
        "MATCH ()-[r:SYNAPSE {id: $id}]->() DELETE r",
        {{"id", id.value()}});

    if (result.is_error()) return util::Error(result.error().message);
    return util::Result<void>::success();
}

util::Result<void> Neo4jGraphRepository::save_synapses_batch(
    const std::vector<const core::Synapse*>& synapses)
{
    if (synapses.empty()) return util::Result<void>::success();

    nlohmann::json synapse_data = nlohmann::json::array();
    for (const auto* synapse : synapses) {
        synapse_data.push_back(synapse_to_params(*synapse));
    }

    auto result = client_.execute_cypher(
        "UNWIND $synapses AS sd "
        "MATCH (src:Neuron {id: sd.source_id}), (tgt:Neuron {id: sd.target_id}) "
        "MERGE (src)-[r:SYNAPSE {id: sd.id}]->(tgt) "
        "SET r.weight = sd.weight, "
        "    r.type = sd.type, "
        "    r.plasticity = sd.plasticity, "
        "    r.delay_ms = sd.delay_ms, "
        "    r.metadata = sd.metadata, "
        "    r.updated_at = datetime() "
        "RETURN count(r)",
        {{"synapses", synapse_data}});

    if (result.is_error()) return util::Error(result.error().message);

    NG_LOG_DEBUG("Batch saved {} synapses to Neo4j", synapses.size());
    return util::Result<void>::success();
}

// ============================================================
// Network-Level Operations
// ============================================================

util::Result<void> Neo4jGraphRepository::save_network(const core::Network& network) {
    // Save all neurons in batch
    std::vector<const core::Neuron*> neurons;
    neurons.reserve(network.neuron_count());
    for (const auto& [id, neuron] : network.neurons()) {
        neurons.push_back(&neuron);
    }
    auto r1 = save_neurons_batch(neurons);
    if (r1.is_error()) return r1;

    // Save all synapses in batch
    std::vector<const core::Synapse*> synapses;
    synapses.reserve(network.synapse_count());
    for (const auto& [id, synapse] : network.synapses()) {
        synapses.push_back(&synapse);
    }
    auto r2 = save_synapses_batch(synapses);
    if (r2.is_error()) return r2;

    NG_LOG_INFO("Saved network to Neo4j: {} neurons, {} synapses",
                network.neuron_count(), network.synapse_count());
    return util::Result<void>::success();
}

util::Result<core::Network> Neo4jGraphRepository::load_network() {
    core::Network network;

    // Load all neurons
    auto neurons_result = client_.execute_cypher(
        "MATCH (n:Neuron) RETURN n");

    if (neurons_result.is_error()) return util::Error(neurons_result.error().message);

    const auto& nr = neurons_result.value();
    if (nr.contains("data") && nr["data"].contains("values")) {
        for (const auto& row : nr["data"]["values"]) {
            try {
                auto neuron = neuron_from_record(row[0]);
                network.add_neuron(std::move(neuron));
            } catch (const std::exception& e) {
                NG_LOG_WARN("Failed to load neuron: {}", e.what());
            }
        }
    }

    // Load all synapses
    auto synapses_result = client_.execute_cypher(
        "MATCH (src:Neuron)-[r:SYNAPSE]->(tgt:Neuron) "
        "RETURN r, src.id AS source_id, tgt.id AS target_id");

    if (synapses_result.is_error()) return util::Error(synapses_result.error().message);

    const auto& sr = synapses_result.value();
    if (sr.contains("data") && sr["data"].contains("values")) {
        for (const auto& row : sr["data"]["values"]) {
            try {
                auto synapse = synapse_from_record(row);
                network.add_synapse(std::move(synapse));
            } catch (const std::exception& e) {
                NG_LOG_WARN("Failed to load synapse: {}", e.what());
            }
        }
    }

    NG_LOG_INFO("Loaded network from Neo4j: {} neurons, {} synapses",
                network.neuron_count(), network.synapse_count());
    return network;
}

util::Result<void> Neo4jGraphRepository::clear_all() {
    auto result = client_.execute_cypher("MATCH (n) DETACH DELETE n");
    if (result.is_error()) return util::Error(result.error().message);

    NG_LOG_INFO("Cleared all Neo4j data");
    return util::Result<void>::success();
}

bool Neo4jGraphRepository::is_connected() const {
    return client_.health_check();
}

} // namespace neurograph::persistence
