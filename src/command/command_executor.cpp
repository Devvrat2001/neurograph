#include "neurograph/command/command_executor.hpp"
#include "neurograph/util/logger.hpp"

namespace neurograph::command {

CommandExecutor::CommandExecutor(core::Network& network)
    : network_(network) {}

ExecutionResult CommandExecutor::execute(const Command& cmd,
                                         const core::NeuronId& neuron_id) {
    switch (cmd.type()) {
        case CommandType::CREATE_NEURON:
            return handle_create_neuron(cmd);
        case CommandType::READ_NEURON:
            return handle_read_neuron(cmd, neuron_id);
        case CommandType::UPDATE_NEURON:
            return handle_update_neuron(cmd, neuron_id);
        case CommandType::DELETE_NEURON:
            return handle_delete_neuron(cmd, neuron_id);
        case CommandType::REPLICATE:
            return handle_replicate(cmd, neuron_id);
        case CommandType::CONNECT:
            return handle_connect(cmd, neuron_id);
        case CommandType::DISCONNECT:
            return handle_disconnect(cmd, neuron_id);
        case CommandType::ACTIVATE:
            return handle_activate(cmd, neuron_id);
        case CommandType::SHUTDOWN:
            return handle_shutdown(cmd, neuron_id);
        case CommandType::SELF_ORGANIZE:
            return handle_self_organize(cmd);
        default:
            return ExecutionResult::fail("Unknown command type: " +
                std::string(command_type_str(cmd.type())));
    }
}

ExecutionResult CommandExecutor::execute_global(const Command& cmd) {
    switch (cmd.type()) {
        case CommandType::CREATE_NEURON:
            return handle_create_neuron(cmd);
        case CommandType::SELF_ORGANIZE:
            return handle_self_organize(cmd);
        default:
            return ExecutionResult::fail(
                "Command type not supported as global: " +
                std::string(command_type_str(cmd.type())));
    }
}

// --- Handlers ---

ExecutionResult CommandExecutor::handle_create_neuron(const Command& cmd) {
    const auto& p = cmd.payload();
    std::string label = p.value("label", "unnamed");

    core::Neuron neuron(core::NeuronId::generate(), label);

    if (p.contains("threshold"))
        neuron.set_threshold(p["threshold"].get<double>());
    if (p.contains("activation_type"))
        neuron.set_activation_type(
            static_cast<core::ActivationType>(p["activation_type"].get<int>()));
    if (p.contains("position")) {
        neuron.set_position(p["position"].get<core::Vec3>());
    }
    if (p.contains("metadata"))
        neuron.set_metadata(p["metadata"]);

    auto result = network_.add_neuron(std::move(neuron));
    if (result.is_error()) {
        return ExecutionResult::fail(result.error().message);
    }

    NG_LOG_INFO("CMD: Created neuron {}", result.value().value());
    return ExecutionResult::ok({{"neuron_id", result.value().value()}});
}

ExecutionResult CommandExecutor::handle_read_neuron(const Command& /*cmd*/,
                                                     const core::NeuronId& nid) {
    const auto* neuron = network_.get_neuron(nid);
    if (!neuron) return ExecutionResult::fail("Neuron not found: " + nid.value());
    return ExecutionResult::ok(neuron->to_json());
}

ExecutionResult CommandExecutor::handle_update_neuron(const Command& cmd,
                                                       const core::NeuronId& nid) {
    auto* neuron = network_.get_neuron(nid);
    if (!neuron) return ExecutionResult::fail("Neuron not found: " + nid.value());

    const auto& p = cmd.payload();
    if (p.contains("label"))
        neuron->set_label(p["label"].get<std::string>());
    if (p.contains("threshold"))
        neuron->set_threshold(p["threshold"].get<double>());
    if (p.contains("activation_type"))
        neuron->set_activation_type(
            static_cast<core::ActivationType>(p["activation_type"].get<int>()));
    if (p.contains("state"))
        neuron->set_state(core::parse_neuron_state(p["state"].get<std::string>()));
    if (p.contains("position"))
        neuron->set_position(p["position"].get<core::Vec3>());
    if (p.contains("metadata"))
        neuron->set_metadata(p["metadata"]);

    NG_LOG_DEBUG("CMD: Updated neuron {}", nid.value());
    return ExecutionResult::ok(neuron->to_json());
}

ExecutionResult CommandExecutor::handle_delete_neuron(const Command& /*cmd*/,
                                                       const core::NeuronId& nid) {
    auto result = network_.remove_neuron(nid);
    if (result.is_error()) return ExecutionResult::fail(result.error().message);

    NG_LOG_INFO("CMD: Deleted neuron {}", nid.value());
    return ExecutionResult::ok({{"deleted", nid.value()}});
}

ExecutionResult CommandExecutor::handle_replicate(const Command& cmd,
                                                   const core::NeuronId& nid) {
    const auto& p = cmd.payload();
    double mutation_rate = p.value("mutation_rate", 0.01);
    bool inherit = p.value("inherit_connections", true);

    auto result = network_.replicate_neuron(nid, mutation_rate, inherit);
    if (result.is_error()) return ExecutionResult::fail(result.error().message);

    NG_LOG_INFO("CMD: Replicated {} -> {}", nid.value(), result.value().value());
    return ExecutionResult::ok({
        {"source_id", nid.value()},
        {"clone_id", result.value().value()}
    });
}

ExecutionResult CommandExecutor::handle_connect(const Command& cmd,
                                                 const core::NeuronId& nid) {
    const auto& p = cmd.payload();
    if (!p.contains("target_id")) {
        return ExecutionResult::fail("CONNECT requires 'target_id' in payload");
    }

    core::NeuronId target(p["target_id"].get<std::string>());
    double weight = p.value("weight", 1.0);
    auto stype = core::parse_synapse_type(p.value("type", "excitatory"));

    auto result = network_.connect(nid, target, weight, stype);
    if (result.is_error()) return ExecutionResult::fail(result.error().message);

    NG_LOG_DEBUG("CMD: Connected {} -> {}", nid.value(), target.value());
    return ExecutionResult::ok({{"synapse_id", result.value().value()}});
}

ExecutionResult CommandExecutor::handle_disconnect(const Command& cmd,
                                                    const core::NeuronId& nid) {
    const auto& p = cmd.payload();
    if (!p.contains("target_id")) {
        return ExecutionResult::fail("DISCONNECT requires 'target_id' in payload");
    }

    core::NeuronId target(p["target_id"].get<std::string>());
    auto result = network_.disconnect(nid, target);
    if (result.is_error()) return ExecutionResult::fail(result.error().message);

    return ExecutionResult::ok({{"disconnected", true}});
}

ExecutionResult CommandExecutor::handle_activate(const Command& cmd,
                                                  const core::NeuronId& nid) {
    auto* neuron = network_.get_neuron(nid);
    if (!neuron) return ExecutionResult::fail("Neuron not found: " + nid.value());

    double input = cmd.payload().value("input", 1.0);
    double output = neuron->activate(input);

    return ExecutionResult::ok({
        {"neuron_id", nid.value()},
        {"input", input},
        {"output", output},
        {"state", std::string(core::neuron_state_str(neuron->state()))}
    });
}

ExecutionResult CommandExecutor::handle_shutdown(const Command& /*cmd*/,
                                                  const core::NeuronId& nid) {
    auto* neuron = network_.get_neuron(nid);
    if (!neuron) return ExecutionResult::fail("Neuron not found: " + nid.value());

    neuron->set_state(core::NeuronState::DISABLED);
    NG_LOG_INFO("CMD: Shutdown neuron {}", nid.value());
    return ExecutionResult::ok({{"neuron_id", nid.value()}, {"state", "disabled"}});
}

ExecutionResult CommandExecutor::handle_self_organize(const Command& cmd) {
    int iterations = cmd.payload().value("iterations", 1);
    network_.self_organize(iterations);

    auto stats = network_.stats();
    NG_LOG_INFO("CMD: Self-organized {} iterations", iterations);
    return ExecutionResult::ok({
        {"iterations", iterations},
        {"cycle", stats.organization_cycle},
        {"neurons", stats.neuron_count},
        {"synapses", stats.synapse_count}
    });
}

} // namespace neurograph::command
