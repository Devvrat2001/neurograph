#pragma once

#include "neurograph/command/command.hpp"
#include "neurograph/core/network.hpp"
#include "neurograph/util/result.hpp"
#include <nlohmann/json.hpp>

namespace neurograph::command {

/// Result of executing a command on a single neuron.
struct ExecutionResult {
    bool success = false;
    nlohmann::json data;
    std::string error;

    static ExecutionResult ok(nlohmann::json data = {}) {
        return {true, std::move(data), ""};
    }
    static ExecutionResult fail(std::string error) {
        return {false, {}, std::move(error)};
    }
};

/// Executes commands on the network.
/// Dispatches based on CommandType, operates on the target neuron or the network as a whole.
class CommandExecutor {
public:
    explicit CommandExecutor(core::Network& network);

    /// Execute a command on a specific neuron.
    ExecutionResult execute(const Command& cmd, const core::NeuronId& neuron_id);

    /// Execute a network-wide command (not neuron-specific).
    ExecutionResult execute_global(const Command& cmd);

private:
    core::Network& network_;

    // --- Per-type handlers ---
    ExecutionResult handle_create_neuron(const Command& cmd);
    ExecutionResult handle_read_neuron(const Command& cmd, const core::NeuronId& nid);
    ExecutionResult handle_update_neuron(const Command& cmd, const core::NeuronId& nid);
    ExecutionResult handle_delete_neuron(const Command& cmd, const core::NeuronId& nid);

    ExecutionResult handle_replicate(const Command& cmd, const core::NeuronId& nid);
    ExecutionResult handle_connect(const Command& cmd, const core::NeuronId& nid);
    ExecutionResult handle_disconnect(const Command& cmd, const core::NeuronId& nid);

    ExecutionResult handle_activate(const Command& cmd, const core::NeuronId& nid);
    ExecutionResult handle_shutdown(const Command& cmd, const core::NeuronId& nid);

    ExecutionResult handle_self_organize(const Command& cmd);
};

} // namespace neurograph::command
