#pragma once

#include "neurograph/command/command.hpp"
#include "neurograph/command/command_verifier.hpp"
#include "neurograph/command/command_executor.hpp"
#include "neurograph/core/network.hpp"
#include <vector>
#include <unordered_map>

namespace neurograph::command {

/// Summary of a command propagation through the network.
struct PropagationResult {
    bool authenticated = false;
    std::string auth_error;

    size_t neurons_visited = 0;
    size_t neurons_succeeded = 0;
    size_t neurons_failed = 0;

    /// Per-neuron execution results.
    std::vector<std::pair<core::NeuronId, ExecutionResult>> results;

    nlohmann::json to_json() const;
};

/// Propagates signed commands through the neural network.
/// Verifies the command signature once, then traverses the network
/// using the specified propagation mode (BFS/DFS/TARGETED/BROADCAST),
/// executing the command on each visited neuron.
class CommandPropagator {
public:
    CommandPropagator(core::Network& network,
                      const CommandVerifier& verifier,
                      CommandExecutor& executor);

    /// Propagate a command starting from entry_neuron.
    /// For TARGETED mode, only executes on the command's target neuron.
    /// For BROADCAST mode, executes on all neurons.
    /// For BFS/DFS, traverses from entry_neuron.
    PropagationResult propagate(const Command& cmd,
                                const core::NeuronId& entry_neuron);

    /// Propagate a command with automatic entry point selection.
    /// Uses the command's target_neuron if set, otherwise picks the first neuron.
    PropagationResult propagate(const Command& cmd);

private:
    core::Network& network_;
    const CommandVerifier& verifier_;
    CommandExecutor& executor_;

    PropagationResult propagate_bfs(const Command& cmd,
                                    const core::NeuronId& start);
    PropagationResult propagate_dfs(const Command& cmd,
                                    const core::NeuronId& start);
    PropagationResult propagate_targeted(const Command& cmd);
    PropagationResult propagate_broadcast(const Command& cmd);
};

} // namespace neurograph::command
