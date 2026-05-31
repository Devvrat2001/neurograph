#include "neurograph/command/command_propagator.hpp"
#include "neurograph/util/logger.hpp"

#include <queue>
#include <stack>
#include <unordered_set>

namespace neurograph::command {

CommandPropagator::CommandPropagator(core::Network& network,
                                     const CommandVerifier& verifier,
                                     CommandExecutor& executor)
    : network_(network), verifier_(verifier), executor_(executor) {}

PropagationResult CommandPropagator::propagate(const Command& cmd,
                                                const core::NeuronId& entry_neuron) {
    // Step 1: Verify signature + authorization
    auto verification = verifier_.verify(cmd);
    if (!verification) {
        PropagationResult result;
        result.authenticated = false;
        result.auth_error = verification.reason;
        NG_LOG_WARN("Command {} rejected: {}", cmd.id(), verification.reason);
        return result;
    }

    NG_LOG_INFO("Command {} ({}) verified, propagating via {}",
                cmd.id(), command_type_str(cmd.type()),
                propagation_mode_str(cmd.propagation_mode()));

    // Step 2: Dispatch based on propagation mode
    switch (cmd.propagation_mode()) {
        case PropagationMode::BFS:
            return propagate_bfs(cmd, entry_neuron);
        case PropagationMode::DFS:
            return propagate_dfs(cmd, entry_neuron);
        case PropagationMode::TARGETED:
            return propagate_targeted(cmd);
        case PropagationMode::BROADCAST:
            return propagate_broadcast(cmd);
        default:
            PropagationResult r;
            r.authenticated = true;
            r.auth_error = "Unknown propagation mode";
            return r;
    }
}

PropagationResult CommandPropagator::propagate(const Command& cmd) {
    // Auto-select entry point
    if (cmd.target_neuron()) {
        return propagate(cmd, *cmd.target_neuron());
    }

    // Use first neuron as entry point
    if (network_.neuron_count() == 0) {
        PropagationResult r;
        r.authenticated = true;
        return r;
    }

    auto first_id = network_.neurons().begin()->first;
    return propagate(cmd, first_id);
}

PropagationResult CommandPropagator::propagate_bfs(const Command& cmd,
                                                    const core::NeuronId& start) {
    PropagationResult result;
    result.authenticated = true;

    if (!network_.has_neuron(start)) {
        NG_LOG_WARN("BFS start neuron not found: {}", start.value());
        return result;
    }

    std::unordered_set<core::NeuronId> visited;
    std::queue<core::NeuronId> queue;
    queue.push(start);
    visited.insert(start);

    // Make a mutable copy of TTL for tracking
    uint32_t remaining_ttl = cmd.ttl();

    while (!queue.empty() && remaining_ttl > 0) {
        core::NeuronId current = queue.front();
        queue.pop();

        // Execute command on this neuron
        auto exec_result = executor_.execute(cmd, current);
        result.results.emplace_back(current, exec_result);
        result.neurons_visited++;

        if (exec_result.success) {
            result.neurons_succeeded++;
        } else {
            result.neurons_failed++;
            NG_LOG_WARN("Command failed on neuron {}: {}", current.value(),
                        exec_result.error);
        }

        // Enqueue unvisited neighbors
        for (const auto& neighbor : network_.get_neighbors(current)) {
            if (!visited.contains(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }

        --remaining_ttl;
    }

    NG_LOG_INFO("BFS propagation complete: {}/{} succeeded",
                result.neurons_succeeded, result.neurons_visited);
    return result;
}

PropagationResult CommandPropagator::propagate_dfs(const Command& cmd,
                                                    const core::NeuronId& start) {
    PropagationResult result;
    result.authenticated = true;

    if (!network_.has_neuron(start)) return result;

    std::unordered_set<core::NeuronId> visited;
    std::stack<core::NeuronId> stack;
    stack.push(start);

    uint32_t remaining_ttl = cmd.ttl();

    while (!stack.empty() && remaining_ttl > 0) {
        core::NeuronId current = stack.top();
        stack.pop();

        if (visited.contains(current)) continue;
        visited.insert(current);

        auto exec_result = executor_.execute(cmd, current);
        result.results.emplace_back(current, exec_result);
        result.neurons_visited++;

        if (exec_result.success) {
            result.neurons_succeeded++;
        } else {
            result.neurons_failed++;
        }

        for (const auto& neighbor : network_.get_neighbors(current)) {
            if (!visited.contains(neighbor)) {
                stack.push(neighbor);
            }
        }

        --remaining_ttl;
    }

    NG_LOG_INFO("DFS propagation complete: {}/{} succeeded",
                result.neurons_succeeded, result.neurons_visited);
    return result;
}

PropagationResult CommandPropagator::propagate_targeted(const Command& cmd) {
    PropagationResult result;
    result.authenticated = true;

    auto target = cmd.target_neuron();
    if (!target) {
        result.neurons_failed = 1;
        result.results.emplace_back(core::NeuronId(""),
            ExecutionResult::fail("TARGETED command has no target neuron"));
        return result;
    }

    auto exec_result = executor_.execute(cmd, *target);
    result.results.emplace_back(*target, exec_result);
    result.neurons_visited = 1;

    if (exec_result.success) {
        result.neurons_succeeded = 1;
    } else {
        result.neurons_failed = 1;
    }

    return result;
}

PropagationResult CommandPropagator::propagate_broadcast(const Command& cmd) {
    PropagationResult result;
    result.authenticated = true;

    // Execute on all neurons (could be parallelized with a thread pool)
    for (const auto& [nid, neuron] : network_.neurons()) {
        auto exec_result = executor_.execute(cmd, nid);
        result.results.emplace_back(nid, exec_result);
        result.neurons_visited++;

        if (exec_result.success) {
            result.neurons_succeeded++;
        } else {
            result.neurons_failed++;
        }
    }

    NG_LOG_INFO("BROADCAST propagation complete: {}/{} succeeded",
                result.neurons_succeeded, result.neurons_visited);
    return result;
}

// --- PropagationResult serialization ---

nlohmann::json PropagationResult::to_json() const {
    nlohmann::json j = {
        {"authenticated", authenticated},
        {"neurons_visited", neurons_visited},
        {"neurons_succeeded", neurons_succeeded},
        {"neurons_failed", neurons_failed}
    };

    if (!auth_error.empty()) {
        j["auth_error"] = auth_error;
    }

    nlohmann::json details = nlohmann::json::array();
    for (const auto& [nid, exec] : results) {
        details.push_back({
            {"neuron_id", nid.value()},
            {"success", exec.success},
            {"data", exec.data},
            {"error", exec.error}
        });
    }
    j["details"] = details;

    return j;
}

} // namespace neurograph::command
