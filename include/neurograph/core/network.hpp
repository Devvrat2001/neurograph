#pragma once

#include "neurograph/core/neuron.hpp"
#include "neurograph/core/synapse.hpp"
#include "neurograph/core/topology.hpp"
#include "neurograph/util/result.hpp"
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <functional>
#include <memory>
#include <vector>
#include <deque>

namespace neurograph::core {

/// Adjacency list: maps each neuron to its outgoing neighbor set.
using AdjacencyList = std::unordered_map<NeuronId, std::unordered_set<NeuronId>>;

/// Snapshot of network statistics.
struct NetworkStats {
    size_t neuron_count = 0;
    size_t synapse_count = 0;
    double avg_activation = 0.0;
    double avg_weight = 0.0;
    int organization_cycle = 0;
};

class Network {
public:
    Network();

    // --- Neuron CRUD ---
    util::Result<NeuronId> add_neuron(Neuron neuron);
    util::Result<void> remove_neuron(const NeuronId& id);
    Neuron* get_neuron(const NeuronId& id);
    const Neuron* get_neuron(const NeuronId& id) const;
    bool has_neuron(const NeuronId& id) const;
    size_t neuron_count() const noexcept { return neurons_.size(); }

    // --- Synapse CRUD ---
    util::Result<SynapseId> add_synapse(Synapse synapse);
    util::Result<void> remove_synapse(const SynapseId& id);
    Synapse* get_synapse(const SynapseId& id);
    const Synapse* get_synapse(const SynapseId& id) const;
    bool has_synapse(const SynapseId& id) const;
    size_t synapse_count() const noexcept { return synapses_.size(); }

    // --- Convenience ---
    /// Create a synapse between two neurons.
    util::Result<SynapseId> connect(const NeuronId& source, const NeuronId& target,
                                    double weight, SynapseType type = SynapseType::EXCITATORY);

    /// Remove all synapses between source and target.
    util::Result<void> disconnect(const NeuronId& source, const NeuronId& target);

    /// Replicate a neuron and optionally inherit its connections.
    util::Result<NeuronId> replicate_neuron(const NeuronId& id,
                                            double mutation_rate = 0.01,
                                            bool inherit_connections = true);

    // --- Topology / Adjacency ---
    std::vector<NeuronId> get_neighbors(const NeuronId& id) const;
    const AdjacencyList& adjacency() const noexcept { return adjacency_; }

    // --- Traversal ---
    /// BFS from start. Visitor returns false to stop traversal.
    void bfs(const NeuronId& start,
             std::function<bool(const NeuronId&)> visitor) const;

    /// DFS from start. Visitor returns false to stop traversal.
    void dfs(const NeuronId& start,
             std::function<bool(const NeuronId&)> visitor) const;

    // --- Self-Organization ---
    void set_topology(std::unique_ptr<Topology> topology);
    void set_topology_config(TopologyConfig config) { topology_config_ = config; }
    const TopologyConfig& topology_config() const noexcept { return topology_config_; }

    /// Run one full self-organization cycle (CPU path). All 9 phases:
    /// rewire, Hebbian-strengthen, prune-synapses, normalize-weights,
    /// prune-dead-neurons, grow-stressed, dynamic-rewire,
    /// mutate-activations, organize-layers.
    ///
    /// **Compatibility caveat:** the rewire + Hebbian-strengthen phases
    /// modify hidden weights without consulting any supervised gradient
    /// signal. When called inside a supervised backprop training loop,
    /// they overwrite weights that the optimizer is tuning and break
    /// convergence (P1-15 found iris collapsed 0.93 → 0.33 with this
    /// running every 10 epochs). For mid-training capacity adaptation
    /// use `growth_only_self_organize` instead — it runs only the
    /// phases that don't fight backprop.
    void self_organize(int iterations = 1);

    /// P1-20: capacity-only variant of `self_organize` for use inside
    /// supervised training loops. Runs three phases that adapt the
    /// network's capacity without overwriting trained weights:
    ///   - Phase 3: prune synapses below the weak/stale thresholds.
    ///   - Phase 5: prune dead hidden neurons (low avg activation).
    ///   - Phase 6: grow new neurons by splitting stressed ones (their
    ///              clones inherit weights with halving + small noise,
    ///              so the function the network computes barely changes).
    /// Skips: rewire, Hebbian-strengthen, weight-normalization,
    /// dynamic-rewire, activation-mutation, layer-organization — each
    /// of which would change the network's function without coordinating
    /// with the supervised optimizer's per-weight state.
    void growth_only_self_organize(int iterations = 1);

    /// Mark neurons as protected IO (won't be pruned/modified by self-org)
    void set_protected_neurons(const std::unordered_set<NeuronId>& ids);
    const std::unordered_set<NeuronId>& protected_neurons() const { return protected_neurons_; }

    /// Report training loss so self-org can make growth/shrink decisions
    void report_training_loss(double loss);

    /// Get current adaptive topology stats for UI display
    struct AdaptiveStats {
        int neurons_pruned_total = 0;
        int neurons_grown_total = 0;
        int synapses_rewired_total = 0;
        int cycles_since_improvement = 0;
        double best_loss = 1e30;
    };
    const AdaptiveStats& adaptive_stats() const { return adaptive_stats_; }

    /// Activate all neurons with given inputs (CPU path).
    void activate_all(const std::vector<double>& inputs);

    // --- Statistics ---
    NetworkStats stats() const;
    int current_cycle() const noexcept { return organization_cycle_; }

    // --- Bulk access ---
    const auto& neurons() const noexcept { return neurons_; }
    const auto& synapses() const noexcept { return synapses_; }

    // --- Serialization ---
    nlohmann::json to_json() const;
    static Network from_json(const nlohmann::json& j);

private:
    std::unordered_map<NeuronId, Neuron> neurons_;
    std::unordered_map<SynapseId, Synapse> synapses_;
    AdjacencyList adjacency_;
    std::unique_ptr<Topology> topology_;
    TopologyConfig topology_config_;
    int organization_cycle_ = 0;

    // Adaptive topology state
    std::unordered_set<NeuronId> protected_neurons_;
    std::deque<double> loss_history_;
    AdaptiveStats adaptive_stats_;

    // Adaptive topology phases (called from self_organize)
    void prune_dead_neurons();
    void grow_neurons();
    void dynamic_rewire();
    void mutate_activations();
    void organize_layers();
};

} // namespace neurograph::core
