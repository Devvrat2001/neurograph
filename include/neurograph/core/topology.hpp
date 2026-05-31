#pragma once

#include "neurograph/core/neuron.hpp"
#include "neurograph/core/synapse.hpp"
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace neurograph::core {

/// P1-19: shared RNG accessor for self-organization sites. Returns a
/// reference to either:
///   - The active `DeterministicSelfOrgScope`'s seeded RNG (if any),
///   - or the default thread-local random_device-seeded RNG.
/// Call sites in `Network::self_organize`, `grow_neurons`,
/// `dynamic_rewire`, `mutate_activations`, `organize_layers`, plus
/// `Neuron::replicate` and `RandomTopology::select_connections` —
/// every random source touched by self-org must use this so the
/// scope guard works uniformly.
std::mt19937& self_org_rng();

// Forward declaration
class Network;

/// P1-19: RAII guard that swaps the RNG used inside `Network::self_organize`
/// (and its callees `grow_neurons`, `dynamic_rewire`, `mutate_activations`,
/// `organize_layers`) from the default `random_device`-seeded thread-local
/// to a deterministic seeded RNG for the current thread. Pushable —
/// nested scopes restore the previous override on destruction.
///
/// The orchestrator wraps its training stage with this guard (alongside
/// `util::DeterministicUuidScope`) so two runs with the same seed produce
/// byte-identical `model.ngm` even when `interleave_self_org=true`.
///
/// Stored thread-local: only affects self-org calls on the same thread.
class DeterministicSelfOrgScope {
public:
    explicit DeterministicSelfOrgScope(std::uint64_t seed);
    ~DeterministicSelfOrgScope();

    DeterministicSelfOrgScope(const DeterministicSelfOrgScope&)            = delete;
    DeterministicSelfOrgScope& operator=(const DeterministicSelfOrgScope&) = delete;
    DeterministicSelfOrgScope(DeterministicSelfOrgScope&&)                 = delete;
    DeterministicSelfOrgScope& operator=(DeterministicSelfOrgScope&&)      = delete;
};

/// Configuration for self-organization algorithms.
struct TopologyConfig {
    // No hard cap on connections — self-org prunes unnecessary ones naturally
    double affinity_threshold = 0.3;
    double pruning_threshold = 0.05;
    int max_idle_cycles = 100;
    double learning_rate = 0.01;
    double max_total_weight = 10.0;

    // --- Adaptive topology (neuron growth/pruning) ---
    bool enable_neuron_pruning = true;
    bool enable_neuron_growth = true;
    bool enable_dynamic_rewiring = true;

    // Neuron pruning: remove hidden neurons with avg activation below this
    double dead_neuron_threshold = 0.01;
    // Minimum cycles before a neuron can be pruned (let it settle first)
    int min_neuron_age = 30;
    // Max hidden neurons (0 = unlimited)
    int max_hidden_neurons = 0;
    // Min hidden neurons to keep
    int min_hidden_neurons = 2;

    // Neuron growth: split stressed neurons when loss plateaus
    // Use loss-stall counter to decide when to grow. If false, grow every
    // self-org cycle (bounded by split ratio and max_hidden_neurons) regardless
    // of training loss — useful when no supervised loss is being reported or
    // when the user wants aggressive continuous expansion.
    bool use_stall_counter = true;
    // How many cycles of no improvement before we try growing (when stall counter enabled)
    int growth_patience = 10;
    // What fraction of stressed neurons to split per growth event (top %)
    double growth_split_ratio = 0.3;  // top 30% of stressed neurons
    // Minimum neurons to split per event (even if ratio says fewer)
    int growth_min_splits = 2;
    // Mutation rate for neuron splitting
    double split_mutation_rate = 0.05;

    // --- P1-16: hard caps for orchestrator-driven training ---------------
    //
    // The pre-P1-16 split-ratio knobs above are stress-relative: they cap
    // splits as a fraction of the *stressed candidate set*. Empirically
    // that's not enough — on small datasets that drive most neurons into
    // the stressed bucket, 30% of the candidates can still mean dozens
    // of splits per cycle, blowing up the network in a few cycles.
    // The two knobs below are absolute caps complementing that.

    /// Total ceiling on `Network::neuron_count()` (input + hidden + output).
    /// `self_organize` refuses any growth that would push the total above
    /// this number. Default 0 = unbounded (matches pre-P1-16 behavior so
    /// existing tests don't change). Orchestrator overrides to e.g. 256.
    int max_neurons = 0;

    /// Hard ceiling on splits per `self_organize` cycle, as a fraction of
    /// the *current total neuron count* (not just stressed candidates).
    /// 0.10 means at most 10% of the network can be split this cycle.
    /// Stricter than `growth_split_ratio` (which is fraction-of-stressed
    /// and so depends on how many candidates the stress detector picked).
    /// Default 0.0 = uncapped. Orchestrator overrides to 0.10 for slow,
    /// stable growth.
    double max_split_fraction_per_cycle = 0.0;

    // Dynamic rewiring: create bypass connections
    int rewire_interval = 5;    // every N cycles
    int max_new_connections = 10; // per rewire event
};

/// Proposed connection from self-organization.
struct ConnectionProposal {
    NeuronId target_id;
    double proposed_weight;
};

/// Abstract interface for topology strategies (self-organization algorithms).
class Topology {
public:
    virtual ~Topology() = default;

    /// Select which neurons to connect to from candidates.
    /// Returns a list of proposed connections with weights.
    virtual std::vector<ConnectionProposal> select_connections(
        const Neuron& neuron,
        const std::vector<const Neuron*>& candidates,
        const Network& network,
        const TopologyConfig& config) const = 0;

    /// Determine if a synapse should be pruned (removed).
    virtual bool should_prune(
        const Synapse& synapse,
        const TopologyConfig& config,
        int current_cycle) const = 0;

    virtual std::string_view name() const = 0;
};

/// Hebbian topology: neurons that fire together wire together.
/// Uses activity correlation + spatial proximity + diversity.
class HebbianTopology : public Topology {
public:
    std::vector<ConnectionProposal> select_connections(
        const Neuron& neuron,
        const std::vector<const Neuron*>& candidates,
        const Network& network,
        const TopologyConfig& config) const override;

    bool should_prune(
        const Synapse& synapse,
        const TopologyConfig& config,
        int current_cycle) const override;

    std::string_view name() const override { return "hebbian"; }
};

/// Random topology: connects to random candidates (baseline comparison).
class RandomTopology : public Topology {
public:
    std::vector<ConnectionProposal> select_connections(
        const Neuron& neuron,
        const std::vector<const Neuron*>& candidates,
        const Network& network,
        const TopologyConfig& config) const override;

    bool should_prune(
        const Synapse& synapse,
        const TopologyConfig& config,
        int current_cycle) const override;

    std::string_view name() const override { return "random"; }
};

} // namespace neurograph::core
