#include "neurograph/core/topology.hpp"
#include "neurograph/core/network.hpp"

#include <algorithm>
#include <random>
#include <cmath>

namespace neurograph::core {

// ============================================================
// HebbianTopology
// ============================================================

std::vector<ConnectionProposal> HebbianTopology::select_connections(
    const Neuron& neuron,
    const std::vector<const Neuron*>& candidates,
    const Network& /*network*/,
    const TopologyConfig& config) const
{
    // Build set of existing outgoing targets for fast lookup
    std::unordered_set<std::string> existing_targets;
    for (const auto& sid : neuron.outgoing_synapses()) {
        existing_targets.insert(sid.value());
    }

    std::vector<ConnectionProposal> scored;

    for (const auto* candidate : candidates) {
        if (candidate->id() == neuron.id()) continue;

        // Use the neuron's built-in affinity scoring
        double score = neuron.evaluate_connection_affinity(*candidate);

        if (score >= config.affinity_threshold) {
            scored.push_back({candidate->id(), score});
        }
    }

    // Sort by score descending — no hard cap on connections,
    // self-org synapse pruning will clean up weak ones naturally
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) {
            return a.proposed_weight > b.proposed_weight;
        });

    return scored;
}

bool HebbianTopology::should_prune(
    const Synapse& synapse,
    const TopologyConfig& config,
    int current_cycle) const
{
    // Prune if weight is below threshold
    if (std::abs(synapse.weight()) < config.pruning_threshold) {
        return true;
    }

    // Prune if idle for too long
    if (synapse.metadata().contains("last_activated_cycle")) {
        int last_cycle = synapse.metadata()["last_activated_cycle"].get<int>();
        if (current_cycle - last_cycle > config.max_idle_cycles) {
            return true;
        }
    }

    return false;
}

// ============================================================
// RandomTopology
// ============================================================

std::vector<ConnectionProposal> RandomTopology::select_connections(
    const Neuron& neuron,
    const std::vector<const Neuron*>& candidates,
    const Network& /*network*/,
    const TopologyConfig& config) const
{
    if (candidates.empty()) return {};

    // Filter out self
    std::vector<const Neuron*> valid;
    for (const auto* c : candidates) {
        if (c->id() != neuron.id()) valid.push_back(c);
    }

    // Shuffle and pick a reasonable number (no hard cap, but don't wire everything).
    // P1-19: shared self-org RNG so the deterministic scope (when active)
    // pins the shuffle order across runs.
    auto& rng = self_org_rng();
    std::shuffle(valid.begin(), valid.end(), rng);

    int count = std::min(static_cast<int>(valid.size()), 20); // random wires a few
    std::uniform_real_distribution<double> weight_dist(0.1, 1.0);

    std::vector<ConnectionProposal> proposals;
    for (int i = 0; i < count; ++i) {
        proposals.push_back({valid[i]->id(), weight_dist(rng)});
    }

    return proposals;
}

bool RandomTopology::should_prune(
    const Synapse& synapse,
    const TopologyConfig& config,
    int /*current_cycle*/) const
{
    return std::abs(synapse.weight()) < config.pruning_threshold;
}

} // namespace neurograph::core
