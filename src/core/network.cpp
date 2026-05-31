#include "neurograph/core/network.hpp"
#include "neurograph/util/logger.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stack>
#include <vector>

namespace neurograph::core {

// ---------------------------------------------------------------------------
// P1-19: deterministic-self-org RNG infrastructure.
//
// Self-org sites in this TU (`grow_neurons`, `dynamic_rewire`,
// `mutate_activations`, `organize_layers`) all need a thread-local RNG.
// Default behavior: random_device-seeded — preserves legacy non-
// deterministic behavior for the legacy advanced UI / standalone callers.
// While a `DeterministicSelfOrgScope` is active on the current thread,
// the helper below returns a seeded RNG owned by the scope, so identical
// seeds produce identical sequences across runs.
// ---------------------------------------------------------------------------

namespace {

thread_local std::vector<std::unique_ptr<std::mt19937>> g_self_org_rng_stack;

// Default fallback RNG when no scope is active. Single instance per
// thread so unrelated calls don't restart the random sequence.
std::mt19937& default_self_org_rng() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

} // namespace

// Internal helper consumed by `grow_neurons`, `dynamic_rewire`,
// `mutate_activations`, and `organize_layers` — see those sites.
std::mt19937& self_org_rng() {
    if (!g_self_org_rng_stack.empty()) return *g_self_org_rng_stack.back();
    return default_self_org_rng();
}

DeterministicSelfOrgScope::DeterministicSelfOrgScope(std::uint64_t seed) {
    g_self_org_rng_stack.push_back(
        std::make_unique<std::mt19937>(static_cast<std::mt19937::result_type>(seed)));
}

DeterministicSelfOrgScope::~DeterministicSelfOrgScope() {
    if (!g_self_org_rng_stack.empty()) g_self_org_rng_stack.pop_back();
}

Network::Network() {
    topology_ = std::make_unique<HebbianTopology>();
}

// ============================================================
// Neuron CRUD
// ============================================================

util::Result<NeuronId> Network::add_neuron(Neuron neuron) {
    NeuronId id = neuron.id();
    if (neurons_.contains(id)) {
        return util::Error("Neuron already exists: " + id.value());
    }
    adjacency_[id] = {};
    neurons_.emplace(id, std::move(neuron));
    NG_LOG_DEBUG("Added neuron: {}", id.value());
    return id;
}

util::Result<void> Network::remove_neuron(const NeuronId& id) {
    auto it = neurons_.find(id);
    if (it == neurons_.end()) {
        return util::Error("Neuron not found: " + id.value());
    }

    // Collect all synapses connected to this neuron
    std::vector<SynapseId> to_remove;
    for (const auto& sid : it->second.outgoing_synapses()) to_remove.push_back(sid);
    for (const auto& sid : it->second.incoming_synapses()) to_remove.push_back(sid);

    // Remove synapses
    for (const auto& sid : to_remove) {
        remove_synapse(sid); // ignores errors for already-removed
    }

    adjacency_.erase(id);
    neurons_.erase(it);
    NG_LOG_DEBUG("Removed neuron: {}", id.value());
    return util::Result<void>::success();
}

Neuron* Network::get_neuron(const NeuronId& id) {
    auto it = neurons_.find(id);
    return (it != neurons_.end()) ? &it->second : nullptr;
}

const Neuron* Network::get_neuron(const NeuronId& id) const {
    auto it = neurons_.find(id);
    return (it != neurons_.end()) ? &it->second : nullptr;
}

bool Network::has_neuron(const NeuronId& id) const {
    return neurons_.contains(id);
}

// ============================================================
// Synapse CRUD
// ============================================================

util::Result<SynapseId> Network::add_synapse(Synapse synapse) {
    SynapseId sid = synapse.id();
    NeuronId src = synapse.source_id();
    NeuronId tgt = synapse.target_id();

    if (!has_neuron(src)) return util::Error("Source neuron not found: " + src.value());
    if (!has_neuron(tgt)) return util::Error("Target neuron not found: " + tgt.value());
    if (synapses_.contains(sid)) return util::Error("Synapse already exists: " + sid.value());

    // Update neuron connection lists
    neurons_.at(src).add_outgoing(sid);
    neurons_.at(tgt).add_incoming(sid);

    // Update adjacency
    adjacency_[src].insert(tgt);

    synapses_.emplace(sid, std::move(synapse));
    NG_LOG_TRACE("Added synapse: {} -> {}", src.value(), tgt.value());
    return sid;
}

util::Result<void> Network::remove_synapse(const SynapseId& id) {
    auto it = synapses_.find(id);
    if (it == synapses_.end()) {
        return util::Error("Synapse not found: " + id.value());
    }

    const auto& syn = it->second;
    NeuronId src = syn.source_id();
    NeuronId tgt = syn.target_id();

    // Update neuron connection lists
    if (auto* s = get_neuron(src)) s->remove_outgoing(id);
    if (auto* t = get_neuron(tgt)) t->remove_incoming(id);

    // Update adjacency (check if other synapses still connect src->tgt)
    if (adjacency_.contains(src)) {
        bool other_exists = false;
        for (const auto& [sid, syn2] : synapses_) {
            if (sid != id && syn2.source_id() == src && syn2.target_id() == tgt) {
                other_exists = true;
                break;
            }
        }
        if (!other_exists) adjacency_[src].erase(tgt);
    }

    synapses_.erase(it);
    return util::Result<void>::success();
}

Synapse* Network::get_synapse(const SynapseId& id) {
    auto it = synapses_.find(id);
    return (it != synapses_.end()) ? &it->second : nullptr;
}

const Synapse* Network::get_synapse(const SynapseId& id) const {
    auto it = synapses_.find(id);
    return (it != synapses_.end()) ? &it->second : nullptr;
}

bool Network::has_synapse(const SynapseId& id) const {
    return synapses_.contains(id);
}

// ============================================================
// Convenience operations
// ============================================================

util::Result<SynapseId> Network::connect(const NeuronId& source, const NeuronId& target,
                                          double weight, SynapseType type) {
    auto synapse = Synapse(SynapseId::generate(), source, target, weight, type);
    return add_synapse(std::move(synapse));
}

util::Result<void> Network::disconnect(const NeuronId& source, const NeuronId& target) {
    std::vector<SynapseId> to_remove;
    for (const auto& [sid, syn] : synapses_) {
        if (syn.source_id() == source && syn.target_id() == target) {
            to_remove.push_back(sid);
        }
    }
    for (const auto& sid : to_remove) {
        auto result = remove_synapse(sid);
        if (result.is_error()) return result;
    }
    return util::Result<void>::success();
}

util::Result<NeuronId> Network::replicate_neuron(const NeuronId& id,
                                                  double mutation_rate,
                                                  bool inherit_connections) {
    const auto* original = get_neuron(id);
    if (!original) return util::Error("Neuron not found: " + id.value());

    Neuron clone = original->replicate(mutation_rate);
    NeuronId clone_id = clone.id();
    auto result = add_neuron(std::move(clone));
    if (result.is_error()) return result;

    if (inherit_connections) {
        // Connect clone to original's outgoing targets
        for (const auto& sid : original->outgoing_synapses()) {
            const auto* syn = get_synapse(sid);
            if (!syn) continue;
            connect(clone_id, syn->target_id(), syn->weight() * 0.5, syn->type());
        }
    }

    NG_LOG_INFO("Replicated neuron {} -> {}", id.value(), clone_id.value());
    return clone_id;
}

// ============================================================
// Topology / Adjacency
// ============================================================

std::vector<NeuronId> Network::get_neighbors(const NeuronId& id) const {
    auto it = adjacency_.find(id);
    if (it == adjacency_.end()) return {};
    return {it->second.begin(), it->second.end()};
}

// ============================================================
// Traversal
// ============================================================

void Network::bfs(const NeuronId& start,
                  std::function<bool(const NeuronId&)> visitor) const {
    if (!has_neuron(start)) return;

    std::unordered_set<NeuronId> visited;
    std::queue<NeuronId> queue;
    queue.push(start);
    visited.insert(start);

    while (!queue.empty()) {
        NeuronId current = queue.front();
        queue.pop();

        if (!visitor(current)) return;

        for (const auto& neighbor : get_neighbors(current)) {
            if (!visited.contains(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }
}

void Network::dfs(const NeuronId& start,
                  std::function<bool(const NeuronId&)> visitor) const {
    if (!has_neuron(start)) return;

    std::unordered_set<NeuronId> visited;
    std::stack<NeuronId> stack;
    stack.push(start);

    while (!stack.empty()) {
        NeuronId current = stack.top();
        stack.pop();

        if (visited.contains(current)) continue;
        visited.insert(current);

        if (!visitor(current)) return;

        for (const auto& neighbor : get_neighbors(current)) {
            if (!visited.contains(neighbor)) {
                stack.push(neighbor);
            }
        }
    }
}

// ============================================================
// Self-Organization
// ============================================================

void Network::set_topology(std::unique_ptr<Topology> topology) {
    topology_ = std::move(topology);
}

void Network::self_organize(int iterations) {
    if (!topology_) return;

    for (int iter = 0; iter < iterations; ++iter) {
        ++organization_cycle_;

        // Build candidate list — only non-protected (hidden) neurons
        // participate in self-organization's wiring phases.
        // Protected neurons (IO) keep their connections as-is.
        std::vector<const Neuron*> candidates;
        candidates.reserve(neurons_.size());
        for (const auto& [id, neuron] : neurons_) {
            if (!protected_neurons_.count(id))
                candidates.push_back(&neuron);
        }

        // Phase 1: Form new connections (hidden neurons only)
        // Collect proposals first, then apply (avoid mutation during iteration)
        struct PendingConnection {
            NeuronId source;
            NeuronId target;
            double weight;
        };
        std::vector<PendingConnection> pending;

        for (const auto& [id, neuron] : neurons_) {
            // Only hidden neurons propose new connections
            if (protected_neurons_.count(id)) continue;

            auto proposals = topology_->select_connections(
                neuron, candidates, *this, topology_config_);
            for (const auto& p : proposals) {
                // Don't connect to protected neurons either
                if (protected_neurons_.count(p.target_id)) continue;
                pending.push_back({id, p.target_id, p.proposed_weight});
            }
        }

        for (const auto& pc : pending) {
            connect(pc.source, pc.target, pc.weight);
        }

        // Phase 2: Strengthen active connections (Hebbian update)
        // Only strengthen synapses between hidden neurons —
        // IO neuron weights are managed by backprop/optimizer
        for (auto& [sid, synapse] : synapses_) {
            // Skip synapses touching protected (IO) neurons
            if (protected_neurons_.count(synapse.source_id()) ||
                protected_neurons_.count(synapse.target_id())) continue;

            const auto* source = get_neuron(synapse.source_id());
            const auto* target = get_neuron(synapse.target_id());
            if (!source || !target) continue;

            double delta = topology_config_.learning_rate *
                           source->activation_level() *
                           target->activation_level();
            synapse.strengthen(delta * synapse.plasticity());

            // Update last_activated_cycle if neurons are active
            if (source->state() == NeuronState::ACTIVE ||
                target->state() == NeuronState::ACTIVE) {
                auto meta = synapse.metadata();
                meta["last_activated_cycle"] = organization_cycle_;
                synapse.set_metadata(std::move(meta));
            }
        }

        // Phase 3: Prune weak/stale connections
        // Never prune synapses connected to IO neurons
        std::vector<SynapseId> to_prune;
        for (const auto& [sid, synapse] : synapses_) {
            if (protected_neurons_.count(synapse.source_id()) ||
                protected_neurons_.count(synapse.target_id())) continue;

            if (topology_->should_prune(synapse, topology_config_, organization_cycle_)) {
                to_prune.push_back(sid);
            }
        }
        for (const auto& sid : to_prune) {
            remove_synapse(sid);
        }

        // Phase 4: Normalize weights (hidden neurons only)
        for (auto& [nid, neuron] : neurons_) {
            if (protected_neurons_.count(nid)) continue;

            double total_weight = 0.0;
            for (const auto& sid : neuron.outgoing_synapses()) {
                const auto* syn = get_synapse(sid);
                if (syn) total_weight += std::abs(syn->weight());
            }
            if (total_weight > topology_config_.max_total_weight) {
                double scale = topology_config_.max_total_weight / total_weight;
                for (const auto& sid : neuron.outgoing_synapses()) {
                    auto* syn = get_synapse(sid);
                    if (syn) syn->set_weight(syn->weight() * scale);
                }
            }
        }

        // Phase 5: Prune dead/useless hidden neurons
        if (topology_config_.enable_neuron_pruning) {
            prune_dead_neurons();
        }

        // Phase 6: Grow new neurons when loss plateaus
        if (topology_config_.enable_neuron_growth) {
            grow_neurons();
        }

        // Phase 7: Dynamic rewiring (create useful new connections)
        if (topology_config_.enable_dynamic_rewiring &&
            organization_cycle_ % topology_config_.rewire_interval == 0) {
            dynamic_rewire();
        }

        // Phase 8: Activation mutation (discover best activation per neuron)
        mutate_activations();

        // Phase 9: Multi-layer organization (depth layout + skip connections)
        organize_layers();

        NG_LOG_DEBUG("Self-organization cycle {} complete: {} neurons, {} synapses "
                     "(pruned={}, grown={}, rewired={})",
                     organization_cycle_, neurons_.size(), synapses_.size(),
                     adaptive_stats_.neurons_pruned_total,
                     adaptive_stats_.neurons_grown_total,
                     adaptive_stats_.synapses_rewired_total);
    }
}

// P1-20: capacity-only variant. See `growth_only_self_organize` doc in
// network.hpp for the rationale. Phases run here:
//   - Phase 3 (prune weak/stale synapses)
//   - Phase 5 (prune dead hidden neurons)
//   - Phase 6 (grow stressed neurons by splitting)
// Skipped vs full self_organize: rewire (Phase 1), Hebbian-strengthen
// (Phase 2), normalize-weights (Phase 4), dynamic-rewire (Phase 7),
// mutate-activations (Phase 8), organize-layers (Phase 9). Each of
// those modifies hidden weights or activation types without consulting
// the supervised gradient and breaks concurrent backprop.
void Network::growth_only_self_organize(int iterations) {
    if (!topology_) return;

    for (int iter = 0; iter < iterations; ++iter) {
        ++organization_cycle_;

        // Phase 3: prune weak/stale connections (skips IO-touching synapses).
        std::vector<SynapseId> to_prune;
        for (const auto& [sid, synapse] : synapses_) {
            if (protected_neurons_.count(synapse.source_id()) ||
                protected_neurons_.count(synapse.target_id())) continue;
            if (topology_->should_prune(synapse, topology_config_, organization_cycle_)) {
                to_prune.push_back(sid);
            }
        }
        for (const auto& sid : to_prune) remove_synapse(sid);

        // Phase 5: prune dead hidden neurons (low avg activation).
        if (topology_config_.enable_neuron_pruning) {
            prune_dead_neurons();
        }

        // Phase 6: grow stressed neurons via Neuron::replicate. Clones
        // inherit incoming + outgoing weights with halving + small
        // weight_noise — preserves the network's pre-split function
        // (sum of original + clone outgoing ≈ original) so backprop's
        // gradient signal stays coherent across the split.
        if (topology_config_.enable_neuron_growth) {
            grow_neurons();
        }

        NG_LOG_DEBUG("Growth-only self-org cycle {}: {} neurons, {} synapses "
                     "(pruned={}, grown={})",
                     organization_cycle_, neurons_.size(), synapses_.size(),
                     adaptive_stats_.neurons_pruned_total,
                     adaptive_stats_.neurons_grown_total);
    }
}

// ============================================================
// Adaptive Topology: Protected Neurons & Loss Tracking
// ============================================================

void Network::set_protected_neurons(const std::unordered_set<NeuronId>& ids) {
    protected_neurons_ = ids;
}

void Network::report_training_loss(double loss) {
    loss_history_.push_back(loss);
    // Keep last 200 loss values
    while (loss_history_.size() > 200) loss_history_.pop_front();

    if (loss < adaptive_stats_.best_loss - topology_config_.pruning_threshold) {
        adaptive_stats_.best_loss = loss;
        adaptive_stats_.cycles_since_improvement = 0;
    } else {
        adaptive_stats_.cycles_since_improvement++;
    }
}

// ============================================================
// Phase 5: Prune Dead Neurons
// ============================================================
// Remove hidden neurons that contribute nothing:
//   - Near-zero average activation (dead neurons)
//   - Completely disconnected (no incoming or no outgoing synapses)
//   - Never protects IO neurons (input/output)

void Network::prune_dead_neurons() {
    // Gather hidden neurons (not protected)
    std::vector<NeuronId> hidden_ids;
    for (auto& [id, neuron] : neurons_) {
        if (protected_neurons_.count(id)) continue;
        hidden_ids.push_back(id);
    }

    // Respect minimum hidden neuron count
    if (static_cast<int>(hidden_ids.size()) <= topology_config_.min_hidden_neurons) return;

    // Score each hidden neuron: how useful is it?
    struct NeuronScore {
        NeuronId id;
        double usefulness = 0.0;
    };
    std::vector<NeuronScore> scores;

    for (auto& hid : hidden_ids) {
        auto* neuron = get_neuron(hid);
        if (!neuron) continue;

        // Check age — don't prune new neurons
        int age = 0;
        if (neuron->metadata().contains("creation_cycle")) {
            age = organization_cycle_ - neuron->metadata()["creation_cycle"].get<int>();
        } else {
            age = organization_cycle_; // existed before tracking started
        }
        if (age < topology_config_.min_neuron_age) continue;

        // Usefulness = combination of:
        // 1. Average activation magnitude (are signals flowing through?)
        double avg_activation = 0.0;
        auto& history = neuron->activation_history();
        if (!history.empty()) {
            for (double a : history) avg_activation += std::abs(a);
            avg_activation /= static_cast<double>(history.size());
        }

        // 2. Connectivity (has both incoming and outgoing?)
        int in_count = static_cast<int>(neuron->incoming_synapses().size());
        int out_count = static_cast<int>(neuron->outgoing_synapses().size());
        bool disconnected = (in_count == 0 || out_count == 0);

        // 3. Weight magnitude of connections (are weights meaningful?)
        double avg_weight = 0.0;
        int total_conns = 0;
        for (auto& sid : neuron->incoming_synapses()) {
            auto* syn = get_synapse(sid);
            if (syn) { avg_weight += std::abs(syn->weight()); total_conns++; }
        }
        for (auto& sid : neuron->outgoing_synapses()) {
            auto* syn = get_synapse(sid);
            if (syn) { avg_weight += std::abs(syn->weight()); total_conns++; }
        }
        if (total_conns > 0) avg_weight /= total_conns;

        double usefulness = avg_activation * 0.5 + avg_weight * 0.3 +
                            (disconnected ? 0.0 : 0.2);

        scores.push_back({hid, usefulness});
    }

    // Sort by usefulness ascending (least useful first)
    std::sort(scores.begin(), scores.end(),
        [](const NeuronScore& a, const NeuronScore& b) {
            return a.usefulness < b.usefulness;
        });

    // Remove neurons below threshold, respecting min count
    int current_hidden = static_cast<int>(hidden_ids.size());
    for (auto& ns : scores) {
        if (current_hidden <= topology_config_.min_hidden_neurons) break;
        if (ns.usefulness >= topology_config_.dead_neuron_threshold) break;

        remove_neuron(ns.id);
        current_hidden--;
        adaptive_stats_.neurons_pruned_total++;
        NG_LOG_INFO("Pruned dead neuron {} (usefulness={:.4f})",
                    ns.id.value(), ns.usefulness);
    }
}

// ============================================================
// Phase 6: Grow Neurons (Split Bottlenecks)
// ============================================================
// When training loss stops improving, the network may need more capacity.
// Strategy: find the most "stressed" hidden neuron (highest activation
// variance = it's trying to represent too many patterns) and split it
// into two neurons, each inheriting half the connections.

void Network::grow_neurons() {
    // Gate growth behind the stall counter only when the user has opted in.
    // When use_stall_counter is false, growth runs every self-org cycle (still
    // respecting max_hidden_neurons and the split_ratio/min_splits bounds).
    if (topology_config_.use_stall_counter) {
        if (adaptive_stats_.cycles_since_improvement < topology_config_.growth_patience) return;
        // Don't grow if no loss was ever reported (no training happening)
        if (loss_history_.empty()) return;
    }

    // Check max neuron limit
    int current_hidden = 0;
    for (auto& [id, _] : neurons_) {
        if (!protected_neurons_.count(id)) current_hidden++;
    }
    if (topology_config_.max_hidden_neurons > 0 &&
        current_hidden >= topology_config_.max_hidden_neurons) return;

    // P1-16: enforce the absolute max_neurons cap (input + hidden + output).
    // Returns silently rather than warning per call — `self_organize` may
    // be invoked many times during training and we don't want to spam the
    // log every cycle once the cap is hit.
    if (topology_config_.max_neurons > 0 &&
        static_cast<int>(neurons_.size()) >= topology_config_.max_neurons) {
        return;
    }

    auto& rng = self_org_rng();   // P1-19: deterministic when scope active

    // Gather IO neuron sets (needed for both seed and split paths)
    std::vector<NeuronId> input_ids, output_ids;
    for (auto& pid : protected_neurons_) {
        auto* n = get_neuron(pid);
        if (!n) continue;
        if (n->label().rfind("input_", 0) == 0 || n->label().rfind("in_", 0) == 0)
            input_ids.push_back(pid);
        else
            output_ids.push_back(pid);
    }

    // ---------------------------------------------------------------
    // Special case: NO hidden neurons exist yet — seed initial layer
    // ---------------------------------------------------------------
    if (current_hidden == 0 && !input_ids.empty() && !output_ids.empty()) {
        // Dynamically compute how many hidden neurons to create based on
        // the problem size. Heuristic: geometric mean of inputs and outputs,
        // clamped to a reasonable range.
        int n_in = static_cast<int>(input_ids.size());
        int n_out = static_cast<int>(output_ids.size());
        int optimal = static_cast<int>(std::ceil(std::sqrt(
            static_cast<double>(n_in) * static_cast<double>(n_out))));
        // Clamp: at least 2*outputs, at most 2*sqrt(inputs*outputs)
        optimal = std::max(optimal, n_out * 2);
        optimal = std::min(optimal, optimal * 2);
        // Hard floor/ceiling for sanity
        optimal = std::clamp(optimal, 4, 512);

        // P1-16: respect the user-set caps. If max_hidden_neurons is set,
        // never seed more than that. If max_neurons (total) is set,
        // never push the total over its ceiling.
        if (topology_config_.max_hidden_neurons > 0) {
            optimal = std::min(optimal, topology_config_.max_hidden_neurons);
        }
        if (topology_config_.max_neurons > 0) {
            const int room = topology_config_.max_neurons
                           - static_cast<int>(neurons_.size());
            optimal = std::min(optimal, std::max(0, room));
        }
        if (optimal == 0) return;

        // Find spatial center between layers
        double avg_in_x = 0, avg_out_x = 0;
        for (auto& iid : input_ids) {
            auto* n = get_neuron(iid);
            if (n && n->position()) avg_in_x += n->position()->x;
        }
        for (auto& oid : output_ids) {
            auto* n = get_neuron(oid);
            if (n && n->position()) avg_out_x += n->position()->x;
        }
        avg_in_x /= std::max(1.0, (double)n_in);
        avg_out_x /= std::max(1.0, (double)n_out);
        double mid_x = (avg_in_x + avg_out_x) / 2.0;

        // Spread hidden neurons spatially
        double spread = std::min(8.0, optimal * 0.3);
        std::uniform_real_distribution<double> pos_y(-spread, spread);
        std::uniform_real_distribution<double> pos_z(-spread * 0.5, spread * 0.5);
        std::uniform_real_distribution<double> wt(0.01, 0.2);

        std::vector<NeuronId> new_hidden_ids;
        for (int h = 0; h < optimal; ++h) {
            auto nid = NeuronId::generate();
            Neuron neuron(nid, "hidden_" + std::to_string(h));
            neuron.set_position(Vec3{mid_x, pos_y(rng), pos_z(rng)});
            neuron.set_activation_type(ActivationType::SIGMOID);
            neuron.set_threshold(0.0);
            auto meta = nlohmann::json::object();
            meta["creation_cycle"] = organization_cycle_;
            neuron.set_metadata(std::move(meta));

            auto res = add_neuron(std::move(neuron));
            if (res.ok()) new_hidden_ids.push_back(nid);
        }

        // Fully connected: all inputs -> all hidden, all hidden -> all outputs
        for (auto& hid : new_hidden_ids) {
            for (auto& iid : input_ids)
                connect(iid, hid, wt(rng));
            for (auto& oid : output_ids)
                connect(hid, oid, wt(rng));
        }

        adaptive_stats_.neurons_grown_total += static_cast<int>(new_hidden_ids.size());
        adaptive_stats_.cycles_since_improvement = 0;
        NG_LOG_INFO("Seeded {} hidden neurons (sqrt({} * {}) = {}, adapted to problem size)",
                    new_hidden_ids.size(), n_in, n_out, optimal);
        return;
    }

    // ---------------------------------------------------------------
    // Normal case: split the top % of most stressed hidden neurons
    // ---------------------------------------------------------------

    struct StressScore {
        NeuronId id;
        double stress = 0.0;
    };
    std::vector<StressScore> stressed;

    for (auto& [id, neuron] : neurons_) {
        if (protected_neurons_.count(id)) continue;
        auto& history = neuron.activation_history();
        if (history.size() < 4) continue; // relaxed from 10 to 4

        // Compute variance of activations
        double sum = 0.0;
        for (double a : history) sum += a;
        double mean = sum / history.size();

        double var_sum = 0.0;
        for (double a : history) var_sum += (a - mean) * (a - mean);
        double variance = var_sum / history.size();

        // Also consider connectivity load (many connections = bottleneck)
        double max_load = static_cast<double>(input_ids.size() + output_ids.size());
        if (max_load < 1.0) max_load = 40.0;
        double load = static_cast<double>(neuron.incoming_synapses().size() +
                                          neuron.outgoing_synapses().size());
        double stress = variance * 0.6 + (load / max_load) * 0.4;

        stressed.push_back({id, stress});
    }

    if (stressed.empty()) return;

    // Sort by stress descending
    std::sort(stressed.begin(), stressed.end(),
        [](const StressScore& a, const StressScore& b) {
            return a.stress > b.stress;
        });

    // Split the top growth_split_ratio % of stressed neurons
    int to_split = static_cast<int>(std::ceil(
        stressed.size() * topology_config_.growth_split_ratio));
    to_split = std::max(to_split, topology_config_.growth_min_splits);
    to_split = std::min(to_split, static_cast<int>(stressed.size()));

    // Respect max hidden neuron cap
    if (topology_config_.max_hidden_neurons > 0) {
        int room = topology_config_.max_hidden_neurons - current_hidden;
        to_split = std::min(to_split, room);
    }

    // P1-16: enforce total max_neurons cap. Same room calculation as
    // max_hidden_neurons but counting against the full network size.
    if (topology_config_.max_neurons > 0) {
        int room_total = topology_config_.max_neurons
                       - static_cast<int>(neurons_.size());
        to_split = std::min(to_split, room_total);
    }

    // P1-16: enforce per-cycle absolute split fraction. This is the
    // headline anti-runaway-growth knob — even with high stress and a
    // permissive growth_split_ratio, we never split more than e.g. 10%
    // of the current network in a single cycle. Together with the
    // total cap above, growth tapers off smoothly rather than
    // exploding when the stress detector flags many candidates at once.
    if (topology_config_.max_split_fraction_per_cycle > 0.0) {
        const int cap = static_cast<int>(std::floor(
            static_cast<double>(neurons_.size())
            * topology_config_.max_split_fraction_per_cycle));
        // At least 1 split — otherwise tiny networks (a handful of
        // hidden neurons) would never be allowed to grow at all under
        // a tight fraction. The caller's max_neurons / max_hidden
        // caps still bound the long-term ceiling.
        to_split = std::min(to_split, std::max(1, cap));
    }
    if (to_split <= 0) return;

    std::uniform_real_distribution<double> weight_noise(-0.1, 0.1);

    for (int g = 0; g < to_split; ++g) {
        auto& target = stressed[g];
        auto* original = get_neuron(target.id);
        if (!original) continue;

        // Create a clone via replication
        Neuron clone = original->replicate(topology_config_.split_mutation_rate);
        NeuronId clone_id = clone.id();

        // Tag with creation cycle
        auto meta = clone.metadata();
        meta["creation_cycle"] = organization_cycle_;
        meta["parent_neuron"] = original->id().value();
        clone.set_metadata(std::move(meta));

        auto result = add_neuron(std::move(clone));
        if (!result.ok()) continue;

        // Clone inherits ALL incoming connections (full view of same inputs).
        // Weight is halved so the combined pair approximates the original's
        // input response; added noise breaks symmetry so the pair diverges.
        auto incoming = original->incoming_synapses(); // copy
        for (auto& sid : incoming) {
            auto* syn = get_synapse(sid);
            if (!syn) continue;
            connect(syn->source_id(), clone_id,
                    syn->weight() * 0.5 + weight_noise(rng), syn->type());
        }

        // Clone inherits ALL outgoing connections too (reaches same outputs).
        // Original's outgoing weights are halved so sum of (orig + clone) ≈ orig,
        // preserving the downstream signal while giving the clone an equal voice.
        // NOTE: we do NOT remove the original's outgoing synapses — doing so can
        // strand neurons that only connected to one output, leaving clones
        // connected only to the input layer and wasting capacity.
        auto outgoing = original->outgoing_synapses(); // copy
        for (auto& sid : outgoing) {
            auto* syn = get_synapse(sid);
            if (!syn) continue;
            double orig_w = syn->weight();
            syn->set_weight(orig_w * 0.5);
            connect(clone_id, syn->target_id(),
                    orig_w * 0.5 + weight_noise(rng), syn->type());
        }

        adaptive_stats_.neurons_grown_total++;
    }

    NG_LOG_INFO("Split {} of {} stressed hidden neurons (top {:.0f}%)",
                to_split, stressed.size(),
                topology_config_.growth_split_ratio * 100.0);

    // Reset patience counter
    adaptive_stats_.cycles_since_improvement = 0;
}

// ============================================================
// Phase 7: Dynamic Rewiring
// ============================================================
// Create new shortcut connections to improve signal flow.
// Strategy: find pairs of neurons that are far apart in the graph
// but have correlated activations — a direct connection would help.

void Network::dynamic_rewire() {
    // Gather hidden neurons
    std::vector<NeuronId> hidden_ids;
    for (auto& [id, _] : neurons_) {
        if (!protected_neurons_.count(id)) hidden_ids.push_back(id);
    }
    if (hidden_ids.size() < 2) return;

    auto& rng = self_org_rng();   // P1-19: deterministic when scope active

    // Sample random pairs of hidden neurons and score them
    struct RewireCandidate {
        NeuronId from;
        NeuronId to;
        double score = 0.0;
    };
    std::vector<RewireCandidate> candidates;

    // Sample up to 50 random pairs
    int sample_count = std::min(50, static_cast<int>(hidden_ids.size() *
                                                      (hidden_ids.size() - 1) / 2));
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(hidden_ids.size()) - 1);

    for (int s = 0; s < sample_count; ++s) {
        int i = idx_dist(rng);
        int j = idx_dist(rng);
        if (i == j) continue;

        auto& n1 = *get_neuron(hidden_ids[i]);
        auto& n2 = *get_neuron(hidden_ids[j]);

        // Skip if already connected
        bool already_connected = false;
        for (auto& sid : n1.outgoing_synapses()) {
            auto* syn = get_synapse(sid);
            if (syn && syn->target_id() == hidden_ids[j]) {
                already_connected = true;
                break;
            }
        }
        if (already_connected) continue;

        // Score: high activity correlation + both are active neurons
        double affinity = n1.evaluate_connection_affinity(n2);

        // Boost score if both neurons have meaningful activation
        auto& h1 = n1.activation_history();
        auto& h2 = n2.activation_history();
        double avg1 = 0.0, avg2 = 0.0;
        if (!h1.empty()) { for (double a : h1) avg1 += std::abs(a); avg1 /= h1.size(); }
        if (!h2.empty()) { for (double a : h2) avg2 += std::abs(a); avg2 /= h2.size(); }
        double activity_bonus = std::min(avg1, avg2);

        double score = affinity * 0.7 + activity_bonus * 0.3;

        if (score > topology_config_.affinity_threshold)
            candidates.push_back({hidden_ids[i], hidden_ids[j], score});
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
        [](const RewireCandidate& a, const RewireCandidate& b) {
            return a.score > b.score;
        });

    // Create top connections
    int created = 0;
    std::uniform_real_distribution<double> wt(0.05, 0.3);
    for (auto& c : candidates) {
        if (created >= topology_config_.max_new_connections) break;
        auto result = connect(c.from, c.to, wt(rng));
        if (result.ok()) {
            created++;
            adaptive_stats_.synapses_rewired_total++;
        }
    }

    if (created > 0) {
        NG_LOG_INFO("Dynamic rewiring: created {} new connections", created);
    }
}

// ============================================================
// Phase 8: Activation Mutation
// ============================================================
// Let self-organization discover the best activation function per neuron.
// Every few cycles, pick the lowest-performing hidden neurons and
// randomly mutate their activation function. If they improve, keep it.

void Network::mutate_activations() {
    if (organization_cycle_ % 10 != 0) return; // every 10 cycles

    static const ActivationType candidates[] = {
        ActivationType::SIGMOID,
        ActivationType::RELU,
        ActivationType::TANH,
        ActivationType::LEAKY_RELU,
        ActivationType::GELU,
        ActivationType::SWISH,
        ActivationType::ELU
    };
    constexpr int num_candidates = 7;

    auto& rng = self_org_rng();   // P1-19: deterministic when scope active
    std::uniform_int_distribution<int> act_dist(0, num_candidates - 1);

    // Find hidden neurons with low average activation (underperforming)
    std::vector<NeuronId> candidates_to_mutate;
    for (auto& [id, neuron] : neurons_) {
        if (protected_neurons_.count(id)) continue;

        auto& history = neuron.activation_history();
        if (history.size() < 8) continue;

        double avg = 0.0;
        for (double a : history) avg += std::abs(a);
        avg /= history.size();

        // Neurons with very low or saturated activations need a different function
        if (avg < 0.05 || avg > 0.95) {
            candidates_to_mutate.push_back(id);
        }
    }

    // Mutate a random subset (up to 20% of candidates)
    std::shuffle(candidates_to_mutate.begin(), candidates_to_mutate.end(), rng);
    int to_mutate = std::max(1, static_cast<int>(candidates_to_mutate.size()) / 5);
    to_mutate = std::min(to_mutate, static_cast<int>(candidates_to_mutate.size()));

    for (int i = 0; i < to_mutate; ++i) {
        auto* neuron = get_neuron(candidates_to_mutate[i]);
        if (!neuron) continue;

        ActivationType old_type = neuron->activation_type();
        ActivationType new_type;
        do {
            new_type = candidates[act_dist(rng)];
        } while (new_type == old_type);

        neuron->set_activation_type(new_type);
    }

    if (to_mutate > 0) {
        NG_LOG_DEBUG("Activation mutation: mutated {} neurons", to_mutate);
    }
}

// ============================================================
// Phase 9: Multi-Layer Organization
// ============================================================
// Organize hidden neurons into depth layers based on their distance
// from input neurons. If a single layer is too wide, split it into
// two layers by inserting relay neurons. Also creates skip/residual
// connections across layers when gradient flow is weak.

void Network::organize_layers() {
    if (organization_cycle_ % 20 != 0) return; // every 20 cycles

    // Compute depth of each hidden neuron (shortest path from any input)
    std::unordered_map<NeuronId, int> depth;
    std::queue<NeuronId> q;

    for (auto& pid : protected_neurons_) {
        auto* n = get_neuron(pid);
        if (!n) continue;
        if (n->label().rfind("input_", 0) == 0 || n->label().rfind("in_", 0) == 0) {
            depth[pid] = 0;
            q.push(pid);
        }
    }

    while (!q.empty()) {
        auto nid = q.front(); q.pop();
        auto* neuron = get_neuron(nid);
        if (!neuron) continue;

        for (auto& sid : neuron->outgoing_synapses()) {
            auto* syn = get_synapse(sid);
            if (!syn) continue;
            auto& target = syn->target_id();
            if (!depth.count(target)) {
                depth[target] = depth[nid] + 1;
                q.push(target);
            }
        }
    }

    // Group hidden neurons by depth
    std::unordered_map<int, std::vector<NeuronId>> layers;
    int max_depth = 0;
    for (auto& [id, d] : depth) {
        if (protected_neurons_.count(id)) continue;
        layers[d].push_back(id);
        max_depth = std::max(max_depth, d);
    }

    if (layers.empty() || max_depth < 1) return;

    // Update spatial positions to reflect layer depth
    // This makes the 3D visualization show actual layer structure
    for (auto& [d, neurons_in_layer] : layers) {
        double x_pos = -8.0 + (16.0 * d) / (max_depth + 1);
        double spacing = std::min(1.0, 8.0 / std::max(1, static_cast<int>(neurons_in_layer.size())));

        for (size_t i = 0; i < neurons_in_layer.size(); ++i) {
            auto* neuron = get_neuron(neurons_in_layer[i]);
            if (!neuron) continue;
            double y = (static_cast<double>(i) - neurons_in_layer.size() / 2.0) * spacing;
            neuron->set_position(Vec3{x_pos, y, 0.0});
        }
    }

    // Check for skip connection opportunities:
    // If any layer has neurons with very weak gradient flow
    // (low activation variance), add a residual connection from
    // an earlier layer to a later layer to bypass the bottleneck
    auto& rng = self_org_rng();   // P1-19: deterministic when scope active
    std::uniform_real_distribution<double> wt(0.01, 0.1);

    for (int d = 1; d < max_depth - 1; ++d) {
        auto it_src = layers.find(d);
        auto it_dst = layers.find(d + 2);  // skip one layer
        if (it_src == layers.end() || it_dst == layers.end()) continue;

        // Check if the intermediate layer (d+1) is a bottleneck
        auto it_mid = layers.find(d + 1);
        if (it_mid == layers.end()) continue;

        double mid_avg_activation = 0.0;
        int count = 0;
        for (auto& mid_id : it_mid->second) {
            auto* n = get_neuron(mid_id);
            if (!n) continue;
            auto& hist = n->activation_history();
            if (hist.empty()) continue;
            for (double a : hist) mid_avg_activation += std::abs(a);
            count += static_cast<int>(hist.size());
        }
        if (count > 0) mid_avg_activation /= count;

        // If middle layer has weak activation, create skip connections
        if (mid_avg_activation < 0.1 && !it_src->second.empty() && !it_dst->second.empty()) {
            // Pick one source and one destination
            std::uniform_int_distribution<int> src_dist(0, static_cast<int>(it_src->second.size()) - 1);
            std::uniform_int_distribution<int> dst_dist(0, static_cast<int>(it_dst->second.size()) - 1);

            auto src_id = it_src->second[src_dist(rng)];
            auto dst_id = it_dst->second[dst_dist(rng)];

            connect(src_id, dst_id, wt(rng));
            adaptive_stats_.synapses_rewired_total++;
            NG_LOG_DEBUG("Skip connection: layer {} -> layer {} (bypass weak layer {})",
                         d, d + 2, d + 1);
        }
    }
}

void Network::activate_all(const std::vector<double>& inputs) {
    // Assign inputs to neurons in iteration order
    size_t i = 0;
    for (auto& [id, neuron] : neurons_) {
        double input = (i < inputs.size()) ? inputs[i] : 0.0;

        // Accumulate incoming signals
        double incoming_sum = input;
        for (const auto& sid : neuron.incoming_synapses()) {
            const auto* syn = get_synapse(sid);
            if (!syn) continue;
            const auto* source = get_neuron(syn->source_id());
            if (!source) continue;
            incoming_sum += syn->propagate(source->activation_level());
        }

        neuron.activate(incoming_sum);
        ++i;
    }
}

// ============================================================
// Statistics
// ============================================================

NetworkStats Network::stats() const {
    NetworkStats s;
    s.neuron_count = neurons_.size();
    s.synapse_count = synapses_.size();
    s.organization_cycle = organization_cycle_;

    if (!neurons_.empty()) {
        double sum_act = 0.0;
        for (const auto& [id, n] : neurons_) sum_act += n.activation_level();
        s.avg_activation = sum_act / neurons_.size();
    }

    if (!synapses_.empty()) {
        double sum_w = 0.0;
        for (const auto& [id, syn] : synapses_) sum_w += std::abs(syn.weight());
        s.avg_weight = sum_w / synapses_.size();
    }

    return s;
}

// ============================================================
// Serialization
// ============================================================

nlohmann::json Network::to_json() const {
    nlohmann::json j;
    j["organization_cycle"] = organization_cycle_;

    nlohmann::json neurons_arr = nlohmann::json::array();
    for (const auto& [id, n] : neurons_) neurons_arr.push_back(n.to_json());
    j["neurons"] = neurons_arr;

    nlohmann::json synapses_arr = nlohmann::json::array();
    for (const auto& [id, s] : synapses_) synapses_arr.push_back(s.to_json());
    j["synapses"] = synapses_arr;

    return j;
}

Network Network::from_json(const nlohmann::json& j) {
    Network net;
    net.organization_cycle_ = j.value("organization_cycle", 0);

    for (const auto& nj : j.at("neurons")) {
        net.add_neuron(Neuron::from_json(nj));
    }

    for (const auto& sj : j.at("synapses")) {
        net.add_synapse(Synapse::from_json(sj));
    }

    return net;
}

} // namespace neurograph::core
