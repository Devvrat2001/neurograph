#include "neurograph/training/backprop.hpp"
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace neurograph::training {

Backprop::Backprop(core::Network& network, IOManager& io_manager)
    : network_(network), io_manager_(io_manager) {}

bool Backprop::needs_rebuild() const {
    return topo_order_.empty()
        || network_.neuron_count() != last_neuron_count_
        || network_.synapse_count() != last_synapse_count_;
}

void Backprop::rebuild_topo_order() {
    topo_order_.clear();
    last_neuron_count_ = network_.neuron_count();
    last_synapse_count_ = network_.synapse_count();

    auto& input_ids = io_manager_.input_ids();
    auto& output_ids = io_manager_.output_ids();

    if (input_ids.empty() || output_ids.empty()) return;

    // Build sets for quick lookup (also cached for dropout)
    std::unordered_set<core::NeuronId> input_set(input_ids.begin(), input_ids.end());
    std::unordered_set<core::NeuronId> output_set(output_ids.begin(), output_ids.end());
    input_set_ = input_set;
    output_set_ = output_set;

    // Step 1: Find all neurons reachable forward from inputs
    std::unordered_set<core::NeuronId> forward_reachable;
    {
        std::queue<core::NeuronId> q;
        for (auto& id : input_ids) {
            q.push(id);
            forward_reachable.insert(id);
        }
        while (!q.empty()) {
            auto nid = q.front(); q.pop();
            auto* neuron = network_.get_neuron(nid);
            if (!neuron) continue;
            for (auto& sid : neuron->outgoing_synapses()) {
                auto* syn = network_.get_synapse(sid);
                if (!syn) continue;
                auto& target = syn->target_id();
                if (forward_reachable.find(target) == forward_reachable.end()) {
                    forward_reachable.insert(target);
                    q.push(target);
                }
            }
        }
    }

    // Step 2: Find all neurons reachable backward from outputs
    std::unordered_set<core::NeuronId> backward_reachable;
    {
        std::queue<core::NeuronId> q;
        for (auto& id : output_ids) {
            q.push(id);
            backward_reachable.insert(id);
        }
        while (!q.empty()) {
            auto nid = q.front(); q.pop();
            auto* neuron = network_.get_neuron(nid);
            if (!neuron) continue;
            for (auto& sid : neuron->incoming_synapses()) {
                auto* syn = network_.get_synapse(sid);
                if (!syn) continue;
                auto& source = syn->source_id();
                if (backward_reachable.find(source) == backward_reachable.end()) {
                    backward_reachable.insert(source);
                    q.push(source);
                }
            }
        }
    }

    // Step 3: Active subgraph = intersection of forward and backward reachable
    std::unordered_set<core::NeuronId> active_set;
    for (auto& id : forward_reachable) {
        if (backward_reachable.count(id))
            active_set.insert(id);
    }

    if (active_set.empty()) return;

    // Step 4: Kahn's algorithm on the active subgraph
    // Compute in-degree for each active neuron (only counting active predecessors)
    std::unordered_map<core::NeuronId, int> in_degree;
    for (auto& id : active_set) in_degree[id] = 0;

    for (auto& id : active_set) {
        auto* neuron = network_.get_neuron(id);
        if (!neuron) continue;
        for (auto& sid : neuron->incoming_synapses()) {
            auto* syn = network_.get_synapse(sid);
            if (!syn) continue;
            if (active_set.count(syn->source_id()))
                in_degree[id]++;
        }
    }

    // Seed with zero in-degree neurons (should include all inputs)
    std::queue<core::NeuronId> q;
    for (auto& [id, deg] : in_degree) {
        if (deg == 0) q.push(id);
    }

    std::unordered_set<core::NeuronId> visited;
    while (!q.empty()) {
        auto nid = q.front(); q.pop();
        if (visited.count(nid)) continue;
        visited.insert(nid);
        topo_order_.push_back(nid);

        auto* neuron = network_.get_neuron(nid);
        if (!neuron) continue;
        for (auto& sid : neuron->outgoing_synapses()) {
            auto* syn = network_.get_synapse(sid);
            if (!syn) continue;
            auto& target = syn->target_id();
            if (active_set.count(target) && !visited.count(target)) {
                in_degree[target]--;
                if (in_degree[target] <= 0)
                    q.push(target);
            }
        }
    }

    // Any active neurons not visited are part of a cycle — append them at the end
    // (their gradients will still propagate, just not in strict order)
    for (auto& id : active_set) {
        if (!visited.count(id))
            topo_order_.push_back(id);
    }
}

ForwardState Backprop::forward(const std::vector<double>& inputs) {
    if (needs_rebuild()) rebuild_topo_order();

    ForwardState state;
    state.topo_order = topo_order_;

    // Clamp inputs
    io_manager_.set_inputs(inputs);
    auto& input_ids = io_manager_.input_ids();
    for (size_t i = 0; i < input_ids.size(); ++i) {
        double val = (i < inputs.size()) ? inputs[i] : 0.0;
        state.input_sums[input_ids[i]] = val;
        state.activations[input_ids[i]] = val;
    }

    // Forward pass in topological order
    std::unordered_set<core::NeuronId> input_set(input_ids.begin(), input_ids.end());

    for (auto& nid : topo_order_) {
        if (input_set.count(nid)) continue; // inputs already set

        auto* neuron = network_.get_neuron(nid);
        if (!neuron) continue;

        // Sum incoming signals
        double input_sum = 0.0;
        for (auto& sid : neuron->incoming_synapses()) {
            auto* syn = network_.get_synapse(sid);
            if (!syn) continue;
            auto it = state.activations.find(syn->source_id());
            if (it != state.activations.end()) {
                input_sum += syn->propagate(it->second);
            }
        }

        // Batch normalization: normalize pre-activation for hidden neurons
        if (batch_norm_ && !output_set_.count(nid)) {
            auto& bn = bn_stats_[nid];
            constexpr double momentum = 0.1;
            constexpr double eps = 1e-5;

            if (training_mode_) {
                // Update running statistics with exponential moving average
                bn.running_mean = (1.0 - momentum) * bn.running_mean + momentum * input_sum;
                double diff = input_sum - bn.running_mean;
                bn.running_var = (1.0 - momentum) * bn.running_var + momentum * diff * diff;
            }

            // Normalize: z_hat = (z - mean) / sqrt(var + eps)
            double z_hat = (input_sum - bn.running_mean) / std::sqrt(bn.running_var + eps);
            // Scale and shift: y = gamma * z_hat + beta
            input_sum = bn.gamma * z_hat + bn.beta;
        }

        state.input_sums[nid] = input_sum;

        // Apply activation function
        double activation = neuron->activate(input_sum);

        // Dropout: randomly zero hidden neuron activations during training
        if (training_mode_ && dropout_rate_ > 0.0 &&
            !output_set_.count(nid)) {
            // This neuron is hidden (not input, not output) — apply dropout
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(dropout_rng_) < dropout_rate_) {
                activation = 0.0;
                state.dropped_neurons.insert(nid);
            } else {
                // Inverted dropout: scale surviving activations
                activation /= (1.0 - dropout_rate_);
            }
        }

        state.activations[nid] = activation;
    }

    return state;
}

GradientMap Backprop::backward(const ForwardState& state,
                                const std::vector<double>& output_gradients) {
    GradientMap grads;

    // Seed output neuron gradients
    auto& output_ids = io_manager_.output_ids();
    for (size_t i = 0; i < output_ids.size() && i < output_gradients.size(); ++i) {
        grads.neuron_gradients[output_ids[i]] = output_gradients[i];
    }

    // Reverse topological order
    auto& input_ids = io_manager_.input_ids();
    std::unordered_set<core::NeuronId> input_set(input_ids.begin(), input_ids.end());

    for (int idx = static_cast<int>(state.topo_order.size()) - 1; idx >= 0; --idx) {
        auto& nid = state.topo_order[idx];
        if (input_set.count(nid)) continue;

        // Skip gradient computation for dropped neurons
        if (state.dropped_neurons.count(nid)) {
            grads.neuron_gradients[nid] = 0.0;
            continue;
        }

        auto it_grad = grads.neuron_gradients.find(nid);
        if (it_grad == grads.neuron_gradients.end()) continue;
        double dL_da = it_grad->second;

        auto* neuron = network_.get_neuron(nid);
        if (!neuron) continue;

        // dL/dz = dL/da * activation'(z)
        auto it_z = state.input_sums.find(nid);
        double z = (it_z != state.input_sums.end()) ? it_z->second : 0.0;

        // Get activation derivative
        auto act_fn = core::make_activation(neuron->activation_type());
        double da_dz = act_fn->derivative(z);
        double dL_dz = dL_da * da_dz;

        // Propagate to each incoming synapse
        for (auto& sid : neuron->incoming_synapses()) {
            auto* syn = network_.get_synapse(sid);
            if (!syn) continue;

            auto it_src = state.activations.find(syn->source_id());
            double src_activation = (it_src != state.activations.end()) ? it_src->second : 0.0;

            double sign = (syn->type() == core::SynapseType::EXCITATORY) ? 1.0 : -1.0;

            // dL/dw = dL/dz * source_activation * sign
            grads.weight_gradients[syn->id()] = dL_dz * src_activation * sign;

            // dL/da_source += dL/dz * weight * sign
            grads.neuron_gradients[syn->source_id()] += dL_dz * syn->weight() * sign;
        }
    }

    return grads;
}

GradientMap Backprop::compute_gradients(const Sample& sample, LossFunction& loss,
                                         double& loss_value) {
    auto state = forward(sample.features);
    auto predicted = io_manager_.read_outputs();

    loss_value = loss.compute(predicted, sample.target);
    auto output_grads = loss.gradient(predicted, sample.target);

    return backward(state, output_grads);
}

void Backprop::accumulate(GradientMap& target, const GradientMap& source) {
    for (auto& [id, grad] : source.weight_gradients)
        target.weight_gradients[id] += grad;
    for (auto& [id, grad] : source.neuron_gradients)
        target.neuron_gradients[id] += grad;
}

void Backprop::scale(GradientMap& grads, double factor) {
    for (auto& [id, grad] : grads.weight_gradients)
        grad *= factor;
    for (auto& [id, grad] : grads.neuron_gradients)
        grad *= factor;
}

} // namespace neurograph::training
