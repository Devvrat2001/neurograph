#pragma once

#include "neurograph/core/network.hpp"
#include "neurograph/training/io_manager.hpp"
#include "neurograph/training/loss.hpp"
#include "neurograph/training/data_loader.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>

namespace neurograph::training {

/// Cached forward pass state needed for backward pass
struct ForwardState {
    std::unordered_map<core::NeuronId, double> input_sums;   // pre-activation (z)
    std::unordered_map<core::NeuronId, double> activations;  // post-activation (a)
    std::vector<core::NeuronId> topo_order;                  // execution order
    std::unordered_set<core::NeuronId> dropped_neurons;      // neurons zeroed by dropout
};

/// Per-synapse and per-neuron gradients
struct GradientMap {
    std::unordered_map<core::NeuronId, double> neuron_gradients;   // dL/da
    std::unordered_map<core::SynapseId, double> weight_gradients;  // dL/dw
};

class Backprop {
public:
    Backprop(core::Network& network, IOManager& io_manager);

    /// Rebuild topological ordering via Kahn's algorithm.
    /// Must be called when network topology changes.
    void rebuild_topo_order();

    /// Forward pass in topological order. Caches state for backward pass.
    ForwardState forward(const std::vector<double>& inputs);

    /// Backward pass: propagate output_gradients (dL/da for output neurons)
    /// back through the network in reverse topological order.
    GradientMap backward(const ForwardState& state,
                         const std::vector<double>& output_gradients);

    /// Full forward + backward for one sample. Returns gradients and loss value.
    GradientMap compute_gradients(const Sample& sample, LossFunction& loss,
                                   double& loss_value);

    /// Check if topology changed since last rebuild
    bool needs_rebuild() const;

    const std::vector<core::NeuronId>& topo_order() const { return topo_order_; }

    /// Set dropout rate (probability of zeroing a hidden neuron during training)
    void set_dropout_rate(double rate) { dropout_rate_ = rate; }

    /// Set training mode (dropout is only applied when training_mode is true)
    void set_training_mode(bool mode) { training_mode_ = mode; }

    /// Enable/disable batch normalization for hidden neurons
    void set_batch_norm(bool enabled) { batch_norm_ = enabled; }

    /// Accumulate gradients from multiple samples (for batching)
    static void accumulate(GradientMap& target, const GradientMap& source);

    /// Scale all gradients by a factor (e.g., 1/batch_size)
    static void scale(GradientMap& grads, double factor);

private:
    core::Network& network_;
    IOManager& io_manager_;
    std::vector<core::NeuronId> topo_order_;
    size_t last_neuron_count_ = 0;
    size_t last_synapse_count_ = 0;

    // Dropout
    double dropout_rate_ = 0.0;
    bool training_mode_ = true;
    std::unordered_set<core::NeuronId> input_set_;   // cached for dropout checks
    std::unordered_set<core::NeuronId> output_set_;  // cached for dropout checks
    std::mt19937 dropout_rng_{std::random_device{}()};

    // Batch normalization: per-neuron running statistics
    bool batch_norm_ = false;
    struct BNStats {
        double running_mean = 0.0;
        double running_var = 1.0;
        double gamma = 1.0;   // learnable scale
        double beta = 0.0;    // learnable shift
    };
    std::unordered_map<core::NeuronId, BNStats> bn_stats_;
};

} // namespace neurograph::training
