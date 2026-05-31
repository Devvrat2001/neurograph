#include "neurograph/training/io_manager.hpp"
#include <algorithm>
#include <cmath>

namespace neurograph::training {

IOManager::IOManager(core::Network& network) : network_(network) {}

util::Result<void> IOManager::ensure_input_neurons(int num_features) {
    // If we already have the right number, keep them
    if (static_cast<int>(input_ids_.size()) == num_features) {
        bool all_exist = true;
        for (auto& id : input_ids_) {
            if (!network_.has_neuron(id)) { all_exist = false; break; }
        }
        if (all_exist) return util::Result<void>::success();
    }

    input_ids_.clear();
    input_ids_.reserve(num_features);

    // For large input counts (like MNIST 784), arrange as a 2D grid
    // instead of a single vertical line. This keeps the visualization compact.
    bool use_grid = (num_features > 50);
    int grid_cols = 1, grid_rows = num_features;
    double spacing = 1.2;

    if (use_grid) {
        // Try to make a square-ish grid; for 784 -> 28x28
        grid_cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_features))));
        grid_rows = (num_features + grid_cols - 1) / grid_cols;
        // Compact spacing so the grid fits within ~12 units total
        spacing = std::min(1.2, 12.0 / std::max(grid_rows, grid_cols));
    }

    for (int i = 0; i < num_features; ++i) {
        auto id = core::NeuronId::generate();
        core::Neuron neuron(id, "input_" + std::to_string(i));

        double x, y, z;
        if (use_grid) {
            int row = i / grid_cols;
            int col = i % grid_cols;
            // Grid centered at x=-10, spread in y and z
            x = -10.0;
            y = (row - grid_rows / 2.0) * spacing;
            z = (col - grid_cols / 2.0) * spacing;
        } else {
            x = -8.0;
            y = (i - num_features / 2.0) * spacing;
            z = 0.0;
        }
        neuron.set_position(core::Vec3{x, y, z});
        neuron.set_activation_type(core::ActivationType::SIGMOID);
        neuron.set_threshold(0.0);

        auto result = network_.add_neuron(std::move(neuron));
        if (!result.ok()) return util::Error("Failed to create input neuron: " + result.error().message);
        input_ids_.push_back(id);
    }
    return util::Result<void>::success();
}

util::Result<void> IOManager::ensure_output_neurons(int num_outputs,
                                                      const std::vector<std::string>& class_labels) {
    class_labels_ = class_labels;

    if (static_cast<int>(output_ids_.size()) == num_outputs) {
        bool all_exist = true;
        for (auto& id : output_ids_) {
            if (!network_.has_neuron(id)) { all_exist = false; break; }
        }
        if (all_exist) return util::Result<void>::success();
    }

    output_ids_.clear();
    output_ids_.reserve(num_outputs);

    // Space outputs evenly across the same vertical range as inputs
    // but on the right side, well separated
    double total_height = std::min(12.0, num_outputs * 1.2);
    double out_spacing = (num_outputs > 1) ? total_height / (num_outputs - 1) : 0.0;

    // P1-17: pick output activation by task heuristic.
    //   - Regression (num_outputs == 1 and no class labels) → LINEAR.
    //     Identity outputs let MSE-loss regression learn unbounded
    //     scalars, and they let CrossEntropyLoss classifiers (which
    //     softmax internally) get raw logits with full gradient flow.
    //   - Classification (num_outputs >= 2 OR class_labels non-empty)
    //     → SIGMOID, preserving legacy behavior so callers of
    //     `load_dataset` that already shipped don't see a behavior
    //     change. The orchestrator's SON path overrides per-neuron
    //     after IO setup when it wants logits.
    const bool looks_like_regression =
        (num_outputs == 1 && class_labels.empty());
    const core::ActivationType out_act = looks_like_regression
        ? core::ActivationType::LINEAR
        : core::ActivationType::SIGMOID;

    for (int i = 0; i < num_outputs; ++i) {
        auto id = core::NeuronId::generate();
        std::string label = "output_" + std::to_string(i);
        if (i < static_cast<int>(class_labels.size()))
            label = "out_" + class_labels[i];
        core::Neuron neuron(id, label);
        double y = (i - num_outputs / 2.0) * out_spacing;
        neuron.set_position(core::Vec3{10.0, y, 0.0});
        neuron.set_activation_type(out_act);
        neuron.set_threshold(0.0);

        auto result = network_.add_neuron(std::move(neuron));
        if (!result.ok()) return util::Error("Failed to create output neuron: " + result.error().message);
        output_ids_.push_back(id);
    }
    return util::Result<void>::success();
}

util::Result<void> IOManager::ensure_io_neurons(int num_features, int num_outputs,
                                                  const std::vector<std::string>& class_labels) {
    auto r1 = ensure_input_neurons(num_features);
    if (!r1.ok()) return r1;
    return ensure_output_neurons(num_outputs, class_labels);
}

void IOManager::set_inputs(const std::vector<double>& features) {
    for (size_t i = 0; i < input_ids_.size() && i < features.size(); ++i) {
        auto* neuron = network_.get_neuron(input_ids_[i]);
        if (neuron) {
            neuron->set_activation_level(features[i]);
            neuron->set_state(core::NeuronState::ACTIVE);
        }
    }
}

std::vector<double> IOManager::read_outputs() const {
    std::vector<double> out;
    out.reserve(output_ids_.size());
    for (auto& id : output_ids_) {
        auto* neuron = network_.get_neuron(id);
        out.push_back(neuron ? neuron->activation_level() : 0.0);
    }
    return out;
}

snapshot::IOMapping IOManager::get_io_mapping() const {
    snapshot::IOMapping mapping;
    mapping.input_features = static_cast<int>(input_ids_.size());
    mapping.output_classes = static_cast<int>(output_ids_.size());
    mapping.class_labels = class_labels_;
    for (auto& id : input_ids_) mapping.input_neuron_ids.push_back(id.value());
    for (auto& id : output_ids_) mapping.output_neuron_ids.push_back(id.value());
    return mapping;
}

util::Result<void> IOManager::load_io_mapping(const snapshot::IOMapping& mapping) {
    input_ids_.clear();
    output_ids_.clear();
    class_labels_ = mapping.class_labels;

    for (auto& id_str : mapping.input_neuron_ids) {
        core::NeuronId id(id_str);
        if (!network_.has_neuron(id))
            return util::Error("Input neuron not found: " + id_str);
        input_ids_.push_back(id);
    }
    for (auto& id_str : mapping.output_neuron_ids) {
        core::NeuronId id(id_str);
        if (!network_.has_neuron(id))
            return util::Error("Output neuron not found: " + id_str);
        output_ids_.push_back(id);
    }
    return util::Result<void>::success();
}

} // namespace neurograph::training
