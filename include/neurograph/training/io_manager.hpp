#pragma once

#include "neurograph/core/network.hpp"
#include "neurograph/snapshot/snapshot_metadata.hpp"
#include "neurograph/util/result.hpp"
#include <vector>
#include <string>

namespace neurograph::training {

class IOManager {
public:
    explicit IOManager(core::Network& network);

    /// Create dedicated input neurons (one per feature). Positions them on the left.
    util::Result<void> ensure_input_neurons(int num_features);

    /// Create dedicated output neurons (one per class/target). Positions them on the right.
    util::Result<void> ensure_output_neurons(int num_outputs,
                                              const std::vector<std::string>& class_labels = {});

    /// Convenience: ensure both input and output neurons exist
    util::Result<void> ensure_io_neurons(int num_features, int num_outputs,
                                          const std::vector<std::string>& class_labels = {});

    /// Clamp input neuron activations to feature values
    void set_inputs(const std::vector<double>& features);

    /// Read raw output neuron activations
    std::vector<double> read_outputs() const;

    /// Build IOMapping for snapshot serialization
    snapshot::IOMapping get_io_mapping() const;

    /// Restore from a saved IOMapping
    util::Result<void> load_io_mapping(const snapshot::IOMapping& mapping);

    const std::vector<core::NeuronId>& input_ids() const { return input_ids_; }
    const std::vector<core::NeuronId>& output_ids() const { return output_ids_; }
    const std::vector<std::string>& class_labels() const { return class_labels_; }

    int num_features() const { return static_cast<int>(input_ids_.size()); }
    int num_outputs() const { return static_cast<int>(output_ids_.size()); }

private:
    core::Network& network_;
    std::vector<core::NeuronId> input_ids_;
    std::vector<core::NeuronId> output_ids_;
    std::vector<std::string> class_labels_;
};

} // namespace neurograph::training
