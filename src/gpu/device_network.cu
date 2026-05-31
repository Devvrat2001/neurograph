#include "neurograph/gpu/device_network.hpp"
#include "neurograph/core/network.hpp"
#include "common.cuh"

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <cstring>

namespace neurograph::gpu {

DeviceNetwork::~DeviceNetwork() {
    free();
}

DeviceNetwork::DeviceNetwork(DeviceNetwork&& other) noexcept
    : data_(other.data_)
    , allocated_(other.allocated_)
    , neuron_index_to_id_(std::move(other.neuron_index_to_id_))
    , synapse_index_to_id_(std::move(other.synapse_index_to_id_))
{
    other.data_ = {};
    other.allocated_ = false;
}

DeviceNetwork& DeviceNetwork::operator=(DeviceNetwork&& other) noexcept {
    if (this != &other) {
        free();
        data_ = other.data_;
        allocated_ = other.allocated_;
        neuron_index_to_id_ = std::move(other.neuron_index_to_id_);
        synapse_index_to_id_ = std::move(other.synapse_index_to_id_);
        other.data_ = {};
        other.allocated_ = false;
    }
    return *this;
}

void DeviceNetwork::allocate(int num_neurons, int num_synapses) {
    free();

    data_.num_neurons = num_neurons;
    data_.num_synapses = num_synapses;
    int N = num_neurons;
    int E = num_synapses;

    // Neuron arrays
    CUDA_CHECK(cudaMalloc(&data_.d_activation_levels, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data_.d_thresholds, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data_.d_outputs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data_.d_states, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&data_.d_activation_types, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&data_.d_positions_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data_.d_positions_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data_.d_positions_z, N * sizeof(float)));

    // CSR arrays
    CUDA_CHECK(cudaMalloc(&data_.d_row_offsets, (N + 1) * sizeof(int)));
    if (E > 0) {
        CUDA_CHECK(cudaMalloc(&data_.d_col_indices, E * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&data_.d_weights, E * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&data_.d_synapse_types, E * sizeof(int8_t)));
    }

    // Zero-initialize outputs
    CUDA_CHECK(cudaMemset(data_.d_outputs, 0, N * sizeof(float)));

    allocated_ = true;
}

void DeviceNetwork::free() {
    if (!allocated_) return;

    cudaFree(data_.d_activation_levels);
    cudaFree(data_.d_thresholds);
    cudaFree(data_.d_outputs);
    cudaFree(data_.d_states);
    cudaFree(data_.d_activation_types);
    cudaFree(data_.d_positions_x);
    cudaFree(data_.d_positions_y);
    cudaFree(data_.d_positions_z);
    cudaFree(data_.d_row_offsets);
    cudaFree(data_.d_col_indices);
    cudaFree(data_.d_weights);
    cudaFree(data_.d_synapse_types);
    cudaFree(data_.d_affinity_scores);

    data_ = {};
    allocated_ = false;
}

void DeviceNetwork::upload_from_host(const core::Network& network) {
    int N = static_cast<int>(network.neuron_count());
    int E = static_cast<int>(network.synapse_count());

    allocate(N, E);

    // Build neuron ID -> contiguous index mapping
    neuron_index_to_id_.clear();
    neuron_index_to_id_.reserve(N);
    std::unordered_map<std::string, int> id_to_index;

    int idx = 0;
    for (const auto& [nid, neuron] : network.neurons()) {
        id_to_index[nid.value()] = idx;
        neuron_index_to_id_.push_back(nid.value());
        ++idx;
    }

    // Fill host arrays
    std::vector<float> h_activations(N), h_thresholds(N);
    std::vector<float> h_pos_x(N), h_pos_y(N), h_pos_z(N);
    std::vector<uint8_t> h_states(N), h_act_types(N);

    idx = 0;
    for (const auto& [nid, neuron] : network.neurons()) {
        h_activations[idx] = static_cast<float>(neuron.activation_level());
        h_thresholds[idx] = static_cast<float>(neuron.threshold());
        h_states[idx] = static_cast<uint8_t>(neuron.state());
        h_act_types[idx] = static_cast<uint8_t>(neuron.activation_type());
        if (neuron.position()) {
            h_pos_x[idx] = static_cast<float>(neuron.position()->x);
            h_pos_y[idx] = static_cast<float>(neuron.position()->y);
            h_pos_z[idx] = static_cast<float>(neuron.position()->z);
        }
        ++idx;
    }

    // Build CSR format for synapses
    // row_offsets[i] = index into col_indices where neuron i's outgoing synapses begin
    std::vector<int> h_row_offsets(N + 1, 0);
    std::vector<int> h_col_indices;
    std::vector<float> h_weights;
    std::vector<int8_t> h_syn_types;

    h_col_indices.reserve(E);
    h_weights.reserve(E);
    h_syn_types.reserve(E);

    synapse_index_to_id_.clear();

    // Count outgoing synapses per neuron
    for (const auto& [sid, syn] : network.synapses()) {
        auto it = id_to_index.find(syn.source_id().value());
        if (it != id_to_index.end()) {
            h_row_offsets[it->second + 1]++;
        }
    }

    // Prefix sum
    for (int i = 0; i < N; ++i) {
        h_row_offsets[i + 1] += h_row_offsets[i];
    }

    // Fill CSR data (using offset tracking for each row)
    std::vector<int> current_offset(h_row_offsets.begin(), h_row_offsets.end() - 1);
    h_col_indices.resize(E);
    h_weights.resize(E);
    h_syn_types.resize(E);
    synapse_index_to_id_.resize(E);

    for (const auto& [sid, syn] : network.synapses()) {
        auto src_it = id_to_index.find(syn.source_id().value());
        auto tgt_it = id_to_index.find(syn.target_id().value());
        if (src_it == id_to_index.end() || tgt_it == id_to_index.end()) continue;

        int src_idx = src_it->second;
        int pos = current_offset[src_idx]++;
        h_col_indices[pos] = tgt_it->second;
        h_weights[pos] = static_cast<float>(syn.weight());
        h_syn_types[pos] = (syn.type() == core::SynapseType::EXCITATORY) ? 1 : -1;
        synapse_index_to_id_[pos] = sid.value();
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(data_.d_activation_levels, h_activations.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_thresholds, h_thresholds.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_states, h_states.data(),
                          N * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_activation_types, h_act_types.data(),
                          N * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_positions_x, h_pos_x.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_positions_y, h_pos_y.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_positions_z, h_pos_z.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_.d_row_offsets, h_row_offsets.data(),
                          (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (E > 0) {
        CUDA_CHECK(cudaMemcpy(data_.d_col_indices, h_col_indices.data(),
                              E * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data_.d_weights, h_weights.data(),
                              E * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data_.d_synapse_types, h_syn_types.data(),
                              E * sizeof(int8_t), cudaMemcpyHostToDevice));
    }
}

void DeviceNetwork::download_to_host(core::Network& network) const {
    if (!allocated_) return;

    int N = data_.num_neurons;
    int E = data_.num_synapses;

    // Download neuron data
    std::vector<float> h_activations(N);
    std::vector<uint8_t> h_states(N);

    CUDA_CHECK(cudaMemcpy(h_activations.data(), data_.d_activation_levels,
                          N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_states.data(), data_.d_states,
                          N * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Update CPU neurons
    for (int i = 0; i < N; ++i) {
        auto* neuron = network.get_neuron(core::NeuronId(neuron_index_to_id_[i]));
        if (!neuron) continue;
        neuron->set_activation_level(static_cast<double>(h_activations[i]));
        neuron->set_state(static_cast<core::NeuronState>(h_states[i]));
    }

    // Download synapse weights
    if (E > 0) {
        std::vector<float> h_weights(E);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), data_.d_weights,
                              E * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < E; ++i) {
            auto* synapse = network.get_synapse(core::SynapseId(synapse_index_to_id_[i]));
            if (!synapse) continue;
            synapse->set_weight(static_cast<double>(h_weights[i]));
        }
    }
}

void DeviceNetwork::launch_activate(cudaStream_t stream) {
    if (!allocated_ || data_.num_neurons == 0) return;
    launch_activate_neurons_kernel(
        data_.d_activation_levels, data_.d_thresholds, data_.d_outputs,
        data_.d_states, data_.d_activation_types,
        data_.d_row_offsets, data_.d_col_indices,
        data_.d_weights, data_.d_synapse_types,
        data_.num_neurons, stream);
}

void DeviceNetwork::launch_propagate(cudaStream_t stream) {
    if (!allocated_ || data_.num_neurons == 0) return;
    launch_propagate_signals_kernel(
        data_.d_activation_levels, data_.d_outputs,
        data_.d_row_offsets, data_.d_col_indices,
        data_.d_weights, data_.d_synapse_types,
        data_.num_neurons, stream);
}

void DeviceNetwork::launch_hebbian_update(float learning_rate, cudaStream_t stream) {
    if (!allocated_ || data_.num_synapses == 0) return;
    launch_hebbian_update_kernel(
        data_.d_activation_levels, data_.d_weights,
        data_.d_row_offsets, data_.d_col_indices,
        learning_rate, data_.num_neurons, data_.num_synapses, stream);
}

void DeviceNetwork::launch_normalize_weights(float max_total, cudaStream_t stream) {
    if (!allocated_ || data_.num_neurons == 0) return;
    launch_normalize_weights_kernel(
        data_.d_weights, data_.d_row_offsets,
        data_.num_neurons, max_total, stream);
}

void DeviceNetwork::launch_compute_affinity(cudaStream_t stream) {
    if (!allocated_ || data_.num_neurons == 0) return;

    // Allocate affinity scores if not already
    int N = data_.num_neurons;
    if (!data_.d_affinity_scores) {
        CUDA_CHECK(cudaMalloc(&data_.d_affinity_scores, N * N * sizeof(float)));
    }

    launch_compute_affinity_kernel(
        data_.d_positions_x, data_.d_positions_y, data_.d_positions_z,
        data_.d_activation_levels, data_.d_affinity_scores,
        N, stream);
}

void DeviceNetwork::self_organize_gpu(core::Network& network, int iterations,
                                       float learning_rate) {
    upload_from_host(network);

    for (int i = 0; i < iterations; ++i) {
        // Step 1: Propagate signals through network
        launch_propagate(nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Apply activation functions
        launch_activate(nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Hebbian weight update
        launch_hebbian_update(learning_rate, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 4: Normalize weights
        launch_normalize_weights(10.0f, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    download_to_host(network);
}

} // namespace neurograph::gpu
