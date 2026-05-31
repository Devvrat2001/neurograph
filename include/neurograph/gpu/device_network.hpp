#pragma once

#include "neurograph/core/activation.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace neurograph::core { class Network; }

namespace neurograph::gpu {

/// Structure-of-Arrays layout for GPU computation.
/// All arrays are device pointers allocated via cudaMalloc.
struct DeviceNetworkData {
    // Neuron arrays [N]
    float* d_activation_levels = nullptr;
    float* d_thresholds = nullptr;
    float* d_outputs = nullptr;
    uint8_t* d_states = nullptr;
    uint8_t* d_activation_types = nullptr;
    float* d_positions_x = nullptr;
    float* d_positions_y = nullptr;
    float* d_positions_z = nullptr;

    // Synapse data in CSR format
    int* d_row_offsets = nullptr;    // [N+1]
    int* d_col_indices = nullptr;    // [E] target neuron index
    float* d_weights = nullptr;      // [E]
    int8_t* d_synapse_types = nullptr; // [E] +1 excitatory, -1 inhibitory

    // Workspace
    float* d_affinity_scores = nullptr; // [N*N] or smaller

    int num_neurons = 0;
    int num_synapses = 0;
};

/// Manages GPU memory and host-device transfers for the neural network.
class DeviceNetwork {
public:
    DeviceNetwork() = default;
    ~DeviceNetwork();

    // Non-copyable
    DeviceNetwork(const DeviceNetwork&) = delete;
    DeviceNetwork& operator=(const DeviceNetwork&) = delete;

    // Movable
    DeviceNetwork(DeviceNetwork&& other) noexcept;
    DeviceNetwork& operator=(DeviceNetwork&& other) noexcept;

    /// Allocate device memory for N neurons and E synapses.
    void allocate(int num_neurons, int num_synapses);

    /// Free all device memory.
    void free();

    /// Upload network data from host (CPU) Network object to device.
    /// Converts from object-oriented to SoA + CSR format.
    void upload_from_host(const core::Network& network);

    /// Download results from device back to host Network object.
    /// Updates activation levels and states.
    void download_to_host(core::Network& network) const;

    /// Access the device data (for passing to kernels).
    const DeviceNetworkData& data() const noexcept { return data_; }
    DeviceNetworkData& data() noexcept { return data_; }

    bool is_allocated() const noexcept { return allocated_; }

    // --- GPU kernel launchers ---
    void launch_activate(cudaStream_t stream = nullptr);
    void launch_propagate(cudaStream_t stream = nullptr);
    void launch_hebbian_update(float learning_rate, cudaStream_t stream = nullptr);
    void launch_normalize_weights(float max_total, cudaStream_t stream = nullptr);
    void launch_compute_affinity(cudaStream_t stream = nullptr);

    /// Run a full self-organization cycle on GPU.
    void self_organize_gpu(core::Network& network, int iterations, float learning_rate);

private:
    DeviceNetworkData data_{};
    bool allocated_ = false;

    // Host-side index mapping (neuron ID -> contiguous index)
    // Stored to map back from GPU results to CPU objects
    std::vector<std::string> neuron_index_to_id_;
    std::vector<std::string> synapse_index_to_id_;
};

// --- Kernel declarations (defined in .cu files) ---
extern "C" {
void launch_activate_neurons_kernel(
    float* activations, const float* thresholds, float* outputs,
    uint8_t* states, const uint8_t* act_types,
    const int* row_offsets, const int* col_indices,
    const float* weights, const int8_t* syn_types,
    int num_neurons, cudaStream_t stream);

void launch_propagate_signals_kernel(
    const float* activations, float* outputs,
    const int* row_offsets, const int* col_indices,
    const float* weights, const int8_t* syn_types,
    int num_neurons, cudaStream_t stream);

void launch_hebbian_update_kernel(
    const float* activations, float* weights,
    const int* row_offsets, const int* col_indices,
    float learning_rate, int num_neurons, int num_synapses,
    cudaStream_t stream);

void launch_normalize_weights_kernel(
    float* weights, const int* row_offsets,
    int num_neurons, float max_total,
    cudaStream_t stream);

void launch_compute_affinity_kernel(
    const float* positions_x, const float* positions_y,
    const float* positions_z, const float* activations,
    float* affinity_scores, int num_neurons,
    cudaStream_t stream);
}

} // namespace neurograph::gpu
