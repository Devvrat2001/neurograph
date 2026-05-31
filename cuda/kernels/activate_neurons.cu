#include "common.cuh"
#include "neurograph/core/activation.hpp"

using namespace neurograph::core;

// NeuronState values mirrored from neuron.hpp (can't include it in CUDA
// because it pulls in nlohmann/json)
namespace {
    constexpr uint8_t NEURON_IDLE     = 0;
    constexpr uint8_t NEURON_ACTIVE   = 1;
    constexpr uint8_t NEURON_DISABLED = 3;
}

/// Each thread activates one neuron.
/// Reads the accumulated output (from propagation), applies activation function,
/// updates activation_level and state.
__global__ void activate_neurons_kernel(
    float* activations,         // [N] current activation levels (updated in-place)
    const float* thresholds,    // [N]
    float* outputs,             // [N] accumulated input from propagation
    uint8_t* states,            // [N] NeuronState
    const uint8_t* act_types,   // [N] ActivationType
    int num_neurons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    // Skip disabled neurons
    if (states[idx] == NEURON_DISABLED) return;

    float input = outputs[idx];
    ActivationType atype = static_cast<ActivationType>(act_types[idx]);
    float result = device_activate(input, atype);

    activations[idx] = result;
    states[idx] = (result >= thresholds[idx]) ? NEURON_ACTIVE : NEURON_IDLE;
}

extern "C" void launch_activate_neurons_kernel(
    float* activations, const float* thresholds, float* outputs,
    uint8_t* states, const uint8_t* act_types,
    const int* /*row_offsets*/, const int* /*col_indices*/,
    const float* /*weights*/, const int8_t* /*syn_types*/,
    int num_neurons, cudaStream_t stream)
{
    int blocks = neurograph::gpu::grid_size(num_neurons);
    activate_neurons_kernel<<<blocks, neurograph::gpu::BLOCK_SIZE, 0, stream>>>(
        activations, thresholds, outputs, states, act_types, num_neurons);
    CUDA_KERNEL_CHECK();
}
