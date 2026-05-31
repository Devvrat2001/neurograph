#include "common.cuh"

/// SpMV-style signal propagation through the synapse CSR matrix.
/// Each thread computes the input sum for one target neuron by iterating
/// over all incoming synapses (stored as CSR rows of the *transpose* graph).
///
/// Since our CSR stores outgoing synapses per source neuron, we do the
/// propagation by having each thread iterate over its outgoing edges
/// and atomically add to the target neuron's output.
__global__ void propagate_signals_kernel(
    const float* activations,    // [N] source neuron activations
    float* outputs,              // [N] accumulated input for each neuron (atomic adds)
    const int* row_offsets,      // [N+1] CSR row pointers
    const int* col_indices,      // [E] target neuron indices
    const float* weights,        // [E] synapse weights
    const int8_t* syn_types,     // [E] +1 or -1
    int num_neurons)
{
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= num_neurons) return;

    float src_activation = activations[src];
    if (src_activation == 0.0f) return; // skip inactive neurons

    int start = row_offsets[src];
    int end = row_offsets[src + 1];

    for (int e = start; e < end; ++e) {
        int tgt = col_indices[e];
        float signal = src_activation * weights[e] * static_cast<float>(syn_types[e]);
        atomicAdd(&outputs[tgt], signal);
    }
}

extern "C" void launch_propagate_signals_kernel(
    const float* activations, float* outputs,
    const int* row_offsets, const int* col_indices,
    const float* weights, const int8_t* syn_types,
    int num_neurons, cudaStream_t stream)
{
    // Zero the outputs first
    cudaMemsetAsync(outputs, 0, num_neurons * sizeof(float), stream);

    int blocks = neurograph::gpu::grid_size(num_neurons);
    propagate_signals_kernel<<<blocks, neurograph::gpu::BLOCK_SIZE, 0, stream>>>(
        activations, outputs, row_offsets, col_indices,
        weights, syn_types, num_neurons);
    CUDA_KERNEL_CHECK();
}
