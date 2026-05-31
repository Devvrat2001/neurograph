#include "common.cuh"

/// Hebbian weight update: delta_w = learning_rate * source_activation * target_activation.
/// Each thread updates one synapse weight.
__global__ void hebbian_update_kernel(
    const float* activations,    // [N] neuron activations
    float* weights,              // [E] synapse weights (updated in-place)
    const int* row_offsets,      // [N+1]
    const int* col_indices,      // [E]
    float learning_rate,
    int num_neurons)
{
    // Iterate over all synapses: for each source neuron, update its outgoing edges
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= num_neurons) return;

    float src_act = activations[src];
    int start = row_offsets[src];
    int end = row_offsets[src + 1];

    for (int e = start; e < end; ++e) {
        int tgt = col_indices[e];
        float tgt_act = activations[tgt];
        float delta = learning_rate * src_act * tgt_act;
        weights[e] += delta;
    }
}

extern "C" void launch_hebbian_update_kernel(
    const float* activations, float* weights,
    const int* row_offsets, const int* col_indices,
    float learning_rate, int num_neurons, int /*num_synapses*/,
    cudaStream_t stream)
{
    int blocks = neurograph::gpu::grid_size(num_neurons);
    hebbian_update_kernel<<<blocks, neurograph::gpu::BLOCK_SIZE, 0, stream>>>(
        activations, weights, row_offsets, col_indices,
        learning_rate, num_neurons);
    CUDA_KERNEL_CHECK();
}
