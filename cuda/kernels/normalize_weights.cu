#include "common.cuh"

#include <cmath>

/// Per-neuron weight normalization.
/// If total outgoing weight exceeds max_total, scale all outgoing weights down.
/// Each thread handles one neuron.
__global__ void normalize_weights_kernel(
    float* weights,              // [E] synapse weights (updated in-place)
    const int* row_offsets,      // [N+1]
    int num_neurons,
    float max_total)
{
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= num_neurons) return;

    int start = row_offsets[src];
    int end = row_offsets[src + 1];
    if (start == end) return; // no outgoing synapses

    // Compute total absolute weight
    float total = 0.0f;
    for (int e = start; e < end; ++e) {
        total += fabsf(weights[e]);
    }

    // Scale if exceeds max
    if (total > max_total) {
        float scale = max_total / total;
        for (int e = start; e < end; ++e) {
            weights[e] *= scale;
        }
    }
}

extern "C" void launch_normalize_weights_kernel(
    float* weights, const int* row_offsets,
    int num_neurons, float max_total,
    cudaStream_t stream)
{
    int blocks = neurograph::gpu::grid_size(num_neurons);
    normalize_weights_kernel<<<blocks, neurograph::gpu::BLOCK_SIZE, 0, stream>>>(
        weights, row_offsets, num_neurons, max_total);
    CUDA_KERNEL_CHECK();
}
