#include "common.cuh"

#include <cmath>

/// Pairwise affinity scoring for self-organization.
/// Computes affinity[i * N + j] = f(spatial_distance, activity_correlation).
/// Each thread computes one (i, j) pair.
///
/// For N > ~10K, this becomes O(N^2) which may need chunking.
/// For medium scale (10K-100K) with block-based tiling, this is feasible.
__global__ void compute_affinity_kernel(
    const float* pos_x,          // [N]
    const float* pos_y,          // [N]
    const float* pos_z,          // [N]
    const float* activations,    // [N]
    float* affinity_scores,      // [N*N] output
    int num_neurons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_neurons * num_neurons;
    if (idx >= total_pairs) return;

    int i = idx / num_neurons;
    int j = idx % num_neurons;

    if (i == j) {
        affinity_scores[idx] = 0.0f; // no self-connection affinity
        return;
    }

    // Spatial proximity score [0, 1]
    float dx = pos_x[i] - pos_x[j];
    float dy = pos_y[i] - pos_y[j];
    float dz = pos_z[i] - pos_z[j];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float spatial = fmaxf(0.0f, 1.0f - dist / 10.0f);

    // Activity similarity [0, 1]
    // Simple product-based correlation (approximation for GPU efficiency)
    float act_i = activations[i];
    float act_j = activations[j];
    float activity = act_i * act_j; // both active = high affinity

    // Weighted combination
    float score = 0.4f * spatial + 0.6f * activity;
    affinity_scores[idx] = score;
}

extern "C" void launch_compute_affinity_kernel(
    const float* positions_x, const float* positions_y,
    const float* positions_z, const float* activations,
    float* affinity_scores, int num_neurons,
    cudaStream_t stream)
{
    int total_pairs = num_neurons * num_neurons;
    int blocks = neurograph::gpu::grid_size(total_pairs);
    compute_affinity_kernel<<<blocks, neurograph::gpu::BLOCK_SIZE, 0, stream>>>(
        positions_x, positions_y, positions_z,
        activations, affinity_scores, num_neurons);
    CUDA_KERNEL_CHECK();
}
