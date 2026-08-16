/*
 * entropy_hip.cpp — ROCm/HIP-accelerated Shannon entropy.
 *
 * Near-identical to the CUDA implementation using HIP's CUDA compatibility layer.
 */

#if defined(BA_HAS_ROCM)

#include "rocm_backend.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ba::rocm {

// ═══════════════════════════════════════════════════════════════════════════
// Device helpers
// ═══════════════════════════════════════════════════════════════════════════

__device__ inline int clamp_bin(int idx, int bin_count) {
    if (idx < 0) return 0;
    if (idx >= bin_count) return bin_count - 1;
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════
// Histogram Kernels
// ═══════════════════════════════════════════════════════════════════════════

__global__ void shannon_histogram_kernel(
    const double* __restrict__ data, size_t count,
    int bin_count, double min_val, double bin_width,
    int* __restrict__ bins
) {
    extern __shared__ int shared_bins[];

    for (int i = threadIdx.x; i < bin_count; i += blockDim.x) {
        shared_bins[i] = 0;
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        double v = data[idx];
        if (isfinite(v)) {
            int bin = clamp_bin((int)((v - min_val) / bin_width), bin_count);
            atomicAdd(&shared_bins[bin], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < bin_count; i += blockDim.x) {
        atomicAdd(&bins[i], shared_bins[i]);
    }
}

__global__ void circular_histogram_kernel(
    const double* __restrict__ data, size_t count,
    int bin_count, double bin_width,
    int* __restrict__ bins
) {
    extern __shared__ int shared_bins[];

    for (int i = threadIdx.x; i < bin_count; i += blockDim.x) {
        shared_bins[i] = 0;
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        double a = data[idx];
        if (isfinite(a)) {
            a = fmod(a, 360.0);
            if (a > 180.0) a -= 360.0;
            if (a < -180.0) a += 360.0;
            if (a == 180.0) a = -180.0;
            int bin = clamp_bin((int)((a + 180.0) / bin_width), bin_count);
            atomicAdd(&shared_bins[bin], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < bin_count; i += blockDim.x) {
        atomicAdd(&bins[i], shared_bins[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host helpers
// ═══════════════════════════════════════════════════════════════════════════

bool hip_is_available() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    return err == hipSuccess && count > 0;
}

static double entropy_from_bins_host(const int* bins, int bin_count, int total) {
    if (total < 2) return 0.0;
    double d_total = static_cast<double>(total);
    double entropy = 0.0;
    for (int i = 0; i < bin_count; ++i) {
        if (bins[i] > 0) {
            double p = static_cast<double>(bins[i]) / d_total;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Entropy
// ═══════════════════════════════════════════════════════════════════════════

// Failure sentinel: valid Shannon entropy is always >= 0. Caller falls back to CPU.
static constexpr double kHipFail = -1.0;

void shannon_entropy_batch_hip(
    const double* flat, const size_t* offsets, const size_t* lengths,
    size_t batch_count, int bin_count, double* out_entropies
) {
    if (!hip_is_available() || !flat || !offsets || !lengths || !out_entropies || bin_count < 1) {
        if (out_entropies) {
            for (size_t b = 0; b < batch_count; ++b) out_entropies[b] = kHipFail;
        }
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    size_t total_elements = 0;
    for (size_t b = 0; b < batch_count; ++b) {
        total_elements = std::max(total_elements, offsets[b] + lengths[b]);
    }

    double* d_flat = nullptr;
    int* d_bins = nullptr;
    std::vector<int> h_bins(static_cast<size_t>(bin_count));

    if (hipMalloc(&d_flat, total_elements * sizeof(double)) != hipSuccess ||
        hipMalloc(&d_bins, bin_count * sizeof(int)) != hipSuccess) {
        if (d_flat) hipFree(d_flat);
        if (d_bins) hipFree(d_bins);
        for (size_t b = 0; b < batch_count; ++b) out_entropies[b] = kHipFail;
        return;
    }

    hipMemcpy(d_flat, flat, total_elements * sizeof(double), hipMemcpyHostToDevice);

    size_t shared_hist = static_cast<size_t>(bin_count) * sizeof(int);

    for (size_t b = 0; b < batch_count; ++b) {
        size_t n = lengths[b];
        if (n < 2) {
            out_entropies[b] = 0.0;
            continue;
        }

        double min_val = 0.0, max_val = 0.0;
        int clean_count = 0;
        bool have_range = false;
        for (size_t i = 0; i < n; ++i) {
            double v = flat[offsets[b] + i];
            if (std::isfinite(v)) {
                if (!have_range) {
                    min_val = max_val = v;
                    have_range = true;
                } else {
                    if (v < min_val) min_val = v;
                    if (v > max_val) max_val = v;
                }
                ++clean_count;
            }
        }
        if (clean_count < 2 || max_val <= min_val) {
            out_entropies[b] = 0.0;
            continue;
        }

        double bin_width = (max_val - min_val) / static_cast<double>(bin_count);
        hipMemset(d_bins, 0, bin_count * sizeof(int));

        hipLaunchKernelGGL(shannon_histogram_kernel, dim3(1), dim3(BLOCK_SIZE),
            shared_hist, 0, d_flat + offsets[b], n, bin_count, min_val, bin_width, d_bins);

        if (hipMemcpy(h_bins.data(), d_bins, bin_count * sizeof(int),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            out_entropies[b] = kHipFail;
            continue;
        }
        out_entropies[b] = entropy_from_bins_host(h_bins.data(), bin_count, clean_count);
    }

    hipFree(d_flat);
    hipFree(d_bins);
}

void circular_shannon_entropy_batch_hip(
    const double* flat, const size_t* offsets, const size_t* lengths,
    size_t batch_count, int bin_count, double* out_entropies
) {
    if (!hip_is_available() || !flat || !offsets || !lengths || !out_entropies || bin_count < 1) {
        if (out_entropies) {
            for (size_t b = 0; b < batch_count; ++b) out_entropies[b] = kHipFail;
        }
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    size_t total_elements = 0;
    for (size_t b = 0; b < batch_count; ++b) {
        total_elements = std::max(total_elements, offsets[b] + lengths[b]);
    }

    double* d_flat = nullptr;
    int* d_bins = nullptr;
    std::vector<int> h_bins(static_cast<size_t>(bin_count));

    if (hipMalloc(&d_flat, total_elements * sizeof(double)) != hipSuccess ||
        hipMalloc(&d_bins, bin_count * sizeof(int)) != hipSuccess) {
        if (d_flat) hipFree(d_flat);
        if (d_bins) hipFree(d_bins);
        for (size_t b = 0; b < batch_count; ++b) out_entropies[b] = kHipFail;
        return;
    }

    hipMemcpy(d_flat, flat, total_elements * sizeof(double), hipMemcpyHostToDevice);

    double bin_width = 360.0 / static_cast<double>(bin_count);
    size_t shared_hist = static_cast<size_t>(bin_count) * sizeof(int);

    for (size_t b = 0; b < batch_count; ++b) {
        size_t n = lengths[b];
        if (n < 2) {
            out_entropies[b] = 0.0;
            continue;
        }

        int clean_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(flat[offsets[b] + i])) ++clean_count;
        }
        if (clean_count < 2) {
            out_entropies[b] = 0.0;
            continue;
        }

        hipMemset(d_bins, 0, bin_count * sizeof(int));

        hipLaunchKernelGGL(circular_histogram_kernel, dim3(1), dim3(BLOCK_SIZE),
            shared_hist, 0, d_flat + offsets[b], n, bin_count, bin_width, d_bins);

        if (hipMemcpy(h_bins.data(), d_bins, bin_count * sizeof(int),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            out_entropies[b] = kHipFail;
            continue;
        }
        out_entropies[b] = entropy_from_bins_host(h_bins.data(), bin_count, clean_count);
    }

    hipFree(d_flat);
    hipFree(d_bins);
}

double shannon_entropy_hip(const double* values, size_t count, int bin_count) {
    if (!values || count < 2 || bin_count < 1) return 0.0;
    size_t offset = 0;
    size_t length = count;
    double out = kHipFail;
    shannon_entropy_batch_hip(values, &offset, &length, 1, bin_count, &out);
    return out;
}

double circular_shannon_entropy_hip(const double* angles, size_t count, int bin_count) {
    if (!angles || count < 2 || bin_count < 1) return 0.0;
    size_t offset = 0;
    size_t length = count;
    double out = kHipFail;
    circular_shannon_entropy_batch_hip(angles, &offset, &length, 1, bin_count, &out);
    return out;
}

} // namespace ba::rocm

#endif // BA_HAS_ROCM
