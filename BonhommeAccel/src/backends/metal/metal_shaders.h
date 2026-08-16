/*
 * metal_shaders.h — Metal Shading Language kernel source strings.
 *
 * Embedded as constexpr C++ strings for runtime compilation via MTLDevice.
 * Apple GPUs do not support double in MSL — all kernels use float.
 * Host converts double↔float; histogram bins stay exact (atomic_uint).
 */

#ifndef BA_METAL_SHADERS_H
#define BA_METAL_SHADERS_H

namespace ba::metal {

constexpr const char* kShannonHistogramKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void shannon_histogram(
    device const float* data       [[buffer(0)]],
    device atomic_uint* bins       [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    constant uint& bin_count       [[buffer(3)]],
    constant float& min_val        [[buffer(4)]],
    constant float& bin_width      [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    float v = data[tid];
    if (!isfinite(v)) return;

    int idx = (int)((v - min_val) / bin_width);
    if (idx < 0) idx = 0;
    if (idx >= (int)bin_count) idx = (int)bin_count - 1;
    atomic_fetch_add_explicit(&bins[idx], 1u, memory_order_relaxed);
}
)MSL";

constexpr const char* kCircularHistogramKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void circular_histogram(
    device const float* data       [[buffer(0)]],
    device atomic_uint* bins       [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    constant uint& bin_count       [[buffer(3)]],
    constant float& bin_width      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    float a = data[tid];
    if (!isfinite(a)) return;

    // Wrap to [-180, 180)
    a = fmod(a, 360.0f);
    if (a > 180.0f) a -= 360.0f;
    if (a < -180.0f) a += 360.0f;
    if (a == 180.0f) a = -180.0f;

    int idx = (int)((a + 180.0f) / bin_width);
    if (idx < 0) idx = 0;
    if (idx >= (int)bin_count) idx = (int)bin_count - 1;
    atomic_fetch_add_explicit(&bins[idx], 1u, memory_order_relaxed);
}
)MSL";

// Threadgroup min/max/count of finite float values (one partial per TG).
// Host reduces TG partials in double-friendly float storage.
// TG size should be a power of two (dispatch uses 256).
constexpr const char* kMinMaxReduceKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void minmax_reduce(
    device const float* data       [[buffer(0)]],
    device float* out_mins         [[buffer(1)]],
    device float* out_maxs         [[buffer(2)]],
    device uint* out_counts        [[buffer(3)]],
    constant uint& count           [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint tid_in_tg                 [[thread_position_in_threadgroup]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup float min_s[256];
    threadgroup float max_s[256];
    threadgroup uint  cnt_s[256];

    float lmin = INFINITY;
    float lmax = -INFINITY;
    uint  lcnt = 0u;
    if (tid < count) {
        float v = data[tid];
        if (isfinite(v)) {
            lmin = v;
            lmax = v;
            lcnt = 1u;
        }
    }

    min_s[tid_in_tg] = lmin;
    max_s[tid_in_tg] = lmax;
    cnt_s[tid_in_tg] = lcnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            min_s[tid_in_tg] = min(min_s[tid_in_tg], min_s[tid_in_tg + stride]);
            max_s[tid_in_tg] = max(max_s[tid_in_tg], max_s[tid_in_tg + stride]);
            cnt_s[tid_in_tg] += cnt_s[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid_in_tg == 0) {
        out_mins[tg_id] = min_s[0];
        out_maxs[tg_id] = max_s[0];
        out_counts[tg_id] = cnt_s[0];
    }
}
)MSL";

// Multi-item adaptive histogram: 2D grid (local element x, batch item y).
// bins layout: item * bin_count + bin (strided planes).
constexpr const char* kShannonHistogramBatchKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void shannon_histogram_batch(
    device const float* data           [[buffer(0)]],
    device const uint* offsets         [[buffer(1)]],
    device const uint* lengths         [[buffer(2)]],
    device atomic_uint* bins           [[buffer(3)]],
    device const float* min_vals       [[buffer(4)]],
    device const float* bin_widths     [[buffer(5)]],
    constant uint& bin_count           [[buffer(6)]],
    constant uint& batch_count         [[buffer(7)]],
    uint2 tid                          [[thread_position_in_grid]]
) {
    uint item = tid.y;
    uint local = tid.x;
    if (item >= batch_count) return;
    uint len = lengths[item];
    if (local >= len) return;

    float width = bin_widths[item];
    if (!(width > 0.0f)) return;

    float v = data[offsets[item] + local];
    if (!isfinite(v)) return;

    int idx = (int)((v - min_vals[item]) / width);
    if (idx < 0) idx = 0;
    if (idx >= (int)bin_count) idx = (int)bin_count - 1;
    atomic_fetch_add_explicit(&bins[item * bin_count + (uint)idx], 1u,
                              memory_order_relaxed);
}
)MSL";

// Multi-item circular histogram: same 2D layout, shared circular bin_width.
constexpr const char* kCircularHistogramBatchKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void circular_histogram_batch(
    device const float* data           [[buffer(0)]],
    device const uint* offsets         [[buffer(1)]],
    device const uint* lengths         [[buffer(2)]],
    device atomic_uint* bins           [[buffer(3)]],
    constant uint& bin_count           [[buffer(4)]],
    constant uint& batch_count         [[buffer(5)]],
    constant float& bin_width          [[buffer(6)]],
    uint2 tid                          [[thread_position_in_grid]]
) {
    uint item = tid.y;
    uint local = tid.x;
    if (item >= batch_count) return;
    uint len = lengths[item];
    if (local >= len) return;

    float a = data[offsets[item] + local];
    if (!isfinite(a)) return;

    a = fmod(a, 360.0f);
    if (a > 180.0f) a -= 360.0f;
    if (a < -180.0f) a += 360.0f;
    if (a == 180.0f) a = -180.0f;

    int idx = (int)((a + 180.0f) / bin_width);
    if (idx < 0) idx = 0;
    if (idx >= (int)bin_count) idx = (int)bin_count - 1;
    atomic_fetch_add_explicit(&bins[item * bin_count + (uint)idx], 1u,
                              memory_order_relaxed);
}
)MSL";

// Threadgroup reduction of x/y sums for Pearson means (float32).
constexpr const char* kPearsonMeansKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void pearson_means(
    device const float* x          [[buffer(0)]],
    device const float* y          [[buffer(1)]],
    device atomic_float* out_sx    [[buffer(2)]],
    device atomic_float* out_sy    [[buffer(3)]],
    constant uint& count           [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint tid_in_tg                 [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup float sx_shared[256];
    threadgroup float sy_shared[256];

    float lsx = 0.0f, lsy = 0.0f;
    if (tid < count) {
        lsx = x[tid];
        lsy = y[tid];
    }

    sx_shared[tid_in_tg] = lsx;
    sy_shared[tid_in_tg] = lsy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            sx_shared[tid_in_tg] += sx_shared[tid_in_tg + stride];
            sy_shared[tid_in_tg] += sy_shared[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid_in_tg == 0) {
        atomic_fetch_add_explicit(out_sx, sx_shared[0], memory_order_relaxed);
        atomic_fetch_add_explicit(out_sy, sy_shared[0], memory_order_relaxed);
    }
}
)MSL";

constexpr const char* kPearsonMomentsKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void pearson_moments(
    device const float* x          [[buffer(0)]],
    device const float* y          [[buffer(1)]],
    device atomic_float* out_sxy   [[buffer(2)]],
    device atomic_float* out_sx2   [[buffer(3)]],
    device atomic_float* out_sy2   [[buffer(4)]],
    constant uint& count           [[buffer(5)]],
    constant float& mean_x         [[buffer(6)]],
    constant float& mean_y         [[buffer(7)]],
    uint tid                       [[thread_position_in_grid]],
    uint tid_in_tg                 [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup float sxy_s[256];
    threadgroup float sx2_s[256];
    threadgroup float sy2_s[256];

    float lsxy = 0.0f, lsx2 = 0.0f, lsy2 = 0.0f;
    if (tid < count) {
        float dx = x[tid] - mean_x;
        float dy = y[tid] - mean_y;
        lsxy = dx * dy;
        lsx2 = dx * dx;
        lsy2 = dy * dy;
    }

    sxy_s[tid_in_tg] = lsxy;
    sx2_s[tid_in_tg] = lsx2;
    sy2_s[tid_in_tg] = lsy2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            sxy_s[tid_in_tg] += sxy_s[tid_in_tg + stride];
            sx2_s[tid_in_tg] += sx2_s[tid_in_tg + stride];
            sy2_s[tid_in_tg] += sy2_s[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid_in_tg == 0) {
        atomic_fetch_add_explicit(out_sxy, sxy_s[0], memory_order_relaxed);
        atomic_fetch_add_explicit(out_sx2, sx2_s[0], memory_order_relaxed);
        atomic_fetch_add_explicit(out_sy2, sy2_s[0], memory_order_relaxed);
    }
}
)MSL";

constexpr const char* kPairwiseScoreKernel = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Upper-triangle |x_i - x_j| for float arrays.
kernel void pairwise_abs_diff(
    device const float* data           [[buffer(0)]],
    device float* scores               [[buffer(1)]],
    constant uint& n                   [[buffer(2)]],
    uint tid                           [[thread_position_in_grid]]
) {
    uint num_pairs = n * (n - 1) / 2;
    if (tid >= num_pairs) return;

    float nf = (float)n;
    float t = (float)tid;
    uint i = (uint)(nf - 0.5f - sqrt((nf - 0.5f) * (nf - 0.5f) - 2.0f * t));
    uint row_start = i * n - i * (i + 1) / 2;
    uint j = tid - row_start + i + 1;

    if (i < n && j < n && j > i) {
        scores[tid] = abs(data[i] - data[j]);
    }
}
)MSL";

} // namespace ba::metal

#endif // BA_METAL_SHADERS_H
