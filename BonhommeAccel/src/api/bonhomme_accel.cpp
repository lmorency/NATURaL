/*
 * bonhomme_accel.cpp — C API implementation.
 *
 * Thin shim that validates parameters, dispatches to the appropriate
 * backend via ba::get_backend(), and translates C++ results to C structs.
 *
 * Dispatch: CUDA / ROCm / Metal / NEON / AVX2 / OpenMP when compiled in,
 * else scalar core. GPU paths fall back to SIMD/scalar on kernel failure
 * and when N is below the GPU break-even threshold (launch + sync dominate).
 */

#include "BonhommeAccel.h"
#include "../dispatch/backend.h"
#include "../core/entropy.h"
#include "../core/correlation.h"
#include "../core/statistics.h"
#include "../core/incomplete_beta.h"
#include "../core/pairwise.h"

#if defined(BA_HAS_NEON) || defined(BA_HAS_AVX2) || defined(BA_HAS_AVX512)
#include "../backends/simd/entropy_simd.h"
#include "../backends/simd/correlation_simd.h"
#endif

#if defined(BA_HAS_OPENMP)
#include "../backends/openmp/entropy_omp.h"
#include "../backends/openmp/pairwise_omp.h"
#endif

#if defined(BA_HAS_METAL)
#include "../backends/metal/metal_backend.h"
#endif

#if defined(BA_HAS_CUDA)
#include "../backends/cuda/cuda_backend.h"
#endif

#if defined(BA_HAS_ROCM)
#include "../backends/rocm/rocm_backend.h"
#endif

#include <cmath>
#include <cstddef>

// ═══════════════════════════════════════════════════════════════════════════
// GPU break-even thresholds
//
// Histogram entropy is O(n) with a tiny bin reduce. GPU path pays:
//   buffer alloc + H2D + kernel launch + waitUntilCompleted/D2H.
// On Apple Silicon that is typically multi-millisecond; NEON finishes
// typical SCI/HRV and docking windows (64–4k samples) in microseconds.
// Tuned conservatively so Metal/CUDA only run when they can win.
// ═══════════════════════════════════════════════════════════════════════════

static constexpr size_t kGpuEntropyMinN     = 8192;
/** Multi-item Metal batches: use GPU when total elements reach this even if
 *  no single item hits kGpuEntropyMinN (amortized multi-hist launch). */
static constexpr size_t kGpuEntropyMinTotal = 32768;
static constexpr size_t kGpuPearsonMinN     = 16384;

// ═══════════════════════════════════════════════════════════════════════════
// Version
// ═══════════════════════════════════════════════════════════════════════════

const char* ba_version(void) {
    return "1.1.1";
}

// ═══════════════════════════════════════════════════════════════════════════
// Backend Management
// ═══════════════════════════════════════════════════════════════════════════

// Keep in sync with kGpuEntropyMinN below — single source for recommend API.
static constexpr size_t kRecommendGpuMinN = 4096;

static BABackend cpu_fallback_backend() {
#if defined(BA_HAS_NEON) && (defined(__aarch64__) || defined(_M_ARM64))
    return BA_BACKEND_NEON;
#elif defined(BA_HAS_AVX512)
    return BA_BACKEND_AVX512;
#elif defined(BA_HAS_AVX2)
    return BA_BACKEND_AVX2;
#elif defined(BA_HAS_OPENMP)
    return BA_BACKEND_OPENMP;
#else
    return BA_BACKEND_SCALAR;
#endif
}

BABackend ba_detect_best_backend(void) {
    return ba::get_backend();
}

BABackend ba_get_active_backend(void) {
    return ba::get_backend();
}

void ba_set_backend_override(BABackend backend) {
    ba::g_active_backend.store(static_cast<int>(backend), std::memory_order_release);
}

void ba_clear_backend_override(void) {
    ba::g_active_backend.store(-1, std::memory_order_release);
}

BABackend ba_recommend_backend_for_n(size_t element_count) {
    // Use the cached active backend rather than re-enumerating hardware per call.
    BABackend peak = ba::get_backend();
    const bool is_gpu =
        peak == BA_BACKEND_METAL || peak == BA_BACKEND_CUDA || peak == BA_BACKEND_ROCM;
    if (is_gpu && element_count < kRecommendGpuMinN) {
        return cpu_fallback_backend();
    }
    return peak;
}

const char* ba_backend_name(BABackend backend) {
    switch (backend) {
        case BA_BACKEND_SCALAR:  return "Scalar";
        case BA_BACKEND_NEON:    return "NEON";
        case BA_BACKEND_AVX2:    return "AVX2";
        case BA_BACKEND_AVX512:  return "AVX-512";
        case BA_BACKEND_OPENMP:  return "OpenMP";
        case BA_BACKEND_METAL:   return "Metal";
        case BA_BACKEND_CUDA:    return "CUDA";
        case BA_BACKEND_ROCM:    return "ROCm";
        default:                 return "Unknown";
    }
}

const char* ba_status_string(BAStatus status) {
    switch (status) {
        case BA_OK:                      return "OK";
        case BA_ERR_NULL_PTR:            return "Null pointer";
        case BA_ERR_INSUFFICIENT_DATA:   return "Insufficient data";
        case BA_ERR_SIZE_MISMATCH:       return "Size mismatch";
        case BA_ERR_INVALID_PARAM:       return "Invalid parameter";
        case BA_ERR_BACKEND_UNAVAILABLE: return "Backend unavailable";
        case BA_ERR_ALLOC_FAILED:        return "Allocation failed";
        default:                         return "Unknown error";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU fallbacks (used when GPU returns failure sentinel or for small-N skip)
// ═══════════════════════════════════════════════════════════════════════════

static double fallback_shannon(const double* values, size_t count, int bin_count) {
#if defined(BA_HAS_NEON)
    return ba::simd::shannon_entropy_neon(values, count, bin_count);
#elif defined(BA_HAS_AVX512)
    return ba::simd::shannon_entropy_avx512(values, count, bin_count);
#elif defined(BA_HAS_AVX2)
    return ba::simd::shannon_entropy_avx2(values, count, bin_count);
#else
    return ba::core::shannon_entropy(values, count, bin_count);
#endif
}

static double fallback_circular(const double* angles, size_t count, int bin_count) {
#if defined(BA_HAS_NEON)
    return ba::simd::circular_shannon_entropy_neon(angles, count, bin_count);
#elif defined(BA_HAS_AVX512)
    return ba::simd::circular_shannon_entropy_avx512(angles, count, bin_count);
#elif defined(BA_HAS_AVX2)
    return ba::simd::circular_shannon_entropy_avx2(angles, count, bin_count);
#else
    return ba::core::circular_shannon_entropy(angles, count, bin_count);
#endif
}

static double fallback_shannon_fixed(const double* values, size_t count, int bin_count,
                                      double domain_min, double domain_max) {
#if defined(BA_HAS_NEON)
    return ba::simd::shannon_entropy_fixed_neon(values, count, bin_count,
                                                 domain_min, domain_max);
#elif defined(BA_HAS_AVX2)
    return ba::simd::shannon_entropy_fixed_avx2(values, count, bin_count,
                                                 domain_min, domain_max);
#else
    return ba::core::shannon_entropy_fixed(values, count, bin_count,
                                            domain_min, domain_max);
#endif
}

static double fallback_pearson(const double* x, const double* y, size_t count) {
#if defined(BA_HAS_NEON)
    return ba::simd::pearson_correlation_neon(x, y, count);
#elif defined(BA_HAS_AVX512)
    return ba::simd::pearson_correlation_avx512(x, y, count);
#elif defined(BA_HAS_AVX2)
    return ba::simd::pearson_correlation_avx2(x, y, count);
#else
    return ba::core::pearson_correlation(x, y, count);
#endif
}

/** Best CPU batch path: OpenMP (SIMD per item) > SIMD loop > scalar. */
static void fallback_shannon_batch(const double* values_flat,
                                    const size_t* offsets, const size_t* lengths,
                                    size_t batch_count, int bin_count,
                                    double* out_entropies) {
#if defined(BA_HAS_OPENMP)
    ba::omp::shannon_entropy_batch_omp(values_flat, offsets, lengths,
                                        batch_count, bin_count, out_entropies);
#elif defined(BA_HAS_AVX512)
    ba::simd::shannon_entropy_batch_avx512(values_flat, offsets, lengths,
                                            batch_count, bin_count, out_entropies);
#elif defined(BA_HAS_AVX2)
    ba::simd::shannon_entropy_batch_avx2(values_flat, offsets, lengths,
                                          batch_count, bin_count, out_entropies);
#elif defined(BA_HAS_NEON)
    for (size_t b = 0; b < batch_count; ++b) {
        out_entropies[b] = ba::simd::shannon_entropy_neon(
            values_flat + offsets[b], lengths[b], bin_count);
    }
#else
    ba::core::shannon_entropy_batch(values_flat, offsets, lengths,
                                     batch_count, bin_count, out_entropies);
#endif
}

static void fallback_circular_batch(const double* values_flat,
                                     const size_t* offsets, const size_t* lengths,
                                     size_t batch_count, int bin_count,
                                     double* out_entropies) {
#if defined(BA_HAS_OPENMP)
    ba::omp::circular_shannon_entropy_batch_omp(values_flat, offsets, lengths,
                                                 batch_count, bin_count, out_entropies);
#elif defined(BA_HAS_NEON)
    for (size_t b = 0; b < batch_count; ++b) {
        out_entropies[b] = ba::simd::circular_shannon_entropy_neon(
            values_flat + offsets[b], lengths[b], bin_count);
    }
#elif defined(BA_HAS_AVX512)
    for (size_t b = 0; b < batch_count; ++b) {
        out_entropies[b] = ba::simd::circular_shannon_entropy_avx512(
            values_flat + offsets[b], lengths[b], bin_count);
    }
#elif defined(BA_HAS_AVX2)
    for (size_t b = 0; b < batch_count; ++b) {
        out_entropies[b] = ba::simd::circular_shannon_entropy_avx2(
            values_flat + offsets[b], lengths[b], bin_count);
    }
#else
    ba::core::circular_shannon_entropy_batch(
        values_flat, offsets, lengths, batch_count, bin_count, out_entropies);
#endif
}

static bool is_gpu_backend(BABackend b) {
    return b == BA_BACKEND_METAL || b == BA_BACKEND_CUDA || b == BA_BACKEND_ROCM;
}

/** Max array length in a batch (for GPU break-even decisions). */
static size_t batch_max_length(const size_t* lengths, size_t batch_count) {
    size_t m = 0;
    for (size_t b = 0; b < batch_count; ++b) {
        if (lengths[b] > m) m = lengths[b];
    }
    return m;
}

/** Sum of lengths across a batch (Metal multi-hist amortization threshold). */
static size_t batch_total_elements(const size_t* lengths, size_t batch_count) {
    size_t t = 0;
    for (size_t b = 0; b < batch_count; ++b) {
        t += lengths[b];
    }
    return t;
}

/**
 * Whether a GPU backend should attempt this batch.
 * Metal: multi-item OK when max_len >= kGpuEntropyMinN OR total >= kGpuEntropyMinTotal
 *   (true multi-histogram amortizes a single wait).
 * CUDA/ROCm: still serial per-item waits — skip multi-item and sub-threshold.
 */
static bool gpu_batch_eligible(BABackend be, const size_t* lengths,
                                size_t batch_count, size_t max_len) {
#if defined(BA_HAS_METAL)
    if (be == BA_BACKEND_METAL) {
        if (max_len >= kGpuEntropyMinN) return true;
        if (batch_total_elements(lengths, batch_count) >= kGpuEntropyMinTotal)
            return true;
        return false;
    }
#else
    (void)lengths;
    (void)batch_count;
#endif
    // CUDA / ROCm (and unknown GPU): multi-item still serial GPU waits.
    if (batch_count > 1) return false;
    return max_len >= kGpuEntropyMinN;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal dispatch helpers
// ═══════════════════════════════════════════════════════════════════════════

static double dispatch_shannon(const double* values, size_t count, int bin_count) {
    BABackend be = ba::get_backend();

    // Skip GPU for small N — launch/sync cost dwarfs the work.
    if (is_gpu_backend(be) && count < kGpuEntropyMinN) {
        return fallback_shannon(values, count, bin_count);
    }

    switch (be) {
#if defined(BA_HAS_CUDA)
        case BA_BACKEND_CUDA: {
            double r = ba::cuda::shannon_entropy_cuda(values, count, bin_count);
            // Failure sentinel (-1); valid entropy is always >= 0.
            return (r >= 0.0) ? r : fallback_shannon(values, count, bin_count);
        }
#endif
#if defined(BA_HAS_ROCM)
        case BA_BACKEND_ROCM: {
            double r = ba::rocm::shannon_entropy_hip(values, count, bin_count);
            return (r >= 0.0) ? r : fallback_shannon(values, count, bin_count);
        }
#endif
#if defined(BA_HAS_METAL)
        case BA_BACKEND_METAL: {
            double r = ba::metal::shannon_entropy_metal(values, count, bin_count);
            return (r >= 0.0) ? r : fallback_shannon(values, count, bin_count);
        }
#endif
#if defined(BA_HAS_NEON)
        case BA_BACKEND_NEON:
            return ba::simd::shannon_entropy_neon(values, count, bin_count);
#endif
#if defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
            return ba::simd::shannon_entropy_avx512(values, count, bin_count);
#endif
#if defined(BA_HAS_AVX2)
        case BA_BACKEND_AVX2:
#if !defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
#endif
            return ba::simd::shannon_entropy_avx2(values, count, bin_count);
#endif
        default:
            return ba::core::shannon_entropy(values, count, bin_count);
    }
}

static double dispatch_circular(const double* angles, size_t count, int bin_count) {
    BABackend be = ba::get_backend();

    if (is_gpu_backend(be) && count < kGpuEntropyMinN) {
        return fallback_circular(angles, count, bin_count);
    }

    switch (be) {
#if defined(BA_HAS_CUDA)
        case BA_BACKEND_CUDA: {
            double r = ba::cuda::circular_shannon_entropy_cuda(angles, count, bin_count);
            return (r >= 0.0) ? r : fallback_circular(angles, count, bin_count);
        }
#endif
#if defined(BA_HAS_ROCM)
        case BA_BACKEND_ROCM: {
            double r = ba::rocm::circular_shannon_entropy_hip(angles, count, bin_count);
            return (r >= 0.0) ? r : fallback_circular(angles, count, bin_count);
        }
#endif
#if defined(BA_HAS_METAL)
        case BA_BACKEND_METAL: {
            double r = ba::metal::circular_shannon_entropy_metal(angles, count, bin_count);
            return (r >= 0.0) ? r : fallback_circular(angles, count, bin_count);
        }
#endif
#if defined(BA_HAS_NEON)
        case BA_BACKEND_NEON:
            return ba::simd::circular_shannon_entropy_neon(angles, count, bin_count);
#endif
#if defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
            return ba::simd::circular_shannon_entropy_avx512(angles, count, bin_count);
#endif
#if defined(BA_HAS_AVX2)
        case BA_BACKEND_AVX2:
#if !defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
#endif
            return ba::simd::circular_shannon_entropy_avx2(angles, count, bin_count);
#endif
        default:
            return ba::core::circular_shannon_entropy(angles, count, bin_count);
    }
}

/** Fixed-domain Shannon: no GPU kernel yet — prefer NEON/AVX2 (incl. Metal fallback). */
static double dispatch_shannon_fixed(const double* values, size_t count, int bin_count,
                                      double domain_min, double domain_max) {
    BABackend be = ba::get_backend();

    switch (be) {
#if defined(BA_HAS_NEON)
        case BA_BACKEND_NEON:
            return ba::simd::shannon_entropy_fixed_neon(values, count, bin_count,
                                                        domain_min, domain_max);
        // Metal has no fixed-domain kernel; use NEON like adaptive small-N fallback.
        case BA_BACKEND_METAL:
            return ba::simd::shannon_entropy_fixed_neon(values, count, bin_count,
                                                        domain_min, domain_max);
#endif
#if defined(BA_HAS_AVX2)
        case BA_BACKEND_AVX2:
        case BA_BACKEND_AVX512:
            // No dedicated fixed-domain AVX-512 kernel; AVX2 path is correct.
            return ba::simd::shannon_entropy_fixed_avx2(values, count, bin_count,
                                                        domain_min, domain_max);
#endif
#if defined(BA_HAS_CUDA) || defined(BA_HAS_ROCM)
        case BA_BACKEND_CUDA:
        case BA_BACKEND_ROCM:
            // No GPU fixed-domain kernel — SIMD/scalar CPU path.
            return fallback_shannon_fixed(values, count, bin_count,
                                           domain_min, domain_max);
#endif
        default:
            return fallback_shannon_fixed(values, count, bin_count,
                                           domain_min, domain_max);
    }
}

static void dispatch_shannon_batch(const double* values_flat,
                                    const size_t* offsets, const size_t* lengths,
                                    size_t batch_count, int bin_count,
                                    double* out_entropies) {
    BABackend be = ba::get_backend();
    const size_t max_len = batch_max_length(lengths, batch_count);

    // GPU eligibility: Metal multi-hist can amortize multi-item batches when
    // total work is large; CUDA/ROCm remain single-item-only (serial waits).
    if (is_gpu_backend(be) &&
        !gpu_batch_eligible(be, lengths, batch_count, max_len)) {
        fallback_shannon_batch(values_flat, offsets, lengths,
                               batch_count, bin_count, out_entropies);
        return;
    }

    switch (be) {
#if defined(BA_HAS_CUDA)
        case BA_BACKEND_CUDA:
            ba::cuda::shannon_entropy_batch_cuda(values_flat, offsets, lengths,
                                                  batch_count, bin_count, out_entropies);
            for (size_t b = 0; b < batch_count; ++b) {
                if (out_entropies[b] < 0.0) {
                    out_entropies[b] = fallback_shannon(
                        values_flat + offsets[b], lengths[b], bin_count);
                }
            }
            return;
#endif
#if defined(BA_HAS_ROCM)
        case BA_BACKEND_ROCM:
            ba::rocm::shannon_entropy_batch_hip(values_flat, offsets, lengths,
                                                 batch_count, bin_count, out_entropies);
            for (size_t b = 0; b < batch_count; ++b) {
                if (out_entropies[b] < 0.0) {
                    out_entropies[b] = fallback_shannon(
                        values_flat + offsets[b], lengths[b], bin_count);
                }
            }
            return;
#endif
#if defined(BA_HAS_METAL)
        case BA_BACKEND_METAL:
            ba::metal::shannon_entropy_batch_metal(values_flat, offsets, lengths,
                                                    batch_count, bin_count, out_entropies);
            for (size_t b = 0; b < batch_count; ++b) {
                if (out_entropies[b] < 0.0) {
                    out_entropies[b] = fallback_shannon(
                        values_flat + offsets[b], lengths[b], bin_count);
                }
            }
            return;
#endif
#if defined(BA_HAS_OPENMP)
        case BA_BACKEND_OPENMP:
            ba::omp::shannon_entropy_batch_omp(values_flat, offsets, lengths,
                                                batch_count, bin_count, out_entropies);
            return;
#endif
#if defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
            ba::simd::shannon_entropy_batch_avx512(values_flat, offsets, lengths,
                                                    batch_count, bin_count, out_entropies);
            return;
#endif
#if defined(BA_HAS_AVX2)
        case BA_BACKEND_AVX2:
#if !defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
#endif
            ba::simd::shannon_entropy_batch_avx2(values_flat, offsets, lengths,
                                                  batch_count, bin_count, out_entropies);
            return;
#endif
#if defined(BA_HAS_NEON)
        case BA_BACKEND_NEON:
            for (size_t b = 0; b < batch_count; ++b) {
                out_entropies[b] = ba::simd::shannon_entropy_neon(
                    values_flat + offsets[b], lengths[b], bin_count);
            }
            return;
#endif
        default:
            ba::core::shannon_entropy_batch(values_flat, offsets, lengths,
                                             batch_count, bin_count, out_entropies);
            return;
    }
}

static void dispatch_circular_batch(const double* values_flat,
                                     const size_t* offsets, const size_t* lengths,
                                     size_t batch_count, int bin_count,
                                     double* out_entropies) {
    BABackend be = ba::get_backend();
    const size_t max_len = batch_max_length(lengths, batch_count);

    if (is_gpu_backend(be) &&
        !gpu_batch_eligible(be, lengths, batch_count, max_len)) {
        fallback_circular_batch(values_flat, offsets, lengths,
                                batch_count, bin_count, out_entropies);
        return;
    }

    switch (be) {
#if defined(BA_HAS_CUDA)
        case BA_BACKEND_CUDA:
            ba::cuda::circular_shannon_entropy_batch_cuda(
                values_flat, offsets, lengths, batch_count, bin_count, out_entropies);
            for (size_t b = 0; b < batch_count; ++b) {
                if (out_entropies[b] < 0.0) {
                    out_entropies[b] = fallback_circular(
                        values_flat + offsets[b], lengths[b], bin_count);
                }
            }
            return;
#endif
#if defined(BA_HAS_ROCM)
        case BA_BACKEND_ROCM:
            ba::rocm::circular_shannon_entropy_batch_hip(
                values_flat, offsets, lengths, batch_count, bin_count, out_entropies);
            for (size_t b = 0; b < batch_count; ++b) {
                if (out_entropies[b] < 0.0) {
                    out_entropies[b] = fallback_circular(
                        values_flat + offsets[b], lengths[b], bin_count);
                }
            }
            return;
#endif
#if defined(BA_HAS_METAL)
        case BA_BACKEND_METAL:
            ba::metal::circular_shannon_entropy_batch_metal(
                values_flat, offsets, lengths, batch_count, bin_count, out_entropies);
            for (size_t b = 0; b < batch_count; ++b) {
                if (out_entropies[b] < 0.0) {
                    out_entropies[b] = fallback_circular(
                        values_flat + offsets[b], lengths[b], bin_count);
                }
            }
            return;
#endif
#if defined(BA_HAS_OPENMP)
        case BA_BACKEND_OPENMP:
            ba::omp::circular_shannon_entropy_batch_omp(
                values_flat, offsets, lengths, batch_count, bin_count, out_entropies);
            return;
#endif
#if defined(BA_HAS_NEON)
        case BA_BACKEND_NEON:
            for (size_t b = 0; b < batch_count; ++b) {
                out_entropies[b] = ba::simd::circular_shannon_entropy_neon(
                    values_flat + offsets[b], lengths[b], bin_count);
            }
            return;
#endif
#if defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
            for (size_t b = 0; b < batch_count; ++b) {
                out_entropies[b] = ba::simd::circular_shannon_entropy_avx512(
                    values_flat + offsets[b], lengths[b], bin_count);
            }
            return;
#endif
#if defined(BA_HAS_AVX2)
        case BA_BACKEND_AVX2:
#if !defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
#endif
            for (size_t b = 0; b < batch_count; ++b) {
                out_entropies[b] = ba::simd::circular_shannon_entropy_avx2(
                    values_flat + offsets[b], lengths[b], bin_count);
            }
            return;
#endif
        default:
            ba::core::circular_shannon_entropy_batch(
                values_flat, offsets, lengths, batch_count, bin_count, out_entropies);
            return;
    }
}

static double dispatch_pearson(const double* x, const double* y, size_t count) {
    BABackend be = ba::get_backend();

    if (is_gpu_backend(be) && count < kGpuPearsonMinN) {
        return fallback_pearson(x, y, count);
    }

    switch (be) {
#if defined(BA_HAS_CUDA)
        case BA_BACKEND_CUDA: {
            double r = ba::cuda::pearson_correlation_cuda(x, y, count);
            return std::isfinite(r) ? r : fallback_pearson(x, y, count);
        }
#endif
#if defined(BA_HAS_ROCM)
        case BA_BACKEND_ROCM: {
            double r = ba::rocm::pearson_correlation_hip(x, y, count);
            return std::isfinite(r) ? r : fallback_pearson(x, y, count);
        }
#endif
#if defined(BA_HAS_METAL)
        case BA_BACKEND_METAL: {
            // Float32 GPU second-moment reduce; fall back to double on failure.
            double r = ba::metal::pearson_correlation_metal(x, y, count);
            if (!std::isfinite(r)) return fallback_pearson(x, y, count);
            // Float32 accumulation loses precision near |r| = 1 (cancellation in
            // the denominator); recompute that boundary region in double so the
            // scientific (CrossDomainValidator) result stays exact.
            if (std::abs(r) > 0.999) return fallback_pearson(x, y, count);
            return r;
        }
#endif
#if defined(BA_HAS_NEON)
        case BA_BACKEND_NEON:
            return ba::simd::pearson_correlation_neon(x, y, count);
#endif
#if defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
            return ba::simd::pearson_correlation_avx512(x, y, count);
#endif
#if defined(BA_HAS_AVX2)
        case BA_BACKEND_AVX2:
#if !defined(BA_HAS_AVX512)
        case BA_BACKEND_AVX512:
#endif
            return ba::simd::pearson_correlation_avx2(x, y, count);
#endif
        default:
            return ba::core::pearson_correlation(x, y, count);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shannon Entropy — Single
// ═══════════════════════════════════════════════════════════════════════════

BAStatus ba_shannon_entropy(
    const double* values, size_t count,
    int bin_count,
    double* out_entropy
) {
    if (!values || !out_entropy) return BA_ERR_NULL_PTR;
    if (count < 2) return BA_ERR_INSUFFICIENT_DATA;
    if (bin_count < 1) return BA_ERR_INVALID_PARAM;

    *out_entropy = dispatch_shannon(values, count, bin_count);
    return BA_OK;
}

BAStatus ba_circular_shannon_entropy(
    const double* angles, size_t count,
    int bin_count,
    double* out_entropy
) {
    if (!angles || !out_entropy) return BA_ERR_NULL_PTR;
    if (count < 2) return BA_ERR_INSUFFICIENT_DATA;
    if (bin_count < 1) return BA_ERR_INVALID_PARAM;

    *out_entropy = dispatch_circular(angles, count, bin_count);
    return BA_OK;
}

BAStatus ba_shannon_entropy_fixed(
    const double* values, size_t count,
    int bin_count,
    double domain_min, double domain_max,
    double* out_entropy
) {
    if (!values || !out_entropy) return BA_ERR_NULL_PTR;
    if (count < 2) return BA_ERR_INSUFFICIENT_DATA;
    if (bin_count < 1 || domain_max <= domain_min) return BA_ERR_INVALID_PARAM;

    *out_entropy = dispatch_shannon_fixed(values, count, bin_count,
                                           domain_min, domain_max);
    return BA_OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// Shannon Entropy — Batch
// ═══════════════════════════════════════════════════════════════════════════

BAStatus ba_shannon_entropy_batch(
    const double* values_flat,
    const size_t* offsets,
    const size_t* lengths,
    size_t batch_count,
    int bin_count,
    double* out_entropies
) {
    if (!values_flat || !offsets || !lengths || !out_entropies) return BA_ERR_NULL_PTR;
    if (batch_count == 0) return BA_ERR_INSUFFICIENT_DATA;
    if (bin_count < 1) return BA_ERR_INVALID_PARAM;

    dispatch_shannon_batch(values_flat, offsets, lengths,
                           batch_count, bin_count, out_entropies);
    return BA_OK;
}

BAStatus ba_circular_shannon_entropy_batch(
    const double* values_flat,
    const size_t* offsets,
    const size_t* lengths,
    size_t batch_count,
    int bin_count,
    double* out_entropies
) {
    if (!values_flat || !offsets || !lengths || !out_entropies) return BA_ERR_NULL_PTR;
    if (batch_count == 0) return BA_ERR_INSUFFICIENT_DATA;
    if (bin_count < 1) return BA_ERR_INVALID_PARAM;

    dispatch_circular_batch(values_flat, offsets, lengths,
                            batch_count, bin_count, out_entropies);
    return BA_OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// Correlation & Regression
// ═══════════════════════════════════════════════════════════════════════════

BAStatus ba_pearson_correlation(
    const double* x, const double* y, size_t count,
    double* out_r
) {
    if (!x || !y || !out_r) return BA_ERR_NULL_PTR;
    if (count < 2) return BA_ERR_INSUFFICIENT_DATA;

    *out_r = dispatch_pearson(x, y, count);
    return BA_OK;
}

BAStatus ba_linear_regression(
    const double* x, const double* y, size_t count,
    double* out_slope, double* out_intercept, double* out_mae
) {
    if (!x || !y || !out_slope || !out_intercept || !out_mae) return BA_ERR_NULL_PTR;
    if (count < 2) return BA_ERR_INSUFFICIENT_DATA;

    auto result = ba::core::linear_regression(x, y, count);
    *out_slope = result.slope;
    *out_intercept = result.intercept;
    *out_mae = result.mae;
    return BA_OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// Descriptive Statistics
// ═══════════════════════════════════════════════════════════════════════════

BAStatus ba_descriptive_stats(
    const double* values, size_t count,
    double* out_mean, double* out_sd
) {
    if (!values || !out_mean || !out_sd) return BA_ERR_NULL_PTR;
    if (count < 1) return BA_ERR_INSUFFICIENT_DATA;

    auto stats = ba::core::descriptive_stats(values, count);
    *out_mean = stats.mean;
    *out_sd = stats.sd;
    return BA_OK;
}

BAStatus ba_zscore_outliers(
    const double* values, size_t count,
    double threshold,
    int32_t* out_outlier_flags
) {
    if (!values || !out_outlier_flags) return BA_ERR_NULL_PTR;
    if (count < 2) return BA_ERR_INSUFFICIENT_DATA;
    if (threshold <= 0.0) return BA_ERR_INVALID_PARAM;

    ba::core::zscore_outliers(values, count, threshold, out_outlier_flags);
    return BA_OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// Pairwise Scoring
// ═══════════════════════════════════════════════════════════════════════════

BAStatus ba_pairwise_scores(
    const void* data,
    size_t n,
    size_t stride,
    BAPairwiseScoreFn score_fn,
    void* user_data,
    double* out_scores
) {
    if (!data || !score_fn || !out_scores) return BA_ERR_NULL_PTR;
    if (n < 2) return BA_ERR_INSUFFICIENT_DATA;
    if (stride == 0) return BA_ERR_INVALID_PARAM;

    switch (ba::get_backend()) {
#if defined(BA_HAS_CUDA)
        case BA_BACKEND_CUDA:
            // Custom host callbacks cannot run on device; host traversal.
            ba::cuda::pairwise_scores_cuda(data, n, stride, score_fn, user_data, out_scores);
            return BA_OK;
#endif
#if defined(BA_HAS_OPENMP)
        case BA_BACKEND_OPENMP:
            ba::omp::pairwise_scores_omp(data, n, stride, score_fn, user_data, out_scores);
            return BA_OK;
#endif
        default:
            // Prefer OpenMP for pairwise even when SIMD/Metal is "best" —
            // host callbacks cannot run on GPU/SIMD.
#if defined(BA_HAS_OPENMP)
            ba::omp::pairwise_scores_omp(data, n, stride, score_fn, user_data, out_scores);
#else
            ba::core::pairwise_scores(data, n, stride, score_fn, user_data, out_scores);
#endif
            return BA_OK;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special Functions
// ═══════════════════════════════════════════════════════════════════════════

BAStatus ba_regularized_incomplete_beta(
    double x, double a, double b,
    double* out_result
) {
    if (!out_result) return BA_ERR_NULL_PTR;
    if (a <= 0.0 || b <= 0.0) return BA_ERR_INVALID_PARAM;
    if (x < 0.0 || x > 1.0) return BA_ERR_INVALID_PARAM;

    *out_result = ba::core::regularized_incomplete_beta(x, a, b);
    return BA_OK;
}

BAStatus ba_pearson_pvalue(
    double r, size_t n,
    double* out_pval
) {
    if (!out_pval) return BA_ERR_NULL_PTR;
    if (n <= 2) return BA_ERR_INSUFFICIENT_DATA;

    *out_pval = ba::core::pearson_pvalue(r, n);
    return BA_OK;
}
