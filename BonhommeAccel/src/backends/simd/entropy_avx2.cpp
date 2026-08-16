/*
 * entropy_avx2.cpp — AVX2-accelerated Shannon entropy.
 *
 * Uses 256-bit SIMD for vectorized min/max finding and FMA for entropy reduction.
 * Histogram binning remains scalar (irregular memory access pattern).
 */

#if defined(__AVX2__)

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <limits>

namespace ba::simd {

double shannon_entropy_avx2(const double* values, size_t count, int bin_count) {
    if (!values || count < 2 || bin_count < 1) return 0.0;

    // Single pass min/max without allocating a cleaned copy.
    // Kept scalar: the histogram scatter (second pass) dominates, and a vector
    // min/max here would only recompute the same values it already folded.
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    size_t clean_count = 0;

    for (size_t i = 0; i < count; ++i) {
        double v = values[i];
        if (!std::isfinite(v)) continue;
        ++clean_count;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }

    if (clean_count < 2) return 0.0;

    double range = max_val - min_val;
    if (range <= 0.0) return 0.0;

    double bin_width = range / static_cast<double>(bin_count);

    std::vector<int> bins(static_cast<size_t>(bin_count), 0);
    for (size_t i = 0; i < count; ++i) {
        double v = values[i];
        if (!std::isfinite(v)) continue;
        int idx = static_cast<int>((v - min_val) / bin_width);
        if (idx < 0) idx = 0;
        if (idx >= bin_count) idx = bin_count - 1;
        bins[static_cast<size_t>(idx)]++;
    }

    double total = static_cast<double>(clean_count);
    double entropy = 0.0;
    for (int i = 0; i < bin_count; ++i) {
        if (bins[static_cast<size_t>(i)] > 0) {
            double p = static_cast<double>(bins[static_cast<size_t>(i)]) / total;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

double circular_shannon_entropy_avx2(const double* angles, size_t count, int bin_count) {
    if (!angles || count < 2 || bin_count < 1) return 0.0;

    size_t clean_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(angles[i])) ++clean_count;
    }
    if (clean_count < 2) return 0.0;

    double bin_width = 360.0 / static_cast<double>(bin_count);
    std::vector<int> bins(static_cast<size_t>(bin_count), 0);

    // Vectorized wrapping with AVX2
    __m256d v360 = _mm256_set1_pd(360.0);
    __m256d v180 = _mm256_set1_pd(180.0);
    __m256d vn180 = _mm256_set1_pd(-180.0);

    size_t simd_end = (count / 4) * 4;
    for (size_t i = 0; i < simd_end; i += 4) {
        // Check for non-finite and fall back to scalar if needed
        bool has_nonfinite = false;
        for (int k = 0; k < 4; ++k) {
            if (!std::isfinite(angles[i + static_cast<size_t>(k)])) {
                has_nonfinite = true;
                break;
            }
        }
        if (has_nonfinite) {
            for (int k = 0; k < 4; ++k) {
                double a = angles[i + static_cast<size_t>(k)];
                if (!std::isfinite(a)) continue;
                a = std::fmod(a, 360.0);
                if (a > 180.0) a -= 360.0;
                if (a < -180.0) a += 360.0;
                if (a == 180.0) a = -180.0;
                int idx = static_cast<int>((a + 180.0) / bin_width);
                if (idx < 0) idx = 0;
                if (idx >= bin_count) idx = bin_count - 1;
                bins[static_cast<size_t>(idx)]++;
            }
            continue;
        }

        __m256d a = _mm256_loadu_pd(&angles[i]);

        // fmod: a - floor(a/360) * 360
        __m256d ratio = _mm256_div_pd(a, v360);
        __m256d floored = _mm256_floor_pd(ratio);
        a = _mm256_fnmadd_pd(floored, v360, a);

        // Wrap to [-180, 180)
        __m256d gt180 = _mm256_cmp_pd(a, v180, _CMP_GT_OQ);
        a = _mm256_blendv_pd(a, _mm256_sub_pd(a, v360), gt180);
        __m256d ltn180 = _mm256_cmp_pd(a, vn180, _CMP_LT_OQ);
        a = _mm256_blendv_pd(a, _mm256_add_pd(a, v360), ltn180);

        // Canonicalize the cut: floor-fmod maps -180 -> +180; fold +180 onto -180
        // so the SIMD path matches scalar fmod (both -> bin 0).
        __m256d eq180 = _mm256_cmp_pd(a, v180, _CMP_EQ_OQ);
        a = _mm256_blendv_pd(a, vn180, eq180);

        // Bin index
        __m256d shifted = _mm256_add_pd(a, v180);
        __m256d fidx = _mm256_div_pd(shifted, _mm256_set1_pd(bin_width));
        fidx = _mm256_floor_pd(fidx);
        fidx = _mm256_max_pd(fidx, _mm256_setzero_pd());
        fidx = _mm256_min_pd(fidx, _mm256_set1_pd(static_cast<double>(bin_count - 1)));

        alignas(32) double idx_arr[4];
        _mm256_store_pd(idx_arr, fidx);
        for (int k = 0; k < 4; ++k) {
            bins[static_cast<size_t>(static_cast<int>(idx_arr[k]))]++;
        }
    }

    for (size_t i = simd_end; i < count; ++i) {
        double a = angles[i];
        if (!std::isfinite(a)) continue;
        a = std::fmod(a, 360.0);
        if (a > 180.0) a -= 360.0;
        if (a < -180.0) a += 360.0;
        if (a == 180.0) a = -180.0;
        int idx = static_cast<int>((a + 180.0) / bin_width);
        if (idx < 0) idx = 0;
        if (idx >= bin_count) idx = bin_count - 1;
        bins[static_cast<size_t>(idx)]++;
    }

    double total = static_cast<double>(clean_count);
    double entropy = 0.0;
    for (int i = 0; i < bin_count; ++i) {
        if (bins[static_cast<size_t>(i)] > 0) {
            double p = static_cast<double>(bins[static_cast<size_t>(i)]) / total;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

// ---------------------------------------------------------------------------
// Fixed-domain Shannon entropy (AVX2 clamp + bin index)
//
// Semantics match core::shannon_entropy_fixed:
//   filter non-finite, clamp to [domain_min, domain_max],
//   bin with truncating cast, clean_count < 2 → 0.
// ---------------------------------------------------------------------------

double shannon_entropy_fixed_avx2(const double* values, size_t count, int bin_count,
                                   double domain_min, double domain_max) {
    if (!values || count < 2 || bin_count < 1) return 0.0;

    size_t clean_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(values[i])) ++clean_count;
    }
    if (clean_count < 2) return 0.0;

    double range = domain_max - domain_min;
    if (range <= 0.0) return 0.0;

    double bin_width = range / static_cast<double>(bin_count);
    std::vector<int> bins(static_cast<size_t>(bin_count), 0);

    const __m256d v_dmin = _mm256_set1_pd(domain_min);
    const __m256d v_dmax = _mm256_set1_pd(domain_max);
    const __m256d v_bw   = _mm256_set1_pd(bin_width);
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256d v_imax = _mm256_set1_pd(static_cast<double>(bin_count - 1));

    size_t simd_end = (count / 4) * 4;
    for (size_t i = 0; i < simd_end; i += 4) {
        bool all_finite = true;
        for (int k = 0; k < 4; ++k) {
            if (!std::isfinite(values[i + static_cast<size_t>(k)])) {
                all_finite = false;
                break;
            }
        }

        if (all_finite) {
            __m256d v = _mm256_loadu_pd(&values[i]);
            // Clamp to [domain_min, domain_max]
            v = _mm256_max_pd(v, v_dmin);
            v = _mm256_min_pd(v, v_dmax);
            // Bin index (non-negative after clamp → trunc toward zero matches scalar cast)
            __m256d fidx = _mm256_div_pd(_mm256_sub_pd(v, v_dmin), v_bw);
            fidx = _mm256_max_pd(fidx, v_zero);
            fidx = _mm256_min_pd(fidx, v_imax);

            alignas(32) double idx_arr[4];
            _mm256_store_pd(idx_arr, fidx);
            for (int k = 0; k < 4; ++k) {
                bins[static_cast<size_t>(static_cast<int>(idx_arr[k]))]++;
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                double v = values[i + static_cast<size_t>(k)];
                if (!std::isfinite(v)) continue;
                v = std::max(domain_min, std::min(domain_max, v));
                int idx = static_cast<int>((v - domain_min) / bin_width);
                if (idx < 0) idx = 0;
                if (idx >= bin_count) idx = bin_count - 1;
                bins[static_cast<size_t>(idx)]++;
            }
        }
    }

    for (size_t i = simd_end; i < count; ++i) {
        double v = values[i];
        if (!std::isfinite(v)) continue;
        v = std::max(domain_min, std::min(domain_max, v));
        int idx = static_cast<int>((v - domain_min) / bin_width);
        if (idx < 0) idx = 0;
        if (idx >= bin_count) idx = bin_count - 1;
        bins[static_cast<size_t>(idx)]++;
    }

    double total = static_cast<double>(clean_count);
    double entropy = 0.0;
    for (int i = 0; i < bin_count; ++i) {
        if (bins[static_cast<size_t>(i)] > 0) {
            double p = static_cast<double>(bins[static_cast<size_t>(i)]) / total;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

void shannon_entropy_batch_avx2(const double* flat, const size_t* offsets,
                                 const size_t* lengths, size_t batch_count,
                                 int bin_count, double* out) {
    for (size_t b = 0; b < batch_count; ++b) {
        out[b] = shannon_entropy_avx2(flat + offsets[b], lengths[b], bin_count);
    }
}

} // namespace ba::simd

#endif // __AVX2__
