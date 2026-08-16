/*
 * entropy_neon.cpp — ARM NEON-accelerated Shannon entropy.
 *
 * Adaptive: scalar min/max (histogram scatter dominates the cost).
 * Circular: 2-wide floor-fmod wrap + bin index (parity with AVX2), with the
 *   ±180 cut canonicalized to -180 so it matches scalar fmod.
 * Fixed-domain: 2-wide clamp + bin index.
 * Entropy reduce over bins is always scalar (bin_count is small).
 */

#if defined(__aarch64__)

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <limits>

namespace ba::simd {

double shannon_entropy_neon(const double* values, size_t count, int bin_count) {
    if (!values || count < 2 || bin_count < 1) return 0.0;

    // Single pass: count finite + min/max, without allocating a cleaned copy.
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

// Scalar wrap + bin — parity with core / AVX2 scalar fallback.
static inline void circular_bin_scalar(double a, double bin_width, int bin_count,
                                        std::vector<int>& bins) {
    if (!std::isfinite(a)) return;
    a = std::fmod(a, 360.0);
    if (a > 180.0) a -= 360.0;
    if (a < -180.0) a += 360.0;
    if (a == 180.0) a = -180.0;
    int idx = static_cast<int>((a + 180.0) / bin_width);
    if (idx < 0) idx = 0;
    if (idx >= bin_count) idx = bin_count - 1;
    bins[static_cast<size_t>(idx)]++;
}

double circular_shannon_entropy_neon(const double* angles, size_t count, int bin_count) {
    if (!angles || count < 2 || bin_count < 1) return 0.0;

    size_t clean_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(angles[i])) ++clean_count;
    }
    if (clean_count < 2) return 0.0;

    const double bin_width = 360.0 / static_cast<double>(bin_count);
    const double inv_width = 1.0 / bin_width;
    std::vector<int> bins(static_cast<size_t>(bin_count), 0);

    // Vectorized wrap + bin (2-wide float64), mirrors AVX2 floor-based fmod.
    // floor-based remainder + wrap to [-180,180) matches std::fmod for this domain.
    const float64x2_t v360   = vdupq_n_f64(360.0);
    const float64x2_t v180   = vdupq_n_f64(180.0);
    const float64x2_t vn180  = vdupq_n_f64(-180.0);
    const float64x2_t vinv_w = vdupq_n_f64(inv_width);
    const float64x2_t vzero  = vdupq_n_f64(0.0);
    const float64x2_t vimax  = vdupq_n_f64(static_cast<double>(bin_count - 1));

    const size_t simd_end = (count / 2) * 2;
    for (size_t i = 0; i < simd_end; i += 2) {
        const double a0 = angles[i];
        const double a1 = angles[i + 1];
        if (!std::isfinite(a0) || !std::isfinite(a1)) {
            circular_bin_scalar(a0, bin_width, bin_count, bins);
            circular_bin_scalar(a1, bin_width, bin_count, bins);
            continue;
        }

        float64x2_t a = vld1q_f64(&angles[i]);

        // fmod: a - floor(a/360) * 360  (vrndmq = round toward −∞)
        float64x2_t ratio = vdivq_f64(a, v360);
        float64x2_t floored = vrndmq_f64(ratio);
        a = vfmsq_f64(a, floored, v360); // a − floored * 360

        // Wrap to [-180, 180)
        uint64x2_t gt180 = vcgtq_f64(a, v180);
        a = vbslq_f64(gt180, vsubq_f64(a, v360), a);
        uint64x2_t ltn180 = vcltq_f64(a, vn180);
        a = vbslq_f64(ltn180, vaddq_f64(a, v360), a);

        // Canonicalize the cut: floor-fmod maps -180 -> +180; fold +180 onto -180
        // so the SIMD path matches scalar fmod (both -> bin 0).
        uint64x2_t eq180 = vceqq_f64(a, v180);
        a = vbslq_f64(eq180, vn180, a);

        // Bin index: floor((a + 180) * inv_width), clamp to [0, bin_count-1]
        float64x2_t fidx = vmulq_f64(vaddq_f64(a, v180), vinv_w);
        fidx = vrndmq_f64(fidx);
        fidx = vmaxq_f64(fidx, vzero);
        fidx = vminq_f64(fidx, vimax);

        alignas(16) double idx_arr[2];
        vst1q_f64(idx_arr, fidx);
        bins[static_cast<size_t>(static_cast<int>(idx_arr[0]))]++;
        bins[static_cast<size_t>(static_cast<int>(idx_arr[1]))]++;
    }

    for (size_t i = simd_end; i < count; ++i) {
        circular_bin_scalar(angles[i], bin_width, bin_count, bins);
    }

    const double total = static_cast<double>(clean_count);
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
// Fixed-domain Shannon entropy (NEON clamp + bin index)
//
// Semantics match core::shannon_entropy_fixed:
//   filter non-finite, clamp to [domain_min, domain_max],
//   bin with truncating cast, clean_count < 2 → 0.
// ---------------------------------------------------------------------------

double shannon_entropy_fixed_neon(const double* values, size_t count, int bin_count,
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

    const float64x2_t v_dmin = vdupq_n_f64(domain_min);
    const float64x2_t v_dmax = vdupq_n_f64(domain_max);
    const float64x2_t v_bw   = vdupq_n_f64(bin_width);
    const float64x2_t v_zero = vdupq_n_f64(0.0);
    const float64x2_t v_imax = vdupq_n_f64(static_cast<double>(bin_count - 1));

    size_t simd_end = (count / 2) * 2;
    for (size_t i = 0; i < simd_end; i += 2) {
        const double a = values[i];
        const double b = values[i + 1];
        const bool fa = std::isfinite(a);
        const bool fb = std::isfinite(b);

        if (fa && fb) {
            // Vector clamp to domain, then bin index (non-negative → trunc == floor).
            float64x2_t v = vld1q_f64(&values[i]);
            v = vmaxq_f64(v, v_dmin);
            v = vminq_f64(v, v_dmax);
            float64x2_t fidx = vdivq_f64(vsubq_f64(v, v_dmin), v_bw);
            // Clamp index to [0, bin_count-1] (domain_max maps to bin_count).
            fidx = vmaxq_f64(fidx, v_zero);
            fidx = vminq_f64(fidx, v_imax);

            alignas(16) double idx_arr[2];
            vst1q_f64(idx_arr, fidx);
            bins[static_cast<size_t>(static_cast<int>(idx_arr[0]))]++;
            bins[static_cast<size_t>(static_cast<int>(idx_arr[1]))]++;
        } else {
            if (fa) {
                double v = std::max(domain_min, std::min(domain_max, a));
                int idx = static_cast<int>((v - domain_min) / bin_width);
                if (idx < 0) idx = 0;
                if (idx >= bin_count) idx = bin_count - 1;
                bins[static_cast<size_t>(idx)]++;
            }
            if (fb) {
                double v = std::max(domain_min, std::min(domain_max, b));
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

} // namespace ba::simd

#endif // __aarch64__
