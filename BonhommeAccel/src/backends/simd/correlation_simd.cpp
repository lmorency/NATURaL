/*
 * correlation_simd.cpp — SIMD-accelerated Pearson correlation.
 *
 * Uses vectorized dot products for sum_xy, sum_x2, sum_y2.
 * AVX2 (4-wide) or NEON (2-wide) depending on platform.
 */

#include <cmath>
#include <cstddef>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ba::simd {

#if defined(__AVX2__)

static inline double hsum_pd(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    __m128d sum2 = _mm_add_pd(sum, _mm_permute_pd(sum, 1));
    return _mm_cvtsd_f64(sum2);
}

double pearson_correlation_avx2(const double* x, const double* y, size_t count) {
    if (!x || !y || count < 2) return 0.0;

    // Fast path: skip the heap copy when every pair is finite.
    bool all_finite = true;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { all_finite = false; break; }
    }

    const double* px = x;
    const double* py = y;
    size_t n = count;

    std::vector<double> cx, cy;
    if (!all_finite) {
        cx.reserve(count);
        cy.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (std::isfinite(x[i]) && std::isfinite(y[i])) {
                cx.push_back(x[i]);
                cy.push_back(y[i]);
            }
        }
        n = cx.size();
        if (n < 2) return 0.0;
        px = cx.data();
        py = cy.data();
    }

    // Mean computation (AVX2)
    size_t simd_end = (n / 4) * 4;
    __m256d vsum_x = _mm256_setzero_pd();
    __m256d vsum_y = _mm256_setzero_pd();

    for (size_t i = 0; i < simd_end; i += 4) {
        vsum_x = _mm256_add_pd(vsum_x, _mm256_loadu_pd(&px[i]));
        vsum_y = _mm256_add_pd(vsum_y, _mm256_loadu_pd(&py[i]));
    }

    double sum_x = hsum_pd(vsum_x);
    double sum_y = hsum_pd(vsum_y);
    for (size_t i = simd_end; i < n; ++i) {
        sum_x += px[i];
        sum_y += py[i];
    }

    double mean_x = sum_x / static_cast<double>(n);
    double mean_y = sum_y / static_cast<double>(n);

    // Correlation (AVX2 vectorized dot products)
    __m256d vmx = _mm256_set1_pd(mean_x);
    __m256d vmy = _mm256_set1_pd(mean_y);
    __m256d v_sum_xy = _mm256_setzero_pd();
    __m256d v_sum_x2 = _mm256_setzero_pd();
    __m256d v_sum_y2 = _mm256_setzero_pd();

    for (size_t i = 0; i < simd_end; i += 4) {
        __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(&px[i]), vmx);
        __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(&py[i]), vmy);
        v_sum_xy = _mm256_fmadd_pd(dx, dy, v_sum_xy);
        v_sum_x2 = _mm256_fmadd_pd(dx, dx, v_sum_x2);
        v_sum_y2 = _mm256_fmadd_pd(dy, dy, v_sum_y2);
    }

    double sxy = hsum_pd(v_sum_xy);
    double sx2 = hsum_pd(v_sum_x2);
    double sy2 = hsum_pd(v_sum_y2);

    for (size_t i = simd_end; i < n; ++i) {
        double dx = px[i] - mean_x;
        double dy = py[i] - mean_y;
        sxy += dx * dy;
        sx2 += dx * dx;
        sy2 += dy * dy;
    }

    double denom = std::sqrt(sx2 * sy2);
    return (denom > 0.0) ? (sxy / denom) : 0.0;
}

#endif // __AVX2__

#if defined(__aarch64__)

double pearson_correlation_neon(const double* x, const double* y, size_t count) {
    if (!x || !y || count < 2) return 0.0;

    // Fast path: skip the heap copy when every pair is finite.
    bool all_finite = true;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { all_finite = false; break; }
    }

    const double* px = x;
    const double* py = y;
    size_t n = count;

    std::vector<double> cx, cy;
    if (!all_finite) {
        cx.reserve(count);
        cy.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (std::isfinite(x[i]) && std::isfinite(y[i])) {
                cx.push_back(x[i]);
                cy.push_back(y[i]);
            }
        }
        n = cx.size();
        if (n < 2) return 0.0;
        px = cx.data();
        py = cy.data();
    }

    size_t simd_end = (n / 2) * 2;
    float64x2_t vsx = vdupq_n_f64(0.0);
    float64x2_t vsy = vdupq_n_f64(0.0);

    for (size_t i = 0; i < simd_end; i += 2) {
        vsx = vaddq_f64(vsx, vld1q_f64(&px[i]));
        vsy = vaddq_f64(vsy, vld1q_f64(&py[i]));
    }

    double sum_x = vgetq_lane_f64(vsx, 0) + vgetq_lane_f64(vsx, 1);
    double sum_y = vgetq_lane_f64(vsy, 0) + vgetq_lane_f64(vsy, 1);
    for (size_t i = simd_end; i < n; ++i) {
        sum_x += px[i];
        sum_y += py[i];
    }

    double mean_x = sum_x / static_cast<double>(n);
    double mean_y = sum_y / static_cast<double>(n);

    float64x2_t vmx = vdupq_n_f64(mean_x);
    float64x2_t vmy = vdupq_n_f64(mean_y);
    float64x2_t v_sxy = vdupq_n_f64(0.0);
    float64x2_t v_sx2 = vdupq_n_f64(0.0);
    float64x2_t v_sy2 = vdupq_n_f64(0.0);

    for (size_t i = 0; i < simd_end; i += 2) {
        float64x2_t dx = vsubq_f64(vld1q_f64(&px[i]), vmx);
        float64x2_t dy = vsubq_f64(vld1q_f64(&py[i]), vmy);
        v_sxy = vfmaq_f64(v_sxy, dx, dy);
        v_sx2 = vfmaq_f64(v_sx2, dx, dx);
        v_sy2 = vfmaq_f64(v_sy2, dy, dy);
    }

    double sxy = vgetq_lane_f64(v_sxy, 0) + vgetq_lane_f64(v_sxy, 1);
    double sx2 = vgetq_lane_f64(v_sx2, 0) + vgetq_lane_f64(v_sx2, 1);
    double sy2 = vgetq_lane_f64(v_sy2, 0) + vgetq_lane_f64(v_sy2, 1);

    for (size_t i = simd_end; i < n; ++i) {
        double dx = px[i] - mean_x;
        double dy = py[i] - mean_y;
        sxy += dx * dy;
        sx2 += dx * dx;
        sy2 += dy * dy;
    }

    double denom = std::sqrt(sx2 * sy2);
    return (denom > 0.0) ? (sxy / denom) : 0.0;
}

#endif // __aarch64__

} // namespace ba::simd
