/*
 * correlation_avx512.cpp — AVX-512-accelerated Pearson correlation.
 *
 * Uses 8-wide double FMA for sum_xy / sum_x2 / sum_y2 after NaN pair filter.
 * Mathematical parity with scalar/AVX2 paths.
 */

#if defined(__AVX512F__)

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ba::simd {

static inline double hsum_pd(__m512d v) {
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d s256 = _mm256_add_pd(lo, hi);
    __m128d lo128 = _mm256_castpd256_pd128(s256);
    __m128d hi128 = _mm256_extractf128_pd(s256, 1);
    __m128d sum = _mm_add_pd(lo128, hi128);
    __m128d sum2 = _mm_add_pd(sum, _mm_permute_pd(sum, 1));
    return _mm_cvtsd_f64(sum2);
}

double pearson_correlation_avx512(const double* x, const double* y, size_t count) {
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

    // Mean computation (AVX-512, 8-wide)
    size_t simd_end = (n / 8) * 8;
    __m512d vsum_x = _mm512_setzero_pd();
    __m512d vsum_y = _mm512_setzero_pd();

    for (size_t i = 0; i < simd_end; i += 8) {
        vsum_x = _mm512_add_pd(vsum_x, _mm512_loadu_pd(&px[i]));
        vsum_y = _mm512_add_pd(vsum_y, _mm512_loadu_pd(&py[i]));
    }

    double sum_x = hsum_pd(vsum_x);
    double sum_y = hsum_pd(vsum_y);
    for (size_t i = simd_end; i < n; ++i) {
        sum_x += px[i];
        sum_y += py[i];
    }

    double mean_x = sum_x / static_cast<double>(n);
    double mean_y = sum_y / static_cast<double>(n);

    // Correlation (AVX-512 vectorized FMA dot products)
    __m512d vmx = _mm512_set1_pd(mean_x);
    __m512d vmy = _mm512_set1_pd(mean_y);
    __m512d v_sum_xy = _mm512_setzero_pd();
    __m512d v_sum_x2 = _mm512_setzero_pd();
    __m512d v_sum_y2 = _mm512_setzero_pd();

    for (size_t i = 0; i < simd_end; i += 8) {
        __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(&px[i]), vmx);
        __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(&py[i]), vmy);
        v_sum_xy = _mm512_fmadd_pd(dx, dy, v_sum_xy);
        v_sum_x2 = _mm512_fmadd_pd(dx, dx, v_sum_x2);
        v_sum_y2 = _mm512_fmadd_pd(dy, dy, v_sum_y2);
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

} // namespace ba::simd

#endif // __AVX512F__
