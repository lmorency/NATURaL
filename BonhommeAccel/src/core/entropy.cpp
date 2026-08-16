/*
 * entropy.cpp — Scalar Shannon entropy implementation.
 *
 * Mirrors BonhommeCore/EntropyCalculator.swift exactly:
 *   - Same 32-bin default histogram
 *   - Same NaN/Inf filtering
 *   - Same circular wrapping: fmod -> [-180, 180)
 *   - Same entropy formula: H = -Sum p_i * log2(p_i)
 *
 * Numerical parity target: < 1e-10 bits difference vs Swift.
 */

#include "entropy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ba::core {

// ---------------------------------------------------------------------------
// Adaptive Shannon entropy (linear domain)
// ---------------------------------------------------------------------------

double shannon_entropy(const double* values, size_t count, int bin_count) {
    if (!values || count < 2 || bin_count < 1) return 0.0;

    // Filter non-finite values and find min/max in one pass
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    size_t clean_count = 0;

    // First pass: count finite values and find range
    for (size_t i = 0; i < count; ++i) {
        double v = values[i];
        if (std::isfinite(v)) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
            ++clean_count;
        }
    }

    if (clean_count < 2) return 0.0;

    double range = max_val - min_val;
    if (range <= 0.0) return 0.0;

    double bin_width = range / static_cast<double>(bin_count);

    // Histogram
    std::vector<int> bins(static_cast<size_t>(bin_count), 0);

    for (size_t i = 0; i < count; ++i) {
        double v = values[i];
        if (!std::isfinite(v)) continue;
        // Clamp to [0, bin_count-1] — guards against FP noise below min_val
        int idx = static_cast<int>((v - min_val) / bin_width);
        if (idx < 0) idx = 0;
        if (idx >= bin_count) idx = bin_count - 1;
        bins[static_cast<size_t>(idx)]++;
    }

    // Entropy
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
// Circular Shannon entropy (torsional angles)
// ---------------------------------------------------------------------------

double circular_shannon_entropy(const double* angles, size_t count, int bin_count) {
    if (!angles || count < 2 || bin_count < 1) return 0.0;

    // Filter non-finite
    size_t clean_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(angles[i])) ++clean_count;
    }
    if (clean_count < 2) return 0.0;

    double bin_width = 360.0 / static_cast<double>(bin_count);

    std::vector<int> bins(static_cast<size_t>(bin_count), 0);

    for (size_t i = 0; i < count; ++i) {
        double a = angles[i];
        if (!std::isfinite(a)) continue;

        // Wrap to [-180, 180) — mirrors Swift's truncatingRemainder(dividingBy:)
        a = std::fmod(a, 360.0);
        if (a > 180.0) a -= 360.0;
        if (a < -180.0) a += 360.0;
        // Canonicalize the cut: +180 ≡ -180 are the same angle; fold both onto -180
        // so SIMD (floor-based) and scalar (fmod-based) wrap agree on this boundary.
        if (a == 180.0) a = -180.0;

        // Map [-180, 180) -> bin index [0, bin_count)
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
// Fixed-domain Shannon entropy
// ---------------------------------------------------------------------------

double shannon_entropy_fixed(const double* values, size_t count, int bin_count,
                              double domain_min, double domain_max) {
    if (!values || count < 2 || bin_count < 1) return 0.0;

    // Filter non-finite (NaN/Inf) — mirrors adaptive/circular paths and Swift
    size_t clean_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(values[i])) ++clean_count;
    }
    if (clean_count < 2) return 0.0;

    double range = domain_max - domain_min;
    if (range <= 0.0) return 0.0;

    double bin_width = range / static_cast<double>(bin_count);

    std::vector<int> bins(static_cast<size_t>(bin_count), 0);

    for (size_t i = 0; i < count; ++i) {
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

// ---------------------------------------------------------------------------
// Batch entropy
// ---------------------------------------------------------------------------

void shannon_entropy_batch(const double* values_flat,
                            const size_t* offsets, const size_t* lengths,
                            size_t batch_count, int bin_count,
                            double* out_entropies) {
    for (size_t b = 0; b < batch_count; ++b) {
        out_entropies[b] = shannon_entropy(
            values_flat + offsets[b], lengths[b], bin_count);
    }
}

void circular_shannon_entropy_batch(const double* values_flat,
                                     const size_t* offsets, const size_t* lengths,
                                     size_t batch_count, int bin_count,
                                     double* out_entropies) {
    for (size_t b = 0; b < batch_count; ++b) {
        out_entropies[b] = circular_shannon_entropy(
            values_flat + offsets[b], lengths[b], bin_count);
    }
}

} // namespace ba::core
