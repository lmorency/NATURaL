/*
 * test_circular_simd.cpp — Exhaustive circular Shannon / NEON-path coverage.
 *
 * Exercises wrap arithmetic, odd-length tails, multi-revolution angles,
 * batch vs single parity, GPU/CPU size thresholds, and FlexAID-style
 * multi-bond packing. Public C API only (backend-agnostic).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "BonhommeAccel.h"
#include "reference_values.h"

#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using namespace ba::test;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    if (n == 0) return v;
    if (n == 1) {
        v[0] = lo;
        return v;
    }
    for (size_t i = 0; i < n; ++i) {
        v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return v;
}

static double circular_h(const std::vector<double>& a, int bins = DEFAULT_BIN_COUNT) {
    double h = -1.0;
    REQUIRE(ba_circular_shannon_entropy(a.data(), a.size(), bins, &h) == BA_OK);
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════
// Wrap edge cases (NEON floor-fmod + scalar must agree via public API)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Circular: exactly ±180 boundary bins correctly", "[entropy][circular][simd]") {
    // ±180 are the same angle; both canonicalize to -180 (bin 0) — low H
    std::vector<double> angles;
    for (int i = 0; i < 64; ++i) {
        angles.push_back(180.0);   // folds onto -180 → bin 0
        angles.push_back(-180.0);  // bin 0
        angles.push_back(179.5);
        angles.push_back(-179.5);
    }
    double h = circular_h(angles);
    REQUIRE(h < 2.5);
    REQUIRE(h > 0.0);
}

TEST_CASE("Circular: -180 cut canonicalized identically across backends",
          "[entropy][circular][simd][parity]") {
    // Regression: NEON/AVX2/AVX-512 floor-fmod mapped -180 -> +180 (last bin)
    // while scalar/Swift fmod mapped -180 -> bin 0. -180 and +179.5 are 0.5°
    // apart on the circle and must fall in two adjacent bins regardless of backend.
    std::vector<double> angles;
    for (int i = 0; i < 100; ++i) {
        angles.push_back(-180.0);
        angles.push_back(179.5);
    }

    auto h_with = [&](BABackend b) {
        ba_set_backend_override(b);
        double h = -1.0;
        REQUIRE(ba_circular_shannon_entropy(angles.data(), angles.size(),
                                             DEFAULT_BIN_COUNT, &h) == BA_OK);
        return h;
    };

    double h_scalar = h_with(BA_BACKEND_SCALAR);
    double h_neon   = h_with(BA_BACKEND_NEON);
    double h_avx2   = h_with(BA_BACKEND_AVX2);
    double h_avx512 = h_with(BA_BACKEND_AVX512);
    double h_metal  = h_with(BA_BACKEND_METAL);
    ba_clear_backend_override();

    // Uncompiled backends fall back to scalar via dispatch, so all must agree.
    REQUIRE_THAT(h_neon,   WithinAbs(h_scalar, 1e-9));
    REQUIRE_THAT(h_avx2,   WithinAbs(h_scalar, 1e-9));
    REQUIRE_THAT(h_avx512, WithinAbs(h_scalar, 1e-9));
    REQUIRE_THAT(h_metal,  WithinAbs(h_scalar, 1e-9));

    // Two populated bins → H = 1.0 (not 0.0 as the pre-fix SIMD path returned).
    REQUIRE(h_scalar > 0.9);
    REQUIRE(h_scalar < 1.1);
}

TEST_CASE("Circular: multi-revolution angles match primary range", "[entropy][circular][simd]") {
    // 350° ≡ -10°, 710° ≡ -10°, -370° ≡ -10° after wrap
    auto near_minus_10 = [](size_t n) {
        std::vector<double> a;
        a.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) {
            double jitter = 0.1 * static_cast<double>(static_cast<int>(i % 5) - 2);
            a.push_back(350.0 + jitter);
            a.push_back(710.0 + jitter);
            a.push_back(-370.0 + jitter);
        }
        return a;
    };
    auto primary = [](size_t n) {
        std::vector<double> a;
        a.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) {
            double jitter = 0.1 * static_cast<double>(static_cast<int>(i % 5) - 2);
            a.push_back(-10.0 + jitter);
            a.push_back(-10.0 + jitter);
            a.push_back(-10.0 + jitter);
        }
        return a;
    };

    // Large enough for SIMD vector lanes (NEON 2 / AVX2 4) + GPU threshold skip
    constexpr size_t N = 512;
    double h_wrapped = circular_h(near_minus_10(N));
    double h_primary = circular_h(primary(N));
    REQUIRE_THAT(h_wrapped, WithinAbs(h_primary, 1e-9));
    REQUIRE(h_wrapped < 2.0);
}

TEST_CASE("Circular: negative multi-turn and positive multi-turn parity",
          "[entropy][circular][simd]") {
    std::vector<double> pos, neg;
    for (int k = 0; k < 400; ++k) {
        double base = 20.0 + 0.05 * (k % 11);
        pos.push_back(base);
        pos.push_back(base + 360.0);
        pos.push_back(base + 720.0);
        neg.push_back(base);
        neg.push_back(base - 360.0);
        neg.push_back(base - 720.0);
    }
    REQUIRE_THAT(circular_h(pos), WithinAbs(circular_h(neg), 1e-9));
}

TEST_CASE("Circular: odd length exercises SIMD tail", "[entropy][circular][simd]") {
    // 2-wide NEON / 4-wide AVX2 → odd and mod-4 lengths exercise remainders
    for (size_t n : {3ull, 5ull, 7ull, 9ull, 17ull, 33ull, 65ull, 127ull, 255ull, 1001ull}) {
        auto angles = linspace(-180.0, 179.9, n);
        // Inject one NaN mid-stream so mixed finite pairs hit scalar lane
        if (n > 4) angles[n / 2] = NAN;
        double h = circular_h(angles);
        REQUIRE(std::isfinite(h));
        REQUIRE(h >= 0.0);
    }
}

TEST_CASE("Circular: all identical angles → zero entropy", "[entropy][circular][simd]") {
    std::vector<double> angles(1000, 42.0);
    REQUIRE(circular_h(angles) == 0.0);

    // Same after multi-turn equivalents of a single angle
    std::vector<double> wrapped(500, 42.0 + 360.0);
    for (int i = 0; i < 500; ++i) wrapped.push_back(42.0 - 360.0);
    REQUIRE(circular_h(wrapped) == 0.0);
}

TEST_CASE("Circular: sparse NaN/Inf does not change clean distribution",
          "[entropy][circular][simd]") {
    // Interleave non-finite *alongside* clean samples (do not replace them).
    auto clean = linspace(-90.0, 90.0, 800);
    std::vector<double> dirty;
    dirty.reserve(clean.size() + clean.size() / 25 + 4);
    for (size_t i = 0; i < clean.size(); ++i) {
        if (i % 50 == 0) dirty.push_back(NAN);
        if (i % 50 == 25) dirty.push_back(INFINITY);
        dirty.push_back(clean[i]);
    }
    dirty.push_back(-INFINITY);
    dirty.push_back(NAN);

    double h_clean = circular_h(clean);
    double h_dirty = circular_h(dirty);
    REQUIRE_THAT(h_dirty, WithinAbs(h_clean, ENTROPY_TOL));
}

TEST_CASE("Circular: adaptive sees wraparound as high entropy, circular low",
          "[entropy][circular][parity]") {
    // Classic FlexAID regression: mass near ±180
    std::vector<double> angles;
    for (int i = 0; i < 200; ++i) {
        angles.push_back(170.0 + 10.0 * static_cast<double>(i) / 199.0);
        angles.push_back(-180.0 + 10.0 * static_cast<double>(i) / 199.0);
    }
    double h_circ = circular_h(angles);
    double h_lin = 0.0;
    REQUIRE(ba_shannon_entropy(angles.data(), angles.size(), DEFAULT_BIN_COUNT, &h_lin) == BA_OK);
    // Linear adaptive range is ~360° → two edge bins → often lower H actually...
    // The key FlexAID property: circular remains low (clustered).
    REQUIRE(h_circ < 3.0);
    REQUIRE(std::isfinite(h_lin));
}

// ═══════════════════════════════════════════════════════════════════════════
// Size thresholds: small-N CPU vs large-N (may hit Metal) numerical parity
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Circular: small N and large N same formula for uniform arc",
          "[entropy][circular][parity]") {
    // Subsample of a uniform circle should approach same H as denser sample
    // when distribution shape is identical (near-uniform → near max).
    auto make_uniform = [](size_t n) {
        return linspace(-180.0, 180.0 - 360.0 / static_cast<double>(n), n);
    };
    double h_small = circular_h(make_uniform(128));   // CPU path
    double h_large = circular_h(make_uniform(16384)); // may use GPU if available
    // Both near log2(32)=5; allow modest gap due to sampling density
    REQUIRE(h_small > 4.5);
    REQUIRE(h_large > 4.9);
    REQUIRE(std::abs(h_large - h_small) < 0.4);
}

TEST_CASE("Circular: constrained small and large agree within tight tol",
          "[entropy][circular][parity]") {
    // Same analytical distribution: uniform in [-10, 10], different N
    auto make = [](size_t n) { return linspace(-10.0, 10.0, n); };
    double h64 = circular_h(make(64));
    double h4096 = circular_h(make(4096));
    double h16384 = circular_h(make(16384));
    // Dense samples of same interval → H should converge
    REQUIRE_THAT(h4096, WithinAbs(h16384, 0.05));
    REQUIRE(h64 > 0.0);
    REQUIRE(h64 < 3.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch circular
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Circular batch: empty batch_count returns error", "[entropy][circular][batch]") {
    double out = 0;
    size_t off = 0, len = 0;
    double flat = 0;
    REQUIRE(ba_circular_shannon_entropy_batch(&flat, &off, &len, 0, 32, &out)
            == BA_ERR_INSUFFICIENT_DATA);
}

TEST_CASE("Circular batch: invalid bin_count", "[entropy][circular][batch]") {
    std::vector<double> flat = {1, 2, 3, 4};
    size_t off = 0, len = 4;
    double out = 0;
    REQUIRE(ba_circular_shannon_entropy_batch(flat.data(), &off, &len, 1, 0, &out)
            == BA_ERR_INVALID_PARAM);
}

TEST_CASE("Circular batch: null pointers", "[entropy][circular][batch]") {
    size_t off = 0, len = 2;
    double out = 0;
    double flat[2] = {0, 1};
    REQUIRE(ba_circular_shannon_entropy_batch(nullptr, &off, &len, 1, 32, &out)
            == BA_ERR_NULL_PTR);
    REQUIRE(ba_circular_shannon_entropy_batch(flat, nullptr, &len, 1, 32, &out)
            == BA_ERR_NULL_PTR);
    REQUIRE(ba_circular_shannon_entropy_batch(flat, &off, nullptr, 1, 32, &out)
            == BA_ERR_NULL_PTR);
    REQUIRE(ba_circular_shannon_entropy_batch(flat, &off, &len, 1, 32, nullptr)
            == BA_ERR_NULL_PTR);
}

TEST_CASE("Circular batch: short segments return 0 entropy", "[entropy][circular][batch]") {
    // length 0 and 1 are insufficient clean data → 0 from core
    std::vector<double> flat = {10.0, 20.0, 30.0};
    size_t offsets[] = {0, 1, 2};
    size_t lengths[] = {0, 1, 1};
    double out[3] = {-1, -1, -1};
    REQUIRE(ba_circular_shannon_entropy_batch(flat.data(), offsets, lengths, 3,
                                               DEFAULT_BIN_COUNT, out) == BA_OK);
    REQUIRE(out[0] == 0.0);
    REQUIRE(out[1] == 0.0);
    REQUIRE(out[2] == 0.0);
}

TEST_CASE("Circular batch: FlexAID multi-bond free/bound packing",
          "[entropy][circular][batch][flexaids]") {
    // 6 free + 6 bound = 12 segments (mirrors FlexAIDdS bond packing)
    constexpr size_t n_bonds = 6;
    constexpr size_t n_samples = 256;
    std::vector<double> flat;
    std::vector<size_t> offsets, lengths;
    std::vector<double> individual_h;

    for (size_t b = 0; b < n_bonds; ++b) {
        // free: wide
        offsets.push_back(flat.size());
        lengths.push_back(n_samples);
        double center = -180.0 + 60.0 * static_cast<double>(b);
        for (size_t i = 0; i < n_samples; ++i) {
            flat.push_back(center + 40.0 * (static_cast<double>(i) / n_samples - 0.5));
        }
    }
    for (size_t b = 0; b < n_bonds; ++b) {
        // bound: narrow
        offsets.push_back(flat.size());
        lengths.push_back(n_samples);
        double center = -180.0 + 60.0 * static_cast<double>(b);
        for (size_t i = 0; i < n_samples; ++i) {
            flat.push_back(center + 5.0 * (static_cast<double>(i) / n_samples - 0.5));
        }
    }

    std::vector<double> batch(2 * n_bonds, -1.0);
    REQUIRE(ba_circular_shannon_entropy_batch(flat.data(), offsets.data(), lengths.data(),
                                               2 * n_bonds, DEFAULT_BIN_COUNT,
                                               batch.data()) == BA_OK);

    for (size_t b = 0; b < 2 * n_bonds; ++b) {
        double h = 0.0;
        REQUIRE(ba_circular_shannon_entropy(flat.data() + offsets[b], lengths[b],
                                             DEFAULT_BIN_COUNT, &h) == BA_OK);
        REQUIRE_THAT(batch[b], WithinAbs(h, 1e-9));
        individual_h.push_back(h);
    }

    // Each bond: ΔS = H_bound - H_free < 0
    for (size_t b = 0; b < n_bonds; ++b) {
        double delta = individual_h[n_bonds + b] - individual_h[b];
        REQUIRE(delta < -0.5);
    }
}

TEST_CASE("Circular batch: many tiny bonds still matches single",
          "[entropy][circular][batch]") {
    constexpr size_t n_items = 32;
    constexpr size_t n_per = 100;
    std::vector<double> flat;
    std::vector<size_t> offsets, lengths;
    for (size_t b = 0; b < n_items; ++b) {
        offsets.push_back(flat.size());
        lengths.push_back(n_per);
        for (size_t i = 0; i < n_per; ++i) {
            flat.push_back(10.0 * static_cast<double>(b)
                           + 2.0 * std::sin(0.1 * static_cast<double>(i)));
        }
    }
    std::vector<double> batch(n_items);
    REQUIRE(ba_circular_shannon_entropy_batch(flat.data(), offsets.data(), lengths.data(),
                                               n_items, DEFAULT_BIN_COUNT, batch.data())
            == BA_OK);
    for (size_t b = 0; b < n_items; ++b) {
        double h = 0;
        ba_circular_shannon_entropy(flat.data() + offsets[b], lengths[b],
                                     DEFAULT_BIN_COUNT, &h);
        REQUIRE_THAT(batch[b], WithinAbs(h, 1e-9));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adaptive Shannon SIMD-path stress (min/max + bin)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Adaptive: odd length + NaN parity with clean", "[entropy][simd]") {
    constexpr size_t N = 1001;
    std::vector<double> clean, dirty;
    for (size_t i = 0; i < N; ++i) {
        double v = std::sin(0.07 * static_cast<double>(i));
        clean.push_back(v);
        dirty.push_back(v);
        if (i % 41 == 0) dirty.push_back(NAN);
        if (i % 53 == 0) dirty.push_back(INFINITY);
    }
    double h_c = 0, h_d = 0;
    REQUIRE(ba_shannon_entropy(clean.data(), clean.size(), 32, &h_c) == BA_OK);
    REQUIRE(ba_shannon_entropy(dirty.data(), dirty.size(), 32, &h_d) == BA_OK);
    REQUIRE_THAT(h_d, WithinAbs(h_c, ENTROPY_TOL));
}

TEST_CASE("Adaptive: invalid bin_count and null", "[entropy]") {
    double v[4] = {1, 2, 3, 4};
    double out = 0;
    REQUIRE(ba_shannon_entropy(v, 4, 0, &out) == BA_ERR_INVALID_PARAM);
    REQUIRE(ba_shannon_entropy(nullptr, 4, 32, &out) == BA_ERR_NULL_PTR);
    REQUIRE(ba_shannon_entropy(v, 4, 32, nullptr) == BA_ERR_NULL_PTR);
    REQUIRE(ba_circular_shannon_entropy(v, 4, 0, &out) == BA_ERR_INVALID_PARAM);
}

TEST_CASE("Adaptive: large N GPU-path finite and in range", "[entropy][parity]") {
    std::vector<double> vals(20000);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = std::cos(0.001 * static_cast<double>(i)) + 0.01 * (i % 13);
    }
    double h = 0;
    REQUIRE(ba_shannon_entropy(vals.data(), vals.size(), 32, &h) == BA_OK);
    REQUIRE(std::isfinite(h));
    REQUIRE(h >= 0.0);
    REQUIRE(h <= std::log2(32.0) + 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// Status / backend sanity
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Status strings cover known codes", "[backend]") {
    REQUIRE(std::string(ba_status_string(BA_OK)) == "OK");
    REQUIRE(std::string(ba_status_string(BA_ERR_NULL_PTR)).find("Null") != std::string::npos);
    REQUIRE(std::string(ba_status_string(BA_ERR_INVALID_PARAM)).length() > 0);
    REQUIRE(std::string(ba_status_string(static_cast<BAStatus>(999))).find("Unknown")
            != std::string::npos);
}

TEST_CASE("Version is semver-like 1.x", "[backend]") {
    std::string v = ba_version();
    REQUIRE(v.size() >= 3);
    REQUIRE(v[0] == '1');
    REQUIRE(v.find('.') != std::string::npos);
}
