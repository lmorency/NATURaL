/*
 * correlation.cpp — Pearson correlation and linear regression (scalar).
 *
 * Mirrors the global pearsonCorrelation() and linearRegression() from
 * BonhommeCore/EntropyCalculator.swift.
 */

#include "correlation.h"

#include <cmath>
#include <vector>

namespace ba::core {

double pearson_correlation(const double* x, const double* y, size_t count) {
    if (!x || !y || count < 2) return 0.0;

    // Fast path: avoid the heap copy when every pair is finite.
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

    double n_dbl = static_cast<double>(n);
    double mean_x = 0.0, mean_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean_x += px[i];
        mean_y += py[i];
    }
    mean_x /= n_dbl;
    mean_y /= n_dbl;

    double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = px[i] - mean_x;
        double dy = py[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    double denom = std::sqrt(sum_x2 * sum_y2);
    if (denom <= 0.0) return 0.0;

    return sum_xy / denom;
}

RegressionResult linear_regression(const double* x, const double* y, size_t count) {
    RegressionResult result{0.0, 0.0, 0.0};
    if (!x || !y || count < 2) return result;

    // Filter non-finite pairs (parity with pearson_correlation)
    std::vector<double> cx, cy;
    cx.reserve(count);
    cy.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(x[i]) && std::isfinite(y[i])) {
            cx.push_back(x[i]);
            cy.push_back(y[i]);
        }
    }

    double n = static_cast<double>(cx.size());
    if (n < 2.0) return result;

    double mean_x = 0.0, mean_y = 0.0;
    for (size_t i = 0; i < cx.size(); ++i) {
        mean_x += cx[i];
        mean_y += cy[i];
    }
    mean_x /= n;
    mean_y /= n;

    double sum_xy = 0.0, sum_x2 = 0.0;
    for (size_t i = 0; i < cx.size(); ++i) {
        double dx = cx[i] - mean_x;
        sum_xy += dx * (cy[i] - mean_y);
        sum_x2 += dx * dx;
    }

    result.slope = (sum_x2 > 0.0) ? (sum_xy / sum_x2) : 0.0;
    result.intercept = mean_y - result.slope * mean_x;

    double total_error = 0.0;
    for (size_t i = 0; i < cx.size(); ++i) {
        double predicted = result.slope * cx[i] + result.intercept;
        total_error += std::abs(cy[i] - predicted);
    }
    result.mae = total_error / n;

    return result;
}

} // namespace ba::core
