import Foundation
#if BONHOMME_ACCEL
import BonhommeAccelSwift
#endif

/// Reusable Shannon entropy calculator for any distribution of continuous values.
///
/// Originally developed for HRV RR-interval analysis (Shannon Collapse Index),
/// this utility generalizes to sleep stage distributions, respiratory rate patterns,
/// activity variability, and any domain where distributional entropy measures
/// coherence or variability.
///
/// Methodology ported from the configurational entropy engine in FlexAIDdS,
/// adapted from molecular torsional distributions to generalized health signal distributions.
public struct EntropyCalculator: Sendable {
    /// Number of histogram bins for entropy calculation.
    public let binCount: Int

    public init(binCount: Int = 32) {
        // binCount < 1 would zero-divide and create an empty histogram.
        self.binCount = max(1, binCount)
    }

    /// Shannon entropy of a distribution of values, binned into a histogram.
    ///
    /// H = -Σ p_i log₂(p_i) for each bin with p_i > 0.
    ///
    /// Uses data-adaptive bin range [min, max]. Appropriate for linear domains
    /// (RR intervals, respiratory rates). For circular domains (torsional angles),
    /// use ``circularShannonEntropy(_:)`` instead.
    ///
    /// - Parameter values: Array of continuous values (e.g., RR intervals in ms,
    ///   sleep stage durations, respiratory rates).
    /// - Returns: Entropy in bits. Higher = more uniform/variable; lower = more concentrated/coherent.
    public func shannonEntropy(_ values: [Double]) -> Double {
        #if BONHOMME_ACCEL
        if values.count >= AccelEntropy.delegationThreshold,
           let result = AccelEntropy.shannonEntropy(values, binCount: binCount) {
            return result
        }
        #endif

        let clean = values.filter { $0.isFinite }
        guard clean.count >= 2 else { return 0 }

        guard let minVal = clean.min(), let maxVal = clean.max() else { return 0 }
        let range = maxVal - minVal
        guard range > 0 else { return 0 }

        let binWidth = range / Double(binCount)
        var bins = [Int](repeating: 0, count: binCount)

        for value in clean {
            var idx = Int((value - minVal) / binWidth)
            if idx < 0 { idx = 0 }
            if idx >= binCount { idx = binCount - 1 }
            bins[idx] += 1
        }

        let total = Double(clean.count)
        var entropy = 0.0
        for count in bins where count > 0 {
            let p = Double(count) / total
            entropy -= p * log2(p)
        }
        return entropy
    }

    /// Shannon entropy for circular distributions (e.g., torsional angles in degrees).
    ///
    /// Unlike ``shannonEntropy(_:)``, this method uses **fixed bins** spanning [-180°, +180°)
    /// so that angles near the boundary (e.g., -179° and +179°) are correctly recognized
    /// as adjacent. This prevents the linear method from artificially inflating entropy
    /// for distributions that wrap around ±180°.
    ///
    /// H = -Σ p_i log₂(p_i) — same formula, domain-appropriate binning.
    ///
    /// - Parameter angles: Torsional angles in degrees. Values outside [-180, 180]
    ///   are wrapped via modular arithmetic.
    /// - Returns: Entropy in bits. Range [0, log₂(binCount)].
    public func circularShannonEntropy(_ angles: [Double]) -> Double {
        #if BONHOMME_ACCEL
        if angles.count >= AccelEntropy.delegationThreshold,
           let result = AccelEntropy.circularShannonEntropy(angles, binCount: binCount) {
            return result
        }
        #endif

        return circularShannonEntropyScalar(angles)
    }

    /// Batch circular Shannon entropy for multiple independent angle arrays.
    ///
    /// Uses Accel batch C API when linked and data is large enough to amortize
    /// dispatch (multi-bond FlexAID∆S path). Falls back to per-array
    /// ``circularShannonEntropy(_:)`` (which may still use single-array Accel).
    ///
    /// - Parameter arrays: One angle sample array per histogram (e.g. one bond).
    /// - Returns: Entropy in bits for each array, same order as `arrays`.
    public func circularShannonEntropyBatch(_ arrays: [[Double]]) -> [Double] {
        guard !arrays.isEmpty else { return [] }

        #if BONHOMME_ACCEL
        let total = arrays.reduce(0) { $0 + $1.count }
        let maxLen = arrays.map(\.count).max() ?? 0
        // Multi-item or large single: one C call beats N Swift/C round-trips.
        if arrays.count >= 2 || maxLen >= AccelEntropy.delegationThreshold,
           total >= AccelEntropy.delegationThreshold {
            var flat: [Double] = []
            var offsets: [Int] = []
            var lengths: [Int] = []
            flat.reserveCapacity(total)
            offsets.reserveCapacity(arrays.count)
            lengths.reserveCapacity(arrays.count)
            for a in arrays {
                offsets.append(flat.count)
                lengths.append(a.count)
                flat.append(contentsOf: a)
            }
            if let results = AccelEntropy.circularShannonEntropyBatch(
                flat: flat, offsets: offsets, lengths: lengths, binCount: binCount
            ) {
                return results
            }
        }
        #endif

        return arrays.map { circularShannonEntropy($0) }
    }

    /// Scalar circular entropy (no Accel). Used by single-path and tests.
    private func circularShannonEntropyScalar(_ angles: [Double]) -> Double {
        let clean = angles.filter { $0.isFinite }
        guard clean.count >= 2 else { return 0 }

        let binWidth = 360.0 / Double(binCount)
        var bins = [Int](repeating: 0, count: binCount)

        for angle in clean {
            // Wrap to [-180, 180)
            var a = angle.truncatingRemainder(dividingBy: 360.0)
            if a > 180.0 { a -= 360.0 }
            if a < -180.0 { a += 360.0 }
            // Canonicalize the cut: +180 ≡ -180 are the same angle; fold both onto -180
            // so they land in the same bin (parity with the SIMD/C++ kernels).
            if a == 180.0 { a = -180.0 }
            // Map [-180, 180) → bin index [0, binCount)
            var idx = Int((a + 180.0) / binWidth)
            if idx < 0 { idx = 0 }
            if idx >= binCount { idx = binCount - 1 }
            bins[idx] += 1
        }

        let total = Double(clean.count)
        var entropy = 0.0
        for count in bins where count > 0 {
            let p = Double(count) / total
            entropy -= p * log2(p)
        }
        return entropy
    }

    /// Map entropy (bits) to a 0–1 coherence score where 1 = maximally coherent.
    ///
    /// Uses ``binCount`` to set the theoretical maximum: log₂(binCount)
    /// (e.g. 5.0 for the default 32-bin histogram).
    ///
    /// - Parameter entropy: Shannon entropy in bits.
    /// - Returns: Score in [0.0, 1.0] where 1.0 = zero entropy (perfect coherence).
    public func entropyToScore(_ entropy: Double) -> Double {
        entropyToScore(entropy, maxEntropy: log2(Double(binCount)))
    }

    /// Map entropy (bits) to a 0–1 coherence score where 1 = maximally coherent.
    ///
    /// - Parameters:
    ///   - entropy: Shannon entropy in bits.
    ///   - maxEntropy: Theoretical maximum entropy for normalization.
    ///     Prefer log₂(binCount) (the default when omitting this parameter via
    ///     ``entropyToScore(_:)``). Override only when scoring against a different
    ///     theoretical ceiling.
    /// - Returns: Score in [0.0, 1.0] where 1.0 = zero entropy (perfect coherence).
    public func entropyToScore(_ entropy: Double, maxEntropy: Double) -> Double {
        guard maxEntropy > 0 else { return 1.0 }
        let clamped = max(0, min(maxEntropy, entropy))
        return 1.0 - (clamped / maxEntropy)
    }

    /// Shannon entropy with fixed bin edges over a known domain.
    ///
    /// Unlike the adaptive overload, this uses caller-specified domain bounds so that
    /// identical distributions always produce identical histograms regardless of sample extremes.
    /// Use for torsional angles (domainMin: -180, domainMax: 180) or any bounded domain.
    ///
    /// Values outside the domain are clamped to the nearest edge.
    ///
    /// - Parameters:
    ///   - values: Array of continuous values.
    ///   - domainMin: Lower bound of the histogram domain (inclusive).
    ///   - domainMax: Upper bound of the histogram domain (inclusive).
    /// - Returns: Entropy in bits.
    public func shannonEntropy(_ values: [Double], domainMin: Double, domainMax: Double) -> Double {
        #if BONHOMME_ACCEL
        if values.count >= AccelEntropy.delegationThreshold,
           let result = AccelEntropy.shannonEntropyFixed(values, binCount: binCount,
                                                          domainMin: domainMin, domainMax: domainMax) {
            return result
        }
        #endif

        // Filter non-finite before binning (parity with adaptive/circular paths and Accel).
        let clean = values.filter { $0.isFinite }
        guard clean.count >= 2 else { return 0 }
        let range = domainMax - domainMin
        guard range > 0 else { return 0 }

        let binWidth = range / Double(binCount)
        var bins = [Int](repeating: 0, count: binCount)

        for value in clean {
            let clamped = max(domainMin, min(domainMax, value))
            var idx = Int((clamped - domainMin) / binWidth)
            if idx < 0 { idx = 0 }
            if idx >= binCount { idx = binCount - 1 }
            bins[idx] += 1
        }

        let total = Double(clean.count)
        var entropy = 0.0
        for count in bins where count > 0 {
            let p = Double(count) / total
            entropy -= p * log2(p)
        }
        return entropy
    }

    /// Compute both entropy and its normalized score in one call.
    ///
    /// Score normalization defaults to maxEntropy = log₂(binCount).
    ///
    /// - Parameter values: Array of continuous values.
    /// - Returns: Tuple of (entropy in bits, coherence score 0–1), or nil if insufficient data.
    public func analyze(_ values: [Double]) -> (entropy: Double, score: Double)? {
        analyze(values, maxEntropy: log2(Double(binCount)))
    }

    /// Compute both entropy and its normalized score in one call.
    ///
    /// - Parameters:
    ///   - values: Array of continuous values.
    ///   - maxEntropy: Theoretical maximum for score normalization.
    ///     Prefer log₂(binCount) (the default when omitting this parameter via
    ///     ``analyze(_:)``). Override only when scoring against a different ceiling.
    /// - Returns: Tuple of (entropy in bits, coherence score 0–1), or nil if insufficient data.
    public func analyze(_ values: [Double], maxEntropy: Double) -> (entropy: Double, score: Double)? {
        guard values.count >= 2 else { return nil }
        let h = shannonEntropy(values)
        let s = entropyToScore(h, maxEntropy: maxEntropy)
        return (entropy: h, score: s)
    }
}

// MARK: - Entropy Event Classification

/// Classified entropy event, mirroring Shannon's `EntropyEvent` enum.
///
/// The unified entropy framework recognizes four states across all three domains
/// (molecular docking, LLM safety, physiological biofeedback):
///
/// | Event | FlexAIDdS | Shannon | NATURaL |
/// |-------|-----------|---------|---------|
/// | collapse | Binding lock-in | Evaluation awareness | Sympathomimetic onset |
/// | expansion | Solvation release | Jailbreak / evasion | Parasympathomimetic onset |
/// | oscillation | Unstable binding site | Adversarial probing | Autonomic instability |
/// | none | Free in solvent | Normal generation | Resting tone |
public enum EntropyEvent: String, Sendable {
    case none = "none"
    case collapse = "collapse"
    case expansion = "expansion"
    case oscillation = "oscillation"
}

/// Sliding-window entropy event detector.
///
/// Tracks entropy over a window of recent measurements and detects three classes
/// of anomaly: collapse (ordering), expansion (disordering), and oscillation
/// (rapid alternation). This mirrors the `CollapseDetector` in Shannon and
/// the `detect_entropy_plateau` in FlexAIDdS.
public struct EntropyEventDetector: Sendable {
    public let windowSize: Int
    /// Z-score threshold for collapse detection (default -3.2σ).
    public let collapseThreshold: Double
    /// Z-score threshold for expansion detection (default +3.2σ).
    public let expansionThreshold: Double
    public let oscillationWindow: Int

    private var window: [Double]
    private var eventHistory: [EntropyEvent]

    public init(
        windowSize: Int = 8,
        collapseThreshold: Double = -3.2,
        expansionThreshold: Double = 3.2,
        oscillationWindow: Int = 5
    ) {
        self.windowSize = max(1, windowSize)
        self.collapseThreshold = collapseThreshold
        self.expansionThreshold = expansionThreshold
        self.oscillationWindow = max(1, oscillationWindow)
        self.window = []
        self.eventHistory = []
    }

    /// Push an entropy value and classify the event.
    public mutating func push(_ entropy: Double) -> (event: EntropyEvent, delta: Double, zScore: Double) {
        window.append(entropy)
        if window.count > windowSize {
            window.removeFirst()
        }

        let count = window.count
        guard count >= 2 else {
            eventHistory.append(.none)
            if eventHistory.count > oscillationWindow { eventHistory.removeFirst() }
            return (.none, 0, 0)
        }

        let mean = window.reduce(0, +) / Double(count)
        let variance = window.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count)
        let std = sqrt(max(0, variance))
        let delta = entropy - mean
        let z = std > 1e-12 ? delta / std : 0.0

        let windowReady = count >= windowSize

        // Classify on the rolling z-score (delta / std), not the raw bit delta:
        // a raw ±3.2-bit swing is ~65% of the full log2(32) range and effectively
        // never fires. Thresholding z keeps collapse/expansion a meaningful
        // "Nσ" event across entropy scales (mirrors FlexAIDdS CollapseDetector).
        var event: EntropyEvent = .none
        if windowReady && z < collapseThreshold {
            event = .collapse
        } else if windowReady && z > expansionThreshold {
            event = .expansion
        }

        eventHistory.append(event)
        if eventHistory.count > oscillationWindow { eventHistory.removeFirst() }

        if windowReady && event != .none {
            let alternations = countAlternations()
            if alternations >= 2 {
                event = .oscillation
            }
        }

        return (event, delta, z)
    }

    /// Reset detector state.
    public mutating func reset() {
        window.removeAll()
        eventHistory.removeAll()
    }

    private func countAlternations() -> Int {
        var count = 0
        for i in 1..<eventHistory.count {
            let prev = eventHistory[i - 1]
            let curr = eventHistory[i]
            if (prev == .collapse && curr == .expansion) ||
               (prev == .expansion && curr == .collapse) {
                count += 1
            }
        }
        return count
    }
}

// MARK: - Shared Statistical Utilities

/// Pearson product-moment correlation coefficient.
///
/// - Parameters:
///   - x: First variable array.
///   - y: Second variable array (must have same count as x).
/// - Returns: Pearson r in [-1, 1], or 0 if insufficient data or zero variance.
public func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
    #if BONHOMME_ACCEL
    if let result = AccelCorrelation.pearsonCorrelation(x, y) {
        return result
    }
    #endif

    guard x.count >= 2, x.count == y.count else { return 0 }

    // Filter non-finite pairs (parity with linearRegression and the C++ Accel core).
    var cx: [Double] = []
    var cy: [Double] = []
    cx.reserveCapacity(x.count)
    cy.reserveCapacity(y.count)
    for i in 0..<x.count {
        if x[i].isFinite && y[i].isFinite {
            cx.append(x[i])
            cy.append(y[i])
        }
    }

    let n = Double(cx.count)
    guard n >= 2 else { return 0 }

    let meanX = cx.reduce(0, +) / n
    let meanY = cy.reduce(0, +) / n

    var sumXY = 0.0
    var sumX2 = 0.0
    var sumY2 = 0.0

    for i in 0..<cx.count {
        let dx = cx[i] - meanX
        let dy = cy[i] - meanY
        sumXY += dx * dy
        sumX2 += dx * dx
        sumY2 += dy * dy
    }

    let denom = sqrt(sumX2 * sumY2)
    guard denom > 0 else { return 0 }
    return sumXY / denom
}

/// Ordinary least-squares linear regression: y = slope × x + intercept.
///
/// - Parameters:
///   - x: Independent variable array.
///   - y: Dependent variable array (must have same count as x).
/// - Returns: Tuple of (slope, intercept, mean absolute error).
public func linearRegression(x: [Double], y: [Double]) -> (slope: Double, intercept: Double, mae: Double) {
    #if BONHOMME_ACCEL
    if let result = AccelCorrelation.linearRegression(x: x, y: y) {
        return result
    }
    #endif

    guard x.count == y.count, x.count >= 2 else {
        return (slope: 0, intercept: 0, mae: 0)
    }

    // Filter non-finite pairs (parity with pearsonCorrelation / Accel)
    var cx: [Double] = []
    var cy: [Double] = []
    cx.reserveCapacity(x.count)
    cy.reserveCapacity(y.count)
    for i in 0..<x.count {
        if x[i].isFinite && y[i].isFinite {
            cx.append(x[i])
            cy.append(y[i])
        }
    }

    let n = Double(cx.count)
    guard n >= 2 else { return (slope: 0, intercept: 0, mae: 0) }

    let meanX = cx.reduce(0, +) / n
    let meanY = cy.reduce(0, +) / n

    var sumXY = 0.0
    var sumX2 = 0.0

    for i in 0..<cx.count {
        let dx = cx[i] - meanX
        sumXY += dx * (cy[i] - meanY)
        sumX2 += dx * dx
    }

    let slope = sumX2 > 0 ? sumXY / sumX2 : 0
    let intercept = meanY - slope * meanX

    var totalError = 0.0
    for i in 0..<cx.count {
        let predicted = slope * cx[i] + intercept
        totalError += abs(cy[i] - predicted)
    }
    let mae = totalError / n

    return (slope: slope, intercept: intercept, mae: mae)
}
