import Foundation

/// Computes the Shannon Collapse Index (SCI) from HRV signals.
///
/// The SCI measures autonomic coherence by calculating the Shannon entropy
/// of RR-interval distributions. Lower entropy indicates higher coherence
/// (focused breathing), while higher entropy indicates variability
/// (distracted or stressed state).
///
/// Methodology ported from the configurational entropy engine in FlexAIDdS,
/// adapted from molecular torsional distributions to cardiac interval distributions.
///
/// ## Fixed RR domain
/// SCI entropy uses a **fixed** physiological RR-interval domain
/// [`rrDomainMinMs`, `rrDomainMaxMs`] (300–1500 ms), not data-adaptive binning.
/// Fixed edges keep histograms comparable across sessions and make collapse
/// meaningful: a narrow cluster occupies few of the full-domain bins.
public struct HRVAnalyzer: SignalAnalyzer, Sendable {
    public let primarySignalType: SignalType = .heartRateVariability

    // MARK: - Fixed RR histogram domain (ms)

    /// Lower bound of the SCI RR-interval histogram domain, in milliseconds.
    ///
    /// 300 ms ≈ 200 BPM — extreme tachycardia. Values below this are clamped
    /// into the first bin so outliers cannot shrink the effective domain.
    public static let rrDomainMinMs: Double = 300

    /// Upper bound of the SCI RR-interval histogram domain, in milliseconds.
    ///
    /// 1500 ms ≈ 40 BPM — extreme bradycardia. Values above this are clamped
    /// into the last bin so outliers cannot expand the effective domain.
    public static let rrDomainMaxMs: Double = 1500

    /// Shared entropy calculator (reusable across sleep, respiratory, activity analyzers).
    private let entropyCalc: EntropyCalculator
    /// Window size in seconds for sliding entropy.
    private let windowSeconds: TimeInterval
    /// Entropy threshold (bits) below which we consider "focused".
    private let collapseThreshold: Double

    /// Number of histogram bins (delegates to EntropyCalculator).
    var binCount: Int { entropyCalc.binCount }

    public init(
        binCount: Int = 32,
        windowSeconds: TimeInterval = 60,
        collapseThreshold: Double = 3.2
    ) {
        self.entropyCalc = EntropyCalculator(binCount: binCount)
        self.windowSeconds = windowSeconds
        self.collapseThreshold = collapseThreshold
    }

    public func analyze(
        signals: [any HealthSignal],
        context: AnalysisContext
    ) -> AnalysisInsight {
        let hrvSignals = signals.compactMap { $0 as? HRVSignal }

        guard !hrvSignals.isEmpty else {
            return AnalysisInsight(
                signalType: .heartRateVariability,
                score: nil,
                trend: .stable,
                status: .normal,
                summary: LocalizedString(
                    en: "No HRV data available yet.",
                    fr: "Aucune donnée VRC disponible pour le moment.",
                    es: "No hay datos de VRC disponibles aún.",
                    ja: "HRVデータはまだありません。",
                    zh: "尚无心率变异性数据。",
                    ko: "아직 HRV 데이터가 없습니다.",
                    ru: "Данные ВСР ещё недоступны.",
                    de: "Noch keine HRV-Daten verfügbar.",
                    ar: "لا تتوفر بيانات تقلب معدل ضربات القلب بعد."
                )
            )
        }

        // Collect all RR intervals within the window
        let cutoff = Date().addingTimeInterval(-windowSeconds)
        let windowedSignals = hrvSignals.filter { $0.timestamp >= cutoff }

        let allRR = windowedSignals.flatMap(\.rrIntervals)
        let entropy = allRR.count >= 4 ? shannonEntropy(allRR) : nil
        let sciScore = entropy.map { entropyToScore($0) }

        // Trend: compare first half vs second half
        let trend = computeTrend(signals: windowedSignals)

        // Cross-reference medication context if available
        let medNote = medicationContextNote(context: context)

        let status: InsightStatus
        if let score = sciScore {
            status = score >= 0.6 ? .normal : (score >= 0.3 ? .advisory : .alert)
        } else {
            status = .normal
        }

        let scoreText = sciScore.map { String(format: "%.0f", $0 * 100) } ?? "--"
        return AnalysisInsight(
            signalType: .heartRateVariability,
            score: sciScore,
            trend: trend,
            status: status,
            summary: LocalizedString(
                en: "Focus coherence: \(scoreText)%.\(medNote)",
                fr: "Cohérence de concentration : \(scoreText) %.\(medNote)",
                es: "Coherencia de concentración: \(scoreText) %.\(medNote)",
                ja: "集中コヒーレンス：\(scoreText)%。\(medNote)",
                zh: "专注一致性：\(scoreText)%。\(medNote)",
                ko: "집중 코히어런스: \(scoreText)%.\(medNote)",
                ru: "Когерентность концентрации: \(scoreText) %.\(medNote)",
                de: "Fokus-Kohärenz: \(scoreText) %.\(medNote)",
                ar: "تماسك التركيز: \(scoreText)٪.\(medNote)"
            )
        )
    }

    // MARK: - Entropy Math (delegates to EntropyCalculator)

    /// Shannon entropy of RR intervals over the fixed physiological domain
    /// [`rrDomainMinMs`, `rrDomainMaxMs`] (300–1500 ms).
    ///
    /// Does **not** use data-adaptive binning — identical distributions always
    /// produce comparable histograms regardless of sample extremes.
    /// Kept as internal API so existing tests continue to work unchanged.
    func shannonEntropy(_ intervals: [Double]) -> Double {
        entropyCalc.shannonEntropy(
            intervals,
            domainMin: Self.rrDomainMinMs,
            domainMax: Self.rrDomainMaxMs
        )
    }

    /// Map entropy (bits) to a 0–1 score where 1 = maximally focused.
    /// Normalizes against log₂(binCount) (5.0 for the default 32 bins).
    private func entropyToScore(_ entropy: Double) -> Double {
        entropyCalc.entropyToScore(entropy)
    }

    private func computeTrend(signals: [HRVSignal]) -> InsightTrend {
        guard signals.count >= 4 else { return .stable }
        let mid = signals.count / 2
        let firstHalf = signals[..<mid].map(\.rmssd)
        let secondHalf = signals[mid...].map(\.rmssd)

        let avgFirst = firstHalf.reduce(0, +) / Double(firstHalf.count)
        let avgSecond = secondHalf.reduce(0, +) / Double(secondHalf.count)
        let delta = avgSecond - avgFirst

        if delta > 5 { return .improving }
        if delta < -5 { return .declining }
        return .stable
    }

    /// If medication signals exist in context, note potential correlation.
    private func medicationContextNote(context: AnalysisContext) -> String {
        guard let medSignals = context.signalsByType[.medication],
              let latest = medSignals.last as? MedicationSignal,
              latest.event == .taken,
              Date().timeIntervalSince(latest.timestamp) < 3600 else {
            return ""
        }
        return " Recent \(latest.name.localized) dose may affect readings."
    }
}
