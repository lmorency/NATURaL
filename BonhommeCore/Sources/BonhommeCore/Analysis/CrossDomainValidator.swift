import Foundation
#if BONHOMME_ACCEL
import BonhommeAccelSwift
#endif

/// Validates the isomorphism between FlexAID∆S configurational entropy (in silico)
/// and DrugResponseAnalyzer HRV entropy (in vivo).
///
/// The hypothesis under test:
///   A drug with large |ΔS_config| in molecular docking should produce
///   large |ΔH_hrv| in cardiac RR-interval distributions, because both
///   measure the entropy cost of a binding event using the same Shannon formula.
///
/// Cross-domain validation proceeds by:
/// 1. Pairing substances that have both in-silico (FlexAIDdSResult or BindingEntropyProfile)
///    and in-vivo (DrugResponseResult) entropy measurements
/// 2. Computing Pearson correlation between |ΔS_config| and |ΔH_hrv|
/// 3. Computing a proper p-value via the t-distribution
/// 4. Reporting R², mean absolute error, and statistical significance
///
/// A significant positive correlation (p < 0.05, n ≥ 5) constitutes independent
/// validation that the entropy-collapse framework generalizes from molecular torsional
/// angles to physiological cardiac intervals.
public struct CrossDomainValidator: Sendable {

    /// A paired observation: one substance's in-silico and in-vivo entropy deltas.
    public struct PairedObservation: Sendable {
        /// Substance identifier.
        public let substanceId: String

        /// ΔS_config from FlexAID∆S (bits, typically negative for binding).
        public let deltaSConfig: Double

        /// ΔH_hrv from DrugResponseAnalyzer (bits).
        public let deltaHHRV: Double

        /// -TΔS at 298K (kcal/mol, positive = entropy penalty).
        public let entropyPenaltyKcal: Double

        /// Effect size from in-vivo analysis (|ΔH| / baseline_H).
        public let inVivoEffectSize: Double

        public init(
            substanceId: String,
            deltaSConfig: Double,
            deltaHHRV: Double,
            entropyPenaltyKcal: Double,
            inVivoEffectSize: Double
        ) {
            self.substanceId = substanceId
            self.deltaSConfig = deltaSConfig
            self.deltaHHRV = deltaHHRV
            self.entropyPenaltyKcal = entropyPenaltyKcal
            self.inVivoEffectSize = inVivoEffectSize
        }
    }

    /// Result of cross-domain validation.
    public struct ValidationResult: Sendable {
        /// Paired observations used in the analysis.
        public let observations: [PairedObservation]

        /// Pearson r between |ΔS_config| and |ΔH_hrv|.
        public let pearsonR: Double

        /// Two-tailed p-value for the Pearson correlation.
        /// Computed via t-distribution: t = r × √(n-2) / √(1-r²).
        public let pValue: Double

        /// R-squared (coefficient of determination).
        /// Fraction of in-vivo variance explained by in-silico entropy.
        public var rSquared: Double { pearsonR * pearsonR }

        /// Number of substances in the analysis.
        public var n: Int { observations.count }

        /// p-value threshold for statistical significance (from AnalysisConfiguration).
        public let significanceLevel: Double

        /// Minimum paired observations required for significance (from AnalysisConfiguration).
        public let minPairs: Int

        /// Whether the correlation is statistically significant.
        /// Uses proper p-value testing against the configured significance level.
        public var isSignificant: Bool {
            pValue < significanceLevel && n >= minPairs
        }

        /// Mean absolute prediction error (bits).
        /// How well |ΔS_config| predicts |ΔH_hrv| via linear regression.
        public let meanAbsError: Double

        /// Linear regression slope: |ΔH_hrv| ≈ slope × |ΔS_config| + intercept.
        /// Represents the scaling factor between molecular and physiological entropy.
        public let regressionSlope: Double

        /// Linear regression intercept.
        public let regressionIntercept: Double

        /// Bilingual summary.
        public var summary: LocalizedString {
            let rText = String(format: "%.3f", pearsonR)
            let r2Text = String(format: "%.3f", rSquared)
            let pText = String(format: "%.4f", pValue)
            let maeText = String(format: "%.2f", meanAbsError)
            let sigText = isSignificant ? "significant" : "not significant"
            let sigFr = isSignificant ? "significative" : "non significative"

            let sigEs = isSignificant ? "significativa" : "no significativa"
            let sigJa = isSignificant ? "有意" : "非有意"
            let sigZh = isSignificant ? "显著" : "不显著"
            let sigKo = isSignificant ? "유의미함" : "유의미하지 않음"
            let sigRu = isSignificant ? "значимая" : "незначимая"
            let sigDe = isSignificant ? "signifikant" : "nicht signifikant"
            let sigAr = isSignificant ? "ذات دلالة إحصائية" : "غير ذات دلالة إحصائية"

            return LocalizedString(
                en: "Cross-domain validation (n=\(n)): r = \(rText), R² = \(r2Text), p = \(pText), MAE = \(maeText) bits. Correlation is \(sigText).",
                fr: "Validation interdomaines (n=\(n)) : r = \(rText), R² = \(r2Text), p = \(pText), MAE = \(maeText) bits. Corrélation \(sigFr).",
                es: "Validación interdominio (n=\(n)): r = \(rText), R² = \(r2Text), p = \(pText), MAE = \(maeText) bits. Correlación \(sigEs).",
                ja: "クロスドメイン検証（n=\(n)）：r = \(rText)、R² = \(r2Text)、p = \(pText)、MAE = \(maeText) ビット。相関は\(sigJa)。",
                zh: "跨域验证（n=\(n)）：r = \(rText)，R² = \(r2Text)，p = \(pText)，MAE = \(maeText) 比特。相关性\(sigZh)。",
                ko: "교차 도메인 검증 (n=\(n)): r = \(rText), R² = \(r2Text), p = \(pText), MAE = \(maeText) 비트. 상관관계 \(sigKo).",
                ru: "Кросс-доменная валидация (n=\(n)): r = \(rText), R² = \(r2Text), p = \(pText), MAE = \(maeText) бит. Корреляция \(sigRu).",
                de: "Domänenübergreifende Validierung (n=\(n)): r = \(rText), R² = \(r2Text), p = \(pText), MAE = \(maeText) Bits. Korrelation \(sigDe).",
                ar: "التحقق عبر المجالات (n=\(n)): r = \(rText)، R² = \(r2Text)، p = \(pText)، MAE = \(maeText) بت. الارتباط \(sigAr)."
            )
        }

        public init(
            observations: [PairedObservation],
            pearsonR: Double,
            pValue: Double,
            meanAbsError: Double,
            regressionSlope: Double,
            regressionIntercept: Double,
            significanceLevel: Double = AnalysisConfiguration.default.crossDomainSignificanceLevel,
            minPairs: Int = AnalysisConfiguration.default.crossDomainMinPairs
        ) {
            self.observations = observations
            self.pearsonR = pearsonR
            self.pValue = pValue
            self.meanAbsError = meanAbsError
            self.regressionSlope = regressionSlope
            self.regressionIntercept = regressionIntercept
            self.significanceLevel = significanceLevel
            self.minPairs = minPairs
        }
    }

    private let dockingAnalyzer: FlexAIDdSAnalyzer
    private let configuration: AnalysisConfiguration

    public init(dockingAnalyzer: FlexAIDdSAnalyzer = FlexAIDdSAnalyzer()) {
        self.dockingAnalyzer = dockingAnalyzer
        self.configuration = .default
    }

    public init(configuration: AnalysisConfiguration, dockingAnalyzer: FlexAIDdSAnalyzer = FlexAIDdSAnalyzer()) {
        self.dockingAnalyzer = dockingAnalyzer
        self.configuration = configuration
    }

    // MARK: - Validation from Raw Results

    /// Validate correlation between FlexAID∆S results and DrugResponse results.
    ///
    /// Pairs results by substanceId and computes Pearson correlation
    /// between |ΔS_config| and |ΔH_hrv|.
    ///
    /// - Parameters:
    ///   - dockingResults: In-silico FlexAID∆S results.
    ///   - drugResponseResults: In-vivo DrugResponseAnalyzer results.
    /// - Returns: ValidationResult, or nil if fewer than `crossDomainMinPairs` paired substances.
    public func validate(
        dockingResults: [FlexAIDdSResult],
        drugResponseResults: [DrugResponseResult]
    ) -> ValidationResult? {
        var dockingBySubstance: [String: FlexAIDdSResult] = [:]
        for r in dockingResults {
            dockingBySubstance[r.substanceId] = r
        }

        var responseBySubstance: [String: DrugResponseResult] = [:]
        for r in drugResponseResults {
            responseBySubstance[r.doseEvent.medicationId] = r
        }

        var pairs: [PairedObservation] = []
        for (substanceId, docking) in dockingBySubstance {
            guard let response = responseBySubstance[substanceId] else { continue }
            pairs.append(PairedObservation(
                substanceId: substanceId,
                deltaSConfig: docking.totalDeltaSConfig,
                deltaHHRV: response.peakDeltaH,
                entropyPenaltyKcal: dockingAnalyzer.entropyPenaltyKcal(
                    deltaSBits: docking.totalDeltaSConfig
                ),
                inVivoEffectSize: response.effectSize
            ))
        }

        return buildResult(from: pairs)
    }

    // MARK: - Validation from Known Profiles

    /// Validate using BindingEntropyProfile known values against DrugResponseResults.
    ///
    /// Useful when actual FlexAID docking has not been run but published/reference
    /// ΔS_config values exist for the substances.
    ///
    /// - Parameter drugResponseResults: In-vivo DrugResponseAnalyzer results.
    /// - Returns: ValidationResult, or nil if fewer than `crossDomainMinPairs` paired substances.
    public func validateFromProfiles(
        drugResponseResults: [DrugResponseResult]
    ) -> ValidationResult? {
        var pairs: [PairedObservation] = []

        for response in drugResponseResults {
            guard let bindingProfile = BindingEntropyProfile.profile(
                for: response.doseEvent.medicationId
            ) else { continue }

            pairs.append(PairedObservation(
                substanceId: response.doseEvent.medicationId,
                deltaSConfig: bindingProfile.expectedDeltaSBits,
                deltaHHRV: response.peakDeltaH,
                entropyPenaltyKcal: bindingProfile.expectedEntropyPenaltyKcal,
                inVivoEffectSize: response.effectSize
            ))
        }

        return buildResult(from: pairs)
    }

    // MARK: - Hybrid Validation

    /// Validate using a mix of actual docking results and known profiles.
    ///
    /// Prefers actual docking results when available; falls back to
    /// BindingEntropyProfile for substances without docking data.
    public func validateHybrid(
        dockingResults: [FlexAIDdSResult],
        drugResponseResults: [DrugResponseResult]
    ) -> ValidationResult? {
        var dockingBySubstance: [String: Double] = [:]
        var penaltyBySubstance: [String: Double] = [:]

        // Actual docking results take priority
        for r in dockingResults {
            dockingBySubstance[r.substanceId] = r.totalDeltaSConfig
            penaltyBySubstance[r.substanceId] = dockingAnalyzer.entropyPenaltyKcal(
                deltaSBits: r.totalDeltaSConfig
            )
        }

        // Fill in from known profiles where docking hasn't been run
        for profile in BindingEntropyProfile.knownProfiles {
            if dockingBySubstance[profile.substanceId] == nil {
                dockingBySubstance[profile.substanceId] = profile.expectedDeltaSBits
                penaltyBySubstance[profile.substanceId] = profile.expectedEntropyPenaltyKcal
            }
        }

        var pairs: [PairedObservation] = []
        for response in drugResponseResults {
            let id = response.doseEvent.medicationId
            guard let deltaS = dockingBySubstance[id],
                  let penalty = penaltyBySubstance[id] else { continue }

            pairs.append(PairedObservation(
                substanceId: id,
                deltaSConfig: deltaS,
                deltaHHRV: response.peakDeltaH,
                entropyPenaltyKcal: penalty,
                inVivoEffectSize: response.effectSize
            ))
        }

        return buildResult(from: pairs)
    }

    // MARK: - Three-Way Validation

    /// A three-way paired observation: computational (FlexAID∆S) vs ITC-measured (SCORPIO)
    /// vs in-vivo (NATURaL HRV) entropy for the same substance.
    public struct ThreeWayObservation: Sendable {
        /// Substance identifier.
        public let substanceId: String

        /// ΔS_config from FlexAID∆S or BindingEntropyProfile (bits).
        public let flexAIDDeltaSBits: Double

        /// -TΔS from SCORPIO ITC (kcal/mol).
        public let scorpioMinusTDeltaSKcal: Double

        /// ΔH_hrv from NATURaL DrugResponseAnalyzer (bits).
        public let naturalDeltaHHRV: Double

        public init(
            substanceId: String,
            flexAIDDeltaSBits: Double,
            scorpioMinusTDeltaSKcal: Double,
            naturalDeltaHHRV: Double
        ) {
            self.substanceId = substanceId
            self.flexAIDDeltaSBits = flexAIDDeltaSBits
            self.scorpioMinusTDeltaSKcal = scorpioMinusTDeltaSKcal
            self.naturalDeltaHHRV = naturalDeltaHHRV
        }
    }

    /// Result of three-way validation with pairwise Pearson correlations and p-values.
    public struct ThreeWayValidationResult: Sendable {
        /// Three-way paired observations.
        public let observations: [ThreeWayObservation]

        /// Pearson r: FlexAID∆S (computational) vs SCORPIO (ITC).
        public let flexAIDvsScorpio: Double

        /// Two-tailed p-value for FlexAID vs SCORPIO (t-distribution / incomplete beta).
        public let flexAIDvsScorpioPValue: Double

        /// Pearson r: FlexAID∆S (computational) vs NATURaL (HRV).
        public let flexAIDvsNatural: Double

        /// Two-tailed p-value for FlexAID vs NATURaL.
        public let flexAIDvsNaturalPValue: Double

        /// Pearson r: SCORPIO (ITC) vs NATURaL (HRV).
        public let scorpioVsNatural: Double

        /// Two-tailed p-value for SCORPIO vs NATURaL.
        public let scorpioVsNaturalPValue: Double

        /// Number of substances in the three-way analysis.
        public var n: Int { observations.count }

        /// Bilingual summary of pairwise correlations and p-values.
        public var summary: LocalizedString {
            let fs = String(format: "%.3f", flexAIDvsScorpio)
            let fn = String(format: "%.3f", flexAIDvsNatural)
            let sn = String(format: "%.3f", scorpioVsNatural)
            let pfs = String(format: "%.4f", flexAIDvsScorpioPValue)
            let pfn = String(format: "%.4f", flexAIDvsNaturalPValue)
            let psn = String(format: "%.4f", scorpioVsNaturalPValue)

            return LocalizedString(
                en: "Three-way validation (n=\(n)): FlexAID↔SCORPIO r=\(fs) (p=\(pfs)), FlexAID↔NATURaL r=\(fn) (p=\(pfn)), SCORPIO↔NATURaL r=\(sn) (p=\(psn)).",
                fr: "Validation tripartite (n=\(n)) : FlexAID↔SCORPIO r=\(fs) (p=\(pfs)), FlexAID↔NATURaL r=\(fn) (p=\(pfn)), SCORPIO↔NATURaL r=\(sn) (p=\(psn)).",
                es: "Validación tripartita (n=\(n)): FlexAID↔SCORPIO r=\(fs) (p=\(pfs)), FlexAID↔NATURaL r=\(fn) (p=\(pfn)), SCORPIO↔NATURaL r=\(sn) (p=\(psn)).",
                ja: "三者間検証（n=\(n)）：FlexAID↔SCORPIO r=\(fs) (p=\(pfs))、FlexAID↔NATURaL r=\(fn) (p=\(pfn))、SCORPIO↔NATURaL r=\(sn) (p=\(psn))。",
                zh: "三方验证（n=\(n)）：FlexAID↔SCORPIO r=\(fs) (p=\(pfs))，FlexAID↔NATURaL r=\(fn) (p=\(pfn))，SCORPIO↔NATURaL r=\(sn) (p=\(psn))。",
                ko: "삼자 검증 (n=\(n)): FlexAID↔SCORPIO r=\(fs) (p=\(pfs)), FlexAID↔NATURaL r=\(fn) (p=\(pfn)), SCORPIO↔NATURaL r=\(sn) (p=\(psn)).",
                ru: "Трёхсторонняя валидация (n=\(n)): FlexAID↔SCORPIO r=\(fs) (p=\(pfs)), FlexAID↔NATURaL r=\(fn) (p=\(pfn)), SCORPIO↔NATURaL r=\(sn) (p=\(psn)).",
                de: "Drei-Wege-Validierung (n=\(n)): FlexAID↔SCORPIO r=\(fs) (p=\(pfs)), FlexAID↔NATURaL r=\(fn) (p=\(pfn)), SCORPIO↔NATURaL r=\(sn) (p=\(psn)).",
                ar: "التحقق الثلاثي (n=\(n)): FlexAID↔SCORPIO r=\(fs) (p=\(pfs))، FlexAID↔NATURaL r=\(fn) (p=\(pfn))، SCORPIO↔NATURaL r=\(sn) (p=\(psn))."
            )
        }

        public init(
            observations: [ThreeWayObservation],
            flexAIDvsScorpio: Double,
            flexAIDvsScorpioPValue: Double,
            flexAIDvsNatural: Double,
            flexAIDvsNaturalPValue: Double,
            scorpioVsNatural: Double,
            scorpioVsNaturalPValue: Double
        ) {
            self.observations = observations
            self.flexAIDvsScorpio = flexAIDvsScorpio
            self.flexAIDvsScorpioPValue = flexAIDvsScorpioPValue
            self.flexAIDvsNatural = flexAIDvsNatural
            self.flexAIDvsNaturalPValue = flexAIDvsNaturalPValue
            self.scorpioVsNatural = scorpioVsNatural
            self.scorpioVsNaturalPValue = scorpioVsNaturalPValue
        }
    }

    /// Validate three-way correlation: FlexAID∆S (computational) vs SCORPIO ITC (measured)
    /// vs NATURaL HRV (in-vivo).
    ///
    /// Only includes substances that have data in all three domains:
    /// 1. FlexAID∆S / BindingEntropyProfile for computational ΔS
    /// 2. ThermodynamicBindingProfile with ITC decomposition for SCORPIO -TΔS
    /// 3. DrugResponseResult for in-vivo ΔH_hrv
    ///
    /// - Parameters:
    ///   - dockingResults: In-silico FlexAID∆S results (optional, falls back to BindingEntropyProfile).
    ///   - drugResponseResults: In-vivo DrugResponseAnalyzer results.
    /// - Returns: ThreeWayValidationResult, or nil if fewer than 3 substances have all three.
    public func validateThreeWay(
        dockingResults: [FlexAIDdSResult] = [],
        drugResponseResults: [DrugResponseResult]
    ) -> ThreeWayValidationResult? {
        // Build FlexAID data map (prefer actual docking, fall back to profiles)
        var flexAIDBySubstance: [String: Double] = [:]
        for r in dockingResults {
            flexAIDBySubstance[r.substanceId] = r.totalDeltaSConfig
        }
        for profile in BindingEntropyProfile.knownProfiles {
            if flexAIDBySubstance[profile.substanceId] == nil {
                flexAIDBySubstance[profile.substanceId] = profile.expectedDeltaSBits
            }
        }

        // Build SCORPIO ITC data map (only primary targets with ITC decomposition)
        var scorpioBySubstance: [String: Double] = [:]
        for profile in ThermodynamicBindingProfile.knownProfiles {
            guard profile.isPrimaryTarget,
                  let thermo = profile.thermodynamics else { continue }
            scorpioBySubstance[profile.substanceId] = thermo.minusTDeltaSKcal
        }

        // Build NATURaL HRV data map
        var hrvBySubstance: [String: Double] = [:]
        for r in drugResponseResults {
            hrvBySubstance[r.doseEvent.medicationId] = r.peakDeltaH
        }

        // Find substances with all three data sources
        var observations: [ThreeWayObservation] = []
        for (substanceId, flexAID) in flexAIDBySubstance {
            guard let scorpio = scorpioBySubstance[substanceId],
                  let hrv = hrvBySubstance[substanceId] else { continue }
            observations.append(ThreeWayObservation(
                substanceId: substanceId,
                flexAIDDeltaSBits: flexAID,
                scorpioMinusTDeltaSKcal: scorpio,
                naturalDeltaHHRV: hrv
            ))
        }

        guard observations.count >= configuration.crossDomainMinPairs else { return nil }

        let flexAIDValues = observations.map { abs($0.flexAIDDeltaSBits) }
        let scorpioValues = observations.map { abs($0.scorpioMinusTDeltaSKcal) }
        let naturalValues = observations.map { abs($0.naturalDeltaHHRV) }
        let n = observations.count

        let rFS = pearsonCorrelation(flexAIDValues, scorpioValues)
        let rFN = pearsonCorrelation(flexAIDValues, naturalValues)
        let rSN = pearsonCorrelation(scorpioValues, naturalValues)

        return ThreeWayValidationResult(
            observations: observations,
            flexAIDvsScorpio: rFS,
            flexAIDvsScorpioPValue: Self.computePValue(r: rFS, n: n),
            flexAIDvsNatural: rFN,
            flexAIDvsNaturalPValue: Self.computePValue(r: rFN, n: n),
            scorpioVsNatural: rSN,
            scorpioVsNaturalPValue: Self.computePValue(r: rSN, n: n)
        )
    }

    // MARK: - Private

    private func buildResult(from pairs: [PairedObservation]) -> ValidationResult? {
        guard pairs.count >= configuration.crossDomainMinPairs else { return nil }

        // Filter out non-finite values
        let cleanPairs = pairs.filter {
            $0.deltaSConfig.isFinite && $0.deltaHHRV.isFinite
        }
        guard cleanPairs.count >= configuration.crossDomainMinPairs else { return nil }

        let x = cleanPairs.map { abs($0.deltaSConfig) }
        let y = cleanPairs.map { abs($0.deltaHHRV) }

        let r = pearsonCorrelation(x, y)

        let regression = linearRegression(x: x, y: y)

        let p = Self.computePValue(r: r, n: cleanPairs.count)

        return ValidationResult(
            observations: cleanPairs,
            pearsonR: r,
            pValue: p,
            meanAbsError: regression.mae,
            regressionSlope: regression.slope,
            regressionIntercept: regression.intercept,
            significanceLevel: configuration.crossDomainSignificanceLevel,
            minPairs: configuration.crossDomainMinPairs
        )
    }

    // MARK: - Statistical Significance

    /// Compute two-tailed p-value for Pearson r using the t-distribution.
    ///
    /// Delegates to C++ accelerator when available, with Swift fallback.
    static func computePValue(r: Double, n: Int) -> Double {
        #if BONHOMME_ACCEL
        if let pval = AccelCorrelation.pearsonPValue(r: r, n: n) {
            return pval
        }
        #endif

        guard n > 2 else { return 1.0 }
        let absR = abs(r)
        guard absR < 1.0 else { return absR >= 1.0 ? 0.0 : 1.0 }

        let df = Double(n - 2)
        let t = absR * sqrt(df) / sqrt(1.0 - absR * absR)

        let x = df / (df + t * t)
        let ibeta = regularizedIncompleteBeta(x: x, a: df / 2.0, b: 0.5)
        return ibeta
    }

    /// Regularized incomplete beta function I_x(a, b) — delegates to C++ when available.
    private static func regularizedIncompleteBeta(x: Double, a: Double, b: Double) -> Double {
        #if BONHOMME_ACCEL
        if let result = AccelCorrelation.regularizedIncompleteBeta(x: x, a: a, b: b) {
            return result
        }
        #endif

        guard x > 0 else { return 0.0 }
        guard x < 1 else { return 1.0 }

        if x > (a + 1.0) / (a + b + 2.0) {
            return 1.0 - regularizedIncompleteBeta(x: 1.0 - x, a: b, b: a)
        }

        let lnPrefactor = a * log(x) + b * log(1.0 - x) - log(a) - lnBeta(a: a, b: b)
        let prefactor = exp(lnPrefactor)

        let maxIterations = 200
        let epsilon = 1.0e-10
        let tiny = 1.0e-30

        var c = 1.0
        var d = 1.0 / max(tiny, 1.0 - (a + b) * x / (a + 1.0))
        var h = d

        for m in 1...maxIterations {
            let dm = Double(m)

            var numerator = dm * (b - dm) * x / ((a + 2.0 * dm - 1.0) * (a + 2.0 * dm))
            d = 1.0 / max(tiny, 1.0 + numerator * d)
            c = max(tiny, 1.0 + numerator / c)
            h *= d * c

            numerator = -(a + dm) * (a + b + dm) * x / ((a + 2.0 * dm) * (a + 2.0 * dm + 1.0))
            d = 1.0 / max(tiny, 1.0 + numerator * d)
            c = max(tiny, 1.0 + numerator / c)
            let delta = d * c
            h *= delta

            if abs(delta - 1.0) < epsilon {
                break
            }
        }

        return prefactor * h
    }

    /// Log of the beta function: ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b).
    private static func lnBeta(a: Double, b: Double) -> Double {
        return lgamma(a) + lgamma(b) - lgamma(a + b)
    }
}
