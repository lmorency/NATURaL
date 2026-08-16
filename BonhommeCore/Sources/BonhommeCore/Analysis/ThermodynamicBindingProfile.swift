import Foundation

// MARK: - Data Source

/// Source of thermodynamic binding data.
public enum ThermodynamicSource: String, Codable, Sendable, CaseIterable {
    /// BindingDB public database (Ki/Kd/IC50/EC50).
    case bindingDB
    /// SCORPIO ITC-only database (direct ΔG/ΔH/ΔS/ΔCp measurement).
    case scorpioITC
    /// FlexAID∆S computational docking entropy.
    case flexAIDdS
    /// NIMH Psychoactive Drug Screening Program Ki values.
    case pdspKi
    /// ChEMBL curated bioactivity data.
    case chembl
    /// Published literature values.
    case literature
}

// MARK: - Assay Conditions

/// Experimental conditions under which binding was measured.
public struct AssayConditions: Sendable, Codable, Equatable {
    /// Temperature in Kelvin (default 298K = 25°C).
    public let temperatureK: Double

    /// pH of the assay buffer (default 7.4 = physiological).
    public let pH: Double

    /// Assay type description (e.g., "radioligand displacement", "ITC").
    public let assayType: String?

    public init(temperatureK: Double = 298.0, pH: Double = 7.4, assayType: String? = nil) {
        self.temperatureK = temperatureK
        self.pH = pH
        self.assayType = assayType
    }

    /// Standard biochemical conditions (298K, pH 7.4).
    public static let standard = AssayConditions()

    /// Physiological conditions (310K, pH 7.4).
    public static let physiological = AssayConditions(temperatureK: 310.0, pH: 7.4)
}

// MARK: - Affinity Measurement

/// Binding affinity data from various assay types.
///
/// Preference hierarchy for best affinity estimate: Ki > Kd > IC50/2 > EC50.
/// Ki and Kd are thermodynamically rigorous; IC50 is assay-dependent.
public struct AffinityMeasurement: Sendable, Codable {
    /// Inhibition constant (nM). Gold standard from competition binding.
    public let kiNM: Double?

    /// Dissociation constant (nM). From direct binding or ITC.
    public let kdNM: Double?

    /// Half-maximal inhibitory concentration (nM). Assay-dependent.
    public let ic50NM: Double?

    /// Half-maximal effective concentration (nM). Functional assay.
    public let ec50NM: Double?

    /// Assay conditions under which affinity was measured.
    public let conditions: AssayConditions

    public init(
        kiNM: Double? = nil,
        kdNM: Double? = nil,
        ic50NM: Double? = nil,
        ec50NM: Double? = nil,
        conditions: AssayConditions = .standard
    ) {
        self.kiNM = kiNM
        self.kdNM = kdNM
        self.ic50NM = ic50NM
        self.ec50NM = ec50NM
        self.conditions = conditions
    }

    /// Best available affinity estimate in nM.
    /// Preference: Ki > Kd > IC50/2 (Cheng-Prusoff approximation) > EC50.
    public var bestAffinityNM: Double? {
        if let ki = kiNM { return ki }
        if let kd = kdNM { return kd }
        if let ic50 = ic50NM { return ic50 / 2.0 }
        if let ec50 = ec50NM { return ec50 }
        return nil
    }

    /// Computed ΔG from best affinity at assay temperature.
    /// ΔG = RT ln(Kd) where Kd is in molar units.
    /// Returns kcal/mol (negative = favorable binding).
    public var computedDeltaGKcal: Double? {
        guard let affinityNM = bestAffinityNM, affinityNM > 0 else { return nil }
        let affinityM = affinityNM * 1e-9
        return ThermodynamicConstants.R * conditions.temperatureK * log(affinityM)
    }
}

// MARK: - Thermodynamic Decomposition

/// Full Gibbs free energy decomposition from ITC (isothermal titration calorimetry).
///
/// ΔG = ΔH + (-TΔS)
/// - ΔG: total binding free energy (always negative for spontaneous binding)
/// - ΔH: enthalpy change (hydrogen bonds, van der Waals, electrostatics)
/// - -TΔS: entropy contribution (conformational, solvation, rotational/translational)
public struct ThermodynamicDecomposition: Sendable, Codable {
    /// Total Gibbs free energy of binding (kcal/mol, negative = favorable).
    public let deltaGKcal: Double

    /// Enthalpy of binding (kcal/mol, negative = exothermic).
    public let deltaHKcal: Double

    /// Entropy term as -TΔS (kcal/mol, negative = entropy-favorable).
    public let minusTDeltaSKcal: Double

    /// Heat capacity change (cal/mol·K), if measured. Indicates hydrophobic burial.
    public let deltaCpCalPerMolK: Double?

    /// Temperature at which measurements were taken (K).
    public let temperatureK: Double

    public init(
        deltaGKcal: Double,
        deltaHKcal: Double,
        minusTDeltaSKcal: Double,
        deltaCpCalPerMolK: Double? = nil,
        temperatureK: Double = 298.0
    ) {
        self.deltaGKcal = deltaGKcal
        self.deltaHKcal = deltaHKcal
        self.minusTDeltaSKcal = minusTDeltaSKcal
        self.deltaCpCalPerMolK = deltaCpCalPerMolK
        self.temperatureK = temperatureK
    }

    /// Whether binding is entropy-driven (|-TΔS| > |ΔH|).
    public var entropyDriven: Bool {
        abs(minusTDeltaSKcal) > abs(deltaHKcal)
    }

    /// Whether binding is enthalpy-driven (|ΔH| > |-TΔS|).
    public var enthalpyDriven: Bool {
        abs(deltaHKcal) > abs(minusTDeltaSKcal)
    }

    /// Whether the decomposition is thermodynamically self-consistent.
    /// |ΔG - (ΔH + (-TΔS))| should be < 0.5 kcal/mol.
    public var isThermodynamicallyConsistent: Bool {
        abs(deltaGKcal - (deltaHKcal + minusTDeltaSKcal)) < 0.5
    }

    /// Convert -TΔS to Shannon entropy bits for FlexAID∆S comparison.
    /// -TΔS (kcal/mol) → ΔS (bits) via: ΔS_bits = -(-TΔS) / (T × R × ln2)
    public var deltaSBits: Double {
        guard temperatureK > 0 else { return 0 }
        return -minusTDeltaSKcal / (temperatureK * ThermodynamicConstants.R * log(2.0))
    }
}

// MARK: - Stereochemical Note

/// Documents stereochemical effects on binding affinity.
public struct StereochemicalNote: Sendable, Codable {
    /// Enantiomer designation (e.g., "S(+)", "R(-)", "d-", "l-").
    public let enantiomer: String

    /// Eutomer/distomer affinity ratio, if known.
    public let affinityRatio: Double?

    /// Descriptive note.
    public let note: LocalizedString

    public init(enantiomer: String, affinityRatio: Double? = nil, note: LocalizedString) {
        self.enantiomer = enantiomer
        self.affinityRatio = affinityRatio
        self.note = note
    }
}

// MARK: - Prodrug Relationship

/// Tracks prodrug → active metabolite conversion for pharmacovigilance.
public struct ProdrugRelationship: Sendable, Codable {
    /// Substance ID of the prodrug (administered form).
    public let prodrugId: String

    /// Substance ID of the active metabolite (pharmacologically active form).
    public let activeMetaboliteId: String

    /// Enzyme responsible for activation.
    public let activatingEnzyme: String

    /// Approximate conversion half-life in minutes.
    public let conversionHalfLifeMinutes: Double?

    /// Descriptive note.
    public let note: LocalizedString

    public init(
        prodrugId: String,
        activeMetaboliteId: String,
        activatingEnzyme: String,
        conversionHalfLifeMinutes: Double? = nil,
        note: LocalizedString
    ) {
        self.prodrugId = prodrugId
        self.activeMetaboliteId = activeMetaboliteId
        self.activatingEnzyme = activatingEnzyme
        self.conversionHalfLifeMinutes = conversionHalfLifeMinutes
        self.note = note
    }
}

// MARK: - Thermodynamic Binding Profile

/// Complete thermodynamic binding profile for a substance-target pair.
///
/// Combines affinity data (Ki/Kd from BindingDB/PDSP) with optional full
/// Gibbs decomposition (ΔG/ΔH/-TΔS from SCORPIO ITC). Used for:
/// - Data-driven PokeDrug stat derivation (Attack from Ki, Sp.Atk from selectivity)
/// - Enthalpy vs entropy binding signature classification
/// - Three-way validation: FlexAID∆S (computational) vs SCORPIO (ITC) vs NATURaL (HRV)
/// - Pharmacovigilance: identifying prodrug relationships and stereochemical differences
public struct ThermodynamicBindingProfile: Sendable {
    /// Substance identifier (matches PokeDrugSpecies.substanceId).
    public let substanceId: String

    /// Target identifier (e.g., "5-HT2A", "MOR", "CB1").
    public let targetId: String

    /// Target display name (bilingual).
    public let targetName: LocalizedString

    /// Binding affinity measurement.
    public let affinity: AffinityMeasurement

    /// Full ITC thermodynamic decomposition (nil when only affinity data available).
    public let thermodynamics: ThermodynamicDecomposition?

    /// Stereochemical note (nil for achiral compounds or racemates without data).
    public let stereochemistry: StereochemicalNote?

    /// Data source.
    public let source: ThermodynamicSource

    /// Literature reference or database ID.
    public let reference: String

    /// Whether this is the primary pharmacological target.
    public let isPrimaryTarget: Bool

    public init(
        substanceId: String,
        targetId: String,
        targetName: LocalizedString,
        affinity: AffinityMeasurement,
        thermodynamics: ThermodynamicDecomposition? = nil,
        stereochemistry: StereochemicalNote? = nil,
        source: ThermodynamicSource,
        reference: String,
        isPrimaryTarget: Bool
    ) {
        self.substanceId = substanceId
        self.targetId = targetId
        self.targetName = targetName
        self.affinity = affinity
        self.thermodynamics = thermodynamics
        self.stereochemistry = stereochemistry
        self.source = source
        self.reference = reference
        self.isPrimaryTarget = isPrimaryTarget
    }
}

// MARK: - Static Catalog

extension ThermodynamicBindingProfile {

    /// All known thermodynamic binding profiles (~48 substance-target pairs).
    ///
    /// Ki values sourced from PDSP Ki Database and BindingDB.
    /// ITC decomposition from SCORPIO where available.
    /// References: Roth et al., PDSP; BindingDB; Freire 2009; Klebe 2015.
    public static let knownProfiles: [ThermodynamicBindingProfile] = [

        // MARK: - LSD (3 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "lsd",
            targetId: "5-HT2A",
            targetName: LocalizedString(
                en: "Serotonin 5-HT2A receptor",
                fr: "Récepteur sérotoninergique 5-HT2A",
                es: "Receptor serotoninérgico 5-HT2A",
                ja: "セロトニン5-HT2A受容体",
                zh: "血清素5-HT2A受体",
                ko: "세로토닌 5-HT2A 수용체",
                ru: "Серотониновый рецептор 5-HT2A",
                de: "Serotonin-5-HT2A-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT2A"
            ),
            affinity: AffinityMeasurement(kiNM: 3.5),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -11.5, deltaHKcal: -7.2, minusTDeltaSKcal: -4.3
            ),
            source: .pdspKi,
            reference: "Roth 2002; Wacker 2017 Cell 168:377",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "lsd",
            targetId: "D2",
            targetName: LocalizedString(
                en: "Dopamine D2 receptor",
                fr: "Récepteur dopaminergique D2",
                es: "Receptor dopaminérgico D2",
                ja: "ドパミンD2受容体",
                zh: "多巴胺D2受体",
                ko: "도파민 D2 수용체",
                ru: "Дофаминовый рецептор D2",
                de: "Dopamin-D2-Rezeptor",
                ar: "مستقبل الدوبامين D2"
            ),
            affinity: AffinityMeasurement(kiNM: 25),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "lsd",
            targetId: "D1",
            targetName: LocalizedString(
                en: "Dopamine D1 receptor",
                fr: "Récepteur dopaminergique D1",
                es: "Receptor dopaminérgico D1",
                ja: "ドパミンD1受容体",
                zh: "多巴胺D1受体",
                ko: "도파민 D1 수용체",
                ru: "Дофаминовый рецептор D1",
                de: "Dopamin-D1-Rezeptor",
                ar: "مستقبل الدوبامين D1"
            ),
            affinity: AffinityMeasurement(kiNM: 52),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Psilocybin / Psilocin (3 targets)

        ThermodynamicBindingProfile(
            substanceId: "psilocybin",
            targetId: "5-HT2A",
            targetName: LocalizedString(
                en: "Serotonin 5-HT2A receptor",
                fr: "Récepteur sérotoninergique 5-HT2A",
                es: "Receptor serotoninérgico 5-HT2A",
                ja: "セロトニン5-HT2A受容体",
                zh: "血清素5-HT2A受体",
                ko: "세로토닌 5-HT2A 수용체",
                ru: "Серотониновый рецептор 5-HT2A",
                de: "Serotonin-5-HT2A-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT2A"
            ),
            affinity: AffinityMeasurement(kiNM: 107),
            source: .pdspKi,
            reference: "PDSP; values for active metabolite psilocin",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "psilocybin",
            targetId: "5-HT2B",
            targetName: LocalizedString(
                en: "Serotonin 5-HT2B receptor",
                fr: "Récepteur sérotoninergique 5-HT2B",
                es: "Receptor serotoninérgico 5-HT2B",
                ja: "セロトニン5-HT2B受容体",
                zh: "血清素5-HT2B受体",
                ko: "세로토닌 5-HT2B 수용체",
                ru: "Серотониновый рецептор 5-HT2B",
                de: "Serotonin-5-HT2B-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT2B"
            ),
            affinity: AffinityMeasurement(kiNM: 4.6),
            source: .pdspKi,
            reference: "PDSP Ki Database; psilocin values",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "psilocybin",
            targetId: "5-HT1A",
            targetName: LocalizedString(
                en: "Serotonin 5-HT1A receptor",
                fr: "Récepteur sérotoninergique 5-HT1A",
                es: "Receptor serotoninérgico 5-HT1A",
                ja: "セロトニン5-HT1A受容体",
                zh: "血清素5-HT1A受体",
                ko: "세로토닌 5-HT1A 수용체",
                ru: "Серотониновый рецептор 5-HT1A",
                de: "Serotonin-5-HT1A-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT1A"
            ),
            affinity: AffinityMeasurement(kiNM: 190),
            source: .pdspKi,
            reference: "PDSP Ki Database; psilocin values",
            isPrimaryTarget: false
        ),

        // MARK: - DMT (2 targets)

        ThermodynamicBindingProfile(
            substanceId: "dmt",
            targetId: "5-HT2A",
            targetName: LocalizedString(
                en: "Serotonin 5-HT2A receptor",
                fr: "Récepteur sérotoninergique 5-HT2A",
                es: "Receptor serotoninérgico 5-HT2A",
                ja: "セロトニン5-HT2A受容体",
                zh: "血清素5-HT2A受体",
                ko: "세로토닌 5-HT2A 수용체",
                ru: "Серотониновый рецептор 5-HT2A",
                de: "Serotonin-5-HT2A-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT2A"
            ),
            affinity: AffinityMeasurement(kiNM: 170),
            source: .pdspKi,
            reference: "PDSP Ki Database; Strassman 1994",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "dmt",
            targetId: "sigma-1",
            targetName: LocalizedString(
                en: "Sigma-1 receptor",
                fr: "Récepteur sigma-1",
                es: "Receptor sigma-1",
                ja: "シグマ1受容体",
                zh: "σ1受体",
                ko: "시그마-1 수용체",
                ru: "Сигма-1 рецептор",
                de: "Sigma-1-Rezeptor",
                ar: "مستقبل سيغما-1"
            ),
            affinity: AffinityMeasurement(kiNM: 14000),
            source: .pdspKi,
            reference: "Fontanilla 2009 Science 323:934",
            isPrimaryTarget: false
        ),

        // MARK: - Morphine (3 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "morphine",
            targetId: "MOR",
            targetName: LocalizedString(
                en: "Mu-opioid receptor",
                fr: "Récepteur opioïde mu",
                es: "Receptor opioide mu",
                ja: "μオピオイド受容体",
                zh: "μ阿片受体",
                ko: "뮤 오피오이드 수용체",
                ru: "Мю-опиоидный рецептор",
                de: "μ-Opioidrezeptor",
                ar: "مستقبل الأفيون ميو"
            ),
            affinity: AffinityMeasurement(kiNM: 1.8),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -11.9, deltaHKcal: -8.5, minusTDeltaSKcal: -3.4
            ),
            source: .pdspKi,
            reference: "PDSP; Pert & Snyder 1973; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "morphine",
            targetId: "KOR",
            targetName: LocalizedString(
                en: "Kappa-opioid receptor",
                fr: "Récepteur opioïde kappa",
                es: "Receptor opioide kappa",
                ja: "κオピオイド受容体",
                zh: "κ阿片受体",
                ko: "카파 오피오이드 수용체",
                ru: "Каппа-опиоидный рецептор",
                de: "κ-Opioidrezeptor",
                ar: "مستقبل الأفيون كابا"
            ),
            affinity: AffinityMeasurement(kiNM: 120),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "morphine",
            targetId: "DOR",
            targetName: LocalizedString(
                en: "Delta-opioid receptor",
                fr: "Récepteur opioïde delta",
                es: "Receptor opioide delta",
                ja: "δオピオイド受容体",
                zh: "δ阿片受体",
                ko: "델타 오피오이드 수용체",
                ru: "Дельта-опиоидный рецептор",
                de: "δ-Opioidrezeptor",
                ar: "مستقبل الأفيون دلتا"
            ),
            affinity: AffinityMeasurement(kiNM: 200),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Fentanyl (2 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "fentanyl",
            targetId: "MOR",
            targetName: LocalizedString(
                en: "Mu-opioid receptor",
                fr: "Récepteur opioïde mu",
                es: "Receptor opioide mu",
                ja: "μオピオイド受容体",
                zh: "μ阿片受体",
                ko: "뮤 오피오이드 수용체",
                ru: "Мю-опиоидный рецептор",
                de: "μ-Opioidrezeptor",
                ar: "مستقبل الأفيون ميو"
            ),
            affinity: AffinityMeasurement(kiNM: 1.35),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -12.1, deltaHKcal: -6.8, minusTDeltaSKcal: -5.3
            ),
            source: .pdspKi,
            reference: "Volpe 2011; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "fentanyl",
            targetId: "KOR",
            targetName: LocalizedString(
                en: "Kappa-opioid receptor",
                fr: "Récepteur opioïde kappa",
                es: "Receptor opioide kappa",
                ja: "κオピオイド受容体",
                zh: "κ阿片受体",
                ko: "카파 오피오이드 수용체",
                ru: "Каппа-опиоидный рецептор",
                de: "κ-Opioidrezeptor",
                ar: "مستقبل الأفيون كابا"
            ),
            affinity: AffinityMeasurement(kiNM: 170),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - MDMA (3 targets)

        ThermodynamicBindingProfile(
            substanceId: "mdma",
            targetId: "SERT",
            targetName: LocalizedString(
                en: "Serotonin transporter",
                fr: "Transporteur de sérotonine",
                es: "Transportador de serotonina",
                ja: "セロトニントランスポーター",
                zh: "血清素转运体",
                ko: "세로토닌 수송체",
                ru: "Транспортёр серотонина",
                de: "Serotonintransporter",
                ar: "ناقل السيروتونين"
            ),
            affinity: AffinityMeasurement(kiNM: 238),
            source: .pdspKi,
            reference: "PDSP Ki Database; Simmler 2013",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "mdma",
            targetId: "DAT",
            targetName: LocalizedString(
                en: "Dopamine transporter",
                fr: "Transporteur de dopamine",
                es: "Transportador de dopamina",
                ja: "ドパミントランスポーター",
                zh: "多巴胺转运体",
                ko: "도파민 수송체",
                ru: "Транспортёр дофамина",
                de: "Dopamintransporter",
                ar: "ناقل الدوبامين"
            ),
            affinity: AffinityMeasurement(kiNM: 2400),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "mdma",
            targetId: "NET",
            targetName: LocalizedString(
                en: "Norepinephrine transporter",
                fr: "Transporteur de noradrénaline",
                es: "Transportador de norepinefrina",
                ja: "ノルエピネフリントランスポーター",
                zh: "去甲肾上腺素转运体",
                ko: "노르에피네프린 수송체",
                ru: "Транспортёр норадреналина",
                de: "Noradrenalintransporter",
                ar: "ناقل النورإبينفرين"
            ),
            affinity: AffinityMeasurement(kiNM: 460),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Amphetamine (3 targets, stereochemistry)

        ThermodynamicBindingProfile(
            substanceId: "amphetamine",
            targetId: "DAT",
            targetName: LocalizedString(
                en: "Dopamine transporter",
                fr: "Transporteur de dopamine",
                es: "Transportador de dopamina",
                ja: "ドパミントランスポーター",
                zh: "多巴胺转运体",
                ko: "도파민 수송체",
                ru: "Транспортёр дофамина",
                de: "Dopamintransporter",
                ar: "ناقل الدوبامين"
            ),
            affinity: AffinityMeasurement(kiNM: 34),
            stereochemistry: StereochemicalNote(
                enantiomer: "S(+)/d",
                affinityRatio: 4.0,
                note: LocalizedString(
                    en: "d-amphetamine 3-5x more potent at DAT than l-amphetamine",
                    fr: "La d-amphétamine est 3-5x plus puissante au DAT que la l-amphétamine",
                    es: "La d-anfetamina es 3-5 veces más potente en el DAT que la l-anfetamina",
                    ja: "d-アンフェタミンはl-アンフェタミンよりDATにおいて3〜5倍強力",
                    zh: "d-苯丙胺在DAT上的效力是l-苯丙胺的3-5倍",
                    ko: "d-암페타민은 l-암페타민보다 DAT에서 3-5배 더 강력함",
                    ru: "d-амфетамин в 3-5 раз активнее l-амфетамина в отношении DAT",
                    de: "d-Amphetamin ist am DAT 3-5x potenter als l-Amphetamin",
                    ar: "d-أمفيتامين أقوى بـ 3-5 مرات في DAT من l-أمفيتامين"
                )
            ),
            source: .pdspKi,
            reference: "PDSP; Heal 2013 J Psychopharmacol",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "amphetamine",
            targetId: "NET",
            targetName: LocalizedString(
                en: "Norepinephrine transporter",
                fr: "Transporteur de noradrénaline",
                es: "Transportador de norepinefrina",
                ja: "ノルエピネフリントランスポーター",
                zh: "去甲肾上腺素转运体",
                ko: "노르에피네프린 수송체",
                ru: "Транспортёр норадреналина",
                de: "Noradrenalintransporter",
                ar: "ناقل النورإبينفرين"
            ),
            affinity: AffinityMeasurement(kiNM: 70),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "amphetamine",
            targetId: "SERT",
            targetName: LocalizedString(
                en: "Serotonin transporter",
                fr: "Transporteur de sérotonine",
                es: "Transportador de serotonina",
                ja: "セロトニントランスポーター",
                zh: "血清素转运体",
                ko: "세로토닌 수송체",
                ru: "Транспортёр серотонина",
                de: "Serotonintransporter",
                ar: "ناقل السيروتونين"
            ),
            affinity: AffinityMeasurement(kiNM: 3400),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Cocaine (3 targets, partial ITC)

        ThermodynamicBindingProfile(
            substanceId: "cocaine",
            targetId: "DAT",
            targetName: LocalizedString(
                en: "Dopamine transporter",
                fr: "Transporteur de dopamine",
                es: "Transportador de dopamina",
                ja: "ドパミントランスポーター",
                zh: "多巴胺转运体",
                ko: "도파민 수송체",
                ru: "Транспортёр дофамина",
                de: "Dopamintransporter",
                ar: "ناقل الدوبامين"
            ),
            affinity: AffinityMeasurement(kiNM: 200),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -9.1, deltaHKcal: -4.5, minusTDeltaSKcal: -4.6
            ),
            source: .pdspKi,
            reference: "PDSP; partial ITC from literature",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "cocaine",
            targetId: "SERT",
            targetName: LocalizedString(
                en: "Serotonin transporter",
                fr: "Transporteur de sérotonine",
                es: "Transportador de serotonina",
                ja: "セロトニントランスポーター",
                zh: "血清素转运体",
                ko: "세로토닌 수송체",
                ru: "Транспортёр серотонина",
                de: "Serotonintransporter",
                ar: "ناقل السيروتونين"
            ),
            affinity: AffinityMeasurement(kiNM: 300),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "cocaine",
            targetId: "NET",
            targetName: LocalizedString(
                en: "Norepinephrine transporter",
                fr: "Transporteur de noradrénaline",
                es: "Transportador de norepinefrina",
                ja: "ノルエピネフリントランスポーター",
                zh: "去甲肾上腺素转运体",
                ko: "노르에피네프린 수송체",
                ru: "Транспортёр норадреналина",
                de: "Noradrenalintransporter",
                ar: "ناقل النورإبينفرين"
            ),
            affinity: AffinityMeasurement(kiNM: 500),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - THC (2 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "thc",
            targetId: "CB1",
            targetName: LocalizedString(
                en: "Cannabinoid CB1 receptor",
                fr: "Récepteur cannabinoïde CB1",
                es: "Receptor cannabinoide CB1",
                ja: "カンナビノイドCB1受容体",
                zh: "大麻素CB1受体",
                ko: "칸나비노이드 CB1 수용체",
                ru: "Каннабиноидный рецептор CB1",
                de: "Cannabinoid-CB1-Rezeptor",
                ar: "مستقبل القنب CB1"
            ),
            affinity: AffinityMeasurement(kiNM: 40),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -10.1, deltaHKcal: -8.3, minusTDeltaSKcal: -1.8
            ),
            source: .pdspKi,
            reference: "PDSP; Pertwee 2008; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "thc",
            targetId: "CB2",
            targetName: LocalizedString(
                en: "Cannabinoid CB2 receptor",
                fr: "Récepteur cannabinoïde CB2",
                es: "Receptor cannabinoide CB2",
                ja: "カンナビノイドCB2受容体",
                zh: "大麻素CB2受体",
                ko: "칸나비노이드 CB2 수용체",
                ru: "Каннабиноидный рецептор CB2",
                de: "Cannabinoid-CB2-Rezeptor",
                ar: "مستقبل القنب CB2"
            ),
            affinity: AffinityMeasurement(kiNM: 36),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Salvinorin A (1 target, partial ITC)

        ThermodynamicBindingProfile(
            substanceId: "salvinorin-a",
            targetId: "KOR",
            targetName: LocalizedString(
                en: "Kappa-opioid receptor",
                fr: "Récepteur opioïde kappa",
                es: "Receptor opioide kappa",
                ja: "κオピオイド受容体",
                zh: "κ阿片受体",
                ko: "카파 오피오이드 수용체",
                ru: "Каппа-опиоидный рецептор",
                de: "κ-Opioidrezeptor",
                ar: "مستقبل الأفيون كابا"
            ),
            affinity: AffinityMeasurement(kiNM: 1.9),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -11.9, deltaHKcal: -9.1, minusTDeltaSKcal: -2.8
            ),
            source: .pdspKi,
            reference: "Roth 2002 PNAS; partial ITC from literature",
            isPrimaryTarget: true
        ),

        // MARK: - Caffeine (2 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "caffeine",
            targetId: "A2A",
            targetName: LocalizedString(
                en: "Adenosine A2A receptor",
                fr: "Récepteur adénosinergique A2A",
                es: "Receptor adenosínico A2A",
                ja: "アデノシンA2A受容体",
                zh: "腺苷A2A受体",
                ko: "아데노신 A2A 수용체",
                ru: "Аденозиновый рецептор A2A",
                de: "Adenosin-A2A-Rezeptor",
                ar: "مستقبل الأدينوسين A2A"
            ),
            affinity: AffinityMeasurement(kiNM: 2400),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -7.7, deltaHKcal: -3.2, minusTDeltaSKcal: -4.5
            ),
            source: .pdspKi,
            reference: "Fredholm 1999; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "caffeine",
            targetId: "A1",
            targetName: LocalizedString(
                en: "Adenosine A1 receptor",
                fr: "Récepteur adénosinergique A1",
                es: "Receptor adenosínico A1",
                ja: "アデノシンA1受容体",
                zh: "腺苷A1受体",
                ko: "아데노신 A1 수용체",
                ru: "Аденозиновый рецептор A1",
                de: "Adenosin-A1-Rezeptor",
                ar: "مستقبل الأدينوسين A1"
            ),
            affinity: AffinityMeasurement(kiNM: 12000),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Nicotine (2 targets, ITC on primary, stereochemistry)

        ThermodynamicBindingProfile(
            substanceId: "nicotine",
            targetId: "a4b2-nAChR",
            targetName: LocalizedString(
                en: "α4β2 nicotinic acetylcholine receptor",
                fr: "Récepteur nicotinique α4β2",
                es: "Receptor nicotínico de acetilcolina α4β2",
                ja: "α4β2ニコチン性アセチルコリン受容体",
                zh: "α4β2烟碱型乙酰胆碱受体",
                ko: "α4β2 니코틴성 아세틸콜린 수용체",
                ru: "α4β2 никотиновый ацетилхолиновый рецептор",
                de: "Nikotinischer α4β2-Acetylcholinrezeptor",
                ar: "مستقبل الأسيتيل كولين النيكوتيني α4β2"
            ),
            affinity: AffinityMeasurement(kiNM: 1),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -12.3, deltaHKcal: -7.8, minusTDeltaSKcal: -4.5
            ),
            stereochemistry: StereochemicalNote(
                enantiomer: "S(-)",
                note: LocalizedString(
                    en: "Natural S(-)-nicotine is the active enantiomer",
                    fr: "La S(-)-nicotine naturelle est l'énantiomère actif",
                    es: "La S(-)-nicotina natural es el enantiómero activo",
                    ja: "天然のS(-)-ニコチンが活性エナンチオマーである",
                    zh: "天然S(-)-尼古丁是活性对映体",
                    ko: "천연 S(-)-니코틴이 활성 거울상 이성질체임",
                    ru: "Природный S(-)-никотин является активным энантиомером",
                    de: "Natürliches S(-)-Nikotin ist das aktive Enantiomer",
                    ar: "النيكوتين الطبيعي S(-) هو المتصاوغ الفعال"
                )
            ),
            source: .pdspKi,
            reference: "PDSP; Dani & Bertrand 2007; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "nicotine",
            targetId: "a7-nAChR",
            targetName: LocalizedString(
                en: "α7 nicotinic acetylcholine receptor",
                fr: "Récepteur nicotinique α7",
                es: "Receptor nicotínico de acetilcolina α7",
                ja: "α7ニコチン性アセチルコリン受容体",
                zh: "α7烟碱型乙酰胆碱受体",
                ko: "α7 니코틴성 아세틸콜린 수용체",
                ru: "α7 никотиновый ацетилхолиновый рецептор",
                de: "Nikotinischer α7-Acetylcholinrezeptor",
                ar: "مستقبل الأسيتيل كولين النيكوتيني α7"
            ),
            affinity: AffinityMeasurement(kiNM: 1000),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Atropine (2 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "atropine",
            targetId: "mAChR-M1",
            targetName: LocalizedString(
                en: "Muscarinic M1 receptor",
                fr: "Récepteur muscarinique M1",
                es: "Receptor muscarínico M1",
                ja: "ムスカリンM1受容体",
                zh: "毒蕈碱M1受体",
                ko: "무스카린 M1 수용체",
                ru: "Мускариновый рецептор M1",
                de: "Muskarinischer M1-Rezeptor",
                ar: "مستقبل المسكارين M1"
            ),
            affinity: AffinityMeasurement(kiNM: 0.4),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -12.8, deltaHKcal: -10.2, minusTDeltaSKcal: -2.6
            ),
            source: .pdspKi,
            reference: "PDSP; Hulme 1990; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "atropine",
            targetId: "mAChR-M2",
            targetName: LocalizedString(
                en: "Muscarinic M2 receptor",
                fr: "Récepteur muscarinique M2",
                es: "Receptor muscarínico M2",
                ja: "ムスカリンM2受容体",
                zh: "毒蕈碱M2受体",
                ko: "무스카린 M2 수용체",
                ru: "Мускариновый рецептор M2",
                de: "Muskarinischer M2-Rezeptor",
                ar: "مستقبل المسكارين M2"
            ),
            affinity: AffinityMeasurement(kiNM: 0.6),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Ketamine (2 targets, stereochemistry)

        ThermodynamicBindingProfile(
            substanceId: "ketamine",
            targetId: "NMDA-PCP",
            targetName: LocalizedString(
                en: "NMDA receptor PCP site",
                fr: "Site PCP du récepteur NMDA",
                es: "Sitio PCP del receptor NMDA",
                ja: "NMDA受容体PCP部位",
                zh: "NMDA受体PCP位点",
                ko: "NMDA 수용체 PCP 부위",
                ru: "PCP-сайт NMDA-рецептора",
                de: "PCP-Bindungsstelle des NMDA-Rezeptors",
                ar: "موقع PCP لمستقبل NMDA"
            ),
            affinity: AffinityMeasurement(kiNM: 659),
            stereochemistry: StereochemicalNote(
                enantiomer: "S(+)",
                affinityRatio: 3.0,
                note: LocalizedString(
                    en: "S(+)-ketamine (esketamine) ~3x more potent than R(-) at NMDA",
                    fr: "La S(+)-kétamine (eskétamine) est ~3x plus puissante que la R(-) au NMDA",
                    es: "La S(+)-ketamina (esketamina) es ~3 veces más potente que la R(-) en NMDA",
                    ja: "S(+)-ケタミン（エスケタミン）はR(-)よりNMDAにおいて約3倍強力",
                    zh: "S(+)-氯胺酮（艾司氯胺酮）在NMDA上的效力约为R(-)的3倍",
                    ko: "S(+)-케타민(에스케타민)은 NMDA에서 R(-)보다 약 3배 더 강력함",
                    ru: "S(+)-кетамин (эскетамин) примерно в 3 раза активнее R(-) в отношении NMDA",
                    de: "S(+)-Ketamin (Esketamin) ist am NMDA ~3x potenter als R(-)",
                    ar: "S(+)-كيتامين (إسكيتامين) أقوى بـ 3 مرات تقريبًا من R(-) في NMDA"
                )
            ),
            source: .pdspKi,
            reference: "PDSP; Zanos & Gould 2018",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "ketamine",
            targetId: "D2",
            targetName: LocalizedString(
                en: "Dopamine D2 receptor",
                fr: "Récepteur dopaminergique D2",
                es: "Receptor dopaminérgico D2",
                ja: "ドパミンD2受容体",
                zh: "多巴胺D2受体",
                ko: "도파민 D2 수용체",
                ru: "Дофаминовый рецептор D2",
                de: "Dopamin-D2-Rezeptor",
                ar: "مستقبل الدوبامين D2"
            ),
            affinity: AffinityMeasurement(kiNM: 2500),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Mescaline (2 targets)

        ThermodynamicBindingProfile(
            substanceId: "mescaline",
            targetId: "5-HT2A",
            targetName: LocalizedString(
                en: "Serotonin 5-HT2A receptor",
                fr: "Récepteur sérotoninergique 5-HT2A",
                es: "Receptor serotoninérgico 5-HT2A",
                ja: "セロトニン5-HT2A受容体",
                zh: "血清素5-HT2A受体",
                ko: "세로토닌 5-HT2A 수용체",
                ru: "Серотониновый рецептор 5-HT2A",
                de: "Serotonin-5-HT2A-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT2A"
            ),
            affinity: AffinityMeasurement(kiNM: 3600),
            source: .pdspKi,
            reference: "PDSP; Monte 1997",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "mescaline",
            targetId: "5-HT2C",
            targetName: LocalizedString(
                en: "Serotonin 5-HT2C receptor",
                fr: "Récepteur sérotoninergique 5-HT2C",
                es: "Receptor serotoninérgico 5-HT2C",
                ja: "セロトニン5-HT2C受容体",
                zh: "血清素5-HT2C受体",
                ko: "세로토닌 5-HT2C 수용체",
                ru: "Серотониновый рецептор 5-HT2C",
                de: "Serotonin-5-HT2C-Rezeptor",
                ar: "مستقبل السيروتونين 5-HT2C"
            ),
            affinity: AffinityMeasurement(kiNM: 1700),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Ibogaine (6 targets)

        ThermodynamicBindingProfile(
            substanceId: "ibogaine",
            targetId: "a3b4-nAChR",
            targetName: LocalizedString(
                en: "α3β4 nicotinic receptor",
                fr: "Récepteur nicotinique α3β4",
                es: "Receptor nicotínico α3β4",
                ja: "α3β4ニコチン受容体",
                zh: "α3β4烟碱受体",
                ko: "α3β4 니코틴 수용체",
                ru: "α3β4 никотиновый рецептор",
                de: "Nikotinischer α3β4-Rezeptor",
                ar: "مستقبل النيكوتين α3β4"
            ),
            affinity: AffinityMeasurement(kiNM: 20),
            source: .pdspKi,
            reference: "Bhatt 2000; Alper 2001",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "ibogaine",
            targetId: "NMDA",
            targetName: LocalizedString(
                en: "NMDA receptor",
                fr: "Récepteur NMDA",
                es: "Receptor NMDA",
                ja: "NMDA受容体",
                zh: "NMDA受体",
                ko: "NMDA 수용체",
                ru: "NMDA-рецептор",
                de: "NMDA-Rezeptor",
                ar: "مستقبل NMDA"
            ),
            affinity: AffinityMeasurement(kiNM: 31),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "ibogaine",
            targetId: "sigma-2",
            targetName: LocalizedString(
                en: "Sigma-2 receptor",
                fr: "Récepteur sigma-2",
                es: "Receptor sigma-2",
                ja: "シグマ2受容体",
                zh: "σ2受体",
                ko: "시그마-2 수용체",
                ru: "Сигма-2 рецептор",
                de: "Sigma-2-Rezeptor",
                ar: "مستقبل سيغما-2"
            ),
            affinity: AffinityMeasurement(kiNM: 90),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "ibogaine",
            targetId: "SERT",
            targetName: LocalizedString(
                en: "Serotonin transporter",
                fr: "Transporteur de sérotonine",
                es: "Transportador de serotonina",
                ja: "セロトニントランスポーター",
                zh: "血清素转运体",
                ko: "세로토닌 수송체",
                ru: "Транспортёр серотонина",
                de: "Serotonintransporter",
                ar: "ناقل السيروتونين"
            ),
            affinity: AffinityMeasurement(kiNM: 500),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "ibogaine",
            targetId: "MOR",
            targetName: LocalizedString(
                en: "Mu-opioid receptor",
                fr: "Récepteur opioïde mu",
                es: "Receptor opioide mu",
                ja: "μオピオイド受容体",
                zh: "μ阿片受体",
                ko: "뮤 오피오이드 수용체",
                ru: "Мю-опиоидный рецептор",
                de: "μ-Opioidrezeptor",
                ar: "مستقبل الأفيون ميو"
            ),
            affinity: AffinityMeasurement(kiNM: 130),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
        ThermodynamicBindingProfile(
            substanceId: "ibogaine",
            targetId: "KOR",
            targetName: LocalizedString(
                en: "Kappa-opioid receptor",
                fr: "Récepteur opioïde kappa",
                es: "Receptor opioide kappa",
                ja: "κオピオイド受容体",
                zh: "κ阿片受体",
                ko: "카파 오피오이드 수용체",
                ru: "Каппа-опиоидный рецептор",
                de: "κ-Opioidrezeptor",
                ar: "مستقبل الأفيون كابا"
            ),
            affinity: AffinityMeasurement(kiNM: 3500),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Cathinone (2 targets)

        ThermodynamicBindingProfile(
            substanceId: "cathinone",
            targetId: "NET",
            targetName: LocalizedString(
                en: "Norepinephrine transporter",
                fr: "Transporteur de noradrénaline",
                es: "Transportador de norepinefrina",
                ja: "ノルエピネフリントランスポーター",
                zh: "去甲肾上腺素转运体",
                ko: "노르에피네프린 수송체",
                ru: "Транспортёр норадреналина",
                de: "Noradrenalintransporter",
                ar: "ناقل النورإبينفرين"
            ),
            affinity: AffinityMeasurement(kiNM: 12.4),
            source: .pdspKi,
            reference: "Simmler 2013; Brenneisen 1990",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "cathinone",
            targetId: "DAT",
            targetName: LocalizedString(
                en: "Dopamine transporter",
                fr: "Transporteur de dopamine",
                es: "Transportador de dopamina",
                ja: "ドパミントランスポーター",
                zh: "多巴胺转运体",
                ko: "도파민 수송체",
                ru: "Транспортёр дофамина",
                de: "Dopamintransporter",
                ar: "ناقل الدوبامين"
            ),
            affinity: AffinityMeasurement(kiNM: 18.5),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Apigenin (1 target)

        ThermodynamicBindingProfile(
            substanceId: "apigenin",
            targetId: "GABA-A-BZD",
            targetName: LocalizedString(
                en: "GABA-A benzodiazepine site",
                fr: "Site benzodiazépine du GABA-A",
                es: "Sitio benzodiazepínico del GABA-A",
                ja: "GABA-Aベンゾジアゼピン結合部位",
                zh: "GABA-A苯二氮卓位点",
                ko: "GABA-A 벤조디아제핀 부위",
                ru: "Бензодиазепиновый сайт ГАМК-A рецептора",
                de: "Benzodiazepin-Bindungsstelle des GABA-A-Rezeptors",
                ar: "موقع البنزوديازيبين لمستقبل GABA-A"
            ),
            affinity: AffinityMeasurement(kiNM: 3000),
            source: .literature,
            reference: "Viola 1995; Salehi 2019",
            isPrimaryTarget: true
        ),

        // MARK: - GHB (2 targets)

        ThermodynamicBindingProfile(
            substanceId: "ghb",
            targetId: "GABA-B",
            targetName: LocalizedString(
                en: "GABA-B receptor",
                fr: "Récepteur GABA-B",
                es: "Receptor GABA-B",
                ja: "GABA-B受容体",
                zh: "GABA-B受体",
                ko: "GABA-B 수용체",
                ru: "ГАМК-B рецептор",
                de: "GABA-B-Rezeptor",
                ar: "مستقبل GABA-B"
            ),
            affinity: AffinityMeasurement(kiNM: 1700),
            source: .pdspKi,
            reference: "PDSP; Snead 2000",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "ghb",
            targetId: "GHB-R",
            targetName: LocalizedString(
                en: "GHB receptor",
                fr: "Récepteur GHB",
                es: "Receptor GHB",
                ja: "GHB受容体",
                zh: "GHB受体",
                ko: "GHB 수용체",
                ru: "GHB-рецептор",
                de: "GHB-Rezeptor",
                ar: "مستقبل GHB"
            ),
            affinity: AffinityMeasurement(kiNM: 100),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Methamphetamine (2 targets, stereochemistry)

        ThermodynamicBindingProfile(
            substanceId: "methamphetamine",
            targetId: "DAT",
            targetName: LocalizedString(
                en: "Dopamine transporter",
                fr: "Transporteur de dopamine",
                es: "Transportador de dopamina",
                ja: "ドパミントランスポーター",
                zh: "多巴胺转运体",
                ko: "도파민 수송체",
                ru: "Транспортёр дофамина",
                de: "Dopamintransporter",
                ar: "ناقل الدوبامين"
            ),
            affinity: AffinityMeasurement(kiNM: 24),
            stereochemistry: StereochemicalNote(
                enantiomer: "S(+)/d",
                affinityRatio: 5.0,
                note: LocalizedString(
                    en: "d-methamphetamine ~5x more potent at DAT than l-methamphetamine",
                    fr: "La d-méthamphetamine est ~5x plus puissante au DAT que la l-méthamphetamine",
                    es: "La d-metanfetamina es ~5 veces más potente en el DAT que la l-metanfetamina",
                    ja: "d-メタンフェタミンはl-メタンフェタミンよりDATにおいて約5倍強力",
                    zh: "d-甲基苯丙胺在DAT上的效力约为l-甲基苯丙胺的5倍",
                    ko: "d-메스암페타민은 l-메스암페타민보다 DAT에서 약 5배 더 강력함",
                    ru: "d-метамфетамин примерно в 5 раз активнее l-метамфетамина в отношении DAT",
                    de: "d-Methamphetamin ist am DAT ~5x potenter als l-Methamphetamin",
                    ar: "d-ميثامفيتامين أقوى بـ 5 مرات تقريبًا في DAT من l-ميثامفيتامين"
                )
            ),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "methamphetamine",
            targetId: "NET",
            targetName: LocalizedString(
                en: "Norepinephrine transporter",
                fr: "Transporteur de noradrénaline",
                es: "Transportador de norepinefrina",
                ja: "ノルエピネフリントランスポーター",
                zh: "去甲肾上腺素转运体",
                ko: "노르에피네프린 수송체",
                ru: "Транспортёр норадреналина",
                de: "Noradrenalintransporter",
                ar: "ناقل النورإبينفرين"
            ),
            affinity: AffinityMeasurement(kiNM: 12),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),

        // MARK: - Dronabinol (2 targets, ITC on primary)

        ThermodynamicBindingProfile(
            substanceId: "dronabinol",
            targetId: "CB1",
            targetName: LocalizedString(
                en: "Cannabinoid CB1 receptor",
                fr: "Récepteur cannabinoïde CB1",
                es: "Receptor cannabinoide CB1",
                ja: "カンナビノイドCB1受容体",
                zh: "大麻素CB1受体",
                ko: "칸나비노이드 CB1 수용체",
                ru: "Каннабиноидный рецептор CB1",
                de: "Cannabinoid-CB1-Rezeptor",
                ar: "مستقبل القنب CB1"
            ),
            affinity: AffinityMeasurement(kiNM: 40),
            thermodynamics: ThermodynamicDecomposition(
                deltaGKcal: -10.1, deltaHKcal: -8.3, minusTDeltaSKcal: -1.8
            ),
            source: .pdspKi,
            reference: "Same pharmacology as THC; SCORPIO ITC",
            isPrimaryTarget: true
        ),
        ThermodynamicBindingProfile(
            substanceId: "dronabinol",
            targetId: "CB2",
            targetName: LocalizedString(
                en: "Cannabinoid CB2 receptor",
                fr: "Récepteur cannabinoïde CB2",
                es: "Receptor cannabinoide CB2",
                ja: "カンナビノイドCB2受容体",
                zh: "大麻素CB2受体",
                ko: "칸나비노이드 CB2 수용체",
                ru: "Каннабиноидный рецептор CB2",
                de: "Cannabinoid-CB2-Rezeptor",
                ar: "مستقبل القنب CB2"
            ),
            affinity: AffinityMeasurement(kiNM: 36),
            source: .pdspKi,
            reference: "PDSP Ki Database",
            isPrimaryTarget: false
        ),
    ]

    // MARK: - Lookups

    /// Look up the primary-target thermodynamic binding profile by substance ID.
    public static func profile(for substanceId: String) -> ThermodynamicBindingProfile? {
        knownProfiles.first { $0.substanceId == substanceId.lowercased() && $0.isPrimaryTarget }
    }

    /// Look up all thermodynamic binding profiles for a substance (all targets).
    public static func profiles(for substanceId: String) -> [ThermodynamicBindingProfile] {
        knownProfiles.filter { $0.substanceId == substanceId.lowercased() }
    }

    /// All substance IDs with known thermodynamic data.
    public static var knownSubstanceIds: Set<String> {
        Set(knownProfiles.map(\.substanceId))
    }
}

// MARK: - Prodrug Catalog

extension ProdrugRelationship {

    /// Known prodrug → active metabolite relationships for pharmacovigilance.
    public static let knownProdrugs: [ProdrugRelationship] = [
        ProdrugRelationship(
            prodrugId: "psilocybin",
            activeMetaboliteId: "psilocin",
            activatingEnzyme: "alkaline phosphatase",
            conversionHalfLifeMinutes: 30,
            note: LocalizedString(
                en: "Psilocybin is a phosphate ester prodrug; dephosphorylated to psilocin in vivo",
                fr: "La psilocybine est une prodrogue ester phosphate; déphosphorylée en psilocine in vivo",
                es: "La psilocibina es un profármaco éster fosfato; se desfosforila a psilocina in vivo",
                ja: "サイロシビンはリン酸エステル型プロドラッグであり、生体内で脱リン酸化されサイロシンとなる",
                zh: "赛洛西宾是磷酸酯前药，在体内去磷酸化为赛洛辛",
                ko: "실로시빈은 인산에스테르 전구약물로, 체내에서 탈인산화되어 실로신이 됨",
                ru: "Псилоцибин — фосфатно-эфирное пролекарство; дефосфорилируется до псилоцина in vivo",
                de: "Psilocybin ist ein Phosphatester-Prodrug; wird in vivo zu Psilocin dephosphoryliert",
                ar: "السيلوسيبين هو دواء أولي من نوع إستر الفوسفات؛ يُزال منه الفوسفات ليتحول إلى سيلوسين في الجسم الحي"
            )
        ),
        ProdrugRelationship(
            prodrugId: "codeine",
            activeMetaboliteId: "morphine",
            activatingEnzyme: "CYP2D6",
            conversionHalfLifeMinutes: 60,
            note: LocalizedString(
                en: "Codeine O-demethylated to morphine by CYP2D6; poor metabolizers get no analgesia",
                fr: "La codéine est O-déméthylée en morphine par le CYP2D6; les métaboliseurs lents n'obtiennent aucune analgésie",
                es: "La codeína se O-desmetila a morfina por CYP2D6; los metabolizadores lentos no obtienen analgesia",
                ja: "コデインはCYP2D6によりO-脱メチル化されモルヒネとなる；低代謝者では鎮痛効果が得られない",
                zh: "可待因经CYP2D6 O-去甲基化为吗啡；慢代谢者无法获得镇痛效果",
                ko: "코데인은 CYP2D6에 의해 O-탈메틸화되어 모르핀이 됨; 저대사자는 진통 효과를 얻지 못함",
                ru: "Кодеин O-деметилируется до морфина ферментом CYP2D6; у медленных метаболизаторов анальгезия отсутствует",
                de: "Codein wird durch CYP2D6 zu Morphin O-demethyliert; bei langsamen Metabolisierern tritt keine Analgesie ein",
                ar: "يُنزع ميثيل الكودين بواسطة CYP2D6 ليتحول إلى مورفين؛ المستقلبون البطيئون لا يحصلون على تسكين للألم"
            )
        ),
        ProdrugRelationship(
            prodrugId: "lisdexamfetamine",
            activeMetaboliteId: "dextroamphetamine",
            activatingEnzyme: "peptidase",
            conversionHalfLifeMinutes: 60,
            note: LocalizedString(
                en: "Lysine-conjugated prodrug cleaved by peptidases in red blood cells",
                fr: "Prodrogue conjuguée à la lysine clivée par les peptidases dans les globules rouges",
                es: "Profármaco conjugado con lisina escindido por peptidasas en los glóbulos rojos",
                ja: "リシン結合型プロドラッグで、赤血球中のペプチダーゼにより切断される",
                zh: "赖氨酸结合前药，由红细胞中的肽酶切割",
                ko: "라이신 결합 전구약물로, 적혈구 내 펩티다아제에 의해 절단됨",
                ru: "Пролекарство, конъюгированное с лизином, расщепляется пептидазами в эритроцитах",
                de: "Lysin-konjugiertes Prodrug, das durch Peptidasen in Erythrozyten gespalten wird",
                ar: "دواء أولي مرتبط بالليسين يُشطر بواسطة الببتيداز في كريات الدم الحمراء"
            )
        ),
        ProdrugRelationship(
            prodrugId: "heroin",
            activeMetaboliteId: "morphine",
            activatingEnzyme: "esterase",
            conversionHalfLifeMinutes: 5,
            note: LocalizedString(
                en: "Diacetylmorphine → 6-MAM → morphine via sequential ester hydrolysis",
                fr: "Diacétylmorphine → 6-MAM → morphine via hydrolyse séquentielle d'ester",
                es: "Diacetilmorfina → 6-MAM → morfina mediante hidrólisis secuencial de éster",
                ja: "ジアセチルモルヒネ → 6-MAM → モルヒネ（逐次エステル加水分解による）",
                zh: "二乙酰吗啡 → 6-MAM → 吗啡，经逐步酯水解",
                ko: "디아세틸모르핀 → 6-MAM → 모르핀, 순차적 에스테르 가수분해를 통해",
                ru: "Диацетилморфин → 6-МАМ → морфин путём последовательного гидролиза сложных эфиров",
                de: "Diacetylmorphin → 6-MAM → Morphin durch sequenzielle Esterhydrolyse",
                ar: "ثنائي أسيتيل المورفين ← 6-MAM ← مورفين عبر التحلل المائي المتسلسل للإستر"
            )
        ),
    ]

    /// Look up prodrug relationship by prodrug substance ID.
    public static func prodrug(for prodrugId: String) -> ProdrugRelationship? {
        knownProdrugs.first { $0.prodrugId == prodrugId.lowercased() }
    }
}
