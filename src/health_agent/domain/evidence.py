"""Age-aware evidence requirements.

A single sufficiency threshold is the wrong shape for this. At 21, lifestyle
dominates risk and almost nobody has had an HbA1c -- demanding bloods refuses
exactly the people for whom an early warning is worth most. At 55, lifestyle
is a poor discriminator: metabolic disease is common, often silent, and an
estimate without bloods is closer to a guess dressed as a number.

So what counts as enough evidence changes with age. This mirrors how screening
actually works -- the NHS Health Check begins at 40 and includes bloods, and
the validated risk equations (QRISK, FINDRISC) lean on them progressively
harder with age.

Two things come out of this:

* **What must be asked for.** Below the bar, the agent asks for exactly the
  missing fields rather than guessing.
* **What may be claimed.** Above the bar but short of the full picture, the
  estimate stands but its confidence is capped and the report says what it
  was based on. An estimate that will not say what it rests on is not honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from health_agent.domain.profile import HealthProfile


class Tier(StrEnum):
    """How strong the evidence behind an assessment is."""

    INSUFFICIENT = "insufficient"
    LIFESTYLE = "lifestyle"      # habits and body composition only
    SCREENING = "screening"      # + recall questions: the FINDRISC inputs
    VITALS = "vitals"            # + blood pressure
    BLOODS = "bloods"            # + glycaemic and lipid markers

    @property
    def label(self) -> str:
        return {
            Tier.INSUFFICIENT: "not enough to estimate",
            Tier.LIFESTYLE: "lifestyle and body composition",
            Tier.SCREENING: "a validated non-invasive screening picture",
            Tier.VITALS: "lifestyle plus blood pressure",
            Tier.BLOODS: "lifestyle, blood pressure and blood markers",
        }[self]

    @property
    def confidence_ceiling(self) -> float:
        """The most certainty an estimate at this tier may claim.

        Thinner evidence cannot produce a confident answer, however sure the
        classifier sounds -- this is what stops a lifestyle-only estimate
        being presented with the authority of one backed by bloods.
        """
        return {
            Tier.INSUFFICIENT: 0.0,
            Tier.LIFESTYLE: 0.55,
            # FINDRISC reaches AUC 0.71-0.77 for undiagnosed diabetes with no
            # blood test at all, so this tier is genuinely informative -- not
            # merely better than refusing.
            Tier.SCREENING: 0.70,
            Tier.VITALS: 0.80,
            Tier.BLOODS: 1.0,
        }[self]


#: Body composition: waist is preferred, height+weight is the fallback.
BODY = ("waist_cm", "height_cm", "weight_kg")
LIFESTYLE_FIELDS = (
    "smoking_status",
    "alcohol_units_per_week",
    "moderate_activity_minutes_per_week",
    "sleep_hours_avg",
    "diet_quality_self_rating",
)
VITALS_FIELDS = ("systolic_bp",)
GLYCAEMIC = ("hba1c_mmol_mol", "fasting_glucose_mmol_l")
LIPIDS = ("total_cholesterol_mmol_l", "hdl_mmol_l", "ldl_mmol_l", "triglycerides_mmol_l")

#: Recall questions -- no test, no equipment, and between them they carry
#: most of what FINDRISC asks for.
RECALL_FIELDS = (
    "on_bp_medication",
    "previously_high_glucose",
    "eats_vegetables_daily",
)

#: How many lifestyle fields must be present before habits mean anything.
MIN_LIFESTYLE = 3
MIN_RECALL = 2


@dataclass(frozen=True, slots=True)
class AgeBand:
    """What this age needs before an estimate is worth giving."""

    max_age: int
    required_tier: Tier
    rationale: str
    #: Tier that would materially improve the estimate, if not already met.
    recommended_tier: Tier | None = None


# Nothing here requires a test to clear. That is deliberate: NHS Health Check
# uptake runs at roughly 46-48% and only covers 40-74, so requiring bloods
# after 45 would refuse about half the people it most needs to reach. The
# recall questions cost nothing, and FINDRISC shows they carry real signal
# without a needle. Bloods raise the ceiling rather than unlock the door.
AGE_BANDS: tuple[AgeBand, ...] = (
    AgeBand(
        max_age=29,
        required_tier=Tier.LIFESTYLE,
        recommended_tier=Tier.SCREENING,
        rationale=(
            "Under 30, absolute risk is low and lifestyle dominates what "
            "there is. Most people this age have never had a blood test, and "
            "requiring one would refuse exactly the people an early warning "
            "helps most."
        ),
    ),
    AgeBand(
        max_age=44,
        required_tier=Tier.SCREENING,
        recommended_tier=Tier.VITALS,
        rationale=(
            "From 30, a few things you can answer from memory -- blood "
            "pressure medication, ever being told your blood sugar was high "
            "-- start carrying real weight. Blood pressure is the next step "
            "up and is free at most pharmacies."
        ),
    ),
    AgeBand(
        max_age=200,
        required_tier=Tier.SCREENING,
        recommended_tier=Tier.BLOODS,
        rationale=(
            "From 45, metabolic and cardiovascular disease is common and "
            "often silent, so bloods matter more than at any younger age. "
            "But only about half of this age group has had them, so the "
            "estimate is still given without -- with its confidence capped, "
            "and with what would sharpen it spelled out."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Evidence:
    tier: Tier
    band: AgeBand
    sufficient: bool
    missing_required: tuple[str, ...] = ()
    missing_recommended: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def confidence_ceiling(self) -> float:
        return self.tier.confidence_ceiling

    def as_dict(self) -> dict:
        return {
            "tier": str(self.tier),
            "basis": self.tier.label,
            "sufficient": self.sufficient,
            "confidence_ceiling": self.confidence_ceiling,
            "missing_required": list(self.missing_required),
            "would_improve": list(self.missing_recommended),
            "why": self.band.rationale,
        }


def band_for(age: int) -> AgeBand:
    return next(b for b in AGE_BANDS if age <= b.max_age)


def _has_body(known: set[str]) -> bool:
    return "waist_cm" in known or {"height_cm", "weight_kg"} <= known


def _lifestyle_count(known: set[str]) -> int:
    return sum(1 for f in LIFESTYLE_FIELDS if f in known)


def achieved_tier(profile: HealthProfile) -> Tier:
    """The strongest tier this profile actually supports."""
    known = profile.known_fields()
    if profile.age is None or not _has_body(known):
        return Tier.INSUFFICIENT
    if _lifestyle_count(known) < MIN_LIFESTYLE:
        return Tier.INSUFFICIENT

    tier = Tier.LIFESTYLE
    if sum(1 for f in RECALL_FIELDS if f in known) >= MIN_RECALL:
        tier = Tier.SCREENING
    if any(f in known for f in VITALS_FIELDS):
        tier = max(tier, Tier.VITALS, key=lambda t: _ORDER[t])
        if any(f in known for f in GLYCAEMIC):
            tier = Tier.BLOODS
    return tier


_ORDER = {
    Tier.INSUFFICIENT: 0,
    Tier.LIFESTYLE: 1,
    Tier.SCREENING: 2,
    Tier.VITALS: 3,
    Tier.BLOODS: 4,
}


def assess(profile: HealthProfile) -> Evidence:
    """What this profile supports, and what is missing to go further."""
    if profile.age is None:
        return Evidence(
            tier=Tier.INSUFFICIENT,
            band=AGE_BANDS[0],
            sufficient=False,
            missing_required=("age",),
            notes=("Age sets both the baseline and what evidence is needed.",),
        )

    band = band_for(profile.age)
    tier = achieved_tier(profile)
    known = profile.known_fields()

    missing: list[str] = []
    if _ORDER[band.required_tier] >= _ORDER[Tier.SCREENING] and (
        sum(1 for f in RECALL_FIELDS if f in known) < MIN_RECALL
    ):
        missing.extend(f for f in RECALL_FIELDS if f not in known)
    if not _has_body(known):
        missing.append("waist_cm" if "height_cm" in known else "height_cm")
        if "weight_kg" not in known and "waist_cm" not in known:
            missing.append("weight_kg")
    if _lifestyle_count(known) < MIN_LIFESTYLE:
        missing.extend(f for f in LIFESTYLE_FIELDS if f not in known)

    if _ORDER[band.required_tier] >= _ORDER[Tier.VITALS] and not any(
        f in known for f in VITALS_FIELDS
    ):
        missing.append("systolic_bp")
    if _ORDER[band.required_tier] >= _ORDER[Tier.BLOODS] and not any(
        f in known for f in GLYCAEMIC
    ):
        missing.append("hba1c_mmol_mol")

    sufficient = _ORDER[tier] >= _ORDER[band.required_tier] and not missing

    recommended: list[str] = []
    target = band.recommended_tier
    if target and _ORDER[tier] < _ORDER[target]:
        if _ORDER[target] >= _ORDER[Tier.SCREENING]:
            recommended.extend(f for f in RECALL_FIELDS if f not in known)
        if _ORDER[target] >= _ORDER[Tier.VITALS] and not any(
            f in known for f in VITALS_FIELDS
        ):
            recommended.append("systolic_bp")
        if _ORDER[target] >= _ORDER[Tier.BLOODS] and not any(
            f in known for f in GLYCAEMIC
        ):
            recommended.append("hba1c_mmol_mol")
    # Past the recommended tier, keep naming the next real improvement.
    if _ORDER[tier] >= _ORDER[Tier.SCREENING] and not any(
        f in known for f in VITALS_FIELDS
    ):
        recommended.append("systolic_bp")
    if _ORDER[tier] >= _ORDER[Tier.VITALS] and not any(
        f in known for f in GLYCAEMIC
    ):
        recommended.append("hba1c_mmol_mol")
    if tier is Tier.BLOODS and not any(f in known for f in LIPIDS):
        recommended.append("total_cholesterol_mmol_l")

    notes: list[str] = []
    if sufficient and recommended:
        notes.append(
            f"Estimated from {tier.label}. "
            f"Adding {', '.join(recommended)} would sharpen it."
        )

    return Evidence(
        tier=tier if sufficient else Tier.INSUFFICIENT,
        band=band,
        sufficient=sufficient,
        missing_required=tuple(dict.fromkeys(missing)),
        missing_recommended=tuple(dict.fromkeys(recommended)),
        notes=tuple(notes),
    )
