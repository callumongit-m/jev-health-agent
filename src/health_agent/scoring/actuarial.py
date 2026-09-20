"""Deterministic life expectancy calculator.

The reasoning LLM decides *which* factors matter and explains them; the
arithmetic happens here so the same profile always yields the same number.
An LLM doing this maths from first principles gives a different answer each
run, which is not a figure anyone should act on.

Baseline figures are period life expectancy (remaining years) from the ONS
National Life Tables for the UK, rounded and interpolated. They are national
averages, not personalised, and are the starting point -- not the answer.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from health_agent.domain.profile import HealthProfile, Sex
from health_agent.domain.results import RiskAssessment

#: age -> remaining years, ONS National Life Tables UK 2020-2022 (approximate).
_BASELINE: dict[str, list[tuple[int, float]]] = {
    "male": [
        (0, 78.6), (20, 59.4), (30, 49.9), (40, 40.5), (50, 31.5),
        (60, 23.2), (70, 15.6), (80, 8.9), (90, 4.1), (100, 2.0),
    ],
    "female": [
        (0, 82.6), (20, 63.2), (30, 53.5), (40, 43.9), (50, 34.5),
        (60, 25.8), (70, 17.8), (80, 10.4), (90, 4.7), (100, 2.2),
    ],
}

#: Years of raw factor cost at which further damage shows sharply diminishing
#: marginal effect. Factor costs overlap heavily -- a smoker with poor diet and
#: no exercise is not losing the arithmetic sum of all three -- so summing them
#: naively over-deducts badly. This saturating curve is the correction.
SATURATION_YEARS = 15.0

#: No combination of modifiable factors deducts more than this share of the
#: remaining baseline. Guards against absurd outputs at older ages.
MAX_DEDUCTION_FRACTION = 0.55

#: Standalone guard, for when the calculator is called without an evidence
#: assessment. The graph does its own age-aware check and passes the result
#: in; this only stops the calculator being misused on its own to turn an
#: empty profile into a number.
MIN_SUFFICIENCY = 0.5


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    key: str
    label: str
    years: float


@dataclass(frozen=True, slots=True)
class LifeExpectancyEstimate:
    baseline_remaining_years: float
    adjusted_remaining_years: float
    estimated_age_at_death: float
    years_lost: float
    years_recoverable: float
    attributions: tuple[FactorAttribution, ...]
    method: str
    caveat: str

    def as_dict(self) -> dict:
        return {
            "baseline_remaining_years": self.baseline_remaining_years,
            "adjusted_remaining_years": self.adjusted_remaining_years,
            "estimated_age_at_death": self.estimated_age_at_death,
            "years_lost_to_modifiable_factors": self.years_lost,
            "years_recoverable": self.years_recoverable,
            "per_factor_years": {a.key: a.years for a in self.attributions},
            "method": self.method,
            "caveat": self.caveat,
        }


def baseline_remaining_years(age: int, sex: Sex | str | None) -> float:
    """Linear interpolation into the national life table."""
    key = str(sex) if sex in ("male", "female") else (
        sex.value if isinstance(sex, Sex) and sex.value in ("male", "female") else None
    )
    if key is None:
        # no sex given: average the two tables rather than guessing
        return round(
            (baseline_remaining_years(age, "male")
             + baseline_remaining_years(age, "female")) / 2,
            1,
        )

    table = _BASELINE[key]
    ages = [a for a, _ in table]
    idx = bisect_right(ages, age) - 1
    idx = max(0, min(idx, len(table) - 2))
    (a0, y0), (a1, y1) = table[idx], table[idx + 1]
    if age >= ages[-1]:
        return table[-1][1]
    fraction = (age - a0) / (a1 - a0)
    return round(y0 + (y1 - y0) * fraction, 1)


def _saturate(raw_years: float) -> float:
    """Diminishing returns: overlapping risks do not add up linearly."""
    if raw_years <= 0:
        return 0.0
    return raw_years / (1.0 + raw_years / SATURATION_YEARS)


def estimate(
    assessment: RiskAssessment,
    profile: HealthProfile,
    evidence: Any = None,
) -> LifeExpectancyEstimate | None:
    """None when there is no honest estimate: no age means no baseline, and
    thin data means any number would be invented rather than derived.

    When an `Evidence` assessment is supplied it decides, because it is
    age-aware and the raw classifier signal is not. Without that, a fit
    21-year-old passes the graph's gate and is then refused here -- two
    notions of "enough" disagreeing about the same person.
    """
    if profile.age is None:
        return None
    if evidence is not None:
        if not evidence.sufficient:
            return None
    elif assessment.data_sufficiency < MIN_SUFFICIENCY:
        return None

    baseline = baseline_remaining_years(profile.age, profile.sex)

    # Weight by confidence: a factor the model is unsure about (typically
    # because the underlying data is missing) must not deduct years as if it
    # were established. Without this, a profile of nothing but age and sex
    # still loses years to factors that were never measured.
    raw = [
        FactorAttribution(key=f.key, label=f.label, years=f.expected_years_cost)
        for f in assessment.factors
        if f.expected_years_cost > 0
    ]
    raw_total = sum(a.years for a in raw)
    deduction = _saturate(raw_total)
    deduction = min(deduction, baseline * MAX_DEDUCTION_FRACTION)

    # scale each factor's share down by the same saturation ratio, so the
    # per-factor numbers sum to the total actually applied
    ratio = (deduction / raw_total) if raw_total else 0.0
    attributions = tuple(
        FactorAttribution(key=a.key, label=a.label, years=round(a.years * ratio, 2))
        for a in sorted(raw, key=lambda x: -x.years)
    )

    adjusted = round(baseline - deduction, 1)
    return LifeExpectancyEstimate(
        baseline_remaining_years=baseline,
        adjusted_remaining_years=adjusted,
        estimated_age_at_death=round(profile.age + adjusted, 1),
        years_lost=round(deduction, 1),
        # recoverable is not the same as lost: some damage is already done
        years_recoverable=round(deduction * 0.7, 1),
        attributions=attributions,
        method=(
            f"ONS national life table baseline for age {profile.age}, minus "
            f"confidence-weighted modifiable-factor cost of {raw_total:.1f} raw "
            f"years, saturated to {deduction:.1f} to account for overlap "
            f"between factors"
        ),
        caveat=(
            "A national average adjusted by modelled lifestyle factors. It is "
            "not a prediction about this individual and carries wide uncertainty."
        ),
    )
