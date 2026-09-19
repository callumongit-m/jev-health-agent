"""Age-aware evidence requirements.

What counts as enough to estimate from changes with age. A flat threshold
refuses the young people early warning helps most, and waves through older
people whose lifestyle data no longer discriminates.
"""

import pytest

from health_agent.domain.evidence import Tier, achieved_tier, assess, band_for
from health_agent.domain.profile import HealthProfile

LIFESTYLE = {
    "sex": "male", "height_cm": 180.0, "weight_kg": 82.0,
    "smoking_status": "never", "alcohol_units_per_week": 4.0,
    "moderate_activity_minutes_per_week": 180, "sleep_hours_avg": 7.0,
}
VITALS = {"systolic_bp": 124, "diastolic_bp": 78}
BLOODS = {"hba1c_mmol_mol": 36}


@pytest.mark.parametrize(
    ("age", "expected_required"),
    [(18, Tier.LIFESTYLE), (29, Tier.LIFESTYLE), (30, Tier.VITALS),
     (44, Tier.VITALS), (45, Tier.BLOODS), (80, Tier.BLOODS)],
)
def test_age_bands_have_no_gaps(age, expected_required):
    assert band_for(age).required_tier is expected_required


def test_a_young_person_is_assessable_without_bloods():
    """The whole point: most 21-year-olds have never had an HbA1c."""
    result = assess(HealthProfile(age=21, **LIFESTYLE))
    assert result.sufficient
    assert result.tier is Tier.LIFESTYLE
    assert result.missing_required == ()


def test_a_young_person_is_still_told_what_would_help():
    result = assess(HealthProfile(age=21, **LIFESTYLE))
    assert "systolic_bp" in result.missing_recommended


def test_middle_age_requires_blood_pressure():
    assert not assess(HealthProfile(age=38, **LIFESTYLE)).sufficient
    assert assess(HealthProfile(age=38, **LIFESTYLE, **VITALS)).sufficient


def test_over_45_requires_bloods():
    without = assess(HealthProfile(age=52, **LIFESTYLE, **VITALS))
    assert not without.sufficient
    assert "hba1c_mmol_mol" in without.missing_required

    with_bloods = assess(HealthProfile(age=52, **LIFESTYLE, **VITALS, **BLOODS))
    assert with_bloods.sufficient
    assert with_bloods.tier is Tier.BLOODS


def test_a_young_person_is_never_asked_for_bloods():
    result = assess(HealthProfile(age=21, **LIFESTYLE))
    assert "hba1c_mmol_mol" not in result.missing_required


def test_waist_alone_satisfies_body_composition():
    """Waist is the better signal, so it should not also demand height+weight."""
    profile = HealthProfile(
        age=21, sex="male", waist_cm=80, height_cm=180,
        smoking_status="never", alcohol_units_per_week=4.0, sleep_hours_avg=7.0,
    )
    assert assess(profile).sufficient


def test_too_few_lifestyle_answers_is_not_enough():
    thin = HealthProfile(age=21, sex="male", height_cm=180, weight_kg=82,
                         smoking_status="never")
    result = assess(thin)
    assert not result.sufficient
    assert result.missing_required


def test_no_age_means_no_estimate():
    """Age sets both the baseline and what evidence is needed."""
    result = assess(HealthProfile(**LIFESTYLE))
    assert not result.sufficient
    assert result.missing_required == ("age",)


def test_thinner_evidence_caps_claimable_confidence():
    ceilings = [t.confidence_ceiling for t in
                (Tier.LIFESTYLE, Tier.VITALS, Tier.BLOODS)]
    assert ceilings == sorted(ceilings), "more evidence must allow more certainty"
    assert Tier.LIFESTYLE.confidence_ceiling < 1.0
    assert Tier.BLOODS.confidence_ceiling == 1.0


def test_achieved_tier_climbs_with_evidence():
    assert achieved_tier(HealthProfile(age=30, **LIFESTYLE)) is Tier.LIFESTYLE
    assert achieved_tier(HealthProfile(age=30, **LIFESTYLE, **VITALS)) is Tier.VITALS
    assert achieved_tier(
        HealthProfile(age=30, **LIFESTYLE, **VITALS, **BLOODS)
    ) is Tier.BLOODS


def test_every_band_explains_itself():
    """A refusal that will not say why is not actionable."""
    from health_agent.domain.evidence import AGE_BANDS

    for band in AGE_BANDS:
        assert len(band.rationale) > 60, band
