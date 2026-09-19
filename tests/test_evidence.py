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
#: No test, no equipment -- answerable from memory by anyone.
RECALL = {
    "on_bp_medication": False,
    "previously_high_glucose": False,
    "eats_vegetables_daily": True,
}
VITALS = {"systolic_bp": 124, "diastolic_bp": 78}
BLOODS = {"hba1c_mmol_mol": 36}


@pytest.mark.parametrize(
    ("age", "expected_required"),
    [(18, Tier.LIFESTYLE), (29, Tier.LIFESTYLE), (30, Tier.SCREENING),
     (44, Tier.SCREENING), (45, Tier.SCREENING), (80, Tier.SCREENING)],
)
def test_age_bands_have_no_gaps(age, expected_required):
    assert band_for(age).required_tier is expected_required


def test_no_age_requires_a_test_to_get_an_estimate():
    """NHS Health Check uptake is ~46% and covers only 40-74. Requiring
    bloods after 45 would refuse about half the people who most need this."""
    from health_agent.domain.evidence import AGE_BANDS, RECALL_FIELDS

    for band in AGE_BANDS:
        assert band.required_tier in (Tier.LIFESTYLE, Tier.SCREENING), (
            f"band up to {band.max_age} requires a measurement or test"
        )
    assert set(RECALL_FIELDS) <= set(HealthProfile.model_fields)


def test_a_young_person_is_assessable_without_bloods():
    """The whole point: most 21-year-olds have never had an HbA1c."""
    result = assess(HealthProfile(age=21, **LIFESTYLE))
    assert result.sufficient
    assert result.tier is Tier.LIFESTYLE
    assert result.missing_required == ()


def test_a_young_person_is_still_told_what_would_help():
    result = assess(HealthProfile(age=21, **LIFESTYLE))
    assert result.missing_recommended, "should name the next improvement"
    assert result.sufficient, "but must not make it a condition"


def test_middle_age_needs_the_recall_answers_but_no_tests():
    assert not assess(HealthProfile(age=38, **LIFESTYLE)).sufficient
    assert assess(HealthProfile(age=38, **LIFESTYLE, **RECALL)).sufficient


def test_blood_pressure_raises_the_ceiling_without_being_required():
    without = assess(HealthProfile(age=38, **LIFESTYLE, **RECALL))
    with_bp = assess(HealthProfile(age=38, **LIFESTYLE, **RECALL, **VITALS))
    assert without.sufficient and with_bp.sufficient
    assert with_bp.confidence_ceiling > without.confidence_ceiling


def test_over_45_is_estimable_without_bloods_but_asked_for_them():
    """Asking is right; refusing turns away half the age group."""
    without = assess(HealthProfile(age=52, **LIFESTYLE, **RECALL))
    assert without.sufficient, "must not refuse someone who has had no tests"
    assert without.tier is Tier.SCREENING
    assert "hba1c_mmol_mol" in without.missing_recommended, "but must ask"

    with_bloods = assess(HealthProfile(age=52, **LIFESTYLE, **RECALL, **VITALS, **BLOODS))
    assert with_bloods.tier is Tier.BLOODS
    assert with_bloods.confidence_ceiling > without.confidence_ceiling


def test_nobody_is_ever_required_to_produce_bloods():
    for age in (21, 38, 52, 75):
        result = assess(HealthProfile(age=age, **LIFESTYLE, **RECALL))
        assert "hba1c_mmol_mol" not in result.missing_required, age
        assert result.sufficient, age


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
                (Tier.LIFESTYLE, Tier.SCREENING, Tier.VITALS, Tier.BLOODS)]
    assert ceilings == sorted(ceilings), "more evidence must allow more certainty"
    assert Tier.LIFESTYLE.confidence_ceiling < 1.0
    assert Tier.BLOODS.confidence_ceiling == 1.0


def test_achieved_tier_climbs_with_evidence():
    assert achieved_tier(HealthProfile(age=30, **LIFESTYLE)) is Tier.LIFESTYLE
    assert achieved_tier(HealthProfile(age=30, **LIFESTYLE, **RECALL)) is Tier.SCREENING
    assert achieved_tier(HealthProfile(age=30, **LIFESTYLE, **RECALL, **VITALS)) is Tier.VITALS
    assert achieved_tier(
        HealthProfile(age=30, **LIFESTYLE, **RECALL, **VITALS, **BLOODS)
    ) is Tier.BLOODS


def test_every_band_explains_itself():
    """A refusal that will not say why is not actionable."""
    from health_agent.domain.evidence import AGE_BANDS

    for band in AGE_BANDS:
        assert len(band.rationale) > 60, band
