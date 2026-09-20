"""Diagnostic thresholds are definitions, not estimates.

Found by an eval against the real model: asked how likely a 27-year-old
with an HbA1c of 60 was to *develop* type 2 diabetes, it said 12 percent.
That is a defensible answer to the question asked -- someone at 60 already
has diabetes -- and a dangerously reassuring thing for that person to read.
"""

import pytest

from health_agent.adapters.core import assess
from health_agent.domain.profile import HealthProfile
from health_agent.domain.thresholds import Urgency, check
from health_agent.scoring import FakeBackend, RiskScorer

HEALTHY = dict(
    sex="female", height_cm=168, weight_kg=60, smoking_status="never",
    alcohol_units_per_week=4, moderate_activity_minutes_per_week=280,
    sleep_hours_avg=8.0, on_bp_medication=False,
    previously_high_glucose=False, eats_vegetables_daily=True,
)


@pytest.fixture
def scorer():
    return RiskScorer(FakeBackend(seed=3, noise=0.0))


@pytest.mark.parametrize(
    ("kwargs", "field", "urgency"),
    [
        ({"hba1c_mmol_mol": 48}, "hba1c_mmol_mol", Urgency.URGENT),
        ({"hba1c_mmol_mol": 60}, "hba1c_mmol_mol", Urgency.URGENT),
        ({"hba1c_mmol_mol": 44}, "hba1c_mmol_mol", Urgency.ROUTINE),
        ({"fasting_glucose_mmol_l": 7.4}, "fasting_glucose_mmol_l", Urgency.URGENT),
        ({"fasting_glucose_mmol_l": 6.4}, "fasting_glucose_mmol_l", Urgency.ROUTINE),
        ({"systolic_bp": 186, "diastolic_bp": 110}, "systolic_bp", Urgency.EMERGENCY),
        ({"systolic_bp": 166, "diastolic_bp": 96}, "systolic_bp", Urgency.URGENT),
        ({"systolic_bp": 144, "diastolic_bp": 88}, "systolic_bp", Urgency.ROUTINE),
        ({"egfr": 26}, "egfr", Urgency.URGENT),
        ({"egfr": 52}, "egfr", Urgency.ROUTINE),
        ({"total_cholesterol_mmol_l": 7.9}, "total_cholesterol_mmol_l", Urgency.ROUTINE),
    ],
)
def test_thresholds_fire_at_the_right_level(kwargs, field, urgency):
    findings = check(HealthProfile(age=45, **kwargs))
    match = next((f for f in findings if f.field == field), None)
    assert match is not None, f"{field} did not fire on {kwargs}"
    assert match.urgency is urgency


@pytest.mark.parametrize("kwargs", [
    {"hba1c_mmol_mol": 38}, {"fasting_glucose_mmol_l": 5.2},
    {"systolic_bp": 124, "diastolic_bp": 78}, {"egfr": 92},
    {"total_cholesterol_mmol_l": 4.6},
])
def test_normal_values_do_not_fire(kwargs):
    assert check(HealthProfile(age=45, **kwargs)) == []


def test_only_the_most_serious_band_per_field_is_reported():
    """A value of 186 passes through three systolic bands. Listing all of
    them is noise."""
    findings = check(HealthProfile(age=45, systolic_bp=186, diastolic_bp=110))
    systolic = [f for f in findings if f.field == "systolic_bp"]
    assert len(systolic) == 1
    assert systolic[0].urgency is Urgency.EMERGENCY


def test_findings_are_ordered_worst_first():
    findings = check(HealthProfile(
        age=60, systolic_bp=186, diastolic_bp=112, hba1c_mmol_mol=44, egfr=55
    ))
    urgencies = [f.urgency for f in findings]
    assert urgencies[0] is Urgency.EMERGENCY
    assert urgencies == sorted(
        urgencies,
        key=lambda u: [Urgency.EMERGENCY, Urgency.URGENT,
                       Urgency.ROUTINE, Urgency.MONITOR].index(u),
    )


def test_every_threshold_says_what_to_do_and_cites_a_source():
    from health_agent.domain.thresholds import THRESHOLDS

    for t in THRESHOLDS:
        assert len(t.meaning) > 30, t.field
        assert len(t.action) > 20, t.field
        assert t.source, t.field


# --- through the graph --------------------------------------------------

def test_a_diagnostic_hba1c_is_stated_not_estimated(scorer):
    """The whole point: this person has diabetes. Telling them their risk
    of developing it is worse than useless."""
    out = assess({"age": 27, "hba1c_mmol_mol": 60, **HEALTHY}, scorer=scorer)
    met = out["clinical_thresholds_met"]
    assert any(f["field"] == "hba1c_mmol_mol" for f in met)
    guidance = out["act_on_this_first"]
    assert "diagnostic threshold for diabetes" in guidance
    assert "not a risk of developing it" in guidance


def test_a_diagnostic_value_does_not_suppress_the_assessment(scorer):
    """Urgent is not the same as an emergency -- they still get their
    picture, with the finding leading it."""
    out = assess({"age": 27, "hba1c_mmol_mol": 60, **HEALTHY}, scorer=scorer)
    assert out["status"] == "complete"
    assert out["risk"]


def test_a_hypertensive_crisis_stops_everything(scorer):
    out = assess({"age": 50, "systolic_bp": 186, "diastolic_bp": 124, **HEALTHY},
                 scorer=scorer)
    assert out["status"] == "seek_care"
    assert "risk" not in out, "must not score someone in a hypertensive crisis"
    assert "today" in out["answer"]


def test_normal_readings_add_nothing_to_the_report(scorer):
    out = assess({"age": 30, "hba1c_mmol_mol": 34, "systolic_bp": 118,
                  "diastolic_bp": 76, "egfr": 95, **HEALTHY}, scorer=scorer)
    assert "clinical_thresholds_met" not in out
