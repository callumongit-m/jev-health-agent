"""What is worth measuring next.

The one thing here a language model cannot do by reasoning. Asked which
test someone should get it produces a sensible-sounding list; this
produces the answer for *this* person, by scoring them across each
unknown's plausible range and seeing how far the answer actually moves.
"""

import pytest

from health_agent.domain.profile import HealthProfile
from health_agent.domain.worth_measuring import CANDIDATES, WORTH_IT, rank
from health_agent.scoring import FakeBackend, RiskScorer

SPARSE = dict(
    age=45, sex="male", height_cm=178.0, weight_kg=96.0,
    smoking_status="current", cigarettes_per_day=15,
    alcohol_units_per_week=26.0, moderate_activity_minutes_per_week=45,
    diet_quality_self_rating=2, perceived_stress_rating=4,
    on_bp_medication=True, previously_high_glucose=False,
    eats_vegetables_daily=False,
)


@pytest.fixture
def scorer():
    return RiskScorer(FakeBackend(seed=6, noise=0.0))


def test_it_only_suggests_things_they_have_not_given(scorer):
    profile = HealthProfile(**SPARSE, systolic_bp=140, diastolic_bp=88)
    assert all(s.field != "systolic_bp" for s in rank(profile, scorer))


def test_nothing_left_to_measure_returns_nothing(scorer):
    full = HealthProfile(
        **SPARSE, systolic_bp=140, diastolic_bp=88, hba1c_mmol_mol=42,
        total_cholesterol_mmol_l=5.2, egfr=80, waist_cm=104,
        resting_hr=70, sleep_hours_avg=6.5,
    )
    assert rank(full, scorer) == []


def test_accessibility_can_beat_raw_informativeness(scorer):
    """The useful answer is not the most informative test in principle, it
    is the most informative thing someone could actually go and do."""
    from health_agent.domain.worth_measuring import Suggestion

    easy = Suggestion("systolic_bp", "blood pressure", 0.16, 1.0, "free")
    hard = Suggestion("egfr", "eGFR", 0.19, 0.3, "blood test")
    assert easy.score > hard.score


def test_things_that_barely_move_the_answer_are_dropped(scorer):
    """Sending someone for a test that changes nothing is worse than saying
    there is nothing worth doing."""
    for suggestion in rank(HealthProfile(**SPARSE), scorer):
        assert suggestion.swing >= WORTH_IT


def test_suggestions_say_how_to_get_the_thing(scorer):
    for suggestion in rank(HealthProfile(**SPARSE), scorer):
        assert len(suggestion.how) > 30, suggestion.label
    for candidate in CANDIDATES:
        assert 0 < candidate.accessibility <= 1.0
        assert len(candidate.probe) >= 3, candidate.field


def test_it_reaches_the_report(scorer):
    from health_agent.adapters.core import assess

    out = assess(SPARSE, scorer=scorer)
    block = out["presentation"]["worth_measuring"]
    assert block["items"]
    assert "for this person" in block["why"]
    for item in block["items"]:
        assert item["what"] and item["how_to_get_it"]
        assert item["how_much_it_would_move_the_estimate"].endswith("%")
