"""Lifetime projection.

The classifier answers a ten-year question, so asking it again at an older
age does not compound -- for someone already at 60 percent the next decade
also comes back at 60 percent, and a chart of that is a flat line. What
people want to see is accumulation, which is a survival composition over
successive decades.
"""

import pytest

from health_agent.domain.profile import HealthProfile
from health_agent.domain.projection import CHARTED, STEP_YEARS, build
from health_agent.scoring import FakeBackend, RiskScorer

PROFILE = dict(
    sex="male", height_cm=180.0, weight_kg=88.0, waist_cm=96.0,
    smoking_status="former", alcohol_units_per_week=10.0,
    moderate_activity_minutes_per_week=90, sleep_hours_avg=6.5,
    diet_quality_self_rating=3, perceived_stress_rating=3,
    on_bp_medication=False, previously_high_glucose=False,
    eats_vegetables_daily=True,
)


@pytest.fixture
def scorer():
    return RiskScorer(FakeBackend(seed=4, noise=0.0))


def test_cumulative_risk_never_falls(scorer):
    """It is the chance of having developed something by an age. That cannot
    go down as the age goes up."""
    projection = build(HealthProfile(age=30, **PROFILE), scorer)
    assert projection is not None
    for key in projection.labels:
        series = [v[key] for _, v in projection.points if key in v]
        assert series == sorted(series), f"{key} went backwards: {series}"


def test_it_accumulates_rather_than_repeating_the_decade(scorer):
    """The whole reason this exists: a flat repeat of the ten-year figure
    tells nobody anything."""
    profile = HealthProfile(age=30, **PROFILE)
    projection = build(profile, scorer)
    single_decade = scorer.score(profile).condition("t2d_10yr").probability

    final = projection.points[-1][1]["t2d_10yr"]
    assert final > single_decade * 1.5, (
        f"lifetime {final} barely exceeds one decade {single_decade}"
    )
    assert final <= 1.0


def test_steps_are_decades_and_stop_at_a_sensible_age(scorer):
    projection = build(HealthProfile(age=40, **PROFILE), scorer)
    ages = [age for age, _ in projection.points]
    assert ages == sorted(ages)
    assert all(b - a == STEP_YEARS for a, b in zip(ages, ages[1:]))


def test_no_age_means_no_projection(scorer):
    assert build(HealthProfile(**PROFILE), scorer) is None


def test_someone_near_the_end_of_the_table_gets_nothing(scorer):
    """One point is not a trend."""
    assert build(HealthProfile(age=84, **PROFILE), scorer) is None


def test_assumptions_travel_with_the_numbers(scorer):
    """A rising line someone cannot contextualise is just frightening."""
    projection = build(HealthProfile(age=30, **PROFILE), scorer)
    assumptions = " ".join(projection.assumptions).lower()
    assert "nothing changes" in assumptions
    assert "independent" in assumptions
    assert "anything else happening first" in assumptions


def test_the_shape_is_chartable(scorer):
    data = build(HealthProfile(age=30, **PROFILE), scorer).as_dict()
    assert data["x_axis"] == "age"
    assert set(data["series"]) <= set(CHARTED)
    for key, points in data["series"].items():
        assert data["labels"][key]
        for point in points:
            assert 0.0 <= point["probability"] <= 1.0
            assert isinstance(point["age"], int)


# --- through the report -------------------------------------------------

def test_the_table_matches_the_risk_figures(scorer):
    """Supplying rows rather than an instruction is what stops the numbers
    drifting in the retelling."""
    from health_agent.adapters.core import assess

    out = assess({"age": 50, **PROFILE}, scorer=scorer)
    table = out["presentation"]["risk_table"]
    assert table["columns"] == ["Condition", "Probability", "Band", "What it means"]
    assert len(table["rows"]) == len(out["risk"])

    by_label = {v["label"]: v for v in out["risk"].values()}
    for row in table["rows"]:
        expected = by_label[row["Condition"]]
        assert row["Probability"] == f"{expected['probability']:.0%}"
        assert row["What it means"], f"{row['Condition']} has no explanation"


def test_the_table_is_ordered_worst_first(scorer):
    from health_agent.adapters.core import assess

    rows = assess({"age": 50, **PROFILE}, scorer=scorer)["presentation"]["risk_table"]["rows"]
    values = [int(r["Probability"].rstrip("%")) for r in rows]
    assert values == sorted(values, reverse=True)


def test_the_contract_forbids_inventing_points(scorer):
    from health_agent.adapters.core import assess

    out = assess({"age": 50, **PROFILE}, scorer=scorer)
    forbidden = " ".join(out["presentation"]["must_not"]).lower()
    assert "invent, extend or smooth" in forbidden
    assert "what will happen" in forbidden
