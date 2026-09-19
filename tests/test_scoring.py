from health_agent.domain.conditions import CONDITIONS, FACTORS
from health_agent.domain.profile import HealthProfile, Sex, SmokingStatus
from health_agent.scoring import FakeBackend, RiskScorer


def _scorer():
    return RiskScorer(FakeBackend(seed=0, noise=0.0))


def test_scorer_answers_every_registered_question():
    a = _scorer().score(HealthProfile(age=50, sex=Sex.MALE, height_cm=175, weight_kg=90))
    assert {c.key for c in a.conditions} == {c.key for c in CONDITIONS}
    assert {f.key for f in a.factors} == {f.key for f in FACTORS}


def test_probabilities_are_bounded():
    a = _scorer().score(HealthProfile(age=80, sex=Sex.MALE, height_cm=170, weight_kg=140,
                                      hba1c_mmol_mol=90, systolic_bp=200, diastolic_bp=110))
    assert all(0.0 <= c.probability <= 1.0 for c in a.conditions)


def test_zero_noise_backend_is_deterministic():
    p = HealthProfile(age=50, sex=Sex.MALE, height_cm=175, weight_kg=90, hba1c_mmol_mol=44)
    s = _scorer()
    assert [c.probability for c in s.score(p).conditions] == [
        c.probability for c in s.score(p).conditions
    ]


def test_backend_omission_is_an_error_not_a_silent_gap():
    import pytest
    from health_agent.scoring.backend import RawResult

    class Broken(FakeBackend):
        def classify(self, state):
            full = super().classify(state)
            return RawResult(answers={k: v for k, v in list(full.answers.items())[:3]},
                             model=full.model)

    with pytest.raises(ValueError, match="omitted answers"):
        RiskScorer(Broken(seed=0)).score(HealthProfile(age=50))


# --- actuarial ---------------------------------------------------------

def test_life_expectancy_refuses_without_age():
    from health_agent.scoring.actuarial import estimate

    p = HealthProfile(sex=Sex.MALE, height_cm=180, weight_kg=85)
    assert estimate(_scorer().score(p), p) is None


def test_life_expectancy_refuses_on_thin_data():
    """An almost-empty profile must not be turned into a number."""
    from health_agent.scoring.actuarial import estimate

    p = HealthProfile(age=41, sex=Sex.FEMALE)
    assessment = _scorer().score(p)
    assert assessment.data_sufficiency < 0.5
    assert estimate(assessment, p) is None


def test_overlapping_factors_do_not_deduct_additively():
    """Naive summing would remove far more years than the evidence supports."""
    from health_agent.scoring.actuarial import estimate

    p = HealthProfile(
        age=55, sex=Sex.MALE, height_cm=175, weight_kg=120, hba1c_mmol_mol=60,
        systolic_bp=170, diastolic_bp=100, smoking_status=SmokingStatus.CURRENT,
        cigarettes_per_day=40, alcohol_units_per_week=60,
        moderate_activity_minutes_per_week=0, sleep_hours_avg=4.5,
        diet_quality_self_rating=1, perceived_stress_rating=5,
        triglycerides_mmol_l=4.5, hdl_mmol_l=0.7, ldl_mmol_l=5.0, alt_u_l=110,
    )
    a = _scorer().score(p)
    e = estimate(a, p)
    assert e is not None
    naive = sum(f.years_cost for f in a.factors)
    assert e.years_lost < naive, "saturation is not being applied"
    assert e.adjusted_remaining_years > 0, "cannot deduct past the baseline"
    assert e.years_lost <= e.baseline_remaining_years * 0.55 + 0.05


def test_baseline_falls_with_age_and_is_higher_for_women():
    from health_agent.scoring.actuarial import baseline_remaining_years as b

    assert b(30, Sex.MALE) > b(50, Sex.MALE) > b(70, Sex.MALE) > b(90, Sex.MALE)
    assert b(40, Sex.FEMALE) > b(40, Sex.MALE)
    assert b(40, Sex.MALE) < b(40, None) < b(40, Sex.FEMALE)
