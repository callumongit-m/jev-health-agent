from health_agent.domain.conditions import CONDITIONS, FACTORS
from health_agent.domain.profile import HealthProfile, Sex
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
