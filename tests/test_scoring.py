from health_agent.domain.conditions import CONDITIONS, FACTORS
import os
import pytest
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


# --- OpenRouter transport ----------------------------------------------

def _fake_openrouter_response(questions: dict) -> dict:
    """A response in OpenRouter's documented Decisions shape."""
    answers = {}
    for key, q in questions.items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.61}
        else:
            legend = {str(i): lvl for i, lvl in enumerate(q["criteria"])}
            answers[key] = {
                "type": "score", "score": 1.99, "confidence": 0.99,
                "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99},
                "legend": legend,
            }
    return {
        "id": "gen-abc123", "model": "typesafe/jev-1.13", "provider": "TypeSafe",
        "answers": answers,
        "usage": {"input_tokens": 312, "output_tokens": 48, "cost": 0.000013},
    }


def test_openrouter_backend_parses_the_documented_response(monkeypatch):
    from health_agent.scoring.backend import OpenRouterBackend

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    backend = OpenRouterBackend()
    captured = {}

    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return _fake_openrouter_response(captured["json"]["questions"])

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json)
        return FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    result = backend.classify({"age": 54, "bmi": 30.9})

    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["json"]["model"] == "~typesafe/jev-latest"
    assert captured["json"]["state"] == {"age": 54, "bmi": 30.9}
    # 7 conditions + 7 factors + sufficiency + 2 acute screens
    assert len(captured["json"]["questions"]) == 17

    assert result.request_id == "gen-abc123"
    assert result.answers["t2d_10yr"].value == 0.61
    smoking = result.answers["smoking_burden"]
    assert smoking.kind == "score" and smoking.confidence == 0.99
    assert smoking.max_level == 3
    # score 1.99 rounds to level 2, which is the light-smoker band
    assert smoking.level_label == "Current light smoker, under ten a day"


def test_openrouter_backend_feeds_the_scorer(monkeypatch):
    """A transport swap must produce the same RiskAssessment shape as Jev."""
    from health_agent.scoring.backend import OpenRouterBackend

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    backend = OpenRouterBackend()

    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return _fake_openrouter_response(backend._questions)

    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, **kw: FakeResponse())

    assessment = RiskScorer(backend).score(HealthProfile(age=54, sex=Sex.MALE))
    assert len(assessment.conditions) == len(CONDITIONS)
    assert len(assessment.factors) == len(FACTORS)
    assert assessment.model == "typesafe/jev-1.13"


def test_openrouter_falls_back_to_the_alternate_path_on_404(monkeypatch):
    """The two published URL variants disagree; a 404 must retry, not fail."""
    from health_agent.scoring.backend import OpenRouterBackend, _OPENROUTER_URLS

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    backend = OpenRouterBackend()
    tried = []

    class Resp:
        def __init__(self, code): self.status_code = code
        def raise_for_status(self): pass
        def json(self): return _fake_openrouter_response(backend._questions)

    def fake_post(url, **kw):
        tried.append(url)
        return Resp(404 if url == _OPENROUTER_URLS[0] else 200)

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    backend.classify({"age": 54})
    assert tried == list(_OPENROUTER_URLS)
    assert backend._url == _OPENROUTER_URLS[1], "should remember what worked"

    backend.classify({"age": 55})
    assert tried[-1] == _OPENROUTER_URLS[1], "should not re-probe once learned"


def test_openrouter_requires_a_key(monkeypatch):
    from health_agent.scoring.backend import OpenRouterBackend

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterBackend()


def test_auto_prefers_typesafe_then_openrouter_then_fake(monkeypatch):
    from health_agent.scoring.backend import get_backend

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert get_backend("auto", seed=1).name.startswith("fake")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert get_backend("auto").name == "openrouter:~typesafe/jev-latest"


def test_env_file_is_loaded_and_empty_values_read_as_absent(tmp_path, monkeypatch):
    """An unfilled `KEY=` line in .env must not look like a configured key."""
    from dotenv import load_dotenv

    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=\nREASONING_MODEL=claude-sonnet-5\n")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("REASONING_MODEL", raising=False)
    load_dotenv(env, override=False)

    assert not os.getenv("TYPESAFE_API_KEY"), "empty value must be falsy"
    assert os.getenv("REASONING_MODEL") == "claude-sonnet-5"


def test_exported_variable_beats_the_env_file(tmp_path, monkeypatch):
    """override=False -- a real exported key must win over a stale .env line."""
    from dotenv import load_dotenv

    env = tmp_path / ".env"
    env.write_text("REASONING_MODEL=from-file\n")
    monkeypatch.setenv("REASONING_MODEL", "from-shell")
    load_dotenv(env, override=False)
    assert os.getenv("REASONING_MODEL") == "from-shell"


# --- unearned confidence ------------------------------------------------

def test_a_factor_with_no_inputs_cannot_claim_confidence():
    """Measured against real Jev: it returned 0.93 confidence for diet on a
    profile containing only age and sex. Everything downstream weights by
    confidence to stop unmeasured factors deducting years or becoming
    someone's top priority, so an unearned confidence defeats all of it."""
    from health_agent.domain.results import NO_EVIDENCE_CEILING

    bare = HealthProfile(age=45, sex=Sex.MALE)
    assessment = _scorer().score(bare)
    for factor in assessment.factors:
        assert factor.has_evidence is False, factor.key
        assert factor.confidence <= NO_EVIDENCE_CEILING, factor.key


def test_supplying_the_inputs_restores_confidence():
    profile = HealthProfile(
        age=45, sex=Sex.MALE, height_cm=180, weight_kg=95,
        smoking_status=SmokingStatus.CURRENT, cigarettes_per_day=20,
        diet_quality_self_rating=2,
    )
    assessment = _scorer().score(profile)
    assert assessment.factor("smoking_burden").has_evidence is True
    assert assessment.factor("adiposity").has_evidence is True
    # still nothing about their sleep
    assert assessment.factor("sleep_debt").has_evidence is False
    assert assessment.factor("sleep_debt").confidence <= 0.2


def test_an_unevidenced_factor_cannot_cost_meaningful_years():
    bare = HealthProfile(age=45, sex=Sex.MALE)
    assessment = _scorer().score(bare)
    assert all(f.expected_years_cost < 1.0 for f in assessment.factors)


def test_every_factor_declares_what_it_needs():
    """A factor with no declared inputs is always treated as evidenced, so a
    new one added without them silently bypasses the cap."""
    from health_agent.domain.conditions import FACTORS

    for spec in FACTORS:
        assert spec.evidence_fields, f"{spec.key} declares no evidence_fields"
        for field in spec.evidence_fields:
            assert field in HealthProfile.model_fields, f"{spec.key}: {field}"


# --- NHS guidance sources ----------------------------------------------

def test_guidance_falls_back_to_the_corpus_without_a_key(monkeypatch):
    """No key, no network, still cited advice. An assessment must not depend
    on a content API being up."""
    from health_agent.evidence.sources import guidance_for

    monkeypatch.delenv("NHS_API_KEY", raising=False)
    for key, kind in (("t2d_10yr", "condition"), ("smoking_burden", "factor")):
        entry = guidance_for(key, kind=kind)
        assert entry and entry["source"] == "NHS"
        assert entry["url"].startswith("https://www.nhs.uk/")
        assert entry["points"]
        assert not entry.get("live")


def test_a_failing_nhs_fetch_does_not_break_the_assessment(monkeypatch, tmp_path):
    """The whole point of the fallback."""
    import httpx

    from health_agent.evidence import sources

    monkeypatch.setenv("NHS_API_KEY", "fake-key")
    monkeypatch.setattr(sources, "_cached", lambda key: None)
    monkeypatch.setattr(sources, "_store", lambda key, payload: None)

    def explode(*args, **kwargs):
        raise httpx.ConnectError("NHS is down")

    monkeypatch.setattr(httpx, "get", explode)
    entry = sources.guidance_for("t2d_10yr", kind="condition")
    assert entry and entry["source"] == "NHS", "should have fallen back"


def test_every_registry_entry_has_an_nhs_slug():
    """A condition or factor added without one silently loses live guidance."""
    from health_agent.domain.conditions import CONDITIONS, FACTORS
    from health_agent.evidence.sources import (
        NHS_CONDITION_SLUGS, NHS_LIVE_WELL_SLUGS,
    )

    assert {c.key for c in CONDITIONS} == set(NHS_CONDITION_SLUGS)
    assert {f.key for f in FACTORS} == set(NHS_LIVE_WELL_SLUGS)
