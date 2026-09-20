"""Adaptive questioning.

A clinician does not work through a form -- they ask, listen, and let the
answer decide what comes next, then stop when the picture is clear. A
questionnaire cannot, because it is written before it knows anything about
you. Measured against a fixed order, asking the highest-value question
next reaches the same accuracy in roughly half the questions.
"""

import pytest

from health_agent.domain.consultation import (
    ALWAYS_WORTH_ASKING, FOUNDATIONS, MAX_QUESTIONS, PROMPTS, STOP_BELOW,
    next_step,
)
from health_agent.domain.profile import HealthProfile
from health_agent.scoring import FakeBackend, RiskScorer

ANSWERS = {
    "age": 23, "sex": "male", "height_cm": 181.6, "weight_kg": 85.0,
    "smoking_status": "never", "on_bp_medication": False,
    "previously_high_glucose": False, "waist_cm": 79.0, "systolic_bp": 118,
    "hba1c_mmol_mol": 34.0, "resting_hr": 58, "sleep_hours_avg": 7.5,
    "egfr": 98.0, "total_cholesterol_mmol_l": 4.4,
    "alcohol_units_per_week": 1.0, "moderate_activity_minutes_per_week": 425,
    "diet_quality_self_rating": 4, "eats_vegetables_daily": True,
    "perceived_stress_rating": 2,
}


@pytest.fixture
def scorer():
    return RiskScorer(FakeBackend(seed=8, noise=0.0))


def _run(scorer, answers, limit=20):
    known, skipped, asked = {}, [], []
    for _ in range(limit):
        step = next_step(HealthProfile(**known), scorer, skipped=skipped)
        if step.done:
            return asked, step
        field = step.question.field
        asked.append(field)
        value = answers.get(field)
        if value is None:
            skipped.append(field)   # "I don't know" is a real answer
        else:
            known[field] = value
    raise AssertionError("consultation never finished")


def test_it_finishes(scorer):
    asked, step = _run(scorer, ANSWERS)
    assert step.done
    assert step.reason


def test_the_essentials_come_first(scorer):
    """There is no estimate at all without these, so nothing clever should
    be asked before them."""
    asked, _ = _run(scorer, ANSWERS)
    foundations = [f for f, _ in FOUNDATIONS]
    assert asked[: len(foundations)] == foundations


def test_it_stops_rather_than_exhausting_the_list(scorer):
    """The point of asking adaptively is being allowed to stop."""
    asked, step = _run(scorer, ANSWERS)
    assert len(asked) < len(ANSWERS), "asked for everything, so it is a form"
    assert "meaningfully change" in step.reason or "Enough" in step.reason


def test_nothing_is_asked_twice(scorer):
    asked, _ = _run(scorer, ANSWERS)
    assert len(asked) == len(set(asked))


def test_every_question_is_in_words(scorer):
    """A raw field name reaching someone -- 'What is your alcohol units per
    week?' -- is what this guards against."""
    known, skipped = {}, []
    for _ in range(20):
        step = next_step(HealthProfile(**known), scorer, skipped=skipped)
        if step.done:
            break
        ask = step.question.ask
        assert "_" not in ask, ask
        assert ask.endswith("?") or "?" in ask, ask
        assert len(ask) > 15, ask
        value = ANSWERS.get(step.question.field)
        if value is None:
            skipped.append(step.question.field)
        else:
            known[step.question.field] = value


def test_the_free_questions_are_asked_before_the_ranking_spends_calls(scorer):
    asked, _ = _run(scorer, ANSWERS)
    free = [f for f, _ in ALWAYS_WORTH_ASKING]
    positions = {f: asked.index(f) for f in free if f in asked}
    assert positions, "the free questions were never asked"
    assert max(positions.values()) < len(FOUNDATIONS) + len(free) + 3


def test_someone_who_knows_nothing_still_terminates(scorer):
    """Every answer unknown must not loop forever."""
    asked, step = _run(scorer, {}, limit=30)
    assert step.done
    assert len(asked) <= MAX_QUESTIONS + len(FOUNDATIONS) + 4


def test_a_question_explains_why_it_is_being_asked(scorer):
    step = next_step(HealthProfile(age=40, sex="male", height_cm=175,
                                   weight_kg=88), scorer)
    assert not step.done
    assert len(step.question.why) > 20


def test_prompts_exist_for_everything_that_can_be_asked(scorer):
    """A field the bands can require but PROMPTS has no entry for reaches
    the person as a field name."""
    from health_agent.domain.evidence import (
        GLYCAEMIC, LIFESTYLE_FIELDS, RECALL_FIELDS, VITALS_FIELDS,
    )

    askable = set(LIFESTYLE_FIELDS) | set(VITALS_FIELDS) | set(GLYCAEMIC)
    covered = (
        set(PROMPTS)
        | {f for f, _ in FOUNDATIONS}
        | {f for f, _ in ALWAYS_WORTH_ASKING}
        | set(RECALL_FIELDS)
    )
    assert askable <= covered, f"no prompt for: {askable - covered}"


def test_i_dont_know_moves_on_rather_than_looping(scorer):
    """Not knowing is one of the commonest answers here, and it used to
    leave the field unknown so the same question came back forever."""
    known, skipped, asked = {}, [], []
    for _ in range(25):
        step = next_step(HealthProfile(**known), scorer, skipped=skipped)
        if step.done:
            break
        field = step.question.field
        assert field not in asked, f"asked for {field} twice"
        asked.append(field)
        # answers only the four essentials, shrugs at everything else
        if field in ("age", "sex", "height_cm", "weight_kg"):
            known[field] = ANSWERS[field]
        else:
            skipped.append(field)
    else:
        raise AssertionError("never terminated when everything was skipped")


def test_skipping_something_required_ends_honestly(scorer):
    """If what is missing is not something they know offhand, say so rather
    than producing a number anyway."""
    known = {"age": 55, "sex": "male", "height_cm": 178.0, "weight_kg": 92.0}
    skipped = ["smoking_status", "on_bp_medication", "previously_high_glucose",
               "eats_vegetables_daily", "alcohol_units_per_week",
               "moderate_activity_minutes_per_week", "sleep_hours_avg",
               "diet_quality_self_rating"]
    step = next_step(HealthProfile(**known), scorer, skipped=skipped)
    assert step.done
    assert "not enough" in step.reason.lower()
