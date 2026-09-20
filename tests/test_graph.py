"""Graph routing: the three terminal states and the tool cycle."""

import json

import pytest

from health_agent.adapters.core import assess
from health_agent.domain.profile import HealthProfile, Sex, SmokingStatus
from health_agent.scoring import FakeBackend, RiskScorer

COMPLETE = HealthProfile(
    age=54, sex=Sex.MALE, height_cm=178, weight_kg=98, systolic_bp=148,
    diastolic_bp=92, hba1c_mmol_mol=46, smoking_status=SmokingStatus.FORMER,
    alcohol_units_per_week=22, moderate_activity_minutes_per_week=40,
    sleep_hours_avg=5.8, diet_quality_self_rating=2, perceived_stress_rating=4,
    on_bp_medication=False, previously_high_glucose=False,
    eats_vegetables_daily=True,
)


@pytest.fixture
def scorer():
    return RiskScorer(FakeBackend(seed=5, noise=0.0))


@pytest.fixture
def narrated(monkeypatch):
    """Opt in to writing the report ourselves. Off by default -- the calling
    model normally does it, on its own budget.

    Settings is a frozen dataclass and graph.py binds it at import, so this
    replaces the binding in that module rather than mutating the object.
    """
    from dataclasses import replace

    from health_agent.graph import graph as graph_module

    monkeypatch.setattr(
        graph_module, "SETTINGS",
        replace(graph_module.SETTINGS, response_mode="narrated"),
    )


def test_complete_profile_reaches_a_scored_answer(scorer, narrated):
    out = assess(COMPLETE, scorer=scorer)
    assert out["status"] == "complete"
    assert out["risk"] and out["factors"]
    assert out["answer"]


def test_life_expectancy_comes_from_the_tool_not_the_prose(scorer, narrated):
    out = assess(COMPLETE, scorer=scorer)
    le = out["life_expectancy"]
    assert le["adjusted_remaining_years"] < le["baseline_remaining_years"]
    assert "ONS national life table" in le["method"]


def test_identical_input_gives_identical_life_expectancy(scorer, narrated):
    """The reason the LLM is handed a calculator instead of doing the maths."""
    first = assess(COMPLETE, scorer=scorer)["life_expectancy"]
    second = assess(COMPLETE, scorer=scorer)["life_expectancy"]
    assert first["adjusted_remaining_years"] == second["adjusted_remaining_years"]


def test_thin_profile_asks_instead_of_guessing(scorer):
    out = assess(HealthProfile(age=41, sex=Sex.FEMALE), scorer=scorer)
    assert out["status"] == "needs_input"
    assert out["questions"]
    assert "life_expectancy" not in out


def test_acute_symptom_short_circuits_before_scoring(scorer):
    out = assess(COMPLETE, raw_text="crushing chest pain radiating to my arm",
                 scorer=scorer)
    assert out["status"] == "seek_care"
    assert "risk" not in out, "must not score someone describing an emergency"
    assert "emergency services" in out["answer"]


def test_symptoms_on_the_profile_are_screened_too(scorer):
    profile = COMPLETE.model_copy(update={"symptoms": ["slurred speech since today"]})
    assert assess(profile, scorer=scorer)["status"] == "seek_care"


@pytest.mark.parametrize("payload_key", ["privacy", "disclaimer"])
@pytest.mark.parametrize(
    "case",
    [
        {"profile": COMPLETE},
        {"profile": HealthProfile(age=41), "raw_text": None},
        {"profile": COMPLETE, "raw_text": "chest pain"},
    ],
    ids=["complete", "needs_input", "seek_care"],
)
def test_every_terminal_state_carries_the_notices(scorer, case, payload_key):
    assert assess(**case, scorer=scorer)[payload_key]


def test_tool_cycle_actually_runs(scorer, narrated):
    """reason -> tools -> reason is the agentic loop; assert it happened."""
    from health_agent.graph.graph import build_graph

    final = build_graph(scorer).invoke({"profile": COMPLETE, "raw_text": None})
    assert "actuarial_calc" in final["tool_calls_made"]
    assert any(getattr(m, "name", None) == "actuarial_calc" for m in final["messages"])


# --- data mode: the calling model writes it up -------------------------

def test_data_mode_returns_a_contract_and_spends_nothing(scorer):
    """Default. The caller is already a capable model in the person's own
    subscription -- running a second one to write prose it will rewrite is
    cost for no benefit."""
    out = assess(COMPLETE, scorer=scorer)
    assert out["status"] == "complete"
    assert out.get("answer") is None, "no prose should be generated"
    assert out["presentation"]["ranked_areas"]
    assert out["risk"]


# --- life expectancy is offered, never volunteered ---------------------

def test_life_expectancy_is_withheld_until_asked_for(scorer):
    """Some people want that number and act on it. For others it lands as a
    death sentence they did not request, so it is never sent unasked."""
    out = assess(COMPLETE, scorer=scorer)
    assert "life_expectancy" not in out
    contract = out["presentation"]
    assert contract["ask_then_stop"].startswith("Would you like to know")

    blob = json.dumps(contract).lower()
    for leaked in ("remaining_years", "age_at_death", "life expectancy you",
                   "you will live"):
        assert leaked not in blob, f"contract leaked {leaked!r}"


def test_the_contract_forbids_volunteering_it(scorer):
    out = assess(COMPLETE, scorer=scorer)
    forbidden = " ".join(out["presentation"]["must_not"]).lower()
    assert "life expectancy" in forbidden and "age at death" in forbidden
    assert "wait" in forbidden, "must_not should say to wait for an answer"

    framing = out["presentation"]["framing"]["ask_then_stop"].lower()
    assert "do not answer it yourself" in framing
    assert "stop there" in framing


def test_years_recoverable_is_still_offered_because_it_motivates(scorer):
    """Recoverable years are the actionable half and are not a mortality
    figure, so they are not withheld."""
    out = assess(COMPLETE, scorer=scorer)
    verbatim = " ".join(out["presentation"]["must_include_verbatim"])
    assert "recoverable" in verbatim


def test_the_calculator_still_runs_behind_the_contract(scorer):
    """Withholding the figure must not mean losing the per-action years."""
    out = assess(COMPLETE, scorer=scorer)
    assert any(a["years_recoverable"] > 0 for a in out["presentation"]["ranked_areas"])


# --- cited guidance ----------------------------------------------------

def test_areas_carry_research_not_guidance(scorer):
    """Fitty shows why an area matters. It does not say what to do about it."""
    out = assess(COMPLETE, scorer=scorer)
    evidence = out["presentation"]["evidence"]
    assert evidence
    for entry in evidence:
        assert entry["research"], entry
        for paper in entry["research"]:
            assert paper["url"].startswith("https://")
            assert paper["cited_by"] > 0


def test_the_scope_statement_is_always_included(scorer):
    """The whole posture: a questionnaire locates the weight, it cannot say
    what to do about it."""
    out = assess(COMPLETE, scorer=scorer)
    verbatim = " ".join(out["presentation"]["must_include_verbatim"])
    assert "not professional medical advice" in verbatim
    assert "has not examined you" in verbatim
    assert "conversation to have with a clinician" in verbatim


def test_the_contract_forbids_turning_an_area_into_an_instruction(scorer):
    out = assess(COMPLETE, scorer=scorer)
    forbidden = " ".join(out["presentation"]["must_not"]).lower()
    assert "turn an area into an instruction" in forbidden
    assert "research as a recommendation" in forbidden


def test_the_contract_forbids_inventing_statistics(scorer):
    out = assess(COMPLETE, scorer=scorer)
    forbidden = " ".join(out["presentation"]["must_not"]).lower()
    assert "invent statistics" in forbidden
    assert "cite it" in forbidden


def test_contract_supplies_the_safety_critical_wording_verbatim(scorer):
    """A host can ignore rules. It is far less likely to mangle a sentence
    handed to it, so the parts that must be right are pre-written."""
    out = assess(COMPLETE, scorer=scorer)
    verbatim = " ".join(out["presentation"]["must_include_verbatim"])
    assert "not a diagnosis" in verbatim
    assert "NOT used to train" in verbatim
    assert "do not sum them" in verbatim.lower()


def test_contract_forbids_the_things_that_actually_hurt(scorer):
    out = assess(COMPLETE, scorer=scorer)
    forbidden = " ".join(out["presentation"]["must_not"]).lower()
    assert "name a condition" in forbidden
    assert "add up" in forbidden
    assert "medication" in forbidden


def test_triage_leads_the_contract_when_present(scorer):
    out = assess(
        {**{k: v for k, v in COMPLETE.model_dump(exclude={"bmi", "waist_to_height", "provenance"}).items() if v not in (None, [], {})}},
        raw_text="headache at night and throwing up in the mornings",
        scorer=scorer,
    )
    assert out["symptom_patterns"] == ["raised_intracranial_pressure"]
    assert out["presentation"]["must_include_verbatim"][0] == out["act_on_this_first"]


def test_narrated_mode_still_produces_prose(scorer, narrated):
    out = assess(COMPLETE, scorer=scorer)
    assert out.get("answer"), "narrated mode must still write a report"
    assert "presentation" not in out


# --- enrichments must not be able to take the report down --------------

def test_a_failing_projection_does_not_lose_the_assessment(scorer, monkeypatch):
    """Found the hard way: an exhausted API balance during the projection
    raised straight through and cost the whole report. The scored
    assessment is what someone came for."""
    from health_agent.domain import projection as projection_rules

    def explode(*args, **kwargs):
        raise RuntimeError("402 Payment Required")

    monkeypatch.setattr(projection_rules, "build", explode)
    out = assess(COMPLETE, scorer=scorer)
    assert out["status"] == "complete"
    assert out["risk"], "the assessment itself must survive"
    assert "projection" not in out["presentation"]


def test_a_failing_suggestion_ranking_does_not_lose_the_assessment(scorer, monkeypatch):
    from health_agent.domain import worth_measuring as measuring_rules

    def explode(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(measuring_rules, "rank", explode)
    out = assess(COMPLETE, scorer=scorer)
    assert out["status"] == "complete"
    assert out["risk"]
    assert "worth_measuring" not in out["presentation"]


def test_both_failing_still_gives_a_usable_report(scorer, monkeypatch):
    from health_agent.domain import projection as projection_rules
    from health_agent.domain import worth_measuring as measuring_rules

    def explode(*args, **kwargs):
        raise RuntimeError("everything is down")

    monkeypatch.setattr(projection_rules, "build", explode)
    monkeypatch.setattr(measuring_rules, "rank", explode)
    out = assess(COMPLETE, scorer=scorer)
    assert out["risk"] and out["presentation"]["risk_table"]["rows"]
    verbatim = " ".join(out["presentation"]["must_include_verbatim"])
    assert "not professional medical advice" in verbatim
