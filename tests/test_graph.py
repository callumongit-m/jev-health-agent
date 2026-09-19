"""Graph routing: the three terminal states and the tool cycle."""

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


def test_life_expectancy_comes_from_the_tool_not_the_prose(scorer):
    out = assess(COMPLETE, scorer=scorer)
    le = out["life_expectancy"]
    assert le["adjusted_remaining_years"] < le["baseline_remaining_years"]
    assert "ONS national life table" in le["method"]


def test_identical_input_gives_identical_life_expectancy(scorer):
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
    assert out["presentation"]["ranked_actions"]
    assert out["risk"] and out["life_expectancy"]


def test_life_expectancy_is_still_computed_without_an_llm(scorer):
    """The calculator is ours; only the writing is delegated."""
    out = assess(COMPLETE, scorer=scorer)
    le = out["life_expectancy"]
    assert le["adjusted_remaining_years"] < le["baseline_remaining_years"]
    assert "ONS national life table" in le["method"]


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
