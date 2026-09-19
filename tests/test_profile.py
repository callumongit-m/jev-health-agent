import pytest
from pydantic import ValidationError

from health_agent.domain.profile import HealthProfile, Sex, SmokingStatus


def test_bmi_computed_and_absent_without_inputs():
    assert HealthProfile(height_cm=180, weight_kg=81).bmi == 25.0
    assert HealthProfile(height_cm=180).bmi is None


def test_diastolic_must_be_below_systolic():
    with pytest.raises(ValidationError, match="diastolic"):
        HealthProfile(systolic_bp=120, diastolic_bp=130)


def test_to_state_omits_empty_and_flattens_enums():
    state = HealthProfile(age=40, sex=Sex.FEMALE, smoking_status=SmokingStatus.NEVER).to_state()
    assert state == {"age": 40, "sex": "female", "smoking_status": "never"}
    assert "family_history" not in state, "empty list should not reach the model"


def test_completeness_rises_with_data():
    sparse = HealthProfile(age=40).completeness()
    rich = HealthProfile(age=40, sex=Sex.MALE, height_cm=180, weight_kg=80,
                         systolic_bp=120, diastolic_bp=78, hba1c_mmol_mol=35).completeness()
    assert 0 < sparse < rich < 1


def test_waist_to_height_prefers_waist_over_bmi_for_a_muscular_build():
    """BMI cannot tell muscle from fat. Someone who lifts can sit at BMI 26
    with a perfectly healthy waist, and the state handed to the classifier
    has to say so or they get read as overweight."""
    lifter = HealthProfile(height_cm=181.6, weight_kg=85, waist_cm=82)
    assert lifter.bmi == 25.8, "BMI alone reads as overweight"
    assert lifter.waist_to_height == 0.452, "waist says otherwise"

    state = lifter.to_state()
    assert state["waist_to_height_ratio"] == 0.452
    assert "muscle from fat" in state["_note_on_bmi"]


def test_waist_absent_leaves_the_state_unchanged():
    state = HealthProfile(height_cm=180, weight_kg=90).to_state()
    assert "waist_to_height_ratio" not in state
    assert "_note_on_bmi" not in state


def test_computed_fields_survive_a_dump_and_rebuild():
    """Every ingest path dumps a profile and reconstructs it. A new computed
    field must not silently break all of them."""
    from health_agent.domain.profile import COMPUTED_FIELDS

    original = HealthProfile(age=21, height_cm=181.6, weight_kg=85, waist_cm=82)
    rebuilt = HealthProfile(
        **original.model_dump(exclude=COMPUTED_FIELDS | {"provenance"})
    )
    assert rebuilt == original
    assert COMPUTED_FIELDS == {"bmi", "waist_to_height"}, \
        "add new computed fields here or round trips will break"
