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
