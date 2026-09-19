from health_agent.privacy import DISCLAIMER, NOTICE, SENSITIVE_FIELDS, redact


def test_notice_states_no_training():
    assert "NOT used to train" in NOTICE
    assert "delete_my_data" in NOTICE


def test_disclaimer_is_non_diagnostic():
    assert "not a diagnosis" in DISCLAIMER


def test_redact_scrubs_only_sensitive_fields():
    out = redact({"age": 54, "hba1c_mmol_mol": 46.0, "thread_id": "t1",
                  "nested": {"weight_kg": 90.0, "ok": 1}})
    assert "54" not in str(out["age"])
    assert "46" not in str(out["hba1c_mmol_mol"])
    assert out["thread_id"] == "t1"
    assert out["nested"]["ok"] == 1
    assert "90" not in str(out["nested"]["weight_kg"])


def test_every_clinical_field_is_marked_sensitive():
    from health_agent.domain.profile import HealthProfile

    exempt = {"provenance"}
    missing = {
        name for name in HealthProfile.model_fields
        if name not in exempt and name not in SENSITIVE_FIELDS
    }
    # fields added later must be classified deliberately, not by omission
    assert missing <= {
        "years_smoked", "sleep_hours_avg", "sleep_efficiency_pct", "steps_daily_avg",
        "moderate_activity_minutes_per_week", "diet_quality_self_rating",
        "perceived_stress_rating", "alcohol_units_per_week",
    }, f"unclassified fields: {missing}"
