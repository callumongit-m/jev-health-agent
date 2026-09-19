"""Privacy notice and data-handling policy.

The notice is not documentation -- it is injected into every surface the agent
exposes (MCP server instructions, every MCP tool description, the A2A agent card,
and every response payload). ``tests/test_privacy.py`` asserts all four.
"""

from __future__ import annotations

NOTICE = (
    "Your health data is NOT used to train any AI model. It is sent only to the "
    "inference APIs required to answer this request, under their no-training terms, "
    "and is discarded when the session ends unless you explicitly opt in to reminders "
    "or trend tracking. You can erase everything at any time with `delete_my_data`."
)

DISCLAIMER = (
    "This is a risk estimate, not a diagnosis, and not a medical device. It "
    "reflects only what it was told and cannot see your full history. Use it "
    "to decide what is worth discussing with a clinician, not instead of one."
)

#: Profile field names that must never reach logs.
SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {
        "age", "sex", "height_cm", "weight_kg", "ethnicity", "postcode",
        "systolic_bp", "diastolic_bp", "resting_hr", "hrv_ms",
        "hba1c_mmol_mol", "fasting_glucose_mmol_l", "total_cholesterol_mmol_l",
        "hdl_mmol_l", "ldl_mmol_l", "triglycerides_mmol_l", "egfr", "alt_u_l",
        "smoking_status", "cigarettes_per_day", "alcohol_units_per_week",
        "family_history", "existing_conditions", "medications", "symptoms",
        "waist_cm", "on_bp_medication", "previously_high_glucose",
    }
)


def redact(payload: dict) -> dict:
    """Return a copy safe to log: sensitive values replaced with a type marker."""
    out = {}
    for key, value in payload.items():
        if key in SENSITIVE_FIELDS and value is not None:
            out[key] = f"<{type(value).__name__} redacted>"
        elif isinstance(value, dict):
            out[key] = redact(value)
        else:
            out[key] = value
    return out
