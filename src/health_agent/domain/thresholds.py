"""Deterministic clinical thresholds.

Some numbers are not a matter of probability. An HbA1c of 48 mmol/mol meets
the diagnostic criterion for diabetes; a blood pressure of 185/120 is a
hypertensive crisis. Those are definitions, and running them through an
estimator can only lose information.

This was found by an eval rather than reasoned about in advance: asked how
likely a 27-year-old with an HbA1c of 60 was to *develop* type 2 diabetes,
the classifier said 12 percent. It was not wrong -- someone at 60 already
has diabetes, so "develop" is the wrong question -- but a person in that
position being told 12 percent is dangerously reassuring.

So thresholds are checked in code, before and alongside the estimate, and
they say what the value means rather than how likely something is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from health_agent.domain.profile import HealthProfile


class Urgency(StrEnum):
    EMERGENCY = "emergency"       # today, unscheduled
    URGENT = "urgent"             # within days
    ROUTINE = "routine"           # worth an appointment
    MONITOR = "monitor"           # worth knowing, not acting on today


@dataclass(frozen=True, slots=True)
class Finding:
    field: str
    value: float
    urgency: Urgency
    meaning: str
    action: str
    source: str

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "urgency": str(self.urgency),
            "meaning": self.meaning,
            "action": self.action,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Threshold:
    field: str
    #: value at or beyond which this fires
    limit: float
    #: True when higher is worse
    above: bool
    urgency: Urgency
    meaning: str
    action: str
    source: str


#: Ordered worst-first per field, so the most serious match wins.
THRESHOLDS: tuple[Threshold, ...] = (
    # --- blood pressure ---
    Threshold(
        "systolic_bp", 180, True, Urgency.EMERGENCY,
        "A systolic reading of 180 or above is a hypertensive crisis.",
        "Get this checked today. If you also have chest pain, breathlessness, "
        "a severe headache or visual changes, call 999.",
        "NICE NG136",
    ),
    Threshold(
        "diastolic_bp", 120, True, Urgency.EMERGENCY,
        "A diastolic reading of 120 or above is a hypertensive crisis.",
        "Get this checked today.",
        "NICE NG136",
    ),
    Threshold(
        "systolic_bp", 160, True, Urgency.URGENT,
        "A systolic reading of 160 or above is stage 2 hypertension -- "
        "high enough that lifestyle change alone rarely brings it down.",
        "See a GP within the next week or two; this usually needs treating.",
        "NICE NG136",
    ),
    Threshold(
        "systolic_bp", 140, True, Urgency.ROUTINE,
        "This is in the hypertensive range.",
        "Worth confirming with a repeat reading and a GP appointment -- a "
        "single high reading is not a diagnosis.",
        "NICE NG136",
    ),
    # --- glycaemia ---
    Threshold(
        "hba1c_mmol_mol", 48, True, Urgency.URGENT,
        "An HbA1c of 48 mmol/mol (6.5%) or above meets the diagnostic "
        "threshold for diabetes. This is not a risk of developing it.",
        "See a GP to have this confirmed and to discuss treatment. Bring the "
        "result with you.",
        "WHO / NICE NG28",
    ),
    Threshold(
        "hba1c_mmol_mol", 42, True, Urgency.ROUTINE,
        "An HbA1c between 42 and 47 mmol/mol is in the prediabetes range.",
        "Worth a GP conversation. At this level, structured lifestyle change "
        "often prevents progression -- the NHS Diabetes Prevention Programme "
        "takes referrals here.",
        "NICE NG38 / WHO",
    ),
    Threshold(
        "fasting_glucose_mmol_l", 7.0, True, Urgency.URGENT,
        "A fasting glucose of 7.0 mmol/L or above meets the diagnostic "
        "threshold for diabetes.",
        "See a GP to have this confirmed.",
        "WHO",
    ),
    Threshold(
        "fasting_glucose_mmol_l", 6.1, True, Urgency.ROUTINE,
        "A fasting glucose between 6.1 and 6.9 mmol/L indicates impaired "
        "fasting glucose.",
        "Worth a GP conversation about prevention.",
        "WHO",
    ),
    # --- kidney ---
    Threshold(
        "egfr", 30, False, Urgency.URGENT,
        "An eGFR below 30 indicates severely reduced kidney function "
        "(stage 4 chronic kidney disease).",
        "See a GP promptly; this usually warrants specialist input.",
        "NICE NG203",
    ),
    Threshold(
        "egfr", 60, False, Urgency.ROUTINE,
        "An eGFR below 60 indicates reduced kidney function.",
        "Worth a GP conversation, and a repeat test to confirm.",
        "NICE NG203",
    ),
    # --- lipids ---
    Threshold(
        "total_cholesterol_mmol_l", 7.5, True, Urgency.ROUTINE,
        "A total cholesterol of 7.5 mmol/L or above can indicate familial "
        "hypercholesterolaemia, which is inherited and treatable.",
        "Worth raising with a GP, especially if heart disease runs in your "
        "family at a young age.",
        "NICE CG71",
    ),
    # --- liver ---
    Threshold(
        "alt_u_l", 150, True, Urgency.ROUTINE,
        "An ALT this far above the normal range warrants investigation.",
        "See a GP to find the cause.",
        "NICE NG50",
    ),
)


def check(profile: HealthProfile) -> list[Finding]:
    """Diagnostic thresholds this person's values already meet.

    One finding per field -- the most serious. Listing every band a value
    passes through is noise.
    """
    findings: list[Finding] = []
    seen: set[str] = set()

    for threshold in THRESHOLDS:
        if threshold.field in seen:
            continue
        value = getattr(profile, threshold.field, None)
        if value is None:
            continue
        crossed = (
            value >= threshold.limit if threshold.above else value < threshold.limit
        )
        if not crossed:
            continue
        seen.add(threshold.field)
        findings.append(
            Finding(
                field=threshold.field,
                value=float(value),
                urgency=threshold.urgency,
                meaning=threshold.meaning,
                action=threshold.action,
                source=threshold.source,
            )
        )

    order = {
        Urgency.EMERGENCY: 0, Urgency.URGENT: 1,
        Urgency.ROUTINE: 2, Urgency.MONITOR: 3,
    }
    return sorted(findings, key=lambda f: order[f.urgency])


def highest_urgency(findings: list[Finding]) -> Urgency | None:
    return findings[0].urgency if findings else None
