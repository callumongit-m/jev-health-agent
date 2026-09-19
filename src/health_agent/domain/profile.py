"""The canonical shape of a person's health data.

Every ingest source (MCP tool arguments, Terra webhook, Apple Health export)
normalises into ``HealthProfile``. The Jev scorer consumes it directly as state,
so field names are written to be self-describing to a model reading them raw.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, computed_field, model_validator


#: Computed fields are serialised by model_dump but rejected on the way back
#: in, so every dump-then-reconstruct must exclude them. Referencing this set
#: rather than spelling them out means adding a computed field cannot quietly
#: break every round trip in the codebase.
COMPUTED_FIELDS: frozenset[str] = frozenset({"bmi", "waist_to_height"})


class Sex(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class SmokingStatus(StrEnum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class Source(StrEnum):
    """Where a field came from. Drives trust and staleness handling."""

    SELF_REPORTED = "self_reported"
    WEARABLE = "wearable"
    LAB = "lab"
    CLINICAL = "clinical"
    DERIVED = "derived"


class FieldMeta(BaseModel):
    """Provenance and recency for a single field."""

    source: Source
    observed_at: datetime
    device: str | None = None

    def age_days(self, *, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        observed = self.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return (now - observed).total_seconds() / 86_400


class HealthProfile(BaseModel):
    """A person's health picture. Every clinical field is optional -- the whole
    point of the gate node is coping with partial data."""

    model_config = {"extra": "forbid"}

    # --- demographics -------------------------------------------------
    age: int | None = Field(default=None, ge=0, le=120)
    sex: Sex | None = None
    height_cm: float | None = Field(default=None, gt=0, le=260)
    weight_kg: float | None = Field(default=None, gt=0, le=500)
    ethnicity: str | None = None
    #: Waist matters more than BMI for metabolic risk, and unlike BMI it does
    #: not mistake muscle for fat -- which matters a great deal for anyone who
    #: lifts. Waist-to-height ratio above 0.5 is the usual threshold.
    waist_cm: float | None = Field(default=None, gt=30, le=250)

    # --- vitals -------------------------------------------------------
    systolic_bp: int | None = Field(default=None, ge=50, le=300)
    diastolic_bp: int | None = Field(default=None, ge=30, le=200)
    resting_hr: int | None = Field(default=None, ge=25, le=220)
    hrv_ms: float | None = Field(default=None, ge=0, le=400)

    # --- labs ---------------------------------------------------------
    hba1c_mmol_mol: float | None = Field(default=None, ge=15, le=200)
    fasting_glucose_mmol_l: float | None = Field(default=None, ge=1, le=40)
    total_cholesterol_mmol_l: float | None = Field(default=None, ge=1, le=20)
    hdl_mmol_l: float | None = Field(default=None, ge=0.1, le=6)
    ldl_mmol_l: float | None = Field(default=None, ge=0.1, le=15)
    triglycerides_mmol_l: float | None = Field(default=None, ge=0.1, le=30)
    egfr: float | None = Field(default=None, ge=1, le=200)
    alt_u_l: float | None = Field(default=None, ge=1, le=1000)

    # --- lifestyle ----------------------------------------------------
    smoking_status: SmokingStatus | None = None
    cigarettes_per_day: int | None = Field(default=None, ge=0, le=100)
    years_smoked: int | None = Field(default=None, ge=0, le=90)
    alcohol_units_per_week: float | None = Field(default=None, ge=0, le=200)
    moderate_activity_minutes_per_week: int | None = Field(default=None, ge=0, le=5000)
    steps_daily_avg: int | None = Field(default=None, ge=0, le=100_000)
    sleep_hours_avg: float | None = Field(default=None, ge=0, le=24)
    sleep_efficiency_pct: float | None = Field(default=None, ge=0, le=100)
    diet_quality_self_rating: int | None = Field(
        default=None, ge=1, le=5, description="1 = poor, 5 = excellent"
    )
    perceived_stress_rating: int | None = Field(
        default=None, ge=1, le=5, description="1 = none, 5 = severe"
    )

    # --- recall questions -------------------------------------------
    # These need no test and no equipment, and they carry most of what a
    # validated non-invasive diabetes score (FINDRISC) asks for. They are
    # what makes an estimate possible for the ~half of over-40s who have
    # never had, or cannot recall, their bloods.
    on_bp_medication: bool | None = Field(
        default=None, description="ever prescribed medication for blood pressure"
    )
    previously_high_glucose: bool | None = Field(
        default=None,
        description="ever told by a clinician their blood sugar was high, "
                    "including in pregnancy",
    )
    eats_vegetables_daily: bool | None = Field(
        default=None, description="vegetables, fruit or berries most days"
    )

    # --- history ------------------------------------------------------
    family_history: list[str] = Field(default_factory=list)
    existing_conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)

    # --- metadata -----------------------------------------------------
    provenance: dict[str, FieldMeta] = Field(default_factory=dict, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bmi(self) -> float | None:
        if self.height_cm is None or self.weight_kg is None:
            return None
        return round(self.weight_kg / (self.height_cm / 100) ** 2, 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def waist_to_height(self) -> float | None:
        """A better adiposity signal than BMI, and the one that stops a
        muscular person being read as overweight."""
        if self.waist_cm is None or self.height_cm is None:
            return None
        return round(self.waist_cm / self.height_cm, 3)

    @model_validator(mode="after")
    def _bp_ordering(self) -> Self:
        if (
            self.systolic_bp is not None
            and self.diastolic_bp is not None
            and self.diastolic_bp >= self.systolic_bp
        ):
            raise ValueError("diastolic_bp must be lower than systolic_bp")
        return self

    # --- helpers ------------------------------------------------------
    def known_fields(self) -> set[str]:
        """Clinical fields that actually carry a value."""
        return {
            name
            for name in type(self).model_fields
            if name != "provenance" and _is_present(getattr(self, name))
        }

    def completeness(self) -> float:
        """Fraction of clinical fields populated. Cheap pre-filter before Jev."""
        total = len(type(self).model_fields) - 1  # exclude provenance
        return round(len(self.known_fields()) / total, 3)

    def stale_fields(self, *, max_age_days: float = 30.0) -> set[str]:
        return {
            name
            for name, meta in self.provenance.items()
            if meta.age_days() > max_age_days
        }

    def to_state(self) -> dict[str, Any]:
        """The dict handed to Jev. Omits empty values so the model is not asked
        to reason about a wall of nulls, and appends recency where it matters."""
        state: dict[str, Any] = {}
        for name in type(self).model_fields:
            if name == "provenance":
                continue
            value = getattr(self, name)
            if not _is_present(value):
                continue
            state[name] = value.value if isinstance(value, StrEnum) else value
        if self.bmi is not None:
            state["bmi"] = self.bmi
        if self.waist_to_height is not None:
            state["waist_to_height_ratio"] = self.waist_to_height
            state["_note_on_bmi"] = (
                "Waist-to-height is present, so prefer it over BMI for "
                "adiposity: BMI cannot distinguish muscle from fat."
            )
        stale = self.stale_fields()
        if stale:
            state["_note_possibly_outdated"] = sorted(stale)
        return state


def rebuild(profile: "HealthProfile", **updates: Any) -> "HealthProfile":
    """A copy with `updates` applied, safe across computed fields."""
    return HealthProfile(
        **(
            profile.model_dump(exclude=COMPUTED_FIELDS | {"provenance"})
            | updates
        ),
        provenance=updates.pop("provenance", profile.provenance),
    )


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, set, str)) and len(value) == 0:
        return False
    return True
