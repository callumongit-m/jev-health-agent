"""Apple Health export ingest -- the free wearables path.

Terra's floor is ~$499/mo, which is not a sensible dependency for this. An
Apple Health export costs nothing, needs no OAuth and no keys, and because
Apple Health already aggregates most wearables (Oura, Whoop, Garmin, Fitbit
and the Watch all write into it) one export covers the same ground for a
single user.

What it does not give you is a live feed -- it is a snapshot the person
exports by hand. The Terra path stays in the tree for when that matters.

Export it on iPhone: Health -> profile picture -> Export All Health Data.
That produces export.zip containing export.xml, which can be hundreds of MB,
so this parses it as a stream and never loads the document into memory.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree

from health_agent.domain.profile import FieldMeta, HealthProfile, Source

#: Apple record type -> (profile field, aggregation)
#: "mean" averages the samples in the window; "sum_per_day" totals each day
#: then averages those daily totals; "last" takes the most recent value.
RECORD_MAP: dict[str, tuple[str, str]] = {
    "HKQuantityTypeIdentifierBodyMass": ("weight_kg", "last"),
    "HKQuantityTypeIdentifierHeight": ("height_cm", "last"),
    "HKQuantityTypeIdentifierRestingHeartRate": ("resting_hr", "mean"),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": ("hrv_ms", "mean"),
    "HKQuantityTypeIdentifierStepCount": ("steps_daily_avg", "sum_per_day"),
    "HKQuantityTypeIdentifierBloodPressureSystolic": ("systolic_bp", "mean"),
    "HKQuantityTypeIdentifierBloodPressureDiastolic": ("diastolic_bp", "mean"),
    "HKQuantityTypeIdentifierAppleExerciseTime": (
        "moderate_activity_minutes_per_week", "sum_per_day"
    ),
    "HKQuantityTypeIdentifierBloodGlucose": ("fasting_glucose_mmol_l", "mean"),
}

#: Fields that must be whole numbers on HealthProfile.
_INT_FIELDS = {
    "resting_hr", "systolic_bp", "diastolic_bp", "steps_daily_avg",
    "moderate_activity_minutes_per_week",
}

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
_ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
}
_IN_BED_VALUE = "HKCategoryValueSleepAnalysisInBed"

#: Only look at the recent past -- a value from three years ago is not this
#: person's current picture.
WINDOW_DAYS = 28


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:  # Apple writes "2026-09-18 07:14:22 +0100"
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None


def _iter_records(path: Path) -> Iterator[dict[str, str]]:
    """Stream <Record> elements, from a .zip or a bare export.xml.

    Elements are cleared as we go, so memory stays flat regardless of export
    size -- these files routinely run to hundreds of megabytes.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(
                (n for n in archive.namelist() if n.endswith("export.xml")), None
            )
            if name is None:
                raise ValueError(f"no export.xml inside {path.name}")
            with archive.open(name) as handle:
                yield from _stream(handle)
    else:
        with path.open("rb") as handle:
            yield from _stream(handle)


def _stream(handle) -> Iterator[dict[str, str]]:
    for _event, element in ElementTree.iterparse(handle, events=("end",)):
        if element.tag == "Record":
            yield element.attrib
        element.clear()


def _to_cm(value: float, unit: str) -> float:
    return value * 100 if unit in ("m",) else value


def _glucose_to_mmol(value: float, unit: str) -> float:
    # Apple may report mg/dL; the profile wants mmol/L.
    return round(value / 18.0182, 2) if "mg" in unit.lower() else value


def parse_export(
    path: str | Path, *, window_days: int = WINDOW_DAYS, now: datetime | None = None
) -> tuple[dict[str, Any], dict[str, FieldMeta]]:
    """Read an Apple Health export into HealthProfile fields plus provenance."""
    path = Path(path)
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    samples: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    per_day: dict[str, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    sleep_by_night: dict[Any, float] = defaultdict(float)
    in_bed_by_night: dict[Any, float] = defaultdict(float)
    latest: dict[str, datetime] = {}

    for attrs in _iter_records(path):
        record_type = attrs.get("type", "")
        start = _parse_dt(attrs.get("startDate"))
        if start is None or start < cutoff:
            continue

        if record_type == SLEEP_TYPE:
            end = _parse_dt(attrs.get("endDate"))
            if end is None:
                continue
            hours = (end - start).total_seconds() / 3600
            night = (start - timedelta(hours=12)).date()
            value = attrs.get("value", "")
            if value in _ASLEEP_VALUES:
                sleep_by_night[night] += hours
                latest["sleep_hours_avg"] = max(
                    latest.get("sleep_hours_avg", end), end
                )
            elif value == _IN_BED_VALUE:
                in_bed_by_night[night] += hours
            continue

        mapped = RECORD_MAP.get(record_type)
        if mapped is None:
            continue
        field, how = mapped
        try:
            value = float(attrs.get("value", ""))
        except ValueError:
            continue

        unit = attrs.get("unit", "")
        if field == "height_cm":
            value = _to_cm(value, unit)
        elif field == "fasting_glucose_mmol_l":
            value = _glucose_to_mmol(value, unit)

        if how == "sum_per_day":
            per_day[field][start.date()] += value
        else:
            samples[field].append((start, value))
        latest[field] = max(latest.get(field, start), start)

    fields: dict[str, Any] = {}

    for field, points in samples.items():
        how = next(h for t, (f, h) in RECORD_MAP.items() if f == field)
        if how == "last":
            fields[field] = round(max(points, key=lambda p: p[0])[1], 2)
        else:
            fields[field] = round(sum(v for _, v in points) / len(points), 2)

    for field, days in per_day.items():
        if not days:
            continue
        daily_mean = sum(days.values()) / len(days)
        # exercise minutes are asked for weekly, steps daily
        fields[field] = (
            round(daily_mean * 7) if field == "moderate_activity_minutes_per_week"
            else round(daily_mean)
        )

    if sleep_by_night:
        fields["sleep_hours_avg"] = round(
            sum(sleep_by_night.values()) / len(sleep_by_night), 2
        )
        shared = set(sleep_by_night) & set(in_bed_by_night)
        if shared:
            efficiency = [
                min(sleep_by_night[n] / in_bed_by_night[n], 1.0) * 100
                for n in shared
                if in_bed_by_night[n] > 0
            ]
            if efficiency:
                fields["sleep_efficiency_pct"] = round(
                    sum(efficiency) / len(efficiency), 1
                )
                latest.setdefault("sleep_efficiency_pct", latest["sleep_hours_avg"])

    for field in list(fields):
        if field in _INT_FIELDS:
            fields[field] = int(round(fields[field]))

    provenance = {
        field: FieldMeta(
            source=Source.WEARABLE,
            observed_at=latest.get(field, now),
            device="apple_health_export",
        )
        for field in fields
    }
    return fields, provenance


def merge_into(profile: HealthProfile, path: str | Path, **kwargs) -> HealthProfile:
    """Apply an export to a profile, without clobbering fresher clinical data."""
    fields, provenance = parse_export(path, **kwargs)
    if not fields:
        return profile

    keep: dict[str, Any] = {}
    for field, value in fields.items():
        existing = profile.provenance.get(field)
        if existing and existing.source in (Source.LAB, Source.CLINICAL):
            if existing.age_days() < provenance[field].age_days():
                continue
        keep[field] = value

    return HealthProfile(
        **(
            profile.model_dump(exclude={"bmi", "provenance"})
            | keep
            | {"provenance": profile.provenance | {k: provenance[k] for k in keep}}
        )
    )
