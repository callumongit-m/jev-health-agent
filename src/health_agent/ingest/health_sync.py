"""Remote Apple Health ingest.

A hosted server cannot read a phone. The practical options, in the order a
person will actually manage them:

1. **Push from the phone on a schedule.** An app like Health Auto Export, or
   an iOS Shortcuts automation, POSTs JSON to a URL on a cadence. This is the
   only one of the three that stays fresh without the person doing anything,
   which makes it the primary path.
2. **Upload an export.** One-off `export.zip`, parsed by `apple_health.py`.
   Zero setup, but a snapshot -- good for a first assessment.
3. **Type it in.** Always available, and the MCP tool schema already covers it.

Apple will not let anything read health data while the phone is locked, so
scheduled pushes land when the device is next unlocked rather than exactly on
time. Freshness here means "within a day", not "live".

The exact Health Auto Export payload is under-documented, so this parser is
deliberately tolerant: it matches metric names loosely, skips what it does
not recognise, and reports both. The response tells you what a real phone
actually sent.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from health_agent.domain.profile import (
    COMPUTED_FIELDS,
    FieldMeta,
    HealthProfile,
    Source,
)

#: Health Auto Export metric name -> (profile field, how to aggregate)
#: Names are matched case-insensitively with separators stripped, so
#: "step_count", "stepCount" and "Step Count" all land in the same place.
METRIC_MAP: dict[str, tuple[str, str]] = {
    "stepcount": ("steps_daily_avg", "mean"),
    "restingheartrate": ("resting_hr", "mean"),
    "heartratevariability": ("hrv_ms", "mean"),
    "heartratevariabilitysdnn": ("hrv_ms", "mean"),
    "weightbodymass": ("weight_kg", "last"),
    "bodymass": ("weight_kg", "last"),
    "height": ("height_cm", "last"),
    "appleexercisetime": ("moderate_activity_minutes_per_week", "weekly"),
    "exercisetime": ("moderate_activity_minutes_per_week", "weekly"),
    "bloodglucose": ("fasting_glucose_mmol_l", "mean"),
    "restingenergy": ("_ignored", "mean"),
}

#: Metrics that carry several values per entry rather than a single `qty`.
COMPOUND = {"bloodpressure", "sleepanalysis"}

_INT_FIELDS = {
    "resting_hr", "systolic_bp", "diastolic_bp", "steps_daily_avg",
    "moderate_activity_minutes_per_week",
}

WINDOW_DAYS = 28


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _parse_date(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _number(entry: dict, *keys: str) -> float | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _to_cm(value: float, units: str) -> float:
    units = units.lower()
    if units in ("m", "metres", "meters"):
        return value * 100
    if units in ("in", "inch", "inches"):
        return value * 2.54
    return value


def _to_kg(value: float, units: str) -> float:
    units = units.lower()
    if units in ("lb", "lbs", "pounds"):
        return value * 0.453592
    if units in ("st", "stone"):
        return value * 6.35029
    return value


def parse_payload(
    payload: dict[str, Any], *, window_days: int = WINDOW_DAYS, now: datetime | None = None
) -> tuple[dict[str, Any], dict[str, FieldMeta], dict[str, Any]]:
    """Returns (fields, provenance, report).

    ``report`` names what was recognised, skipped and out of window, so an
    unfamiliar payload is diagnosable from the HTTP response rather than
    silently producing nothing.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, list):
        return {}, {}, {"error": "no `data.metrics` array in payload"}

    buckets: dict[str, list[tuple[datetime, float]]] = {}
    daily: dict[str, dict[Any, float]] = {}
    sleep: list[tuple[datetime, float, float | None]] = []
    recognised: list[str] = []
    skipped: list[str] = []
    stale = 0

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        raw_name = str(metric.get("name", ""))
        name = _norm(raw_name)
        units = str(metric.get("units", ""))
        entries = metric.get("data") or []
        if not isinstance(entries, list) or not entries:
            continue

        if name not in METRIC_MAP and name not in COMPOUND:
            skipped.append(raw_name)
            continue

        used = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            when = _parse_date(entry.get("date") or entry.get("startDate"))
            if when is None:
                continue
            if when < cutoff:
                stale += 1
                continue

            if name == "bloodpressure":
                systolic = _number(entry, "systolic", "systolicValue")
                diastolic = _number(entry, "diastolic", "diastolicValue")
                if systolic:
                    buckets.setdefault("systolic_bp", []).append((when, systolic))
                if diastolic:
                    buckets.setdefault("diastolic_bp", []).append((when, diastolic))
                used = used or bool(systolic or diastolic)
                continue

            if name == "sleepanalysis":
                asleep = _number(entry, "asleep", "totalSleep", "value", "qty")
                if asleep is None:
                    parts = [
                        _number(entry, p) or 0.0 for p in ("core", "deep", "rem")
                    ]
                    asleep = sum(parts) or None
                in_bed = _number(entry, "inBed", "inBedTime")
                if asleep is not None:
                    sleep.append((when, asleep, in_bed))
                    used = True
                continue

            field, how = METRIC_MAP[name]
            if field == "_ignored":
                continue
            value = _number(entry, "qty", "value", "Avg", "avg")
            if value is None:
                continue
            if field == "height_cm":
                value = _to_cm(value, units)
            elif field == "weight_kg":
                value = _to_kg(value, units)
            elif field == "fasting_glucose_mmol_l" and "mg" in units.lower():
                value = value / 18.0182

            if how == "weekly":
                daily.setdefault(field, {})
                daily[field][when.date()] = daily[field].get(when.date(), 0.0) + value
            else:
                buckets.setdefault(field, []).append((when, value))
            used = True

        if used:
            recognised.append(raw_name)

    fields: dict[str, Any] = {}
    latest: dict[str, datetime] = {}

    for field, points in buckets.items():
        how = "last" if field in ("weight_kg", "height_cm") else "mean"
        if how == "last":
            when, value = max(points, key=lambda p: p[0])
        else:
            value = sum(v for _, v in points) / len(points)
            when = max(w for w, _ in points)
        fields[field] = round(value, 2)
        latest[field] = when

    for field, days in daily.items():
        if days:
            fields[field] = round(sum(days.values()) / len(days) * 7)
            latest[field] = now

    if sleep:
        fields["sleep_hours_avg"] = round(sum(a for _, a, _ in sleep) / len(sleep), 2)
        latest["sleep_hours_avg"] = max(w for w, _, _ in sleep)
        with_bed = [(a, b) for _, a, b in sleep if b]
        if with_bed:
            fields["sleep_efficiency_pct"] = round(
                sum(min(a / b, 1.0) for a, b in with_bed) / len(with_bed) * 100, 1
            )
            latest["sleep_efficiency_pct"] = latest["sleep_hours_avg"]

    for field in list(fields):
        if field in _INT_FIELDS:
            fields[field] = int(round(fields[field]))

    provenance = {
        field: FieldMeta(
            source=Source.WEARABLE,
            observed_at=latest.get(field, now),
            device="apple_health_sync",
        )
        for field in fields
    }
    report = {
        "recognised": sorted(set(recognised)),
        "skipped": sorted(set(skipped))[:40],
        "skipped_count": len(set(skipped)),
        "out_of_window_samples": stale,
        "fields": sorted(fields),
    }
    return fields, provenance, report


def merge_into(profile: HealthProfile, fields: dict, provenance: dict) -> HealthProfile:
    """Apply a sync without clobbering fresher clinical readings."""
    keep: dict[str, Any] = {}
    for field, value in fields.items():
        existing = profile.provenance.get(field)
        if existing and existing.source in (Source.LAB, Source.CLINICAL):
            if existing.age_days() < provenance[field].age_days():
                continue
        keep[field] = value
    return HealthProfile(
        **(
            profile.model_dump(exclude=COMPUTED_FIELDS | {"provenance"})
            | keep
            | {"provenance": profile.provenance | {k: provenance[k] for k in keep}}
        )
    )


def new_sync_token() -> str:
    """A capability token tying a phone to one assessment thread."""
    return secrets.token_urlsafe(24)
