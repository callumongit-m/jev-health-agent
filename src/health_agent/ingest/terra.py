"""Terra ingest -- Apple Health and wearables through one integration.

Terra normalises Apple Health, Fitbit, Oura, Garmin, Whoop and others into a
single schema delivered by webhook, so this is one receiver rather than an
OAuth flow and a parser per device.

Data arrives as daily samples; the agent wants a stable picture, so the
normaliser averages over a window and records how fresh each field is. A value
from three months ago should not be treated like one from this morning, which
is what `HealthProfile.provenance` is for.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Iterable

from health_agent.domain.profile import (
    COMPUTED_FIELDS,
    FieldMeta,
    HealthProfile,
    Source,
)

WIDGET_URL = "https://api.tryterra.co/v2/auth/generateWidgetSession"

#: Providers worth offering. Terra supports far more; these cover most people.
PROVIDERS = ("APPLE", "FITBIT", "OURA", "GARMIN", "WHOOP", "GOOGLE", "SAMSUNG")

#: How many days of samples to average over.
WINDOW_DAYS = 28


def widget_session(*, reference_id: str | None = None) -> dict[str, Any]:
    """A URL the person opens to connect a device.

    Returns a clearly-marked stub when Terra is not configured, rather than
    pretending to have produced a working link.
    """
    dev_id, api_key = os.getenv("TERRA_DEV_ID"), os.getenv("TERRA_API_KEY")
    if not (dev_id and api_key):
        return {
            "configured": False,
            "note": "Wearable connection is not configured on this deployment. "
                    "Health data can still be entered directly.",
            "providers": list(PROVIDERS),
        }

    import httpx

    response = httpx.post(
        WIDGET_URL,
        headers={"dev-id": dev_id, "x-api-key": api_key},
        json={
            "reference_id": reference_id or "anonymous",
            "providers": ",".join(PROVIDERS),
            "language": "en",
        },
        timeout=15.0,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "configured": True,
        "url": body.get("url"),
        "session_id": body.get("session_id"),
        "expires_in_seconds": body.get("expires_in"),
        "providers": list(PROVIDERS),
    }


def verify_webhook(body: bytes, signature_header: str | None) -> bool:
    """Terra signs webhooks as `t=<ts>,v1=<hex hmac of "ts.body">`.

    Returns False when unconfigured: an unverifiable webhook carrying health
    data must be rejected, not trusted by default.
    """
    secret = os.getenv("TERRA_WEBHOOK_SECRET")
    if not (secret and signature_header):
        return False

    parts = dict(
        piece.split("=", 1) for piece in signature_header.split(",") if "=" in piece
    )
    timestamp, provided = parts.get("t"), parts.get("v1")
    if not (timestamp and provided):
        return False

    expected = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, provided)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 2) if present else None


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def normalise(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, FieldMeta]]:
    """Terra webhook payload -> HealthProfile fields plus provenance.

    Handles the `daily`, `sleep` and `body` payload types; anything else is
    ignored rather than guessed at.
    """
    payload_type = payload.get("type", "")
    records: list[dict] = payload.get("data") or []
    if not records:
        return {}, {}

    device = _dig(payload, "user", "provider") or "unknown"
    observed = _latest_timestamp(records) or datetime.now(timezone.utc)
    fields: dict[str, Any] = {}

    if payload_type == "daily":
        fields["steps_daily_avg"] = _int(
            _mean(_dig(r, "distance_data", "steps") for r in records)
        )
        fields["moderate_activity_minutes_per_week"] = _weekly_minutes(records)
        fields["resting_hr"] = _int(
            _mean(_dig(r, "heart_rate_data", "summary", "resting_hr_bpm") for r in records)
        )
        fields["hrv_ms"] = _mean(
            _dig(r, "heart_rate_data", "summary", "avg_hrv_rmssd") for r in records
        )

    elif payload_type == "sleep":
        fields["sleep_hours_avg"] = _hours(
            _mean(
                _dig(r, "sleep_durations_data", "asleep", "duration_asleep_state_seconds")
                for r in records
            )
        )
        fields["sleep_efficiency_pct"] = _pct(
            _mean(_dig(r, "sleep_durations_data", "sleep_efficiency") for r in records)
        )

    elif payload_type == "body":
        fields["weight_kg"] = _mean(
            _dig(r, "measurements_data", "measurements", 0, "weight_kg")
            if isinstance(_dig(r, "measurements_data", "measurements"), list)
            else None
            for r in records
        ) or _mean(
            m.get("weight_kg")
            for r in records
            for m in (_dig(r, "measurements_data", "measurements") or [])
        )
        fields["systolic_bp"] = _int(
            _mean(
                s.get("systolic_bp")
                for r in records
                for s in (_dig(r, "blood_pressure_data", "blood_pressure_samples") or [])
            )
        )
        fields["diastolic_bp"] = _int(
            _mean(
                s.get("diastolic_bp")
                for r in records
                for s in (_dig(r, "blood_pressure_data", "blood_pressure_samples") or [])
            )
        )

    fields = {k: v for k, v in fields.items() if v is not None}
    provenance = {
        key: FieldMeta(source=Source.WEARABLE, observed_at=observed, device=device)
        for key in fields
    }
    return fields, provenance


def _latest_timestamp(records: list[dict]) -> datetime | None:
    stamps = []
    for record in records:
        raw = _dig(record, "metadata", "end_time") or _dig(record, "metadata", "start_time")
        if not raw:
            continue
        try:
            stamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            continue
    return max(stamps) if stamps else None


def _weekly_minutes(records: list[dict]) -> int | None:
    """Terra reports active duration per day; the profile wants weekly minutes."""
    daily = [
        _dig(r, "active_durations_data", "activity_seconds") for r in records
    ]
    mean_seconds = _mean(daily)
    return None if mean_seconds is None else int(round(mean_seconds / 60 * 7))


def _int(value: float | None) -> int | None:
    return None if value is None else int(round(value))


def _hours(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds / 3600, 2)


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 1) if value <= 1.0 else round(value, 1)


def merge_into(profile: HealthProfile, payload: dict[str, Any]) -> HealthProfile:
    """Apply a webhook to a profile. Device data wins over stale self-report,
    but never silently overwrites a fresher lab or clinical value."""
    fields, provenance = normalise(payload)
    if not fields:
        return profile

    keep: dict[str, Any] = {}
    for key, value in fields.items():
        existing = profile.provenance.get(key)
        if existing and existing.source in (Source.LAB, Source.CLINICAL):
            if existing.age_days() < provenance[key].age_days():
                continue  # the clinical reading is fresher; leave it alone
        keep[key] = value

    return HealthProfile(
        **(
            profile.model_dump(exclude=COMPUTED_FIELDS | {"provenance"})
            | keep
            | {"provenance": profile.provenance | {k: provenance[k] for k in keep}}
        )
    )
