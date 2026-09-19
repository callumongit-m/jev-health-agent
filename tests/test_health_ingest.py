"""Remote Apple Health ingest -- a hosted server cannot read a phone."""

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from health_agent.adapters.a2a_server import build_app
from health_agent.adapters.sync_store import create_sync
from health_agent.ingest.health_sync import parse_payload

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def _when(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S %z")


def _metric(name, units, data):
    return {"name": name, "units": units, "data": data}


PAYLOAD = {"data": {"metrics": [
    _metric("step_count", "count", [
        {"date": _when(1), "qty": 2100}, {"date": _when(2), "qty": 2600}]),
    _metric("resting_heart_rate", "count/min", [{"date": _when(1), "qty": 79}]),
    _metric("weight_body_mass", "kg", [{"date": _when(1), "qty": 97.5}]),
    _metric("height", "m", [{"date": _when(9), "qty": 1.78}]),
    _metric("blood_pressure", "mmHg", [
        {"date": _when(2), "systolic": 147, "diastolic": 93}]),
    _metric("sleep_analysis", "hr", [{"date": _when(1), "asleep": 5.4, "inBed": 6.6}]),
    _metric("apple_exercise_time", "min", [{"date": _when(1), "qty": 8}]),
    _metric("unmapped_metric", "x", [{"date": _when(1), "qty": 1}]),
]}}


@pytest.fixture
def client():
    return TestClient(build_app())


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {create_sync()['token']}"}


# --- parser -------------------------------------------------------------

def test_parses_the_common_payload_shape():
    fields, _, report = parse_payload(PAYLOAD, now=NOW)
    assert fields["weight_kg"] == 97.5
    assert fields["height_cm"] == 178.0, "metres must convert"
    assert fields["resting_hr"] == 79
    assert fields["systolic_bp"] == 147 and fields["diastolic_bp"] == 93
    assert fields["steps_daily_avg"] == 2350
    assert fields["moderate_activity_minutes_per_week"] == 56
    assert fields["sleep_hours_avg"] == 5.4
    assert fields["sleep_efficiency_pct"] == pytest.approx(81.8, abs=0.5)
    assert "unmapped_metric" in report["skipped"]


@pytest.mark.parametrize("name", ["step_count", "stepCount", "Step Count", "STEPCOUNT"])
def test_metric_names_match_loosely(name):
    """The payload shape varies by app version; matching must not be brittle."""
    payload = {"data": {"metrics": [_metric(name, "count", [{"date": _when(1), "qty": 5000}])]}}
    fields, _, _ = parse_payload(payload, now=NOW)
    assert fields["steps_daily_avg"] == 5000


def test_imperial_units_convert():
    payload = {"data": {"metrics": [
        _metric("weight_body_mass", "lb", [{"date": _when(1), "qty": 200}]),
        _metric("height", "in", [{"date": _when(1), "qty": 70}]),
    ]}}
    fields, _, _ = parse_payload(payload, now=NOW)
    assert fields["weight_kg"] == pytest.approx(90.7, abs=0.2)
    assert fields["height_cm"] == pytest.approx(177.8, abs=0.2)


def test_old_samples_are_excluded_and_counted():
    payload = {"data": {"metrics": [
        _metric("resting_heart_rate", "count/min", [
            {"date": _when(1), "qty": 60}, {"date": _when(400), "qty": 90}])]}}
    fields, _, report = parse_payload(payload, now=NOW)
    assert fields["resting_hr"] == 60
    assert report["out_of_window_samples"] == 1


def test_sleep_falls_back_to_summing_phases():
    payload = {"data": {"metrics": [_metric("sleep_analysis", "hr", [
        {"date": _when(1), "core": 4.0, "deep": 1.2, "rem": 1.3}])]}}
    fields, _, _ = parse_payload(payload, now=NOW)
    assert fields["sleep_hours_avg"] == pytest.approx(6.5, abs=0.01)


def test_a_payload_with_no_metrics_is_reported_not_crashed():
    _, _, report = parse_payload({"nonsense": True}, now=NOW)
    assert "error" in report


# --- endpoints ----------------------------------------------------------

def test_push_requires_a_valid_token(client):
    assert client.post("/ingest/apple-health", json=PAYLOAD).status_code == 401
    assert client.post(
        "/ingest/apple-health", json=PAYLOAD,
        headers={"Authorization": "Bearer nope"}
    ).status_code == 401


def test_push_accepts_and_rescores(client, auth):
    body = client.post("/ingest/apple-health", json=PAYLOAD, headers=auth).json()
    assert body["accepted"] is True
    assert "weight_kg" in body["known_fields"]
    assert "NOT used to train" in body["privacy"]


def test_unrecognised_payload_is_diagnosable_not_a_400(client, auth):
    """The payload shape varies by app; the caller must be able to see why
    nothing was understood rather than getting an opaque rejection."""
    body = client.post(
        "/ingest/apple-health", headers=auth,
        json={"data": {"metrics": [_metric("mystery", "x", [{"date": _when(1), "qty": 5}])]}},
    ).json()
    assert body["accepted"] is False
    assert body["report"]["skipped"] == ["mystery"]


def test_repeated_pushes_accumulate_into_one_profile(client, auth):
    client.post("/ingest/apple-health", headers=auth, json={"data": {"metrics": [
        _metric("weight_body_mass", "kg", [{"date": _when(2), "qty": 97.5}])]}})
    second = client.post("/ingest/apple-health", headers=auth, json={"data": {"metrics": [
        _metric("resting_heart_rate", "count/min", [{"date": _when(1), "qty": 79}])]}}).json()
    assert {"weight_kg", "resting_hr"} <= set(second["known_fields"]), \
        "a later push must extend the profile, not replace it"


def test_zip_upload_path(client, auth):
    stamp = (NOW - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S %z")
    xml = (
        '<?xml version="1.0"?><HealthData>'
        f'<Record type="HKQuantityTypeIdentifierBodyMass" unit="kg" value="88.0" '
        f'startDate="{stamp}" endDate="{stamp}"/></HealthData>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("apple_health_export/export.xml", xml)
    buffer.seek(0)

    body = client.post(
        "/ingest/apple-health/upload", headers=auth,
        files={"file": ("export.zip", buffer, "application/zip")},
    ).json()
    assert body["accepted"] is True
    assert "weight_kg" in body["known_fields"]


def test_upload_requires_a_token(client):
    assert client.post(
        "/ingest/apple-health/upload",
        files={"file": ("export.zip", io.BytesIO(b"x"), "application/zip")},
    ).status_code == 401
