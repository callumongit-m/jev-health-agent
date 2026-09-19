"""Both adapters over one core: same behaviour, different transport."""

import json

import pytest

from health_agent.privacy import NOTICE

COMPLETE = {
    "age": 62, "sex": "male", "height_cm": 175, "weight_kg": 86,
    "systolic_bp": 158, "diastolic_bp": 96, "hba1c_mmol_mol": 41,
    "smoking_status": "current", "cigarettes_per_day": 25,
    "alcohol_units_per_week": 30, "moderate_activity_minutes_per_week": 20,
    "sleep_hours_avg": 6.0, "diet_quality_self_rating": 2,
    "perceived_stress_rating": 4,
}


def _unwrap(result):
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


# --- MCP ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_exposes_the_expected_tools():
    from health_agent.adapters.mcp_server import server

    names = {t.name for t in await server.list_tools()}
    assert names == {
        "assess_health", "import_apple_health", "connect_wearable",
        "set_health_reminder", "delete_my_data",
    }


@pytest.mark.asyncio
async def test_privacy_notice_is_on_every_mcp_surface():
    """The notice is structural, not documentation."""
    from health_agent.adapters.mcp_server import server

    assert NOTICE in (server.instructions or "")
    for tool in await server.list_tools():
        assert NOTICE in (tool.description or ""), tool.name


@pytest.mark.asyncio
async def test_mcp_schema_is_the_intake_form():
    """The host LLM fills this from conversation, so it must be complete."""
    from health_agent.adapters.mcp_server import server

    tool = next(t for t in await server.list_tools() if t.name == "assess_health")
    props = (tool.input_schema or {}).get("properties", {})
    for field in ("age", "sex", "height_cm", "weight_kg", "smoking_status",
                  "hba1c_mmol_mol", "family_history", "notes"):
        assert field in props, f"{field} missing from the MCP intake schema"
    assert not (tool.input_schema or {}).get("required"), "every field must be optional"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (COMPLETE, "complete"),
        ({"age": 41}, "needs_input"),
        ({**COMPLETE, "notes": "crushing chest pain on stairs"}, "seek_care"),
    ],
    ids=["complete", "needs_input", "seek_care"],
)
async def test_mcp_routes_all_three_paths(args, expected):
    from health_agent.adapters.mcp_server import server

    payload = _unwrap(await server.call_tool("assess_health", args))
    assert payload["status"] == expected
    assert NOTICE in payload["privacy"]
    if expected != "complete":
        assert "risk" not in payload, "numbers leaked on a non-scoring path"


# --- A2A ---------------------------------------------------------------

def test_agent_card_declares_a_reachable_transport():
    """Without supported_interfaces a peer fails with 'no compatible transports'."""
    from health_agent.adapters.a2a_server import build_agent_card

    card = build_agent_card("http://localhost:9000")
    assert card.supported_interfaces
    assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
    assert card.capabilities.streaming is True
    assert NOTICE in card.description
    assert [s.id for s in card.skills] == ["health-risk-assessment"]


def test_a2a_parses_json_and_prose_input():
    from health_agent.adapters.a2a_server import _parse_input

    fields, notes = _parse_input('{"age": 54, "notes": "I smoke"}')
    assert fields == {"age": 54} and notes == "I smoke"

    fields, notes = _parse_input("I'm 41 and pretty sedentary")
    assert fields == {} and notes == "I'm 41 and pretty sedentary"

    fields, notes = _parse_input("{not json")
    assert fields == {} and notes == "{not json"


# --- Terra webhook -----------------------------------------------------

def _signed(body: bytes, secret: str = "testsecret") -> dict:
    import hashlib, hmac, time

    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return {"content-type": "application/json", "terra-signature": f"t={ts},v1={sig}"}


DAILY = {
    "type": "daily",
    "user": {"provider": "OURA", "reference_id": "t-test"},
    "data": [{
        "metadata": {"end_time": "2026-09-18T00:00:00Z"},
        "distance_data": {"steps": 1800},
        "heart_rate_data": {"summary": {"resting_hr_bpm": 82}},
        "active_durations_data": {"activity_seconds": 120},
    }],
}


@pytest.fixture
def webhook_client(monkeypatch):
    from fastapi.testclient import TestClient
    from health_agent.adapters.a2a_server import build_app

    monkeypatch.setenv("TERRA_WEBHOOK_SECRET", "testsecret")
    return TestClient(build_app())


def test_unsigned_webhook_is_rejected(webhook_client):
    """Health data arriving unverifiable must not be trusted."""
    body = json.dumps(DAILY).encode()
    assert webhook_client.post("/webhooks/terra", content=body,
                               headers={"content-type": "application/json"}).status_code == 401


def test_forged_signature_is_rejected(webhook_client):
    body = json.dumps(DAILY).encode()
    headers = _signed(body, secret="wrong-secret")
    assert webhook_client.post("/webhooks/terra", content=body,
                               headers=headers).status_code == 401


def test_wearable_update_refreshes_an_existing_thread(webhook_client):
    """The answer to 'how does this not go stale'."""
    from health_agent.adapters import webhook
    from health_agent.domain.profile import HealthProfile, Sex, SmokingStatus
    from health_agent.scoring import FakeBackend, RiskScorer
    from health_agent.adapters.core import assess

    before_profile = HealthProfile(
        age=54, sex=Sex.MALE, height_cm=178, weight_kg=98, systolic_bp=148,
        diastolic_bp=92, hba1c_mmol_mol=46, smoking_status=SmokingStatus.FORMER,
        alcohol_units_per_week=22, sleep_hours_avg=5.8, diet_quality_self_rating=2,
        perceived_stress_rating=4, moderate_activity_minutes_per_week=240,
        steps_daily_avg=11000,
    )
    webhook.remember("t-test", before_profile)

    scorer = RiskScorer(FakeBackend(seed=1, noise=0.0))
    before = assess(before_profile, scorer=scorer)

    body = json.dumps(DAILY).encode()
    result = webhook_client.post("/webhooks/terra", content=body,
                                 headers=_signed(body)).json()
    assert result["updated"] is True
    assert "steps_daily_avg" in result["fields_changed"]

    after_profile = webhook._PROFILES["t-test"]
    assert after_profile.steps_daily_avg == 1800
    assert after_profile.provenance["steps_daily_avg"].source.value == "wearable"

    after = assess(after_profile, scorer=scorer)
    assert (after["factors"]["activity_deficit"]["years_cost"]
            > before["factors"]["activity_deficit"]["years_cost"]), \
        "becoming sedentary should raise the activity deficit"


def test_unknown_thread_is_not_silently_created(webhook_client):
    payload = {**DAILY, "user": {**DAILY["user"], "reference_id": "never-seen"}}
    body = json.dumps(payload).encode()
    result = webhook_client.post("/webhooks/terra", content=body,
                                 headers=_signed(body)).json()
    assert result["updated"] is False


# --- Apple Health import over MCP --------------------------------------

@pytest.mark.asyncio
async def test_apple_health_import_tool(tmp_path):
    from datetime import datetime, timedelta, timezone

    from health_agent.adapters.mcp_server import server

    now = datetime.now(timezone.utc)
    stamp = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S %z")
    xml = (
        '<?xml version="1.0"?><HealthData>'
        f'<Record type="HKQuantityTypeIdentifierBodyMass" unit="kg" value="88.2" '
        f'startDate="{stamp}" endDate="{stamp}"/>'
        f'<Record type="HKQuantityTypeIdentifierRestingHeartRate" unit="count/min" '
        f'value="71" startDate="{stamp}" endDate="{stamp}"/>'
        "</HealthData>"
    )
    export = tmp_path / "export.xml"
    export.write_text(xml)

    payload = _unwrap(await server.call_tool("import_apple_health", {"path": str(export)}))
    assert payload["imported"] is True
    assert payload["fields"]["weight_kg"] == 88.2
    assert payload["fields"]["resting_hr"] == 71
    assert NOTICE in payload["privacy"]
    assert "age or sex" in payload["next_step"], "must prompt for the baseline inputs"


@pytest.mark.asyncio
async def test_apple_health_import_explains_the_remote_case(tmp_path):
    """A hosted server cannot read the person's disk; say so usefully."""
    from health_agent.adapters.mcp_server import server

    payload = _unwrap(
        await server.call_tool("import_apple_health", {"path": str(tmp_path / "nope.zip")})
    )
    assert payload["imported"] is False
    assert "running remotely" in payload["hint"]


@pytest.mark.asyncio
async def test_apple_health_import_handles_an_empty_export(tmp_path):
    from health_agent.adapters.mcp_server import server

    export = tmp_path / "export.xml"
    export.write_text('<?xml version="1.0"?><HealthData></HealthData>')
    payload = _unwrap(await server.call_tool("import_apple_health", {"path": str(export)}))
    assert payload["imported"] is False
    assert "no usable records" in payload["error"]
