"""Endpoint auth. Without it, anyone with the URL spends your API credits."""

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from health_agent.adapters.auth import protect

TOKEN = "t" * 32


def _app():
    async def ok(_request):
        return JSONResponse({"ok": True})

    return Starlette(routes=[
        Route("/mcp", ok, methods=["GET", "POST"]),
        Route("/healthz", ok, methods=["GET"]),
    ])


def test_missing_token_is_rejected(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    client = TestClient(protect(_app()))
    assert client.get("/mcp").status_code == 401


@pytest.mark.parametrize("header", [
    "", "Bearer", "Bearer ", "Bearer wrong", "Basic " + TOKEN, TOKEN,
])
def test_malformed_or_wrong_credentials_are_rejected(monkeypatch, header):
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    client = TestClient(protect(_app()))
    response = client.get("/mcp", headers={"Authorization": header} if header else {})
    assert response.status_code == 401, header


def test_correct_token_passes(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    client = TestClient(protect(_app()))
    assert client.get("/mcp", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_health_check_stays_public(monkeypatch):
    """A platform has to be able to see the container is alive."""
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    client = TestClient(protect(_app()))
    assert client.get("/healthz").status_code == 200


def test_deployment_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN is not set"):
        protect(_app(), require=True)


def test_a_weak_token_is_refused(monkeypatch):
    """A short token is worse than none -- it looks like security."""
    monkeypatch.setenv("MCP_AUTH_TOKEN", "hunter2")
    with pytest.raises(RuntimeError, match="too short"):
        protect(_app(), require=True)


def test_local_development_stays_open_but_warns(monkeypatch, caplog):
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    with caplog.at_level("WARNING"):
        client = TestClient(protect(_app()))
    assert client.get("/mcp").status_code == 200
    assert "OPEN" in caplog.text


def test_the_token_is_never_echoed(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    client = TestClient(protect(_app()))
    body = client.get("/mcp", headers={"Authorization": "Bearer wrong"}).text
    assert TOKEN not in body
