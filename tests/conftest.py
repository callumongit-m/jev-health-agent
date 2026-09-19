"""Test isolation.

Once a real .env exists, any test that does not inject a scorer would reach
for `get_backend("auto")`, find a live key, and hit a paid API. Tests must be
free, offline and deterministic, so every API key is stripped for the whole
session and the network is left unreachable by construction.
"""

from __future__ import annotations

import pytest

_PROVIDER_KEYS = (
    "TYPESAFE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_DECISIONS_URL",
    "ANTHROPIC_API_KEY",
    "TERRA_DEV_ID",
    "TERRA_API_KEY",
    "TERRA_WEBHOOK_SECRET",
    "TAVILY_API_KEY",
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Strip provider credentials so `auto` resolves to the offline backends.

    Individual tests that need a key present set it themselves with
    monkeypatch.setenv, which still works -- this only clears the default.
    """
    for key in _PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OFFLINE_REASONER", "1")
