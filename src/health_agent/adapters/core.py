"""The single entry point both adapters call. No transport knowledge here."""

from __future__ import annotations

import json
from typing import Any

from health_agent.domain.profile import HealthProfile
from health_agent.graph.graph import build_graph
from health_agent.graph.nodes import build_payload
from health_agent.scoring.scorer import RiskScorer

_GRAPH = None


def _graph(scorer: RiskScorer | None = None):
    global _GRAPH
    if scorer is not None:
        return build_graph(scorer)
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def assess(
    profile: HealthProfile | dict[str, Any],
    *,
    raw_text: str | None = None,
    thread_id: str | None = None,
    scorer: RiskScorer | None = None,
) -> dict[str, Any]:
    """Run the agent end to end and return the adapter-neutral payload."""
    if isinstance(profile, dict):
        profile = HealthProfile(**{k: v for k, v in profile.items() if v is not None})

    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    final = _graph(scorer).invoke(
        {"profile": profile, "raw_text": raw_text}, config=config
    )

    payload = build_payload(final)
    # life expectancy is produced by the actuarial tool inside the loop; surface
    # it at the top level so callers do not have to parse the prose
    for message in reversed(final.get("messages") or []):
        if getattr(message, "name", None) == "actuarial_calc":
            try:
                data = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                break
            if "error" not in data:
                payload["life_expectancy"] = data
            break
    return payload
