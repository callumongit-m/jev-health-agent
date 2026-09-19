"""Tools available to the reasoning node.

The reason <-> tools cycle is the agentic part: probabilities differ per person,
so the evidence the model needs differs per person.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool

_LAST_CONTEXT: dict[str, Any] = {}


def set_context(*, assessment, profile) -> None:
    """The calculator tool needs the current assessment, but the LLM must not
    be able to pass made-up numbers into it -- so it is injected here rather
    than accepted as a tool argument."""
    _LAST_CONTEXT["assessment"] = assessment
    _LAST_CONTEXT["profile"] = profile


@tool
def actuarial_calc() -> str:
    """Compute this person's life expectancy from their risk assessment.

    Use this instead of doing the arithmetic yourself: it applies national life
    tables, weights each factor by how confident the classifier was, and
    accounts for overlap between factors. Returns the baseline, the adjusted
    estimate, years lost, years still recoverable, and a per-factor breakdown.
    """
    from health_agent.scoring.actuarial import estimate

    assessment = _LAST_CONTEXT.get("assessment")
    profile = _LAST_CONTEXT.get("profile")
    if assessment is None or profile is None:
        return json.dumps({"error": "no assessment in context"})

    result = estimate(assessment, profile)
    if result is None:
        return json.dumps(
            {
                "error": "insufficient data",
                "reason": "age is unknown, or the data is too thin for an "
                          "honest estimate. Say so rather than guessing.",
            }
        )
    return json.dumps(result.as_dict(), indent=2)


@tool
def health_evidence_search(query: str) -> str:
    """Search for current clinical evidence on a risk factor or intervention.

    Use for specifics you should not assert from memory: effect sizes, what a
    guideline currently recommends, or how much a given change is shown to help.
    Keep the query short and clinical, e.g. "HbA1c reduction from 150 minutes
    weekly moderate exercise".
    """
    if not os.getenv("TAVILY_API_KEY"):
        return json.dumps(
            {
                "unavailable": True,
                "note": "No search backend configured. Rely on well-established "
                        "general knowledge and say plainly when a number is "
                        "approximate rather than citing a source you cannot check.",
            }
        )
    try:
        from langchain_tavily import TavilySearch

        hits = TavilySearch(max_results=4, topic="general").invoke({"query": query})
        return json.dumps(hits)[:4000]
    except Exception as exc:  # a failed search must not kill the assessment
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool
def set_health_reminder(what: str, cadence: str) -> str:
    """Schedule a recurring nudge for the person, e.g. a weekly weigh-in or a
    quarterly HbA1c recheck.

    `cadence` is plain English: "daily", "weekly", "every 3 months".
    Only call this when the person has asked to be reminded.
    """
    from health_agent.store.reminders import add_reminder

    reminder = add_reminder(what=what, cadence=cadence)
    return json.dumps({"scheduled": True, **reminder})


TOOLS = [actuarial_calc, health_evidence_search, set_health_reminder]
