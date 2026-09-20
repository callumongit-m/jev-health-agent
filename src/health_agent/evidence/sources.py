"""The evidence layer.

Fitty does not tell anyone what to do. It has a person's answers to a
questionnaire and nothing else -- no examination, no history, no bloods
unless they happened to have them. That is not enough to prescribe a course
of action, and pretending otherwise is the failure mode that matters here.

What it can honestly do is name which areas of someone's life carry the most
weight in their own picture, and show the research behind why those areas
matter at all. So this layer supplies published systematic reviews, from
Europe PMC, attached to the factors the report discusses.

Deliberately no guidance corpus. Guidance tells people what to do, and the
moment it is in the payload the report starts reading like instruction.
"""

from __future__ import annotations

from typing import Any

from health_agent.evidence import literature


def collect(
    condition_keys: list[str],
    factor_keys: list[str],
    *,
    with_research: bool = True,
) -> list[dict[str, Any]]:
    """Research for the factors this report is going to discuss.

    Scoped to the top findings. A wall of citations is not evidence, it is
    noise, and it makes the report look more authoritative than it is.
    """
    if not with_research:
        return []

    out: list[dict[str, Any]] = []
    for key in factor_keys:
        papers = literature.for_factor(key)
        if not papers:
            continue
        out.append(
            {
                "factor": key,
                "why_it_matters": (
                    "Published systematic reviews on this area. They are "
                    "here to show the area is worth attention, not to "
                    "prescribe anything."
                ),
                "research": papers,
            }
        )
    return out
