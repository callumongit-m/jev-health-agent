"""The presentation contract: what a host LLM must say, and must not.

When this agent is reached over MCP, the caller is already a capable model
sitting in the person's own subscription. Paying for a second model to write
prose the first one will rewrite anyway is pure cost for no benefit -- so by
default the agent returns structured findings plus this contract, and the
host does the writing.

The catch is that a contract is a request, not a guarantee. A host can ignore
instructions, and some of what must be right here is safety-critical: not
summing overlapping year figures, not naming a condition from symptoms, not
presenting a capped certainty as settled.

So the safety-critical wording is supplied as **finished sentences to quote**
rather than rules to follow. Quoting is a much lower bar than complying, and
it degrades gracefully: a host that ignores the rules but reproduces the text
still tells the person the right thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from health_agent.domain.evidence import Evidence
from health_agent.domain.results import RiskAssessment
from health_agent.privacy import DISCLAIMER, NOTICE


@dataclass(frozen=True, slots=True)
class Presentation:
    headline: str
    must_include: tuple[str, ...]
    must_not: tuple[str, ...]
    ranked_actions: tuple[dict[str, Any], ...]
    framing: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "must_include_verbatim": list(self.must_include),
            "must_not": list(self.must_not),
            "ranked_actions": list(self.ranked_actions),
            "framing": self.framing,
        }


#: Phrasings per factor, as (already-doing-well, needs-work). Telling a former
#: smoker to "stop smoking" is the kind of thing that costs you their trust in
#: everything else on the list, so severity picks the wording.
ACTION_TEXT: dict[str, tuple[str, str]] = {
    "smoking_burden": (
        "Stay stopped. The excess risk keeps falling the longer you do, and "
        "most of it goes within about fifteen years.",
        "Stop smoking. Nothing else on this list comes close.",
    ),
    "adiposity": (
        "Hold your current waist measurement -- it is already working for you.",
        "Bring your waist down. Under half your height is the target.",
    ),
    "activity_deficit": (
        "Keep the training up; it is doing a lot of work here.",
        "Get to 150 minutes of moderate activity a week -- a brisk 30-minute "
        "walk, five days.",
    ),
    "sleep_debt": (
        "Protect the sleep you are getting; it is in the right range.",
        "Protect seven to nine hours at consistent times. Start with what is "
        "cutting it short rather than trying to fix everything.",
    ),
    "alcohol_burden": (
        "Alcohol is not costing you anything at this level.",
        "Bring alcohol within 14 units a week, with several drink-free days.",
    ),
    "diet_quality": (
        "Diet is broadly working; look at the specific gaps rather than "
        "overhauling it.",
        "Shift toward whole foods, vegetables and fibre; cut ultra-processed "
        "food.",
    ),
    "stress_load": (
        "Stress is not a significant cost for you at the moment.",
        "Treat sustained stress as a health problem rather than a personality "
        "trait -- it carries a measurable cost here.",
    ),
}

#: Above this normalised severity, the factor needs work rather than holding.
NEEDS_WORK = 0.34


def action_for(factor) -> str:
    variants = ACTION_TEXT.get(factor.key)
    if variants is None:
        return f"Improve {factor.label.lower()}."
    return variants[1] if factor.severity > NEEDS_WORK else variants[0]


def build(
    assessment: RiskAssessment,
    evidence: Evidence,
    life_expectancy: dict[str, Any] | None,
    urgent_guidance: str | None = None,
) -> Presentation:
    top = assessment.top_conditions(3)
    must: list[str] = []

    # Triage leads, always, and verbatim.
    if urgent_guidance:
        must.append(urgent_guidance)

    headline = (
        f"Based on {evidence.tier.label}. "
        + (
            f"Highest: {top[0].label.lower()} at {top[0].probability:.0%}."
            if top
            else ""
        )
    )

    if evidence.confidence_ceiling < 1.0:
        must.append(
            f"This estimate is based on {evidence.tier.label}, so treat the "
            f"numbers as indicative rather than settled."
        )
    if evidence.missing_recommended:
        readable = {
            "systolic_bp": "a blood pressure reading",
            "hba1c_mmol_mol": "an HbA1c blood test",
            "total_cholesterol_mmol_l": "a cholesterol test",
            "on_bp_medication": "whether you take blood pressure medication",
            "previously_high_glucose": "whether you have been told your blood "
                                       "sugar was high",
            "eats_vegetables_daily": "whether you eat vegetables most days",
        }
        items = [readable.get(f, f) for f in evidence.missing_recommended][:2]
        must.append(f"Getting {' and '.join(items)} would sharpen this materially.")

    actions: list[dict[str, Any]] = []
    per_factor = (life_expectancy or {}).get("per_factor_years") or {}
    for factor in assessment.top_factors(4):
        years = per_factor.get(factor.key, factor.expected_years_cost)
        if years <= 0.05:
            continue
        actions.append(
            {
                "action": action_for(factor),
                "years_recoverable": round(years, 2),
                "current": factor.level_label,
                "confidence": factor.confidence,
            }
        )

    if life_expectancy:
        total = life_expectancy.get("years_recoverable")
        must.append(
            f"About {total} years look recoverable in total. That is less "
            f"than the individual figures added together, because these "
            f"factors overlap -- do not sum them."
        )

    must += [DISCLAIMER, NOTICE]

    return Presentation(
        headline=headline,
        must_include=tuple(must),
        must_not=(
            "Do not name a condition the person might have based on symptoms. "
            "Relay the urgency and what to tell a clinician, and stop there.",
            "Do not add up the per-action year figures. They overlap; the "
            "total is given separately.",
            "Do not state a probability that is not in `risk`, and do not "
            "round one into a stronger claim than it makes.",
            "Do not present a capped certainty as settled -- check "
            "`certainty_capped_by_evidence`.",
            "Do not recommend starting, stopping or changing any medication.",
        ),
        ranked_actions=tuple(actions),
        framing={
            "tone": "Direct and specific. This is someone's health, not a "
                    "sales page -- no false reassurance, no alarmism.",
            "order": "Urgent triage first if present, then what stands out "
                     "and why, then the ranked actions, then the caveats.",
            "length": "Aim for something readable in under two minutes.",
        },
    )
