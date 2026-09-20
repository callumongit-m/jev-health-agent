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
    ranked_areas: tuple[dict[str, Any], ...]
    framing: dict[str, str]
    offer: str | None = None
    evidence: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "headline": self.headline,
            "must_include_verbatim": list(self.must_include),
            "must_not": list(self.must_not),
            "ranked_areas": list(self.ranked_areas),
            "framing": self.framing,
        }
        if self.offer:
            payload["ask_then_stop"] = self.offer
        if self.evidence:
            payload["evidence"] = list(self.evidence)
        return payload


#: Life expectancy is the one output nobody should receive unasked. Some
#: people want the number and act on it; for others it lands as a death
#: sentence they did not request. So it is offered, and only produced if
#: they say yes.
LIFE_EXPECTANCY_OFFER = (
    "Would you like to know your life expectancy, calculated from your data?"
)


# Areas, not instructions. Fitty has a questionnaire and nothing else --
# no examination, no history, often no bloods -- which is enough to say
# where the weight sits in someone's picture and not enough to tell them
# what to do about it. So each of these names the area and its direction,
# and stops short of a prescription.
AREA_TEXT: dict[str, tuple[str, str]] = {
    "smoking_burden": (
        "Smoking is not currently costing you anything here.",
        "Smoking is the heaviest single factor in your picture.",
    ),
    "adiposity": (
        "Body composition is working in your favour.",
        "Body composition is carrying real weight in your picture.",
    ),
    "activity_deficit": (
        "Your activity level is doing a lot of work for you.",
        "Physical activity is one of the larger gaps in your picture.",
    ),
    "sleep_debt": (
        "Your sleep is in a good range.",
        "Sleep is showing up as a meaningful factor for you.",
    ),
    "alcohol_burden": (
        "Alcohol is not a significant factor at your level.",
        "Alcohol intake is contributing measurably here.",
    ),
    "diet_quality": (
        "Diet is broadly working in your favour.",
        "Diet is one of the areas carrying weight in your picture.",
    ),
    "stress_load": (
        "Stress is not a significant factor for you at the moment.",
        "Sustained stress is registering as a real factor, not just a mood.",
    ),
}

#: Above this normalised severity, the area is a gap rather than a strength.
NEEDS_WORK = 0.34


def area_for(factor) -> str:
    variants = AREA_TEXT.get(factor.key)
    if variants is None:
        return f"{factor.label} is a factor in your picture."
    return variants[1] if factor.severity > NEEDS_WORK else variants[0]


#: Said on every report that reaches someone. This is the whole posture:
#: a questionnaire can locate where the weight sits, and cannot tell anyone
#: what to do about it.
SCOPE_STATEMENT = (
    "Fitty is a consultation, not professional medical advice. It has your "
    "answers and nothing else -- it has not examined you, has not seen your "
    "history, and does not have the information needed to tell you what you "
    "specifically should do. What it can do is show which areas carry the "
    "most weight in your own picture, and point at the research on why those "
    "areas matter. Deciding what to actually change, and how, is a "
    "conversation to have with a clinician."
)

def build(
    assessment: RiskAssessment,
    evidence: Evidence,
    life_expectancy: dict[str, Any] | None,
    urgent_guidance: str | None = None,
    guidance: list[dict[str, Any]] | None = None,
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

    areas: list[dict[str, Any]] = []
    per_factor = (life_expectancy or {}).get("per_factor_years") or {}
    for factor in assessment.top_factors(4):
        years = per_factor.get(factor.key, factor.expected_years_cost)
        if years <= 0.05:
            continue
        areas.append(
            {
                "area": area_for(factor),
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

    must += [SCOPE_STATEMENT, DISCLAIMER, NOTICE]

    offer = LIFE_EXPECTANCY_OFFER if life_expectancy else None

    return Presentation(
        headline=headline,
        must_include=tuple(must),
        must_not=(
            "Do not state a life expectancy, an age at death, or years "
            "remaining. Ask the `ask_then_stop` question instead and wait. "
            "Some people want that number; for others it arrives as a death "
            "sentence they did not ask for.",
            "Do not name a condition the person might have based on symptoms. "
            "Relay the urgency and what to tell a clinician, and stop there.",
            "Do not add up the per-area year figures. They overlap; the "
            "total is given separately.",
            "Do not turn an area into an instruction. Naming physical "
            "activity as a factor is supported; telling someone to walk for "
            "thirty minutes five days a week is not -- that is advice this "
            "has no basis for giving. Say what carries weight and leave the "
            "what-to-do-about-it to a clinician.",
            "Do not present the research as a recommendation. It shows the "
            "area matters; it does not say what this person should do.",
            "Do not state a probability that is not in `risk`, and do not "
            "round one into a stronger claim than it makes.",
            "Do not present a capped certainty as settled -- check "
            "`certainty_capped_by_evidence`.",
            "Do not recommend starting, stopping or changing any medication.",
            "Where `evidence` covers a point you are making, cite it with "
            "its source and link rather than asserting the fact yourself. "
            "Do not invent statistics, effect sizes or guideline numbers "
            "that are not in `evidence`.",
            "An entry's `research` list is there to show what a "
            "recommendation rests on -- name the review and link it. Do not "
            "quote findings from it, and do not treat a paper title as "
            "advice. The advice comes from the guidance; the research is "
            "the reason to believe it.",
        ),
        ranked_areas=tuple(areas),
        framing={
            "tone": "Direct and specific. This is someone's health, not a "
                    "sales page -- no false reassurance, no alarmism.",
            "order": "Urgent triage first if present, then what stands out "
                     "and why, then the areas by weight, then the caveats, "
                     "and end by asking the `ask_then_stop` question.",
            "length": "Aim for something readable in under two minutes.",
            "ask_then_stop": "Put this question at the very end and stop "
                             "there. Do not answer it yourself, do not hint "
                             "at the figure, and do not mention a life "
                             "expectancy, an age, or years remaining "
                             "anywhere in this reply. If they say yes, call "
                             "`life_expectancy`. If they say no, drop it and "
                             "never raise it again.",
        },
        offer=offer,
        evidence=tuple(guidance or ()),
    )
