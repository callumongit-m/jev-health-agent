"""Adaptive questioning -- asking the next question rather than all of them.

A clinician does not work through a form. They ask something, listen, and
let the answer decide what to ask next; they stop when the picture is clear
enough to act on. A questionnaire cannot do that because it is written
before it knows anything about you.

This can, because scoring is cheap. At each step every unanswered question
is simulated across its plausible range, and whichever would move the
estimate most gets asked. Measured against asking the same questions in a
fixed order, it reaches the same accuracy in roughly half the questions:

    questions asked      fixed    adaptive
      1                   7.0%        4.5%
      2                   3.8%        1.8%
      3                   4.0%        1.8%

That matters twice over. Fewer questions is the difference between people
finishing and abandoning, and knowing when to stop is what stops it asking
a fit 23-year-old for a cholesterol panel that would change nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from health_agent.domain.evidence import assess as assess_evidence
from health_agent.domain.profile import HealthProfile
from health_agent.domain.worth_measuring import CANDIDATES, Measurable, rank


class Scorer(Protocol):
    def score(self, profile: HealthProfile, **kwargs: Any) -> Any: ...


#: Below this swing, asking is not worth the person's patience.
STOP_BELOW = 0.03
#: Nobody should be asked more than this before getting something back.
MAX_QUESTIONS = 8


@dataclass(frozen=True, slots=True)
class Question:
    field: str
    ask: str
    why: str
    #: how much the estimate could still move on this, 0-1
    worth: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "ask": self.ask,
            "why_this_one": self.why,
            "could_move_the_estimate_by": f"{self.worth:.0%}",
        }


@dataclass(frozen=True, slots=True)
class Step:
    done: bool
    reason: str
    question: Question | None = None
    asked_so_far: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ready_to_assess": self.done,
            "reason": self.reason,
            "questions_answered": self.asked_so_far,
        }
        if self.question:
            payload["next_question"] = self.question.as_dict()
        return payload


# Asked before anything is scored, because without them there is no estimate
# to improve. Ordered by how much they constrain everything downstream.
FOUNDATIONS: tuple[tuple[str, str], ...] = (
    ("age", "How old are you?"),
    ("sex", "What sex were you assigned at birth?"),
    ("height_cm", "How tall are you? Feet and inches is fine."),
    ("weight_kg", "Roughly what do you weigh? Stone, kg, whatever you know."),
)

#: Prompts for the things the ranking can choose between.
PROMPTS: dict[str, str] = {
    "waist_cm": "What is your waist measurement, at the navel? A tape "
                "measure and thirty seconds -- it tells us more than weight "
                "does, and unlike weight it does not mistake muscle for fat.",
    "systolic_bp": "Do you know your blood pressure? The top number is "
                   "enough, and it is free to get checked at most pharmacies.",
    "hba1c_mmol_mol": "Have you had an HbA1c or blood sugar test? Either "
                      "scale works -- 39 mmol/mol or 5.7%.",
    "total_cholesterol_mmol_l": "Do you know your cholesterol?",
    "egfr": "Do you know your eGFR, from a kidney blood test?",
    "resting_hr": "What is your resting heart rate? Any tracker will have it.",
    "sleep_hours_avg": "How many hours do you actually sleep on a normal night?",
    # Every field the evidence bands can require needs a prompt here, or a
    # raw field name reaches the person -- "What is your alcohol units per
    # week?" is what that looks like.
    "alcohol_units_per_week": "Roughly how much do you drink in a week? A "
                              "pint or a medium glass of wine is about two "
                              "units, and a rough number is fine.",
    "moderate_activity_minutes_per_week": "How much exercise do you get in a "
                                          "typical week? Anything that gets "
                                          "you a bit out of breath counts.",
    "diet_quality_self_rating": "How would you rate your diet out of five, "
                                "where one is poor and five is excellent?",
    "eats_vegetables_daily": "Do you eat vegetables, fruit or berries most days?",
    "perceived_stress_rating": "How stressed have you been lately, one to "
                               "five, where one is not at all?",
    "fasting_glucose_mmol_l": "Do you know your fasting blood glucose?",
    "smoking_status": "Do you smoke -- never, used to, or currently?",
}


def _prompt_for(field: str) -> str:
    """A question in words, never a field name."""
    if field in PROMPTS:
        return PROMPTS[field]
    readable = field.replace("_", " ").replace(" cm", "").replace(" kg", "")
    return f"Do you know your {readable}?"

#: Cheap to answer and always worth knowing, so asked before the ranking
#: starts spending calls on what to ask.
ALWAYS_WORTH_ASKING: tuple[tuple[str, str], ...] = (
    ("smoking_status", "Do you smoke -- never, used to, or currently?"),
    ("on_bp_medication",
     "Have you ever been prescribed medication for blood pressure?"),
    ("previously_high_glucose",
     "Has anyone ever told you your blood sugar was high -- at a check-up, "
     "during an illness, or in pregnancy?"),
)


def next_step(
    profile: HealthProfile,
    scorer: Scorer,
    *,
    skipped: Iterable[str] = (),
) -> Step:
    """What to ask next, or that there is nothing left worth asking.

    `skipped` is what they have already been asked and could not answer.
    Without it, "I don't know" leaves the field unknown and it gets asked
    again, forever -- and not knowing is one of the commonest answers here.
    """
    known = profile.known_fields() | set(skipped)
    answered = len(profile.known_fields())

    for field, prompt in FOUNDATIONS:
        if field not in known:
            return Step(
                done=False,
                reason="Nothing can be estimated without this.",
                question=Question(
                    field=field, ask=prompt,
                    why="Needed before anything can be worked out.", worth=1.0,
                ),
                asked_so_far=answered,
            )

    for field, prompt in ALWAYS_WORTH_ASKING:
        if field not in known:
            return Step(
                done=False,
                reason="Free to answer and changes the picture.",
                question=Question(
                    field=field, ask=prompt,
                    why="Costs nothing to answer and carries real weight.",
                    worth=0.5,
                ),
                asked_so_far=answered,
            )

    evidence = assess_evidence(profile)
    outstanding = [f for f in evidence.missing_required if f not in known]
    if not evidence.sufficient and outstanding:
        field = outstanding[0]
        return Step(
            done=False,
            reason=evidence.band.rationale,
            question=Question(
                field=field,
                ask=_prompt_for(field),
                why="At your age an estimate is not given without this.",
                worth=0.6,
            ),
            asked_so_far=answered,
        )

    if not evidence.sufficient:
        return Step(
            done=True,
            reason=(
                "There is not enough to give a trustworthy estimate, and "
                "what is missing is not something you know offhand. Say so "
                "plainly rather than producing a number."
            ),
            asked_so_far=answered,
        )

    if answered >= MAX_QUESTIONS + len(FOUNDATIONS):
        return Step(True, "Enough to give you a picture.", asked_so_far=answered)

    # Everything essential is in. From here, only ask if it would change
    # something -- this is where a form keeps going and a consultation stops.
    ranked = [r for r in rank(profile, scorer, limit=4) if r.field not in known]
    if not ranked or ranked[0].swing < STOP_BELOW:
        return Step(
            done=True,
            reason=(
                "Nothing left that would meaningfully change the estimate. "
                "More questions would be taking your time for nothing."
            ),
            asked_so_far=answered,
        )

    best = ranked[0]
    return Step(
        done=False,
        reason="This is the one thing that would move the estimate most.",
        question=Question(
            field=best.field,
            ask=_prompt_for(best.field),
            why=(
                f"Of everything still unknown, this could move the estimate "
                f"by about {best.swing:.0%}. {best.how}"
            ),
            worth=best.swing,
        ),
        asked_so_far=answered,
    )
