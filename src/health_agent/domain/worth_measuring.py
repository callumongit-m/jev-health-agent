"""What is worth measuring next.

The one thing here a language model cannot do by reasoning. Asked which
test someone should get, a model produces a sensible-sounding list. This
produces the answer for *this* person: take each thing they have not told
us, score them again across its plausible range, and see how far the answer
actually moves. Something that swings the picture by twenty points is worth
getting; something that moves it by two is not.

That needs a scorer that is cheap enough to call a few dozen times and
stable enough that the difference between two runs means something. It
costs about two seconds and a fraction of a penny.

The swing is then weighted by how hard the thing is to obtain, because the
useful answer is not "the most informative test in principle" -- it is
"the most informative thing you could actually go and do". A blood pressure
reading is free at any pharmacy; an eGFR needs bloods and an appointment.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from health_agent.domain.profile import COMPUTED_FIELDS, HealthProfile


class Scorer(Protocol):
    def score(self, profile: HealthProfile, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class Measurable:
    field: str
    label: str
    #: plausible values spanning healthy to concerning
    probe: tuple[float, ...]
    #: 1.0 free and immediate, 0.2 needs an appointment and a needle
    accessibility: float
    how: str


CANDIDATES: tuple[Measurable, ...] = (
    Measurable(
        "systolic_bp", "blood pressure", (110, 130, 150, 175), 1.0,
        "Free at most pharmacies, no appointment. Machines in some "
        "supermarkets too, or a home monitor for about twenty pounds.",
    ),
    Measurable(
        "hba1c_mmol_mol", "HbA1c (average blood sugar)", (32, 40, 48, 60), 0.4,
        "A blood test. Your GP can do it, and home finger-prick kits are "
        "around twenty pounds.",
    ),
    Measurable(
        "total_cholesterol_mmol_l", "cholesterol", (3.8, 5.0, 6.2, 7.6), 0.4,
        "A blood test, often bundled with an NHS Health Check if you are "
        "over forty.",
    ),
    Measurable(
        "egfr", "kidney function (eGFR)", (95, 75, 55, 35), 0.3,
        "A blood test, usually ordered alongside others rather than on its own.",
    ),
    Measurable(
        "waist_cm", "waist measurement", (78, 90, 102, 115), 1.0,
        "A tape measure, at the navel, breathing out. Thirty seconds.",
    ),
    Measurable(
        "resting_hr", "resting heart rate", (52, 62, 75, 90), 0.9,
        "Any fitness tracker, or count your pulse for a minute before "
        "getting up.",
    ),
    Measurable(
        "sleep_hours_avg", "how much you actually sleep", (5.0, 6.5, 7.5, 8.5), 0.8,
        "A week of paying attention, or whatever your phone already records.",
    ),
)


@dataclass(frozen=True, slots=True)
class Suggestion:
    field: str
    label: str
    swing: float
    accessibility: float
    how: str

    @property
    def score(self) -> float:
        return self.swing * self.accessibility

    def as_dict(self) -> dict[str, Any]:
        return {
            "what": self.label,
            "how_much_it_would_move_the_estimate": f"{self.swing:.0%}",
            "how_to_get_it": self.how,
        }


#: Below this the answer barely moves and saying so is more useful than
#: sending someone for a test.
WORTH_IT = 0.04


def rank(
    profile: HealthProfile,
    scorer: Scorer,
    *,
    conditions: tuple[str, ...] = ("t2d_10yr", "cvd_10yr", "hypertension", "ckd"),
    limit: int = 3,
) -> list[Suggestion]:
    """Rank what this person has not told us by how much it would matter."""
    unknown = [c for c in CANDIDATES if getattr(profile, c.field, None) is None]
    if not unknown:
        return []

    base = profile.model_dump(exclude=COMPUTED_FIELDS | {"provenance"})
    jobs = [(c, value) for c in unknown for value in c.probe]

    with ThreadPoolExecutor(max_workers=min(12, len(jobs))) as pool:
        scored = list(
            pool.map(
                lambda job: (
                    job[0],
                    scorer.score(HealthProfile(**(base | {job[0].field: job[1]}))),
                ),
                jobs,
            )
        )

    suggestions: list[Suggestion] = []
    for candidate in unknown:
        runs = [a for c, a in scored if c is candidate]
        spreads = []
        for key in conditions:
            try:
                values = [a.condition(key).probability for a in runs]
            except KeyError:
                continue
            spreads.append(max(values) - min(values))
        if not spreads:
            continue
        swing = sum(spreads) / len(spreads)
        if swing < WORTH_IT:
            continue
        suggestions.append(
            Suggestion(
                field=candidate.field, label=candidate.label, swing=swing,
                accessibility=candidate.accessibility, how=candidate.how,
            )
        )

    # by what it would actually be worth going and doing, not by raw
    # informativeness -- an eGFR may swing more than a blood pressure and
    # still be the worse suggestion
    suggestions.sort(key=lambda s: -s.score)
    return suggestions[:limit]
