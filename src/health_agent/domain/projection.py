"""Risk over a lifetime, if nothing changes.

The classifier answers a ten-year question. Asking it again at an older age
does not compound -- for a 55-year-old already at 60 percent, the next decade
comes back at 60 percent too, and a chart of that is a flat line that tells
nobody anything.

What people actually want to see is accumulation: not "what are the odds in
the next ten years" but "where does this end up if I carry on like this".
That is a survival composition over successive decades, each one scored on
its own, and it turns a flat line into the actual story -- for a fit
21-year-old with diabetes in the family, a 21 percent decade risk becomes an
81 percent chance of getting there by 81.

Three things it assumes, all stated in the output because each one is a
reason the curve overstates:

* Lifestyle stays exactly as it is now. Nobody's does.
* Each decade is conditionally independent of the last. This is the weak
  assumption, and it is why the far end of the curve runs high: composing
  four decades of a 40 percent risk gets to 87 percent, which is above what
  cohort studies show for almost any of these conditions. The shape and the
  ordering between conditions survive that; the absolute number at the far
  end should not be quoted as a lifetime risk. Said in the output, not just
  here.
* Nothing else happens first. There is no competing mortality in this model,
  so a lifetime figure is "if you live that long".
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from health_agent.domain.profile import COMPUTED_FIELDS, HealthProfile

#: A decade at a time: the classifier's question is a ten-year one, so this
#: is the natural step and anything finer is interpolation dressed as data.
STEP_YEARS = 10
# Three decades, not a whole lifetime. Composing ten-year risks indefinitely
# drives anything non-trivial to near-certainty -- a 45-year-old smoker came
# out at 97% for diabetes and 100% for hypertension by 85, which is both
# uninformative and wrong, and it made every counterfactual look identical
# because they all saturate. Over thirty years the curve still shows the
# shape and the differences between paths remain visible.
HORIZON_YEARS = 30
#: Never project past the top of the life tables regardless.
MAX_AGE = 85
#: Conditions worth charting. The rest add lines without adding information.
CHARTED = ("t2d_10yr", "cvd_10yr", "hypertension", "ckd")


class Scorer(Protocol):
    def score(self, profile: HealthProfile, *, free_text: str | None = ...) -> Any: ...


@dataclass(frozen=True, slots=True)
class Projection:
    #: age -> {condition key -> cumulative probability by that age}
    points: tuple[tuple[int, dict[str, float]], ...]
    labels: dict[str, str]
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "x_axis": "age",
            "y_axis": "cumulative probability of having developed it by then",
            "series": {
                key: [
                    {"age": age, "probability": values[key]}
                    for age, values in self.points
                    if key in values
                ]
                for key in self.labels
            },
            "labels": self.labels,
            "assumptions": list(self.assumptions),
        }


def build(
    profile: HealthProfile,
    scorer: Scorer,
    *,
    max_age: int = MAX_AGE,
    conditions: tuple[str, ...] = CHARTED,
) -> Projection | None:
    """Re-score at decade steps and compose. None when age is unknown.

    Costs one classifier call per decade. That is affordable only because
    the classifier is cheap; the same sweep on a language model would be
    both slow and pointless.
    """
    if profile.age is None:
        return None

    ceiling = min(max_age, profile.age + HORIZON_YEARS)
    ages = list(range(profile.age, ceiling + 1, STEP_YEARS))
    if len(ages) < 2:
        return None

    base = profile.model_dump(exclude=COMPUTED_FIELDS | {"provenance"})

    # The decades are independent calls, so run them together -- sequentially
    # this adds four seconds to every assessment, concurrently about one.
    with ThreadPoolExecutor(max_workers=len(ages)) as pool:
        assessments = list(
            pool.map(
                lambda age: scorer.score(HealthProfile(**(base | {"age": age}))),
                ages,
            )
        )

    survival = {key: 1.0 for key in conditions}
    points: list[tuple[int, dict[str, float]]] = []
    labels: dict[str, str] = {}

    for age, assessment in zip(ages, assessments):
        reached: dict[str, float] = {}
        for key in conditions:
            try:
                risk = assessment.condition(key)
            except KeyError:
                continue
            labels.setdefault(key, risk.label)
            # chance of getting through this decade without it, compounded
            survival[key] *= 1.0 - risk.probability
            reached[key] = round(1.0 - survival[key], 3)
        points.append((age + STEP_YEARS, reached))

    return Projection(
        points=tuple(points),
        labels=labels,
        assumptions=(
            "Assumes nothing changes -- same weight, same habits, same "
            "everything -- which is the point of the chart rather than a "
            "prediction.",
            "Each decade is treated as independent of the last. That is the "
            "biggest weakness here and it overstates -- read the shape and "
            "the ordering between conditions, not the absolute figure at "
            "the far end.",
            "It does not account for anything else happening first, so read "
            "a late figure as 'if you get there'.",
            "It stops at thirty years. Compounding decade risks further than "
            "that drives almost any starting point to near-certainty, which "
            "stops being informative.",
        ),
    )
