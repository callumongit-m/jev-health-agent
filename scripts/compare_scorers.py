#!/usr/bin/env python
"""Head-to-head: does the decision model earn its place over an LLM?

Runs both estimators over the same profiles and measures the three things
that decide whether a probability is worth showing anyone:

  stability    same input, repeated -- does the number hold still?
  ordering     does higher risk actually score higher?
  sensitivity  does each declared driver move its condition?

Deliberately bounded so it costs pennies. Reports, does not argue.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from evals.cases_loader import load_orderings, load_personas
from health_agent.domain.profile import COMPUTED_FIELDS, HealthProfile
from health_agent.scoring.backend import get_backend
from health_agent.scoring.scorer import RiskScorer

DRIVERS = [
    ("t2d_10yr", "hba1c_mmol_mol", 32.0, 75.0),
    ("cvd_10yr", "cigarettes_per_day", 0.0, 40.0),
    ("cvd_10yr", "systolic_bp", 112.0, 185.0),
    ("hypertension", "systolic_bp", 112.0, 185.0),
    ("ckd", "egfr", 105.0, 28.0),
    ("nafld", "alt_u_l", 18.0, 120.0),
]

BASE = dict(
    age=52, sex="male", height_cm=178.0, weight_kg=88.0, waist_cm=98.0,
    smoking_status="former", alcohol_units_per_week=10.0,
    moderate_activity_minutes_per_week=120, sleep_hours_avg=6.5,
    diet_quality_self_rating=3, perceived_stress_rating=3,
    on_bp_medication=False, previously_high_glucose=False,
    eats_vegetables_daily=True, systolic_bp=132, diastolic_bp=82,
    hba1c_mmol_mol=38.0, egfr=88.0, alt_u_l=30.0, cigarettes_per_day=0,
)


def _pin(field: str, value: float) -> HealthProfile:
    coerced = int(round(value)) if field in ("systolic_bp", "cigarettes_per_day") else value
    update = {field: coerced}
    # pinning cigarettes on a "former" smoker is a contradictory profile, and
    # penalising an estimator for not reading it is measuring the test
    if field == "cigarettes_per_day":
        update["smoking_status"] = "current" if value > 0 else "never"
    return HealthProfile(**(BASE | update))


def evaluate(scorer: RiskScorer, repeats: int) -> dict:
    name = scorer.backend.name
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")

    # 1. stability -- the same profile, over and over
    baseline = HealthProfile(**BASE)
    runs = [scorer.score(baseline) for _ in range(repeats)]
    spreads = {
        c.key: max(r.condition(c.key).probability for r in runs)
        - min(r.condition(c.key).probability for r in runs)
        for c in runs[0].conditions
    }
    worst_key = max(spreads, key=spreads.get)
    sigma = statistics.pstdev(
        [r.condition(worst_key).probability for r in runs]
    )
    print(f"stability   worst spread {spreads[worst_key]:.3f} on {worst_key} "
          f"(sd {sigma:.3f}) over {repeats} identical runs")

    # 2. ordering
    personas = load_personas()
    ok = 0
    pairs = load_orderings()
    for higher, lower, condition in pairs:
        hi = scorer.score(personas[higher].profile).condition(condition).probability
        lo = scorer.score(personas[lower].profile).condition(condition).probability
        passed = hi > lo
        ok += passed
        if not passed:
            print(f"            MISS {condition}: {higher} {hi:.2f} !> {lower} {lo:.2f}")
    print(f"ordering    {ok}/{len(pairs)} pairs ranked correctly")

    # 3. sensitivity -- healthy to severe on each declared driver
    moves, flat = [], []
    for condition, field, healthy, severe in DRIVERS:
        low = scorer.score(_pin(field, healthy)).condition(condition).probability
        high = scorer.score(_pin(field, severe)).condition(condition).probability
        delta = high - low
        moves.append(delta)
        if delta < 0.07:
            flat.append(f"{condition}/{field} {low:.2f}->{high:.2f}")
    print(f"sensitivity {len(DRIVERS) - len(flat)}/{len(DRIVERS)} drivers move "
          f"materially (median {statistics.median(moves):+.3f})")
    for miss in flat:
        print(f"            FLAT {miss}")

    malformed = getattr(scorer.backend, "malformed", 0)
    coerced = getattr(scorer.backend, "coerced", 0)
    if malformed or coerced:
        print(f"output shape {malformed} responses needed recovery, "
              f"{coerced} fields came back the wrong type")

    return {
        "name": name,
        "malformed": malformed,
        "coerced": coerced,
        "sigma": sigma,
        "worst_spread": spreads[worst_key],
        "ordering": ok / len(pairs),
        "sensitivity": (len(DRIVERS) - len(flat)) / len(DRIVERS),
        "median_move": statistics.median(moves),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--backends", default="openrouter,llm")
    args = parser.parse_args()

    results = []
    for kind in args.backends.split(","):
        try:
            results.append(evaluate(RiskScorer(get_backend(kind.strip())), args.repeats))
        except Exception as exc:
            print(f"\n{kind}: FAILED -- {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
    print(f"{'estimator':32} {'sd':>7} {'ordering':>9} {'sensitivity':>12} {'bad output':>11}")
    for r in results:
        bad = r.get("malformed", 0) + r.get("coerced", 0)
        print(f"{r['name']:32} {r['sigma']:>7.3f} {r['ordering']:>9.0%} "
              f"{r['sensitivity']:>12.0%} {bad:>11}")
    print("\nlower sd is better; ordering and sensitivity higher is better")
    return 0


if __name__ == "__main__":
    sys.exit(main())
