#!/usr/bin/env python
"""Score a persona (or a JSON profile) and print the assessment.

    python scripts/demo_cli.py --persona mid_metabolic_risk
    python scripts/demo_cli.py --json '{"age": 44, "height_cm": 180, "weight_kg": 95}'
"""

from __future__ import annotations

import argparse
import json
import sys

from health_agent.domain.profile import HealthProfile
from health_agent.privacy import DISCLAIMER, NOTICE
from health_agent.scoring.backend import get_backend
from health_agent.scoring.scorer import RiskScorer

BAR = "#"


def render(assessment) -> str:
    lines = [
        f"backend {assessment.model}   {assessment.latency_ms} ms",
        f"data sufficiency {assessment.data_sufficiency:.0%}   "
        f"profile completeness {assessment.completeness:.0%}",
        "",
        "RISK",
    ]
    for c in sorted(assessment.conditions, key=lambda x: -x.probability):
        bar = BAR * round(c.probability * 30)
        lines.append(f"  {c.label:<26} {c.probability:6.1%}  {bar}")

    lines += ["", "MODIFIABLE FACTORS  (years recoverable)"]
    for f in sorted(assessment.factors, key=lambda x: -x.years_cost):
        lines.append(
            f"  {f.label:<26} {f.years_cost:5.1f} yr  "
            f"conf {f.confidence:.2f}  {f.level_label}"
        )
    lines += ["", f"total years at risk from modifiable factors: "
                  f"{assessment.total_years_at_risk}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--persona", help="id from evals/cases/personas.yaml")
    source.add_argument("--json", help="a HealthProfile as JSON")
    parser.add_argument("--text", help="free text the person also said")
    parser.add_argument(
        "--agent", action="store_true",
        help="run the whole agent (graph + reasoning) rather than scoring only",
    )
    parser.add_argument("--backend", default="auto", choices=["auto", "jev", "fake"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.persona:
        from evals.cases_loader import load_personas

        personas = load_personas()
        if args.persona not in personas:
            print(f"unknown persona. available: {', '.join(personas)}", file=sys.stderr)
            return 2
        profile = personas[args.persona].profile
    else:
        profile = HealthProfile(**json.loads(args.json))

    scorer = RiskScorer(get_backend(args.backend, seed=args.seed))

    if not args.agent:
        print(render(scorer.score(profile)))
        print("\n" + DISCLAIMER + "\n" + NOTICE)
        return 0

    from health_agent.adapters.core import assess
    from health_agent.graph.reasoner import reasoner_name

    out = assess(profile, raw_text=args.text, scorer=scorer)
    print(f"status   {out['status']}   (reasoner: {reasoner_name()})")
    print()

    if out["status"] == "seek_care":
        print("RED FLAGS:", ", ".join(out["red_flags"]))
        print()
        print(out["answer"])
    elif out["status"] == "needs_input":
        print(out["answer"])
        for q in out["questions"]:
            print(f"  - {q}")
    else:
        print(render(scorer.score(profile)))
        le = out.get("life_expectancy")
        if le:
            print()
            print("LIFE EXPECTANCY")
            print(f"  baseline for age/sex  {le['baseline_remaining_years']:.1f} years")
            print(f"  adjusted estimate     {le['adjusted_remaining_years']:.1f} years")
            print(f"  age at death approx   {le['estimated_age_at_death']:.1f}")
            print(f"  recoverable           {le['years_recoverable']:.1f} years")
        print()
        print("AGENT")
        for line in (out["answer"] or "").splitlines():
            print("  " + line)

    print("\n" + out["disclaimer"] + "\n" + out["privacy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
