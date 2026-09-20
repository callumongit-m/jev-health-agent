#!/usr/bin/env python
"""Measure a backend's run-to-run noise.

Every monotonicity and sensitivity tolerance is derived from `noise_sigma`,
so a guessed value means either spurious failures or checks that cannot fail.
This measures it: score unchanged profiles many times and take the worst
per-condition standard deviation across them.

Worst, not mean -- a tolerance sized for the average condition is too tight
for the noisiest one, and that is where the false failures come from.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from evals.cases_loader import load_personas
from health_agent.scoring.backend import get_backend
from health_agent.scoring.scorer import RiskScorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--personas", default="lean_young_healthy,mid_metabolic_risk,heavy_smoker_cvd",
        help="noise varies by where a profile sits, so sample across the range",
    )
    parser.add_argument("--out", default="evals/noise_profile.json")
    args = parser.parse_args()

    scorer = RiskScorer(get_backend(args.backend))
    personas = load_personas()
    chosen = [p.strip() for p in args.personas.split(",")]

    print(f"backend {scorer.backend.name}, {args.repeats} repeats x {len(chosen)} profiles")
    print(f"{args.repeats * len(chosen)} calls\n")

    per_condition: dict[str, list[float]] = {}
    report: dict[str, dict] = {}

    for pid in chosen:
        profile = personas[pid].profile
        runs = [scorer.score(profile) for _ in range(args.repeats)]
        stats = {}
        for condition in runs[0].conditions:
            values = [r.condition(condition.key).probability for r in runs]
            sd = statistics.pstdev(values)
            stats[condition.key] = {
                "mean": round(statistics.fmean(values), 4),
                "sd": round(sd, 4),
                "spread": round(max(values) - min(values), 4),
            }
            per_condition.setdefault(condition.key, []).append(sd)
        worst = max(stats, key=lambda k: stats[k]["sd"])
        print(f"{pid:22} worst {worst} sd={stats[worst]['sd']:.4f} "
              f"spread={stats[worst]['spread']:.4f}")
        report[pid] = stats

    all_sds = [sd for sds in per_condition.values() for sd in sds]
    worst_sd = max(all_sds)
    p95 = sorted(all_sds)[int(len(all_sds) * 0.95) - 1] if len(all_sds) > 1 else worst_sd

    print(f"\nacross every condition and profile:")
    print(f"  median sd  {statistics.median(all_sds):.4f}")
    print(f"  p95 sd     {p95:.4f}")
    print(f"  worst sd   {worst_sd:.4f}   <- set noise_sigma to this")

    noisiest = max(per_condition, key=lambda k: max(per_condition[k]))
    print(f"  noisiest condition: {noisiest}")

    payload = {
        "backend": scorer.backend.name,
        "repeats": args.repeats,
        "profiles": chosen,
        "median_sd": round(statistics.median(all_sds), 4),
        "p95_sd": round(p95, 4),
        "worst_sd": round(worst_sd, 4),
        "noisiest_condition": noisiest,
        "per_profile": report,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
