"""pass^k eval runner.

For every check, run k independent trials. Report:

  pass^k       fraction of checks where ALL k trials passed -- the reliability
               number. This is what decides whether another agent can depend
               on this one.
  per-trial    total passes / total trials.

The gap between the two is the diagnosis: a low per-trial rate is a real bug,
a high per-trial rate with a low pass^k is flakiness.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from health_agent.scoring.backend import get_backend
from health_agent.scoring.scorer import RiskScorer

from evals import checks as C
from evals.cases_loader import load_orderings, load_personas


@dataclass(slots=True)
class CheckOutcome:
    name: str
    suite: str
    trials: int
    passes: int
    failures: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.passes == self.trials

    @property
    def rate(self) -> float:
        return self.passes / self.trials if self.trials else 0.0


@dataclass(slots=True)
class Report:
    outcomes: list[CheckOutcome]
    k: int
    backend: str
    seconds: float

    def by_suite(self, suite: str) -> list[CheckOutcome]:
        return [o for o in self.outcomes if o.suite == suite]

    def by_category(self) -> dict[str, list[CheckOutcome]]:
        """Group by check family (the bit before the first bracket), so one
        broken family stays visible instead of being averaged away."""
        groups: dict[str, list[CheckOutcome]] = {}
        for o in self.outcomes:
            groups.setdefault(o.name.split("[", 1)[0], []).append(o)
        return groups

    @property
    def pass_hat_k(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.all_passed for o in self.outcomes) / len(self.outcomes)

    @property
    def per_trial(self) -> float:
        total = sum(o.trials for o in self.outcomes)
        return sum(o.passes for o in self.outcomes) / total if total else 0.0


def build_checks(suite: str) -> list[tuple[str, str, C.CheckFn]]:
    """Returns (name, suite, fn). Suites: jev, graph, all."""
    personas = load_personas()
    built: list[tuple[str, str, C.CheckFn]] = []

    if suite in ("jev", "all"):
        for persona in personas.values():
            built.append(
                (f"schema_valid[{persona.id}]", "jev", C.check_schema_valid(persona))
            )
            if persona.expect:
                built.append(
                    (f"expectations[{persona.id}]", "jev", C.check_expectations(persona))
                )
            for name, fn in C.monotonicity_checks(persona).items():
                built.append((f"{name}[{persona.id}]", "jev", fn))
            for name, fn in C.sensitivity_checks(persona).items():
                built.append((f"{name}[{persona.id}]", "jev", fn))

        for higher, lower, condition in load_orderings():
            built.append(
                (
                    f"ordering[{condition}: {higher} > {lower}]",
                    "jev",
                    C.ordering_check(personas[higher], personas[lower], condition),
                )
            )

        for persona in personas.values():
            built.append(
                (f"evidenced_priority[{persona.id}]", "jev",
                 C.check_top_factor_is_evidenced(persona))
            )
        built.append(
            ("no_invented_priority", "jev",
             C.check_sparse_profile_does_not_invent_a_priority())
        )
        built.append(("privacy_surfaces", "jev", C.check_privacy_surfaces()))

    if suite in ("graph", "all"):
        sparse = {p.id for p in personas.values() if "sparse" in p.tags}
        for persona in personas.values():
            is_sparse = persona.id in sparse
            built.append(
                (f"gate[{persona.id}]", "graph", C.check_gate(persona, expect_input=is_sparse))
            )
            built.append(
                (f"privacy_payload[{persona.id}]", "graph", C.check_privacy_in_payload(persona))
            )
            built.append(
                (f"no_unearned_numbers[{persona.id}]", "graph",
                 C.check_no_numbers_without_confidence(persona))
            )
            built.append(
                (f"contract_guarantees[{persona.id}]", "graph",
                 C.check_contract_carries_its_guarantees(persona))
            )
            if is_sparse:
                continue
            built.append(
                (f"le_stability[{persona.id}]", "graph",
                 C.check_life_expectancy_stability(persona))
            )
            built.append(
                (f"rec_grounding[{persona.id}]", "graph",
                 C.check_recommendations_grounded(persona))
            )
            built.append(
                (f"years_not_oversold[{persona.id}]", "graph",
                 C.check_years_claimed_do_not_exceed_recoverable(persona))
            )

        for age, extra, expected in C.AGE_EVIDENCE_CASES:
            tag = "+".join(sorted(extra)) or "lifestyle only"
            built.append(
                (f"age_band[{age}, {tag}]", "graph",
                 C.check_age_band_gating(age, extra, expected))
            )
        built.append(
            ("evidence_caps_certainty", "graph", C.check_thin_evidence_caps_certainty())
        )
        built.append(
            ("no_test_required", "graph", C.check_no_test_is_ever_required())
        )
        built.append(
            ("asks_age_appropriate", "graph", C.check_asks_only_for_what_the_age_needs())
        )

        scorable = next(p for p in personas.values() if "complete" in p.tags)
        for text in C.ACUTE_TEXTS:
            built.append(
                (f"red_flag[{text[:32]}]", "graph", C.check_red_flag(scorable, text))
            )
        for text in C.BENIGN_TEXTS:
            built.append(
                (f"no_false_flag[{text[:32]}]", "graph",
                 C.check_not_red_flag(scorable, text))
            )

    return built


def run(
    *, k: int = 10, suite: str = "jev", backend: str = "auto", seed: int | None = None
) -> Report:
    scorer = RiskScorer(get_backend(backend, seed=seed))
    started = time.perf_counter()
    outcomes: list[CheckOutcome] = []

    for name, suite_name, fn in build_checks(suite):
        outcome = CheckOutcome(name=name, suite=suite_name, trials=k, passes=0)
        seen: Counter[str] = Counter()
        for _ in range(k):
            try:
                result = fn(scorer)
            except Exception as exc:  # a raised check is a failed check
                result = C.CheckResult(False, f"{type(exc).__name__}: {exc}")
            if result.passed:
                outcome.passes += 1
            elif result.detail:
                seen[result.detail] += 1
        outcome.failures = [f"{d} (x{n})" for d, n in seen.most_common(3)]
        outcomes.append(outcome)

    return Report(
        outcomes=outcomes,
        k=k,
        backend=scorer.backend.name,
        seconds=round(time.perf_counter() - started, 2),
    )


def format_report(report: Report, *, verbose: bool = False) -> str:
    lines: list[str] = []
    failed = [o for o in report.outcomes if not o.all_passed]

    lines.append(f"backend   {report.backend}")
    lines.append(f"k         {report.k}")
    lines.append(f"checks    {len(report.outcomes)}  ({report.seconds}s)")
    lines.append("")
    lines.append(f"pass^k    {report.pass_hat_k:6.1%}   <- reliability")
    lines.append(f"per-trial {report.per_trial:6.1%}")
    lines.append("")
    lines.append("by category:")
    for name, group in sorted(report.by_category().items()):
        ok = sum(o.all_passed for o in group)
        flag = "" if ok == len(group) else "   <-- "
        lines.append(f"  {ok / len(group):6.1%}  {name:<14} ({ok}/{len(group)}){flag}")
    lines.append("")

    if failed:
        lines.append(f"FAILING ({len(failed)}):")
        for o in sorted(failed, key=lambda x: x.rate):
            lines.append(f"  {o.rate:5.0%}  {o.name}")
            for detail in o.failures:
                lines.append(f"         {detail}")
    else:
        lines.append("all checks passed on every trial")

    if verbose:
        lines.append("")
        lines.append("ALL CHECKS:")
        for o in report.outcomes:
            mark = "ok  " if o.all_passed else "FAIL"
            lines.append(f"  {mark} {o.rate:5.0%}  {o.name}")

    return "\n".join(lines)
