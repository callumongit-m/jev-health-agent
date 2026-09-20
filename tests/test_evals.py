"""The eval suite, wired into pytest as a regression gate."""

from evals.runner import format_report, run


def test_jev_suite_is_fully_reliable_offline():
    report = run(k=5, suite="jev", backend="fake", seed=3)
    assert report.pass_hat_k == 1.0, "\n" + format_report(report)


def test_suite_detects_a_model_that_ignores_an_input():
    """A suite that cannot fail proves nothing."""
    from health_agent.scoring import backend as B

    original = B.FakeBackend.classify
    B.FakeBackend.classify = lambda self, state: original(
        self, {**state, "hba1c_mmol_mol": 34}
    )
    try:
        report = run(k=3, suite="jev", backend="fake", seed=3)
    finally:
        B.FakeBackend.classify = original

    failed = [o.name for o in report.outcomes if not o.all_passed]
    assert any("sensitivity[t2d_10yr/hba1c_mmol_mol]" in n for n in failed), failed


def test_suite_detects_a_model_that_ignores_improvement():
    """Monotonicity only tests the worsening direction, which leaves every
    recommendation untested. A counterfactual sweep found the consequence:
    tripling someone's exercise came back with their risk slightly up."""
    from health_agent.scoring import backend as B

    original = B.FakeBackend.classify

    def inverted_activity(self, state):
        # Read activity backwards, so exercising more looks like exercising
        # less. Clamping both sides equally would prove nothing -- that is
        # the very blind spot this check exists to cover.
        flipped = dict(state)
        activity = flipped.get("moderate_activity_minutes_per_week")
        if activity is not None:
            flipped["moderate_activity_minutes_per_week"] = max(0, 400 - activity)
        return original(self, flipped)

    B.FakeBackend.classify = inverted_activity
    try:
        report = run(k=3, suite="jev", backend="fake", seed=3)
    finally:
        B.FakeBackend.classify = original

    failed = [o.name for o in report.outcomes if not o.all_passed]
    assert any("improvement[" in n for n in failed), failed
