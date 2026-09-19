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
