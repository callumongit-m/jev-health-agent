"""Research backing via Europe PMC.

Open and keyless, which matters because the NHS Content API needs an
onboarding review. It also covers the thing NHS pages do not: how much an
intervention actually helps, which is where a model invents a figure.
"""

import pytest

from health_agent.evidence import literature


def test_queries_restrict_to_the_title():
    """An unrestricted search matches anything mentioning a term in passing --
    physical activity and mortality returned a review of olfactory
    dysfunction. Requiring the concept in the title is what fixes it."""
    for key, query in literature.FACTOR_QUERIES.items():
        assert "TITLE:" in query, key


def test_the_window_excludes_the_newest_papers():
    """Sorting by citations over a settled window surfaces what the field
    leans on. Including this year surfaces whatever is newest about
    anything, with no citations to judge it by."""
    from datetime import datetime, timezone

    assert literature.LATEST_YEAR <= datetime.now(timezone.utc).year - 2
    assert literature.EARLIEST_YEAR < literature.LATEST_YEAR


def test_every_factor_has_a_query():
    from health_agent.domain.conditions import FACTORS

    assert {f.key for f in FACTORS} == set(literature.FACTOR_QUERIES)


def test_a_failed_search_returns_nothing_rather_than_raising(monkeypatch):
    """A literature service being down must not cost someone their report."""
    import httpx

    def explode(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", explode)
    assert literature.search("TITLE:anything") == []


def test_uncited_papers_are_dropped(monkeypatch):
    """At this age, no citations means the field has not picked it up."""
    import httpx

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"resultList": {"result": [
                {"title": "Never cited", "pubYear": "2019", "citedByCount": 0,
                 "doi": "10.1/a", "journalTitle": "J"},
                {"title": "Well cited", "pubYear": "2019", "citedByCount": 500,
                 "doi": "10.1/b", "journalTitle": "J"},
            ]}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Response())
    papers = literature.search("TITLE:x", limit=5)
    assert [p.title for p in papers] == ["Well cited"]


def test_a_paper_always_has_a_resolvable_link():
    paper = literature.Paper(
        title="T", journal="J", year="2019", doi=None, pmid="123",
        cited_by=10, open_access=True,
    )
    assert paper.url == "https://europepmc.org/article/MED/123"
    with_doi = literature.Paper(
        title="T", journal="J", year="2019", doi="10.1/x", pmid="123",
        cited_by=10, open_access=True,
    )
    assert with_doi.url == "https://doi.org/10.1/x"


def test_markup_is_stripped_from_titles(monkeypatch):
    import httpx

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"resultList": {"result": [
                {"title": "A <i>meta</i>-analysis.", "pubYear": "2019",
                 "citedByCount": 5, "doi": "10.1/a", "journalTitle": "J"},
            ]}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Response())
    assert literature.search("TITLE:x")[0].title == "A meta -analysis"


# --- through the report -------------------------------------------------

def test_guidance_carries_the_ogl_attribution():
    """NHS content is reusable under the Open Government Licence, on
    condition of attribution. That condition travels with the content."""
    from health_agent.evidence.corpus import OGL_ATTRIBUTION, for_factor

    entry = for_factor("smoking_burden").as_dict()
    assert entry["licence"] == OGL_ATTRIBUTION
    assert "Open Government Licence" in entry["licence"]


def test_the_contract_separates_advice_from_research():
    """A paper title is not advice. Research says why to believe the
    guidance, it does not replace it."""
    from health_agent.adapters.core import assess
    from health_agent.scoring import FakeBackend, RiskScorer

    out = assess(
        {"age": 55, "sex": "male", "height_cm": 178, "weight_kg": 94,
         "smoking_status": "current", "cigarettes_per_day": 15,
         "alcohol_units_per_week": 24, "moderate_activity_minutes_per_week": 60,
         "sleep_hours_avg": 5.5, "diet_quality_self_rating": 2,
         "on_bp_medication": True, "previously_high_glucose": False,
         "eats_vegetables_daily": False},
        scorer=RiskScorer(FakeBackend(seed=3, noise=0.0)),
    )
    forbidden = " ".join(out["presentation"]["must_not"]).lower()
    assert "do not quote findings" in forbidden
    assert "not treat a paper title as" in forbidden
