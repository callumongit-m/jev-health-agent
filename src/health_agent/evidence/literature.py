"""Research backing, via Europe PMC.

Open, keyless, no registration -- which matters because the NHS Content API
needs an onboarding review, and this is available today.

It is also the more useful of the two for the thing that actually matters.
NHS pages say *what* to do and change over years; the number people want is
*how much it helps*, and that is where a language model will otherwise
invent a plausible figure. This attaches a real citation to that claim.

Two rules keep it honest:

* **Reviews only.** Restricted to systematic reviews and meta-analyses from
  the last decade. A single primary study is not evidence for advice, and
  picking one that agrees with you is worse than citing nothing.
* **It supports advice, it does not generate it.** The recommendation comes
  from published guidance; this says what the research behind it is. An
  abstract is not patient-facing advice and must not be relayed as such.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from health_agent.config import SETTINGS

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
#: Literature moves slowly and we only ever want well-established reviews.
CACHE_TTL = timedelta(days=90)
# Window deliberately ends a few years back. Europe PMC's relevance sort is
# recency-biased, and a review published this year has no citations yet, so
# sorting by citations over a settled window is what surfaces the paper the
# field actually leans on rather than the newest one about anything.
EARLIEST_YEAR = 2016
LATEST_YEAR = datetime.now(timezone.utc).year - 2
TIMEOUT = 10.0


@dataclass(frozen=True, slots=True)
class Paper:
    title: str
    journal: str
    year: str
    doi: str | None
    pmid: str | None
    cited_by: int
    open_access: bool

    @property
    def url(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return f"https://europepmc.org/article/MED/{self.pmid}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "cited_by": self.cited_by,
            "open_access": self.open_access,
            "url": self.url,
        }


# Queries restrict to TITLE. Searching the full text matches any paper that
# mentions a term in passing -- an unrestricted search for physical activity
# and mortality returned a review of olfactory dysfunction. Requiring the
# concept in the title is what makes the result about the thing.
FACTOR_QUERIES: dict[str, str] = {
    "smoking_burden": 'TITLE:"smoking cessation"',
    "adiposity": '(TITLE:"weight loss" OR TITLE:"waist circumference") AND TITLE:risk',
    "activity_deficit": 'TITLE:"physical activity" AND TITLE:mortality',
    "sleep_debt": 'TITLE:"sleep duration"',
    "alcohol_burden": 'TITLE:"alcohol consumption" AND (TITLE:mortality OR TITLE:risk)',
    "diet_quality": '(TITLE:"dietary pattern" OR TITLE:"diet quality") AND '
                    '(TITLE:"cardiovascular" OR TITLE:"type 2 diabetes" OR TITLE:mortality)',
    "stress_load": '(TITLE:"psychosocial stress" OR TITLE:"work stress" OR '
                   'TITLE:"job strain") AND (TITLE:"cardiovascular" OR TITLE:hypertension)',
}


def _db() -> sqlite3.Connection:
    path = Path(SETTINGS.checkpoint_path).with_suffix(".literature.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS papers (
               key TEXT PRIMARY KEY,
               payload TEXT NOT NULL,
               fetched_at TEXT NOT NULL
           )"""
    )
    return conn


def _cached(key: str) -> list[dict] | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM papers WHERE key=?", (key,)
        ).fetchone()
    if row is None:
        return None
    if datetime.now(timezone.utc) - datetime.fromisoformat(row[1]) > CACHE_TTL:
        return None
    return json.loads(row[0])


def _store(key: str, payload: list[dict]) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?)",
            (key, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )


def _clean(text: str | None) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", str(text or "")).split())


def search(query: str, *, limit: int = 2) -> list[Paper]:
    """Systematic reviews and meta-analyses matching a query."""
    full = (
        f"({query}) AND "
        f'(PUB_TYPE:"systematic review" OR PUB_TYPE:"meta-analysis") AND '
        f"(FIRST_PDATE:[{EARLIEST_YEAR} TO {LATEST_YEAR}])"
    )
    try:
        import httpx

        response = httpx.get(
            ENDPOINT,
            params={
                "query": full,
                "format": "json",
                "pageSize": limit * 3,
                "resultType": "core",
                "sort": "CITED desc",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        logger.warning("Europe PMC search failed: %s", exc)
        return []

    papers: list[Paper] = []
    for record in body.get("resultList", {}).get("result", []):
        title = _clean(record.get("title"))
        if not title:
            continue
        papers.append(
            Paper(
                title=title.rstrip("."),
                journal=_clean(record.get("journalTitle")) or "unknown journal",
                year=str(record.get("pubYear") or ""),
                doi=record.get("doi"),
                pmid=record.get("pmid"),
                cited_by=int(record.get("citedByCount") or 0),
                open_access=record.get("isOpenAccess") == "Y",
            )
        )

    # Already citation-sorted by the API; drop anything uncited, which at
    # this age means the field has not picked it up.
    return [p for p in papers if p.cited_by > 0][:limit]


def for_factor(key: str, *, limit: int = 2) -> list[dict[str, Any]]:
    """Reviews supporting the advice for one modifiable factor."""
    query = FACTOR_QUERIES.get(key)
    if not query:
        return []

    cached = _cached(key)
    if cached is not None:
        return cached

    papers = [p.as_dict() for p in search(query, limit=limit)]
    if papers:
        _store(key, papers)
    return papers
