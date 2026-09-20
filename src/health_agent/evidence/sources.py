"""Where guidance comes from, and how it is cached.

With `NHS_API_KEY` set, this fetches the live NHS Content API and caches the
result so a later assessment does not refetch. Without one, it falls back to
the curated corpus. Either way the caller gets the same shape, with a source
URL attached, so recommendations can be cited rather than asserted.

Live guidance is cached rather than fetched per assessment: NHS pages change
on the order of months, and a health assessment should not fail because a
content API is having a bad afternoon.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from health_agent.config import SETTINGS
from health_agent.evidence import corpus

logger = logging.getLogger(__name__)

NHS_BASE = "https://api.nhs.uk/conditions"
#: NHS content changes slowly; a stale month is better than a failed request.
CACHE_TTL = timedelta(days=30)

#: registry key -> nhs.uk conditions slug
NHS_SLUGS: dict[str, str] = {
    "t2d_10yr": "type-2-diabetes",
    "cvd_10yr": "cardiovascular-disease",
    "hypertension": "high-blood-pressure-hypertension",
    "metabolic_syndrome": "metabolic-syndrome",
    "sleep_apnoea": "sleep-apnoea",
    "nafld": "non-alcoholic-fatty-liver-disease",
    "ckd": "kidney-disease",
}


def _db() -> sqlite3.Connection:
    path = Path(SETTINGS.checkpoint_path).with_suffix(".evidence.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS guidance (
               key TEXT PRIMARY KEY,
               payload TEXT NOT NULL,
               fetched_at TEXT NOT NULL
           )"""
    )
    return conn


def _cached(key: str) -> dict[str, Any] | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM guidance WHERE key=?", (key,)
        ).fetchone()
    if row is None:
        return None
    fetched = datetime.fromisoformat(row[1])
    if datetime.now(timezone.utc) - fetched > CACHE_TTL:
        return None
    return json.loads(row[0])


def _store(key: str, payload: dict[str, Any]) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO guidance VALUES (?,?,?)",
            (key, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )


def _fetch_nhs(key: str) -> dict[str, Any] | None:
    """Live NHS Content API. Returns None when unavailable for any reason --
    an assessment must never fail because a content API is down."""
    slug = NHS_SLUGS.get(key)
    api_key = os.getenv("NHS_API_KEY")
    if not (slug and api_key):
        return None

    try:
        import httpx

        response = httpx.get(
            f"{NHS_BASE}/{slug}/",
            headers={"subscription-key": api_key, "Accept": "application/json"},
            timeout=8.0,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        logger.warning("NHS fetch failed for %s: %s", key, exc)
        return None

    points: list[str] = []
    for section in (body.get("hasPart") or [])[:4]:
        text = section.get("description") or section.get("text") or ""
        text = " ".join(str(text).split())
        if 40 < len(text) < 400:
            points.append(text)

    if not points:
        return None
    return {
        "title": body.get("name") or key.replace("_", " ").title(),
        "points": points[:3],
        "source": "NHS",
        "url": body.get("url") or f"https://www.nhs.uk/conditions/{slug}/",
        "retrieved": datetime.now(timezone.utc).date().isoformat(),
    }


def guidance_for(key: str, *, kind: str = "condition") -> dict[str, Any] | None:
    """Live NHS if configured and cached, else the curated corpus."""
    if kind == "condition":
        cached = _cached(key)
        if cached is not None:
            return cached
        fetched = _fetch_nhs(key)
        if fetched is not None:
            _store(key, fetched)
            return fetched
        entry = corpus.for_condition(key)
    else:
        entry = corpus.for_factor(key)

    return entry.as_dict() if entry else None


def collect(condition_keys: list[str], factor_keys: list[str]) -> list[dict[str, Any]]:
    """Guidance for what the report is actually going to talk about.

    Scoped to the top findings rather than everything, because a wall of
    citations is not evidence, it is noise.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in condition_keys:
        entry = guidance_for(key, kind="condition")
        if entry and entry["url"] not in seen:
            seen.add(entry["url"])
            out.append(entry)
    for key in factor_keys:
        entry = guidance_for(key, kind="factor")
        if entry and entry["url"] not in seen:
            seen.add(entry["url"])
            out.append(entry)
    return out
