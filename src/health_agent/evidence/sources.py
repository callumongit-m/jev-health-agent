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
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from health_agent.config import SETTINGS
from health_agent.evidence import corpus

logger = logging.getLogger(__name__)

# NHS Website Content API v2. Two content areas matter here: `conditions`
# (the Health A-Z) for what each condition is, and `live-well` for the
# lifestyle factors. v1 is retired. Rate limit is 4,000 requests an hour,
# which caching keeps us nowhere near.
#
# There are two NHS platforms and they differ in host *and* auth header,
# which is easy to get wrong -- both return 401 with a key meant for the
# other. Probed against the live endpoints:
#
#   api.service.nhs.uk/nhs-website-content   header `apikey`
#       NHS England API platform (Apigee). This is what the v2 catalogue
#       entry on digital.nhs.uk issues keys for.
#           {"fault":{"faultstring":"Failed to resolve API Key variable
#            request.header.apikey"}}
#
#   api.nhs.uk                                header `subscription-key`
#       The older NHS website developer portal (Azure APIM).
#           {"statusCode":401,"message":"Access denied due to missing
#            subscription key..."}
#
# The sandbox (sandbox.api.service.nhs.uk) needs no key, which is what the
# docs mean by testing without one -- though it was returning 503 when last
# probed.
NHS_PLATFORM_BASE = "https://api.service.nhs.uk/nhs-website-content"
NHS_LEGACY_BASE = "https://api.nhs.uk"
NHS_BASE = os.getenv("NHS_API_BASE", NHS_PLATFORM_BASE)


def _auth_header(api_key: str, base: str) -> dict[str, str]:
    """The two platforms disagree about what the header is called."""
    if "api.service.nhs.uk" in base:
        return {"apikey": api_key}
    return {"subscription-key": api_key}
#: NHS content changes slowly; a stale month is better than a failed request.
CACHE_TTL = timedelta(days=30)

#: condition key -> Health A-Z slug
NHS_CONDITION_SLUGS: dict[str, str] = {
    "t2d_10yr": "type-2-diabetes",
    "cvd_10yr": "cardiovascular-disease",
    "hypertension": "high-blood-pressure-hypertension",
    "metabolic_syndrome": "metabolic-syndrome",
    "sleep_apnoea": "sleep-apnoea",
    "nafld": "non-alcoholic-fatty-liver-disease",
    "ckd": "kidney-disease",
}

#: factor key -> Live Well path. These are the modifiable factors, and Live
#: Well is where NHS puts the advice rather than the pathology.
NHS_LIVE_WELL_SLUGS: dict[str, str] = {
    "smoking_burden": "quit-smoking/nhs-stop-smoking-services-help-you-quit",
    "adiposity": "healthy-weight/managing-your-weight/start-losing-weight",
    "activity_deficit": "exercise/exercise-guidelines/physical-activity-guidelines-for-adults-aged-19-to-64",
    "sleep_debt": "sleep-and-tiredness/how-to-get-to-sleep",
    "alcohol_burden": "alcohol-advice/calculating-alcohol-units",
    "diet_quality": "eat-well/how-to-eat-a-balanced-diet/eating-a-balanced-diet",
    "stress_load": "mental-wellbeing/stress-anxiety-depression/how-to-deal-with-stress",
}

#: Backwards-compatible alias.
NHS_SLUGS = NHS_CONDITION_SLUGS


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


def _extract_points(body: dict[str, Any]) -> list[str]:
    """Pull readable prose out of an NHS content document.

    NHS uses schema.org, so the body is usually a WebPage with `hasPart`
    sections. The exact nesting varies by page type, so this walks a few
    known shapes rather than assuming one.
    """
    points: list[str] = []

    def add(text: Any) -> None:
        cleaned = " ".join(str(text or "").split())
        # strip any markup that slipped through
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if 40 < len(cleaned) < 400 and cleaned not in points:
            points.append(cleaned)

    add(body.get("description"))
    for section in (body.get("hasPart") or [])[:6]:
        if not isinstance(section, dict):
            continue
        add(section.get("description"))
        add(section.get("text"))
        for nested in (section.get("hasPart") or [])[:3]:
            if isinstance(nested, dict):
                add(nested.get("text") or nested.get("description"))
        if len(points) >= 3:
            break

    return points[:3]


def _fetch_nhs(key: str, *, kind: str) -> dict[str, Any] | None:
    """Live NHS Website Content API v2.

    Returns None for any failure -- a missing key, a bad response, a slug
    that has moved. An assessment must never fail because a content API is
    having a bad afternoon; the curated corpus covers it.
    """
    if kind == "condition":
        slug, area = NHS_CONDITION_SLUGS.get(key), "conditions"
    else:
        slug, area = NHS_LIVE_WELL_SLUGS.get(key), "live-well"

    api_key = os.getenv("NHS_API_KEY")
    if not (slug and api_key):
        return None

    try:
        import httpx

        response = httpx.get(
            f"{NHS_BASE.rstrip('/')}/{area}/{slug}",
            headers={**_auth_header(api_key, NHS_BASE), "Accept": "application/json"},
            timeout=8.0,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        logger.warning("NHS fetch failed for %s/%s: %s", area, slug, exc)
        return None

    points = _extract_points(body)
    if not points:
        # A shape we do not recognise. Log the keys, not the content, so the
        # mapping can be fixed without guessing at the schema.
        logger.warning(
            "NHS returned %s/%s in an unrecognised shape; top-level keys: %s",
            area, slug, sorted(body)[:12],
        )
        return None
    return {
        "title": body.get("name") or key.replace("_", " ").title(),
        "points": points[:3],
        "source": "NHS",
        "url": body.get("url") or f"https://www.nhs.uk/{area}/{slug}/",
        "api_base": NHS_BASE,
        "retrieved": datetime.now(timezone.utc).date().isoformat(),
        "live": True,
    }


def guidance_for(key: str, *, kind: str = "condition") -> dict[str, Any] | None:
    """Live NHS if configured and reachable, else the curated corpus."""
    cache_key = f"{kind}:{key}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    fetched = _fetch_nhs(key, kind=kind)
    if fetched is not None:
        _store(cache_key, fetched)
        return fetched

    entry = corpus.for_condition(key) if kind == "condition" else corpus.for_factor(key)
    return entry.as_dict() if entry else None


def collect(
    condition_keys: list[str],
    factor_keys: list[str],
    *,
    with_research: bool = True,
) -> list[dict[str, Any]]:
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
            if with_research:
                # Guidance says what to do; the reviews say how much it
                # helps, which is the number a model would otherwise invent.
                from health_agent.evidence import literature

                papers = literature.for_factor(key)
                if papers:
                    entry = {**entry, "research": papers}
            out.append(entry)
    return out
