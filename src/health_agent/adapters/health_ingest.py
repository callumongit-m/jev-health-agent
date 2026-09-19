"""Remote Apple Health ingest endpoints.

Two ways in, both authenticated by the sync token from `start_health_sync`:

  POST /ingest/apple-health          scheduled JSON push from the phone
  POST /ingest/apple-health/upload   one-off export.zip

Both resume the same assessment thread and re-score, so the person's picture
updates without them starting over.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, UploadFile
from fastapi import File as FastAPIFile

from health_agent.adapters import sync_store
from health_agent.domain.profile import COMPUTED_FIELDS, HealthProfile
from health_agent.ingest import health_sync
from health_agent.privacy import NOTICE, redact

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest/apple-health", tags=["ingest"])

#: An export.zip is tens of MB; anything far larger is not a health export.
MAX_UPLOAD_BYTES = 300 * 1024 * 1024


def _token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    return authorization.split(" ", 1)[1].strip()


def _session(authorization: str | None) -> tuple[str, HealthProfile]:
    token = _token(authorization)
    record = sync_store.resolve(token)
    if record is None:
        raise HTTPException(401, "unknown or expired sync token")
    stored = record.get("profile_json")
    profile = HealthProfile(**json.loads(stored)) if stored else HealthProfile()
    return token, profile


def _apply(token: str, profile: HealthProfile, fields: dict, provenance: dict) -> dict:
    updated = health_sync.merge_into(profile, fields, provenance)
    sync_store.remember_profile(
        token, updated.model_dump_json(exclude=COMPUTED_FIELDS | {"provenance"})
    )
    record = sync_store.resolve(token)
    thread_id = record["thread_id"] if record else None

    from health_agent.adapters.core import assess

    result = assess(updated, thread_id=thread_id)
    return {
        "status": result["status"],
        "data_sufficiency": result.get("data_sufficiency"),
        "known_fields": sorted(updated.known_fields()),
        "privacy": NOTICE,
    }


@router.post("")
async def push(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Scheduled push from Health Auto Export or an iOS Shortcut."""
    token, profile = _session(authorization)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")

    fields, provenance, report = health_sync.parse_payload(payload)
    # never log values, only which fields arrived
    logger.info("health sync: %s", redact(fields))

    if not fields:
        # Report rather than 400: the payload shape varies by app and version,
        # and the caller needs to see what was and was not understood.
        return {
            "accepted": False,
            "report": report,
            "hint": "No recognised metrics. `report.skipped` lists the metric "
                    "names that arrived, which is what to map next.",
            "privacy": NOTICE,
        }

    return {"accepted": True, "report": report, **_apply(token, profile, fields, provenance)}


@router.post("/upload")
async def upload(
    authorization: str | None = Header(default=None),
    file: UploadFile = FastAPIFile(...),
) -> dict[str, Any]:
    """One-off export.zip, for people who would rather not install an app."""
    token, profile = _session(authorization)

    body = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "export larger than 300MB")

    suffix = ".zip" if (file.filename or "").endswith(".zip") else ".xml"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        handle.write(body)
        handle.flush()
        from health_agent.ingest.apple_health import parse_export

        try:
            fields, provenance = parse_export(Path(handle.name))
        except Exception as exc:
            raise HTTPException(400, f"could not read that export: {exc}")

    if not fields:
        return {
            "accepted": False,
            "hint": "No health records in the last 28 days. The export may be "
                    "old, or recording may be off.",
            "privacy": NOTICE,
        }

    return {"accepted": True, **_apply(token, profile, fields, provenance)}
