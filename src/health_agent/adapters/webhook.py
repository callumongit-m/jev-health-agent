"""Terra webhook receiver -- the answer to "how does this not go stale".

New device data resumes the *existing* checkpointed thread and re-scores,
rather than starting a fresh conversation. The profile lives in the graph
checkpointer keyed by thread_id, so an Oura sync three weeks later updates the
same picture.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from health_agent.domain.profile import HealthProfile
from health_agent.ingest.terra import merge_into, normalise, verify_webhook
from health_agent.privacy import redact

logger = logging.getLogger(__name__)

router = APIRouter()

#: thread_id -> profile. A real deployment reads this from the graph
#: checkpointer; kept in memory here so the path is exercisable without one.
_PROFILES: dict[str, HealthProfile] = {}


def remember(thread_id: str, profile: HealthProfile) -> None:
    _PROFILES[thread_id] = profile


def forget(thread_id: str) -> bool:
    return _PROFILES.pop(thread_id, None) is not None


@router.post("/webhooks/terra")
async def terra_webhook(
    request: Request,
    terra_signature: str | None = Header(default=None, alias="terra-signature"),
) -> dict[str, Any]:
    body = await request.body()

    # An unverifiable webhook carrying health data is rejected, not trusted.
    if not verify_webhook(body, terra_signature):
        raise HTTPException(status_code=401, detail="invalid or missing signature")

    payload = await request.json()
    thread_id = (payload.get("user") or {}).get("reference_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="no reference_id on payload")

    fields, _ = normalise(payload)
    if not fields:
        return {"updated": False, "reason": "no usable fields in payload"}

    # never log the values themselves
    logger.info("terra update for thread %s: %s", thread_id, redact(fields))

    existing = _PROFILES.get(thread_id)
    if existing is None:
        return {"updated": False, "reason": "unknown thread; nothing to update"}

    updated = merge_into(existing, payload)
    _PROFILES[thread_id] = updated

    from health_agent.adapters.core import assess

    result = assess(updated, thread_id=thread_id)
    return {
        "updated": True,
        "fields_changed": sorted(fields),
        "status": result["status"],
        "data_sufficiency": result.get("data_sufficiency"),
    }
