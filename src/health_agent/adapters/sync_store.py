"""Sync tokens: the link between a phone and an assessment thread.

A token is a capability -- whoever holds it can post health data to one
thread. So it is generated server-side, never derived from anything
guessable, and expires.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from health_agent.config import SETTINGS
from health_agent.ingest.health_sync import new_sync_token

TTL = timedelta(days=180)


def base_url() -> str:
    return os.getenv("PUBLIC_BASE_URL", "http://localhost:9000").rstrip("/")


def _connect() -> sqlite3.Connection:
    path = Path(SETTINGS.checkpoint_path).with_suffix(".sync.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS syncs (
               token TEXT PRIMARY KEY,
               thread_id TEXT NOT NULL,
               created_at TEXT NOT NULL,
               expires_at TEXT NOT NULL,
               last_seen_at TEXT,
               profile_json TEXT
           )"""
    )
    return conn


def create_sync(*, thread_id: str | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    token = new_sync_token()
    thread = thread_id or str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO syncs (token, thread_id, created_at, expires_at) "
            "VALUES (?,?,?,?)",
            (token, thread, now.isoformat(), (now + TTL).isoformat()),
        )

    url = f"{base_url()}/ingest/apple-health"
    return {
        "sync_url": url,
        "token": token,
        "thread_id": thread,
        "expires": (now + TTL).date().isoformat(),
        "setup": {
            "recommended": (
                "Install 'Health Auto Export' from the App Store, add an "
                "Automation, choose REST API, set the URL and add a header "
                f"'Authorization: Bearer {token}'. Set it to run daily. "
                "Apple blocks health access while the phone is locked, so it "
                "will sync the next time the phone is unlocked."
            ),
            "one_off_alternative": (
                "If they would rather not install anything: iPhone Health app "
                "-> profile picture -> Export All Health Data, then upload the "
                f"export.zip to {base_url()}/ingest/apple-health/upload with "
                "the same Authorization header. That is a snapshot, not a feed."
            ),
            "no_tech_option": (
                "They can also just tell you their numbers and you pass them "
                "to assess_health directly."
            ),
        },
        "note": "Treat the token like a password -- it lets anyone post health "
                "data to this assessment.",
    }


def resolve(token: str) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT thread_id, profile_json FROM syncs "
            "WHERE token=? AND expires_at > ?",
            (token, now),
        ).fetchone()
    return {"thread_id": row[0], "profile_json": row[1]} if row else None


def remember_profile(token: str, profile_json: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE syncs SET profile_json=?, last_seen_at=? WHERE token=?",
            (profile_json, datetime.now(timezone.utc).isoformat(), token),
        )


def delete_for_thread(thread_id: str) -> int:
    with _connect() as conn:
        return conn.execute(
            "DELETE FROM syncs WHERE thread_id=?", (thread_id,)
        ).rowcount
