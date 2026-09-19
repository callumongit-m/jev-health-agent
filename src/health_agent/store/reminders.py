"""Reminder storage.

Persisting anything is opt-in: a reminder is the one thing the person has
explicitly asked us to remember, so it is the only thing that outlives the
session. Everything else stays in the checkpointer.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from health_agent.config import SETTINGS

_CADENCES = {
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "fortnightly": timedelta(weeks=2),
    "monthly": timedelta(days=30),
    "quarterly": timedelta(days=91),
    "every 3 months": timedelta(days=91),
    "every 6 months": timedelta(days=182),
    "yearly": timedelta(days=365),
}

#: Reminders expire rather than living forever. Renewed by continued use.
TTL = timedelta(days=365)


def _connect() -> sqlite3.Connection:
    path = Path(SETTINGS.checkpoint_path).with_suffix(".reminders.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reminders (
               id TEXT PRIMARY KEY,
               subject_ref TEXT,
               what TEXT NOT NULL,
               cadence TEXT NOT NULL,
               next_due TEXT NOT NULL,
               expires_at TEXT NOT NULL
           )"""
    )
    return conn


def _interval(cadence: str) -> timedelta:
    return _CADENCES.get(cadence.strip().lower(), timedelta(weeks=1))


def add_reminder(*, what: str, cadence: str, subject_ref: str = "default") -> dict:
    now = datetime.now(timezone.utc)
    record = {
        "id": str(uuid.uuid4()),
        "subject_ref": subject_ref,
        "what": what,
        "cadence": cadence,
        "next_due": (now + _interval(cadence)).isoformat(),
        "expires_at": (now + TTL).isoformat(),
    }
    with _connect() as conn:
        conn.execute(
            "INSERT INTO reminders VALUES (:id,:subject_ref,:what,:cadence,"
            ":next_due,:expires_at)",
            record,
        )
    return record


def list_reminders(subject_ref: str = "default") -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id,subject_ref,what,cadence,next_due,expires_at FROM reminders "
            "WHERE subject_ref=? AND expires_at > ?",
            (subject_ref, now),
        ).fetchall()
    keys = ["id", "subject_ref", "what", "cadence", "next_due", "expires_at"]
    return [dict(zip(keys, row)) for row in rows]


def delete_all(subject_ref: str = "default") -> int:
    """Hard delete. Backs the `delete_my_data` tool."""
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM reminders WHERE subject_ref=?", (subject_ref,))
        return cursor.rowcount
