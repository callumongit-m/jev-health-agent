"""Health risk agent.

Loading .env happens here rather than in config.py: any entry point that
imports *anything* from this package needs the environment populated, and
several import `scoring.backend` without ever touching `config`.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# override=False so a real exported variable always beats a stale .env line.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
