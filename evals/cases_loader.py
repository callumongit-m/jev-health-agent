"""Loads persona cases and ordering pairs from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from health_agent.domain.profile import HealthProfile

CASES_DIR = Path(__file__).parent / "cases"


@dataclass(frozen=True, slots=True)
class Persona:
    id: str
    profile: HealthProfile
    tags: tuple[str, ...] = ()
    expect: dict[str, Any] = field(default_factory=dict)


def load_personas() -> dict[str, Persona]:
    raw = yaml.safe_load((CASES_DIR / "personas.yaml").read_text())
    return {
        entry["id"]: Persona(
            id=entry["id"],
            profile=HealthProfile(**entry["profile"]),
            tags=tuple(entry.get("tags", ())),
            expect=entry.get("expect", {}),
        )
        for entry in raw
    }


def load_orderings() -> list[tuple[str, str, str]]:
    raw = yaml.safe_load((CASES_DIR / "orderings.yaml").read_text())
    return [tuple(row) for row in raw]  # type: ignore[misc]
