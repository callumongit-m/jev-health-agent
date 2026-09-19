"""Environment and feature flags."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

LifeExpectancyMode = Literal["actuarial", "llm_raw"]

# Module-level defaults: with slots=True, class attribute access returns a
# descriptor rather than the default value, so defaults live here instead.
DEFAULT_REASONING_MODEL = "claude-sonnet-5"
DEFAULT_LIFE_EXPECTANCY_MODE: LifeExpectancyMode = "actuarial"
DEFAULT_SUFFICIENCY_THRESHOLD = 0.5
DEFAULT_CHECKPOINT_PATH = "health_agent.sqlite"


@dataclass(frozen=True, slots=True)
class Settings:
    reasoning_model: str = DEFAULT_REASONING_MODEL
    reasoning_temperature: float = 0.2
    max_tool_iterations: int = 6

    #: "actuarial" -- the LLM calls a deterministic calculator (default).
    #: "llm_raw"   -- the LLM does the arithmetic itself. Useful for measuring
    #:                how much run-to-run spread that actually costs.
    life_expectancy_mode: LifeExpectancyMode = DEFAULT_LIFE_EXPECTANCY_MODE

    #: Below this Jev data-sufficiency the graph asks for more instead of guessing.
    sufficiency_threshold: float = DEFAULT_SUFFICIENCY_THRESHOLD

    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH

    @property
    def has_anthropic_key(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    @property
    def has_typesafe_key(self) -> bool:
        return bool(os.getenv("TYPESAFE_API_KEY"))

    @property
    def has_search_key(self) -> bool:
        return bool(os.getenv("TAVILY_API_KEY"))


def load() -> Settings:
    return Settings(
        reasoning_model=os.getenv("REASONING_MODEL", DEFAULT_REASONING_MODEL),
        life_expectancy_mode=os.getenv(  # type: ignore[arg-type]
            "LIFE_EXPECTANCY_MODE", DEFAULT_LIFE_EXPECTANCY_MODE
        ),
        sufficiency_threshold=float(
            os.getenv("SUFFICIENCY_THRESHOLD", DEFAULT_SUFFICIENCY_THRESHOLD)
        ),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH),
    )


SETTINGS = load()
