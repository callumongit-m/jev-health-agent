"""Environment and feature flags."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

LifeExpectancyMode = Literal["actuarial", "llm_raw"]

#: "data"     -- return findings plus a presentation contract and let the
#:               calling model write it up. No LLM cost on our side, which is
#:               the right default when the caller is already a capable model.
#: "narrated" -- write the report ourselves, on our key. For callers that have
#:               no model of their own, or when we need to control the prose.
ResponseMode = Literal["data", "narrated"]

# Module-level defaults: with slots=True, class attribute access returns a
# descriptor rather than the default value, so defaults live here instead.
DEFAULT_REASONING_MODEL = "claude-sonnet-5"
DEFAULT_LIFE_EXPECTANCY_MODE: LifeExpectancyMode = "actuarial"
DEFAULT_SUFFICIENCY_THRESHOLD = 0.4
#: Age-banded field requirements are the primary gate; the classifier only
#: overrides them when it objects strongly, so this sits well below them.
DEFAULT_CLASSIFIER_VETO = 0.15
DEFAULT_RESPONSE_MODE: "ResponseMode" = "data"
DEFAULT_CHECKPOINT_PATH = "health_agent.sqlite"

#: Anthropic's server-side web search runs on Anthropic's infrastructure -- no
#: separate search provider or API key. The dynamic-filtering variant needs
#: Claude 4.6 or later; older models take the basic one.
_WEB_SEARCH_DYNAMIC = "web_search_20260209"
_WEB_SEARCH_BASIC = "web_search_20250305"
_DYNAMIC_FILTERING_MODELS = (
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5",
)


def web_search_tool(model: str, *, max_uses: int = 3) -> dict:
    """The server-side web search tool definition for a given model."""
    tool_type = (
        _WEB_SEARCH_DYNAMIC
        if any(model.startswith(m) for m in _DYNAMIC_FILTERING_MODELS)
        else _WEB_SEARCH_BASIC
    )
    return {"type": tool_type, "name": "web_search", "max_uses": max_uses}


@dataclass(frozen=True, slots=True)
class Settings:
    reasoning_model: str = DEFAULT_REASONING_MODEL
    #: Server-side web search. Costs per search on top of tokens.
    enable_web_search: bool = True
    reasoning_temperature: float = 0.2
    max_tool_iterations: int = 6

    #: "actuarial" -- the LLM calls a deterministic calculator (default).
    #: "llm_raw"   -- the LLM does the arithmetic itself. Useful for measuring
    #:                how much run-to-run spread that actually costs.
    life_expectancy_mode: LifeExpectancyMode = DEFAULT_LIFE_EXPECTANCY_MODE

    #: Retained for the standalone actuarial guard.
    sufficiency_threshold: float = DEFAULT_SUFFICIENCY_THRESHOLD
    #: Only a strong classifier objection overrides the age-band requirements.
    classifier_veto_threshold: float = DEFAULT_CLASSIFIER_VETO
    #: Who writes the report. See ResponseMode.
    response_mode: ResponseMode = DEFAULT_RESPONSE_MODE

    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH

    @property
    def has_anthropic_key(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    @property
    def has_typesafe_key(self) -> bool:
        return bool(os.getenv("TYPESAFE_API_KEY"))

    @property
    def web_search_tool(self) -> dict:
        return web_search_tool(self.reasoning_model)


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
        enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "1") not in ("0", "false"),
        classifier_veto_threshold=float(
            os.getenv("CLASSIFIER_VETO_THRESHOLD", DEFAULT_CLASSIFIER_VETO)
        ),
        response_mode=os.getenv("RESPONSE_MODE", DEFAULT_RESPONSE_MODE),  # type: ignore[arg-type]
    )


SETTINGS = load()
