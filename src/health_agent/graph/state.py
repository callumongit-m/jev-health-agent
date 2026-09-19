"""Graph state."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from health_agent.domain.profile import HealthProfile
from health_agent.domain.results import RiskAssessment

Status = Literal["pending", "seek_care", "needs_input", "complete"]


class AgentState(TypedDict, total=False):
    """Checkpointed per thread_id, so a later wearable update resumes the same
    conversation and re-scores instead of starting over."""

    messages: Annotated[list, add_messages]

    #: free text the caller supplied alongside (or instead of) structured fields
    raw_text: str | None
    profile: HealthProfile
    evidence: Any
    assessment: RiskAssessment
    life_expectancy: dict[str, Any] | None

    red_flags: list[str]
    symptom_patterns: list[str]
    urgent_guidance: str | None
    clarifying_questions: list[str]
    status: Status
    answer: str | None
    presentation: dict[str, Any] | None
    tool_calls_made: list[str]
