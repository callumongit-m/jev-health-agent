"""The reasoning model, with an offline stand-in.

``OfflineReasoner`` is not a mock that skips the interesting part: it emits real
tool calls and consumes real ToolMessages, so the reason <-> tools cycle in the
graph is exercised identically whether or not an API key is present.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from health_agent.config import SETTINGS

SYSTEM_PROMPT = """\
You are a health risk analyst. A calibrated classifier has already produced \
probabilities for this person; your job is to explain and act on them, not to \
re-estimate them.

Rules:
- Never restate a probability the classifier did not give you, and never invent one.
- Where a probability's certainty is low, speak in ranges and say why.
- For life expectancy, call `actuarial_calc`. Do not do the arithmetic yourself: \
your answer must be reproducible, and mental arithmetic is not.
- Use web search when you would otherwise assert an effect size from memory -- \
what a guideline currently recommends, or how much a given change is shown to \
help. Cite what you find. If you did not search, say the figure is approximate \
rather than implying a source you did not check.
- Only call `set_health_reminder` if the person asked to be reminded.
- Rank suggestions by years recoverable, highest first. Be concrete and specific: \
"walk 30 minutes after dinner, five days a week" beats "exercise more".
- This is not a diagnosis. Say so, once, without hedging every sentence.
"""


class OfflineReasoner:
    """Deterministic stand-in that still drives the tool loop."""

    def __init__(self, tools: Sequence[Any] | None = None) -> None:
        self._tools = list(tools or [])

    def bind_tools(self, tools: Sequence[Any]) -> "OfflineReasoner":
        return OfflineReasoner(tools)

    def invoke(self, messages: list[BaseMessage], **_: Any) -> AIMessage:
        tool_results = {
            m.name: m.content for m in messages if isinstance(m, ToolMessage)
        }

        if "actuarial_calc" not in tool_results:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "actuarial_calc", "args": {}, "id": "offline-le-1"}
                ],
            )

        return AIMessage(content=self._compose(tool_results["actuarial_calc"]))

    @staticmethod
    def _compose(actuarial_json: str) -> str:
        try:
            data = json.loads(actuarial_json)
        except json.JSONDecodeError:
            data = {}

        if data.get("error"):
            return (
                "There is not enough information yet to estimate life expectancy "
                f"({data.get('reason', data['error'])}). The risk probabilities "
                "above still stand on the data provided."
            )

        per_factor = data.get("per_factor_years", {})
        ranked = sorted(per_factor.items(), key=lambda kv: -kv[1])[:3]
        lines = [
            f"Estimated remaining life expectancy: "
            f"{data.get('adjusted_remaining_years')} years "
            f"(national baseline for this age and sex is "
            f"{data.get('baseline_remaining_years')}).",
            f"Roughly {data.get('years_lost_to_modifiable_factors')} years are "
            f"attributable to modifiable factors, of which about "
            f"{data.get('years_recoverable')} look recoverable.",
            "",
            "Biggest levers, highest first:",
        ]
        lines += [f"  - {key.replace('_', ' ')}: {years} years" for key, years in ranked]
        lines += ["", data.get("caveat", "")]
        return "\n".join(lines)


def get_reasoner(tools: Sequence[Any]) -> Any:
    """Real model when a key is present, offline stand-in otherwise.

    Anthropic's server-side web search is bound alongside the client tools.
    ``bind_tools`` passes a raw Anthropic tool schema straight through, and
    because the search executes on Anthropic's side its results come back in
    the same response -- it never produces a ``tool_call`` for the graph's
    ToolNode to handle, so the routing needs no special case.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return OfflineReasoner().bind_tools(tools)

    from langchain_anthropic import ChatAnthropic

    bound: list[Any] = list(tools)
    if SETTINGS.enable_web_search:
        bound.append(SETTINGS.web_search_tool)

    return ChatAnthropic(
        model=SETTINGS.reasoning_model,
        temperature=SETTINGS.reasoning_temperature,
        max_tokens=2000,
    ).bind_tools(bound)


def reasoner_name() -> str:
    return SETTINGS.reasoning_model if os.getenv("ANTHROPIC_API_KEY") else "offline"
