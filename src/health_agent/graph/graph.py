"""Graph wiring.

    ingest -> screen --(red flag)--> END
                 |
              (clear)
                 v
             classify (Jev, one batched call)
                 |
               gate --(thin data)--> clarify -> END      [status: needs_input]
                 |
               (ok)
                 v
              reason <----------+
                 |  |           |
                 |  +-> tools --+      actuarial_calc / evidence search / reminder
                 v
              respond -> END            [status: complete]

Two cycles matter. reason <-> tools is the agentic one. clarify is the other:
Jev's sufficiency signal decides whether there is enough to score at all, and
if not the graph returns targeted questions to the calling agent instead of
guessing. A cheap calibrated signal gating an expensive reasoning step is the
whole reason a System One model belongs in a graph.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from health_agent.config import SETTINGS
from health_agent.graph.nodes import make_nodes
from health_agent.graph.state import AgentState
from health_agent.graph.tools import TOOLS
from health_agent.scoring.scorer import RiskScorer


def _after_screen(state: AgentState) -> Literal["classify", "__end__"]:
    return END if state.get("red_flags") else "classify"


def _gate(state: AgentState) -> Literal["reason", "clarify"]:
    """Two signals, and both have to agree.

    The deterministic one is age-aware: what counts as enough evidence at 21
    is not what counts at 55, and a flat threshold turns away the young
    people an early warning helps most.

    Jev's calibrated sufficiency is kept as a veto rather than the primary
    gate. It can see things a field checklist cannot -- contradictory values,
    a profile that does not hang together -- but only a strong objection
    overrides a profile that meets its age band, otherwise the deterministic
    rule means nothing.
    """
    evidence = state["evidence"]
    if not evidence.sufficient:
        return "clarify"
    if state["assessment"].data_sufficiency < SETTINGS.classifier_veto_threshold:
        return "clarify"
    return "reason"


def _after_gate(state: AgentState) -> Literal["reason", "compose", "clarify"]:
    """In data mode the calling model writes the report, so no LLM runs here.

    Life expectancy still needs the actuarial calculator, so `compose` calls
    it directly rather than asking a model to call it as a tool.
    """
    decision = _gate(state)
    if decision == "clarify":
        return "clarify"
    return "reason" if SETTINGS.response_mode == "narrated" else "compose"


def _after_reason(state: AgentState) -> Literal["tools", "respond"]:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if getattr(last, "tool_calls", None):
        if len(state.get("tool_calls_made") or []) > SETTINGS.max_tool_iterations:
            return "respond"  # circuit breaker on a runaway tool loop
        return "tools"
    return "respond"


def build_graph(scorer: RiskScorer | None = None, *, checkpointer=None):
    nodes = make_nodes(scorer or RiskScorer())

    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "screen")
    builder.add_conditional_edges("screen", _after_screen, ["classify", END])
    builder.add_conditional_edges(
        "classify", _after_gate, ["reason", "compose", "clarify"]
    )
    builder.add_edge("compose", END)
    builder.add_edge("clarify", END)
    builder.add_conditional_edges("reason", _after_reason, ["tools", "respond"])
    builder.add_edge("tools", "reason")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)
