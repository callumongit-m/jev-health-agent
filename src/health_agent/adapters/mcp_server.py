"""MCP adapter.

This is the consumer-facing front door: a person pastes this server's URL into
Claude, ChatGPT or another MCP host, and the host LLM collects their details
through the tool schema and calls us. The tool's input schema IS the intake
form -- the host does the natural-language parsing for free.

Contrast with the A2A adapter, which exposes an opaque agent rather than tools.
Both are thin; all behaviour lives in `adapters/core.py`.

    uv run python -m health_agent.adapters.mcp_server --transport streamable-http
"""

from __future__ import annotations

import argparse
import json
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from health_agent.adapters.core import assess
from health_agent.privacy import DISCLAIMER, NOTICE

INSTRUCTIONS = f"""\
Estimates a person's probability of developing common chronic conditions, their \
life expectancy, and what would most improve both.

Collect what you can conversationally and pass it to `assess_health` -- every \
field is optional, and the tool will tell you what else it needs rather than \
guessing. Do not invent values you have not been told.

{NOTICE}

{DISCLAIMER}"""

server = MCPServer(
    name="health-risk-agent",
    title="Health Risk Agent",
    instructions=INSTRUCTIONS,
    version="0.2.0",
)


def _privacy_footer(payload: dict[str, Any]) -> dict[str, Any]:
    """The notice is not optional on any response."""
    payload.setdefault("privacy", NOTICE)
    payload.setdefault("disclaimer", DISCLAIMER)
    return payload


@server.tool(
    name="assess_health",
    title="Assess health risk",
    description=(
        "Estimate this person's probability of developing common chronic "
        "conditions, their life expectancy, and the changes that would help "
        "most. Pass whatever you know; omit what you do not. If the data is "
        "too thin the tool returns targeted follow-up questions instead of a "
        "guess -- ask them and call again.\n\n" + NOTICE
    ),
)
def assess_health(
    age: Annotated[int | None, Field(None, ge=0, le=120)] = None,
    sex: Annotated[Literal["male", "female", "other"] | None, Field(None)] = None,
    height_cm: Annotated[float | None, Field(None, gt=0, le=260)] = None,
    weight_kg: Annotated[float | None, Field(None, gt=0, le=500)] = None,
    systolic_bp: Annotated[int | None, Field(None, ge=50, le=300, description="top number")] = None,
    diastolic_bp: Annotated[int | None, Field(None, ge=30, le=200)] = None,
    resting_hr: Annotated[int | None, Field(None, ge=25, le=220)] = None,
    hba1c_mmol_mol: Annotated[float | None, Field(None, ge=15, le=200, description="IFCC mmol/mol, not DCCT %")] = None,
    fasting_glucose_mmol_l: Annotated[float | None, Field(None, ge=1, le=40)] = None,
    total_cholesterol_mmol_l: Annotated[float | None, Field(None, ge=1, le=20)] = None,
    hdl_mmol_l: Annotated[float | None, Field(None, ge=0.1, le=6)] = None,
    ldl_mmol_l: Annotated[float | None, Field(None, ge=0.1, le=15)] = None,
    triglycerides_mmol_l: Annotated[float | None, Field(None, ge=0.1, le=30)] = None,
    egfr: Annotated[float | None, Field(None, ge=1, le=200)] = None,
    alt_u_l: Annotated[float | None, Field(None, ge=1, le=1000)] = None,
    smoking_status: Annotated[Literal["never", "former", "current"] | None, Field(None)] = None,
    cigarettes_per_day: Annotated[int | None, Field(None, ge=0, le=100)] = None,
    alcohol_units_per_week: Annotated[float | None, Field(None, ge=0, le=200)] = None,
    moderate_activity_minutes_per_week: Annotated[int | None, Field(None, ge=0, le=5000)] = None,
    steps_daily_avg: Annotated[int | None, Field(None, ge=0, le=100_000)] = None,
    sleep_hours_avg: Annotated[float | None, Field(None, ge=0, le=24)] = None,
    sleep_efficiency_pct: Annotated[float | None, Field(None, ge=0, le=100)] = None,
    diet_quality_self_rating: Annotated[int | None, Field(None, ge=1, le=5, description="1 poor, 5 excellent")] = None,
    perceived_stress_rating: Annotated[int | None, Field(None, ge=1, le=5, description="1 none, 5 severe")] = None,
    family_history: Annotated[list[str] | None, Field(None, description="e.g. ['type 2 diabetes', 'heart disease']")] = None,
    existing_conditions: Annotated[list[str] | None, Field(None)] = None,
    medications: Annotated[list[str] | None, Field(None)] = None,
    symptoms: Annotated[list[str] | None, Field(None)] = None,
    notes: Annotated[str | None, Field(None, description="anything they said in their own words")] = None,
    thread_id: Annotated[str | None, Field(None, description="reuse to update an earlier assessment")] = None,
) -> dict[str, Any]:
    fields = {k: v for k, v in locals().items() if k not in ("notes", "thread_id")}
    return _privacy_footer(
        assess({k: v for k, v in fields.items() if v is not None},
               raw_text=notes, thread_id=thread_id)
    )


@server.tool(
    name="start_health_sync",
    title="Connect Apple Health",
    description=(
        "Set up an automatic feed of the person's Apple Health data, so their "
        "assessment stays current instead of going stale. Returns a private "
        "URL and token plus setup steps to read out to them.\n\n"
        "Apple Health already aggregates Apple Watch, Oura, Whoop, Garmin and "
        "Fitbit, so this one connection covers all of them. Use it whenever "
        "someone mentions a wearable, a watch, or not wanting to type numbers "
        "in.\n\n"
        "Note: this agent runs on a server and cannot read files from the "
        "person's phone or computer. The link is how their data reaches it."
        "\n\n" + NOTICE
    ),
)
def start_health_sync(
    thread_id: Annotated[str | None, Field(None, description="reuse to keep one assessment updated")] = None,
) -> dict[str, Any]:
    from health_agent.adapters.sync_store import create_sync

    return _privacy_footer(create_sync(thread_id=thread_id))


@server.tool(
    name="connect_wearable",
    title="Connect a wearable or Apple Health",
    description=(
        "Get a link the person opens to connect Apple Health, Fitbit, Oura, "
        "Garmin or Whoop. Once connected their assessment refreshes as new "
        "data arrives, instead of going stale.\n\n" + NOTICE
    ),
)
def connect_wearable(
    thread_id: Annotated[str | None, Field(None)] = None,
) -> dict[str, Any]:
    from health_agent.ingest.terra import widget_session

    return _privacy_footer(widget_session(reference_id=thread_id))


@server.tool(
    name="set_health_reminder",
    title="Set a health reminder",
    description=(
        "Schedule a recurring nudge, e.g. a weekly weigh-in or a quarterly "
        "HbA1c recheck. Only use this when the person has asked for it -- it "
        "is the one thing that is stored beyond the session.\n\n" + NOTICE
    ),
)
def set_health_reminder(
    what: Annotated[str, Field(description="e.g. 'weigh in and log it'")],
    cadence: Annotated[str, Field(description="daily, weekly, monthly, every 3 months")],
    thread_id: Annotated[str | None, Field(None)] = None,
) -> dict[str, Any]:
    from health_agent.store.reminders import add_reminder

    return _privacy_footer(
        {"scheduled": True, **add_reminder(what=what, cadence=cadence,
                                           subject_ref=thread_id or "default")}
    )


@server.tool(
    name="delete_my_data",
    title="Delete everything stored",
    description=(
        "Permanently erase everything stored for this person: reminders, any "
        "connected device link, and the saved conversation state. Irreversible."
        "\n\n" + NOTICE
    ),
)
def delete_my_data(
    thread_id: Annotated[str | None, Field(None)] = None,
) -> dict[str, Any]:
    from health_agent.store.reminders import delete_all

    removed = delete_all(subject_ref=thread_id or "default")
    return _privacy_footer(
        {
            "deleted": True,
            "reminders_removed": removed,
            "note": "Nothing further is retained. Assessment data was never "
                    "stored beyond the session.",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="streamable-http is what a remote MCP connector needs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        server.run("stdio")
    else:
        # host/port are transport kwargs in mcp 2.x, not server settings
        server.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
