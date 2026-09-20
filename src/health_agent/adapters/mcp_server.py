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
import os
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from health_agent.adapters.core import assess
from health_agent.privacy import DISCLAIMER, NOTICE

INSTRUCTIONS = f"""\
Fitty estimates a person's probability of developing common chronic \
conditions and shows which areas of their life carry the most weight in \
that picture, with the published research behind each one.

It is a consultation, not medical advice. It works from a questionnaire -- \
no examination, no history, often no bloods -- which is enough to say where \
the weight sits and not enough to tell anyone what to do about it. Report \
what carries weight; leave what to do about it to a clinician.

Collect what you can conversationally and pass it to `assess_health` -- every \
field is optional, and the tool will tell you what else it needs rather than \
guessing. Do not invent values you have not been told.

Life expectancy is calculated but deliberately not returned. The response \
gives you a question to ask; only call `life_expectancy` if they say yes.

{NOTICE}

{DISCLAIMER}"""

server = MCPServer(
    name="fitty",
    title="Fitty",
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
        "conditions and which areas of their life carry the most weight in "
        "that picture.\n\n"
        "ASK FOR THESE, IN THIS ORDER. They are ordered by how much they "
        "change the answer:\n"
        "  1. Age and sex -- without age there is no estimate at all.\n"
        "  2. Waist measurement, and height and weight. Ask for waist "
        "explicitly; people do not volunteer it. It matters more than BMI "
        "and, unlike BMI, it does not mistake muscle for fat -- without it "
        "anyone who trains reads as overweight.\n"
        "  3. The three recall questions: ever prescribed blood pressure "
        "medication, ever told their blood sugar was high (including in "
        "pregnancy), and whether they eat vegetables most days. No test "
        "needed, and from 30 onwards an estimate is not given without them.\n"
        "  4. Lifestyle: smoking, alcohol units a week, exercise minutes a "
        "week, typical sleep, diet 1-5, stress 1-5.\n"
        "  5. Anything measured they happen to know -- blood pressure, "
        "HbA1c, cholesterol -- and any current symptoms.\n\n"
        "Ask for the lot in one message rather than interrogating them one "
        "field at a time, and tell them to skip whatever they do not know.\n\n"
        "Measurements accept any unit -- \"5'11\", 13 stone 4, 32in, 5.7%\" "
        "all work. Pass them as the person said them; do not convert, and do "
        "not ask them to.\n\n"
        "Pass whatever you know and omit what you do not. Nothing here needs "
        "a blood test: three recall questions (blood pressure medication, "
        "ever being told their blood sugar was high, whether they eat "
        "vegetables most days) are enough to produce an estimate at any age. "
        "If the data is still too thin the tool returns targeted follow-up "
        "questions instead of a guess -- ask them and call again.\n\n"
        "If they mention any current symptoms, pass them in `symptoms` or "
        "`notes`. Combinations that warrant urgent attention are picked up "
        "there and returned as `act_on_this_first`.\n\n"
        "YOU write the report. The response returns findings plus a "
        "`presentation` object: include every string in "
        "`must_include_verbatim` as written, follow `must_not`, and build "
        "the report around `ranked_areas` in the order given. Render "
        "`risk_table` as a markdown table and `projection` as a line chart "
        "-- those arrive as data precisely so the figures cannot drift in "
        "the retelling. Those "
        "sentences are worded the way they are for safety reasons -- "
        "reproduce them rather than rephrasing them.\n\n"
        "Fitty names where the weight sits in someone's picture. It does "
        "not tell them what to do about it, and neither should you -- a "
        "questionnaire is not grounds for a treatment plan. Say what "
        "carries weight, cite the research showing the area matters, and "
        "leave the specifics to a clinician.\n\n" + NOTICE
    ),
)
def assess_health(
    age: Annotated[int | None, Field(None, ge=0, le=120)] = None,
    sex: Annotated[Literal["male", "female", "other"] | None, Field(None)] = None,
    height_cm: Annotated[str | float | None, Field(None, description="any unit: 180, 5'11\", 1.8m, 71in")] = None,
    weight_kg: Annotated[str | float | None, Field(None, description="any unit: 85kg, 187lb, 13 stone 4")] = None,
    waist_cm: Annotated[str | float | None, Field(None, description="ASK FOR THIS -- any unit, 82cm or 32in. It matters more than BMI and does not mistake muscle for fat, so without it anyone who trains reads as overweight")] = None,
    systolic_bp: Annotated[int | None, Field(None, ge=50, le=300, description="top number")] = None,
    diastolic_bp: Annotated[int | None, Field(None, ge=30, le=200)] = None,
    resting_hr: Annotated[int | None, Field(None, ge=25, le=220)] = None,
    hba1c_mmol_mol: Annotated[str | float | None, Field(None, description="either scale: 39 mmol/mol or 5.7%")] = None,
    fasting_glucose_mmol_l: Annotated[str | float | None, Field(None, description="mmol/L or mg/dL")] = None,
    total_cholesterol_mmol_l: Annotated[str | float | None, Field(None, description="mmol/L or mg/dL")] = None,
    hdl_mmol_l: Annotated[str | float | None, Field(None, description="mmol/L or mg/dL")] = None,
    ldl_mmol_l: Annotated[str | float | None, Field(None, description="mmol/L or mg/dL")] = None,
    triglycerides_mmol_l: Annotated[str | float | None, Field(None, description="mmol/L or mg/dL")] = None,
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
    on_bp_medication: Annotated[bool | None, Field(None, description="ever prescribed blood pressure medication")] = None,
    previously_high_glucose: Annotated[bool | None, Field(None, description="ever told their blood sugar was high, including in pregnancy")] = None,
    eats_vegetables_daily: Annotated[bool | None, Field(None, description="vegetables, fruit or berries most days")] = None,
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
    name="life_expectancy",
    title="Life expectancy, if they asked for it",
    description=(
        "Call this ONLY after the person has been asked whether they want "
        "their life expectancy and has said yes. `assess_health` deliberately "
        "withholds it and gives you the question to ask.\n\n"
        "Some people want that number and act on it; for others it arrives "
        "as a death sentence they did not request, so it is never volunteered."
        "\n\nPass the same details you gave `assess_health`. Returns the "
        "national baseline, the adjusted estimate, what is driving the gap, "
        "and how much of it looks recoverable.\n\n" + NOTICE
    ),
)
def life_expectancy(
    age: Annotated[int, Field(ge=0, le=120)],
    sex: Annotated[Literal["male", "female", "other"] | None, Field(None)] = None,
    height_cm: Annotated[str | float | None, Field(None)] = None,
    weight_kg: Annotated[str | float | None, Field(None)] = None,
    waist_cm: Annotated[str | float | None, Field(None)] = None,
    smoking_status: Annotated[Literal["never", "former", "current"] | None, Field(None)] = None,
    cigarettes_per_day: Annotated[int | None, Field(None, ge=0, le=100)] = None,
    alcohol_units_per_week: Annotated[float | None, Field(None, ge=0, le=200)] = None,
    moderate_activity_minutes_per_week: Annotated[int | None, Field(None, ge=0, le=5000)] = None,
    sleep_hours_avg: Annotated[float | None, Field(None, ge=0, le=24)] = None,
    diet_quality_self_rating: Annotated[int | None, Field(None, ge=1, le=5)] = None,
    perceived_stress_rating: Annotated[int | None, Field(None, ge=1, le=5)] = None,
    systolic_bp: Annotated[int | None, Field(None, ge=50, le=300)] = None,
    diastolic_bp: Annotated[int | None, Field(None, ge=30, le=200)] = None,
    hba1c_mmol_mol: Annotated[str | float | None, Field(None)] = None,
    on_bp_medication: Annotated[bool | None, Field(None)] = None,
    previously_high_glucose: Annotated[bool | None, Field(None)] = None,
    eats_vegetables_daily: Annotated[bool | None, Field(None)] = None,
) -> dict[str, Any]:
    from health_agent.domain.profile import HealthProfile
    from health_agent.domain.units import normalise
    from health_agent.scoring.actuarial import estimate
    from health_agent.scoring.backend import get_backend
    from health_agent.scoring.scorer import RiskScorer

    fields = {k: v for k, v in locals().items() if v is not None and k != "Any"}
    clean, conversions, problems = normalise(fields)
    profile = HealthProfile(**{k: v for k, v in clean.items()
                               if k in HealthProfile.model_fields})

    from health_agent.domain import evidence as evidence_rules

    assessment = RiskScorer(get_backend()).score(profile)
    # Same age-aware check the assessment used. Without it this falls back
    # to a raw classifier threshold and refuses people the assessment just
    # accepted -- which is exactly what it did to a fit 23-year-old.
    evidence = evidence_rules.assess(profile)
    result = estimate(assessment, profile, evidence)
    if result is None:
        missing = list(evidence.missing_required) or ["age"]
        return _privacy_footer(
            {
                "available": False,
                "reason": "Not enough to give an honest estimate yet.",
                "need": missing,
                "say_this": (
                    "Ask for what is in `need` and try again. Do not guess "
                    "at a figure, and do not imply one."
                ),
            }
        )

    payload = result.as_dict()
    payload["available"] = True
    payload["how_to_present"] = (
        "Give the adjusted figure and the national baseline together, so it "
        "reads as a comparison rather than a verdict. Then the recoverable "
        "years and what is driving them. Say plainly that this is a national "
        "average adjusted for lifestyle, not a prediction about them, and "
        "that the point of it is the part they can change."
    )
    if conversions:
        payload["units_interpreted"] = conversions
    if problems:
        payload["unit_problems"] = problems
    return _privacy_footer(payload)


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


def build_http_app(*, require_auth: bool = False):
    """The ASGI app a host serves, wrapped in bearer-token auth.

    Also carries the health endpoints a platform needs to know the container
    is alive, and the wearable ingest routes so one deployment serves both.
    """
    import uvicorn  # noqa: F401  (imported here so stdio has no dependency)
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from health_agent.adapters.auth import protect
    from health_agent.config import SETTINGS

    async def healthz(_request):
        return JSONResponse(
            {
                "ok": True,
                "response_mode": SETTINGS.response_mode,
                "scoring": "jev" if SETTINGS.has_typesafe_key
                           else "openrouter" if os.getenv("OPENROUTER_API_KEY")
                           else "offline-fake",
            }
        )

    app = server.streamable_http_app(host="0.0.0.0")
    app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))
    return protect(app, require=require_auth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="streamable-http is what a remote MCP connector needs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--require-auth", action="store_true",
        help="refuse to start without MCP_AUTH_TOKEN. Use this in deployment.",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        server.run("stdio")
        return

    import uvicorn

    uvicorn.run(
        build_http_app(require_auth=args.require_auth),
        host=args.host, port=args.port, log_level="info",
    )


if __name__ == "__main__":
    main()
