"""A2A adapter.

Where MCP exposes *tools with schemas* for a host LLM to drive, A2A exposes an
**opaque agent** that owns a task. A peer sends a message, this agent runs its
own graph, and the peer gets task updates back. The peer never sees the tools,
the Jev call or the reasoning loop -- only the task lifecycle.

That difference is why the clarify branch matters here: when Jev says the data
is too thin, the task moves to `input-required` rather than failing or guessing,
and the peer agent is responsible for going and getting the answers.

    uv run python -m health_agent.adapters.a2a_server --port 9000
    curl localhost:9000/.well-known/agent-card.json
"""

from __future__ import annotations

import argparse
import json
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value

from health_agent.adapters.core import assess
from health_agent.domain.profile import HealthProfile
from health_agent.privacy import DISCLAIMER, NOTICE

logger = logging.getLogger(__name__)

RPC_URL = "/"
CARD_URL = "/.well-known/agent-card.json"


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Fitty",
        description=(
            "Estimates a person's probability of developing common chronic "
            "conditions, their life expectancy, and the changes that would "
            "most improve both. Send whatever health data you have as JSON or "
            "plain text; if it is too thin the task moves to input-required "
            "with specific questions rather than returning a guess. "
            f"{NOTICE} {DISCLAIMER}"
        ),
        version="0.4.0",
        documentation_url="https://github.com/callumongit-m/jev-health-agent",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        # Without this a peer resolves the card and then fails with "no
        # compatible transports found" -- the card must say how to reach us.
        supported_interfaces=[
            AgentInterface(
                url=base_url.rstrip("/") + RPC_URL,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="health-risk-assessment",
                name="Assess health risk",
                description=(
                    "Given age, sex, body measurements, vitals, labs and "
                    "lifestyle, return calibrated probabilities per condition, "
                    "a life expectancy estimate, and ranked interventions."
                ),
                tags=["health", "risk", "prevention", "life-expectancy"],
                examples=[
                    '{"age": 54, "sex": "male", "height_cm": 178, '
                    '"weight_kg": 98, "smoking_status": "former"}',
                    "I'm 41, female, fairly sedentary, about 9 stone, "
                    "and my mum has type 2 diabetes.",
                ],
                input_modes=["text/plain", "application/json"],
                output_modes=["text/plain", "application/json"],
            ),
        ],
    )


def _to_value(payload: dict) -> Value:
    return ParseDict(payload, Value())


def _summarise(payload: dict) -> str:
    """A readable fallback when the payload carries a contract, not prose."""
    presentation = payload.get("presentation") or {}
    lines = [presentation.get("headline", "Assessment complete.")]
    for area in presentation.get("ranked_areas", [])[:3]:
        lines.append(f"- {area['area']} (~{area['years_recoverable']} years)")
    lines += list(presentation.get("must_include_verbatim", []))
    return "\n".join(l for l in lines if l)


def _parse_input(text: str) -> tuple[dict, str | None]:
    """A peer may send JSON, prose, or JSON with prose around it."""
    stripped = (text or "").strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                notes = data.pop("notes", None) or data.pop("raw_text", None)
                return data, notes
        except json.JSONDecodeError:
            pass
    return {}, stripped or None


class HealthAgentExecutor(AgentExecutor):
    """Wraps the same graph both adapters share."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # The Task object itself must reach the queue before any status
        # update, or the handler rejects the stream with
        # "Agent should enqueue Task before TaskStatusUpdateEvent".
        if context.current_task is None:
            await event_queue.enqueue_event(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                )
            )

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        fields, notes = _parse_input(context.get_user_input() or "")
        try:
            profile = HealthProfile(
                **{k: v for k, v in fields.items() if v is not None}
            )
        except Exception as exc:
            await updater.failed(
                updater.new_agent_message(
                    [Part(text=f"Could not read that health data: {exc}")]
                )
            )
            return

        try:
            payload = assess(profile, raw_text=notes, thread_id=context.context_id)
        except Exception:
            logger.exception("assessment failed")  # never log the profile itself
            await updater.failed(
                updater.new_agent_message(
                    [Part(text="The assessment failed. Nothing was stored.")]
                )
            )
            return

        # In data mode there is no prose -- the peer writes it from the
        # contract. Give it a usable text part either way so a peer that only
        # reads text still gets something meaningful.
        text = payload.get("answer") or _summarise(payload)
        parts = [Part(text=text), Part(data=_to_value(payload))]

        if payload["status"] == "needs_input":
            # The peer agent must go and ask; we will not invent the answers.
            questions = "\n".join(f"- {q}" for q in payload.get("questions", []))
            await updater.requires_input(
                updater.new_agent_message(
                    [
                        Part(text=f"{payload['answer']}\n\n{questions}"),
                        Part(data=_to_value(payload)),
                    ]
                )
            )
            return

        await updater.add_artifact(parts, name="health-risk-assessment")
        await updater.complete(updater.new_agent_message(parts))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_app(base_url: str = "http://localhost:9000"):
    from fastapi import FastAPI

    card = build_agent_card(base_url)
    handler = DefaultRequestHandler(
        agent_executor=HealthAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI(title="Fitty (A2A)")

    # Wearable updates arrive here and resume the existing thread.
    from health_agent.adapters.health_ingest import router as health_router
    from health_agent.adapters.webhook import router as webhook_router

    app.include_router(health_router)
    app.include_router(webhook_router)

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card, card_url=CARD_URL),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=RPC_URL),
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(f"http://{args.host}:{args.port}"),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
