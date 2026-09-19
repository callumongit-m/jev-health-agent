#!/usr/bin/env python
"""A peer agent that drives the health agent over A2A.

This is the point of the A2A adapter: a separate agent discovers the card,
sends a task, and handles the lifecycle -- including `input-required`, where
the health agent refuses to guess and hands back questions for the peer to
answer.

    uv run python -m health_agent.adapters.a2a_server --port 9000   # terminal 1
    uv run python scripts/a2a_peer_demo.py                          # terminal 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from a2a.client import create_client
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState

COMPLETE_PROFILE = {
    "age": 62, "sex": "male", "height_cm": 175, "weight_kg": 86,
    "systolic_bp": 158, "diastolic_bp": 96, "hba1c_mmol_mol": 41,
    "ldl_mmol_l": 4.9, "hdl_mmol_l": 0.85, "triglycerides_mmol_l": 2.4,
    "smoking_status": "current", "cigarettes_per_day": 25,
    "alcohol_units_per_week": 30, "moderate_activity_minutes_per_week": 20,
    "sleep_hours_avg": 6.0, "diet_quality_self_rating": 2,
    "perceived_stress_rating": 4, "family_history": ["heart disease"],
}

THIN_PROFILE = {"age": 41, "sex": "female"}

ACUTE = {"age": 58, "sex": "male", "height_cm": 175, "weight_kg": 95,
         "notes": "I get crushing chest pain walking up stairs"}


def _message(payload: dict, context_id: str) -> SendMessageRequest:
    return SendMessageRequest(
        message=Message(
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            role=Role.ROLE_USER,
            parts=[Part(text=json.dumps(payload))],
        )
    )


def _parts_text(parts) -> str:
    return "".join(p.text for p in parts if p.text)


def _parts_data(parts) -> dict:
    from google.protobuf.json_format import MessageToDict

    for part in parts:
        if part.HasField("data"):
            return MessageToDict(part.data)
    return {}


async def run_case(client, label: str, payload: dict) -> None:
    context_id = str(uuid.uuid4())
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")

    states: list[str] = []
    text = ""
    data: dict = {}

    # StreamResponse is a oneof: task | message | status_update | artifact_update
    async for response in client.send_message(_message(payload, context_id)):
        which = response.WhichOneof("payload") or next(
            (
                f.name
                for f in response.DESCRIPTOR.fields
                if response.HasField(f.name)
            ),
            None,
        )
        if which == "task":
            states.append(TaskState.Name(response.task.status.state))
        elif which == "status_update":
            update = response.status_update
            states.append(TaskState.Name(update.status.state))
            if update.status.HasField("message"):
                text = _parts_text(update.status.message.parts) or text
                data = _parts_data(update.status.message.parts) or data
        elif which == "artifact_update":
            parts = response.artifact_update.artifact.parts
            text = _parts_text(parts) or text
            data = _parts_data(parts) or data
        elif which == "message":
            text = _parts_text(response.message.parts) or text
            data = _parts_data(response.message.parts) or data

    print("lifecycle:", " -> ".join(dict.fromkeys(states)) or "(none)")
    if text:
        print()
        print("\n".join("  " + line for line in text.splitlines()[:12]))
    if data.get("risk"):
        top = sorted(data["risk"].items(), key=lambda kv: -kv[1]["probability"])[:3]
        print("\n  top risks:", [(k, round(v["probability"], 3)) for k, v in top])
    if data.get("lifeExpectancy"):
        le = data["lifeExpectancy"]
        print(f"  life expectancy: {le['adjustedRemainingYears']} yrs "
              f"(baseline {le['baselineRemainingYears']})")
    if data:
        print("  status:", data.get("status"),
              "| privacy notice returned:",
              "NOT used to train" in data.get("privacy", ""))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:9000")
    args = parser.parse_args()

    client = await create_client(args.url)
    print(f"discovered agent at {args.url}/.well-known/agent-card.json")

    try:
        await run_case(client, "1. Complete profile -> expect COMPLETED", COMPLETE_PROFILE)
        await run_case(client, "2. Thin profile -> expect INPUT_REQUIRED", THIN_PROFILE)
        await run_case(client, "3. Acute symptom -> expect seek_care, no scoring", ACUTE)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
