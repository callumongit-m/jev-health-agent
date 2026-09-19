"""Graph nodes."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from health_agent import safety
from health_agent.config import SETTINGS
from health_agent.domain.profile import HealthProfile
from health_agent.graph import tools as agent_tools
from health_agent.graph.reasoner import SYSTEM_PROMPT, get_reasoner
from health_agent.graph.state import AgentState
from health_agent.privacy import DISCLAIMER, NOTICE
from health_agent.scoring.scorer import RiskScorer

#: Highest-value missing fields, most informative first. The clarify node asks
#: for these rather than dumping the whole form back at the caller.
CLARIFY_PRIORITY: tuple[tuple[str, str], ...] = (
    ("age", "How old are you?"),
    ("sex", "What sex were you assigned at birth? It changes the baseline risks."),
    ("height_cm", "How tall are you, in cm?"),
    ("weight_kg", "How much do you weigh, in kg?"),
    ("smoking_status", "Do you smoke -- never, formerly, or currently?"),
    ("systolic_bp", "Do you know your blood pressure? The top number is enough."),
    ("hba1c_mmol_mol", "Have you had an HbA1c test? If so, what was it?"),
    (
        "moderate_activity_minutes_per_week",
        "Roughly how many minutes a week do you do moderate exercise?",
    ),
    ("sleep_hours_avg", "How many hours do you typically sleep?"),
    ("alcohol_units_per_week", "Roughly how many units of alcohol per week?"),
    ("family_history", "Any heart disease, stroke or diabetes in close family?"),
)

MAX_CLARIFYING_QUESTIONS = 5


def make_nodes(scorer: RiskScorer):
    """Nodes close over the scorer so tests can inject a backend."""

    def ingest(state: AgentState) -> dict:
        profile = state.get("profile") or HealthProfile()
        return {"profile": profile, "status": "pending", "tool_calls_made": []}

    def screen(state: AgentState) -> dict:
        """Red flags before anything else: never score someone mid-emergency."""
        profile: HealthProfile = state["profile"]
        texts = [state.get("raw_text"), *profile.symptoms]
        flags = safety.detect(*texts)
        if not flags:
            return {"red_flags": []}
        return {
            "red_flags": [f.key for f in flags],
            "status": "seek_care",
            "answer": safety.advice_for(flags),
        }

    def classify(state: AgentState) -> dict:
        """One Jev call. No LLM involved."""
        return {"assessment": scorer.score(state["profile"])}

    def clarify(state: AgentState) -> dict:
        """Return targeted questions to the CALLING agent rather than guessing."""
        profile: HealthProfile = state["profile"]
        known = profile.known_fields()
        questions = [
            text for field, text in CLARIFY_PRIORITY if field not in known
        ][:MAX_CLARIFYING_QUESTIONS]
        if not questions:
            questions = ["Anything else about your health that might be relevant?"]
        return {
            "clarifying_questions": questions,
            "status": "needs_input",
            "answer": (
                "I need a bit more before I can give you numbers worth trusting."
            ),
        }

    def reason(state: AgentState) -> dict:
        """The LLM turn. Loops back through tools until it stops calling them."""
        assessment = state["assessment"]
        profile = state["profile"]
        agent_tools.set_context(assessment=assessment, profile=profile)

        messages = list(state.get("messages") or [])
        if not messages:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_brief(state)),
            ]

        reasoner = get_reasoner(agent_tools.TOOLS)
        reply = reasoner.invoke(messages)

        called = [tc["name"] for tc in getattr(reply, "tool_calls", []) or []]
        return {
            "messages": ([] if state.get("messages") else messages) + [reply],
            "tool_calls_made": list(state.get("tool_calls_made") or []) + called,
        }

    def respond(state: AgentState) -> dict:
        final = next(
            (
                m.content
                for m in reversed(state.get("messages") or [])
                if isinstance(m, AIMessage) and m.content
            ),
            "",
        )
        return {"status": "complete", "answer": final}

    return {
        "ingest": ingest,
        "screen": screen,
        "classify": classify,
        "clarify": clarify,
        "reason": reason,
        "respond": respond,
    }


def _brief(state: AgentState) -> str:
    """What the reasoning model sees: the classifier's findings, not raw data."""
    assessment = state["assessment"]
    profile: HealthProfile = state["profile"]

    lines = [
        "Classifier output for this person. Explain it and act on it.",
        "",
        f"Age {profile.age}, sex {profile.sex}, BMI {profile.bmi}.",
        f"Data sufficiency {assessment.data_sufficiency:.0%}.",
        "",
        "Probabilities (with certainty -- low certainty means speak in ranges):",
    ]
    for c in sorted(assessment.conditions, key=lambda x: -x.probability):
        lines.append(
            f"  {c.label}: {c.probability:.0%} ({c.band}, certainty {c.certainty:.2f})"
        )

    lines += ["", "Modifiable factors, by years recoverable:"]
    for f in sorted(assessment.factors, key=lambda x: -x.years_cost):
        lines.append(
            f"  {f.label}: {f.level_label} "
            f"({f.years_cost} years, confidence {f.confidence:.2f})"
        )

    if state.get("raw_text"):
        lines += ["", f"They also said: {state['raw_text']}"]

    lines += [
        "",
        "Call actuarial_calc for life expectancy. Then give: what stands out "
        "and why, and a ranked set of concrete changes with the years each "
        "could recover.",
    ]
    return "\n".join(lines)


def build_payload(state: AgentState) -> dict:
    """The response both adapters return. Privacy notice is not optional."""
    assessment = state.get("assessment")
    payload: dict = {
        "status": state.get("status", "pending"),
        "privacy": NOTICE,
        "disclaimer": DISCLAIMER,
        "answer": state.get("answer"),
    }
    if state.get("red_flags"):
        payload["red_flags"] = state["red_flags"]
    if state.get("clarifying_questions"):
        payload["questions"] = state["clarifying_questions"]
        payload["data_sufficiency"] = (
            round(assessment.data_sufficiency, 3) if assessment else 0.0
        )
    # Withhold numbers in both non-scoring states. On seek_care they are
    # irrelevant and dangerous; on needs_input the gate has already judged them
    # untrustworthy, and a calling agent given numbers will present them as
    # final regardless of the caveat attached.
    if assessment is not None and state.get("status") not in ("seek_care", "needs_input"):
        payload["risk"] = {
            c.key: {
                "label": c.label,
                "probability": round(c.probability, 4),
                "band": c.band,
                "certainty": c.certainty,
            }
            for c in assessment.conditions
        }
        payload["factors"] = {
            f.key: {
                "label": f.label,
                "level": f.level_label,
                "years_cost": f.years_cost,
                "confidence": f.confidence,
            }
            for f in assessment.factors
        }
        payload["data_sufficiency"] = round(assessment.data_sufficiency, 3)
        payload["model"] = assessment.model
    if state.get("life_expectancy"):
        payload["life_expectancy"] = state["life_expectancy"]
    return payload
