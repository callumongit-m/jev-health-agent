"""Graph nodes."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from health_agent import safety
from health_agent.config import SETTINGS
from health_agent.domain import evidence as evidence_rules
from health_agent.domain import thresholds as threshold_rules
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
    ("on_bp_medication",
     "Have you ever been prescribed medication for blood pressure?"),
    ("previously_high_glucose",
     "Has a doctor or nurse ever told you your blood sugar was high -- at a "
     "check-up, during an illness, or in pregnancy?"),
    ("eats_vegetables_daily",
     "Do you eat vegetables, fruit or berries most days?"),
    ("waist_cm", "What is your waist measurement? Inches or cm, either is "
                 "fine. It tells us more than BMI, and unlike BMI it does "
                 "not mistake muscle for fat."),
    ("smoking_status", "Do you smoke -- never, formerly, or currently?"),
    ("systolic_bp", "Do you know your blood pressure? The top number is enough."),
    ("hba1c_mmol_mol",
     "Have you had an HbA1c or fasting glucose test? At your age it makes a "
     "real difference to the estimate -- a GP or a home test can do it."),
    ("fasting_glucose_mmol_l", "Do you know your fasting blood glucose?"),
    ("total_cholesterol_mmol_l", "Do you know your total cholesterol?"),
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
        if flags:
            return {
                "red_flags": [f.key for f in flags],
                "symptom_patterns": [],
                "status": "seek_care",
                "answer": safety.advice_for(flags),
            }

        # Values that already meet a diagnostic threshold are a matter of
        # definition, not probability, so they are checked in code. An
        # HbA1c of 60 is diabetes; asking a model how likely someone is to
        # "develop" it returns a reassuring number to someone who has it.
        findings = threshold_rules.check(profile)
        emergency = [
            f for f in findings
            if f.urgency is threshold_rules.Urgency.EMERGENCY
        ]
        if emergency:
            return {
                "red_flags": [f"{f.field}_critical" for f in emergency],
                "symptom_patterns": [],
                "threshold_findings": [f.as_dict() for f in findings],
                "status": "seek_care",
                "answer": "\n\n".join(
                    f"{f.meaning} {f.action}" for f in emergency
                ),
            }

        # Patterns are not emergencies, so they do not stop the assessment.
        # They ride alongside it: the person still gets their risk picture,
        # and is told which of what they described is worth acting on now.
        patterns = safety.detect_patterns(*texts)
        guidance = safety.pattern_guidance(patterns) or ""
        if findings:
            lines = [
                "Some of your readings already mean something definite, "
                "separately from any risk estimate:",
                "",
            ]
            for f in findings:
                lines.append(f"  {f.meaning} {f.action} ({f.source})")
            guidance = (guidance + "\n\n" + "\n".join(lines)).strip()

        return {
            "red_flags": [],
            "symptom_patterns": [p.key for p in patterns],
            "threshold_findings": [f.as_dict() for f in findings],
            "urgent_guidance": guidance or None,
        }

    def classify(state: AgentState) -> dict:
        """One Jev call, then the age-aware evidence check. No LLM involved.

        The call also carries the acute screens, so the classifier sees the
        person's own words before any of this is shown to them.
        """
        from health_agent.domain.conditions import ACUTE_THRESHOLD, URGENT_THRESHOLD

        profile = state["profile"]
        free_text = " ".join(
            t for t in [state.get("raw_text"), *profile.symptoms] if t
        )
        assessment = scorer.score(profile, free_text=free_text or None)
        result: dict = {
            "assessment": assessment,
            "evidence": evidence_rules.assess(profile),
        }

        if assessment.acute_probability >= ACUTE_THRESHOLD:
            result |= {
                "status": "seek_care",
                "red_flags": ["classifier_acute"],
                "answer": safety.EMERGENCY,
            }
        elif assessment.urgent_probability >= URGENT_THRESHOLD:
            existing = state.get("urgent_guidance") or ""
            note = (
                "From what you have described, this is worth having assessed "
                "by a clinician within the next few days rather than waiting "
                "to see whether it passes. Contact your GP, or call 111 if "
                "you cannot get an appointment."
            )
            result["urgent_guidance"] = (existing + "\n\n" + note).strip()
        return result

    def clarify(state: AgentState) -> dict:
        """Ask for exactly what this person's age band needs, and say why.

        Asking a 21-year-old for an HbA1c they have never had is how you lose
        them. Asking a 55-year-old to skip it is how you publish a number you
        cannot stand behind.
        """
        profile: HealthProfile = state["profile"]
        evidence = state.get("evidence") or evidence_rules.assess(profile)
        prompts = dict(CLARIFY_PRIORITY)

        questions = [
            prompts.get(field, f"What is your {field.replace('_', ' ')}?")
            for field in evidence.missing_required
        ][:MAX_CLARIFYING_QUESTIONS]

        if not questions:
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
                "I need a little more before the numbers would be worth "
                f"trusting. {evidence.band.rationale}"
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

    def compose(state: AgentState) -> dict:
        """Findings plus a presentation contract. No model involved.

        The calling model is already sitting in the person's conversation --
        it has the context, and the person is already paying for it. Running
        a second model to produce prose the first will rewrite is cost for
        nothing.
        """
        from health_agent.domain import presentation as presentation_rules
        from health_agent.domain import projection as projection_rules
        from health_agent.scoring.actuarial import estimate

        assessment = state["assessment"]
        profile = state["profile"]
        life = estimate(assessment, profile, state["evidence"])
        life_dict = life.as_dict() if life else None

        # Published guidance for what the report will actually discuss, so
        # recommendations can be cited rather than asserted.
        from health_agent.evidence import sources

        guidance = sources.collect(
            [c.key for c in assessment.top_conditions(3)],
            [
                f.key
                for f in assessment.top_factors(3)
                if f.severity > presentation_rules.NEEDS_WORK
            ],
        )

        # Where the numbers land if nothing changes. One classifier call
        # per decade, run concurrently, which is only affordable because
        # the classifier is cheap.
        projection = projection_rules.build(profile, scorer)

        contract = presentation_rules.build(
            assessment,
            state["evidence"],
            life_dict,
            state.get("urgent_guidance"),
            guidance=guidance,
            projection=projection.as_dict() if projection else None,
        )

        return {
            "status": "complete",
            # held back from the payload -- see LIFE_EXPECTANCY_OFFER. The
            # calculator still runs, because the per-action years come from
            # it; only the mortality figure is withheld until asked for.
            "life_expectancy_private": life_dict,
            "presentation": contract.as_dict(),
            "answer": None,
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
        "compose": compose,
        "reason": reason,
        "respond": respond,
    }


def _brief(state: AgentState) -> str:
    """What the reasoning model sees: the classifier's findings, not raw data."""
    assessment = state["assessment"]
    profile: HealthProfile = state["profile"]

    evidence = state.get("evidence")
    body = (
        f"waist-to-height {profile.waist_to_height}"
        if profile.waist_to_height is not None
        else f"BMI {profile.bmi} (no waist given, so this may overstate body "
             f"fat for someone who trains)"
    )
    lines = [
        "Classifier output for this person. Explain it and act on it.",
        "",
        f"Age {profile.age}, sex {profile.sex}, {body}.",
        f"Data sufficiency {assessment.data_sufficiency:.0%}.",]
    if evidence is not None:
        lines += [
            f"Evidence basis: {evidence.tier.label}. Say what the estimate "
            f"rests on, and do not imply more certainty than that supports.",
        ]
        if evidence.missing_recommended:
            lines.append(
                "Getting "
                + ", ".join(evidence.missing_recommended)
                + " would sharpen it -- suggest it once, concretely, without "
                "making it a condition of acting on what is already clear."
            )
    lines += [
        "",
        "Probabilities (with certainty -- low certainty means speak in ranges):",
    ]
    ceiling = evidence.confidence_ceiling if evidence is not None else 1.0
    for c in sorted(assessment.conditions, key=lambda x: -x.probability):
        shown = min(c.certainty, ceiling)
        capped = " [capped by evidence available]" if c.certainty > ceiling else ""
        lines.append(
            f"  {c.label}: {c.probability:.0%} ({c.band}, "
            f"certainty {shown:.2f}){capped}"
        )

    lines += [
        "",
        "Modifiable factors, ranked by confidence-weighted years. These are "
        "SEVERITY WEIGHTS FOR RANKING ONLY. They overlap heavily and DO NOT "
        "ADD UP -- quoting them as independent gains, or summing them, "
        "overstates the benefit several times over. For any years figure you "
        "give the person, use the numbers actuarial_calc returns, which "
        "account for that overlap. A low confidence means the data for that "
        "factor is thin: do not present it as a priority.",
    ]
    for f in assessment.top_factors(len(assessment.factors)):
        lines.append(
            f"  {f.label}: {f.level_label} "
            f"({f.expected_years_cost} expected years, "
            f"{f.years_cost} raw, confidence {f.confidence:.2f})"
        )

    if state.get("raw_text"):
        lines += ["", f"They also said: {state['raw_text']}"]

    if state.get("urgent_guidance"):
        lines += [
            "",
            "SYMPTOM TRIAGE -- lead with this, before the risk numbers:",
            state["urgent_guidance"],
            "",
            "Pass this on as written. Do not name a condition it might be, "
            "do not speculate about a diagnosis, and do not soften the "
            "urgency. Naming something invites self-treatment and, when a "
            "keyword match is wrong, causes real fear for no reason.",
        ]

    lines += [
        "",
        "Call actuarial_calc for life expectancy. Then give: what stands out "
        "and why, and a ranked set of concrete changes. Quote years only from "
        "actuarial_calc's per_factor_years and years_recoverable, and make "
        "clear that the total recoverable is less than the sum of the parts.",
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
    if state.get("urgent_guidance"):
        payload["act_on_this_first"] = state["urgent_guidance"]
        payload["symptom_patterns"] = state.get("symptom_patterns") or []
    if state.get("threshold_findings"):
        payload["clinical_thresholds_met"] = state["threshold_findings"]
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
        # Thin evidence caps how certain the report may sound, however
        # confident the classifier is. A lifestyle-only estimate and one
        # backed by bloods should not read with the same authority.
        ceiling = (
            state["evidence"].confidence_ceiling
            if state.get("evidence") is not None
            else 1.0
        )
        payload["risk"] = {
            c.key: {
                "label": c.label,
                "probability": round(c.probability, 4),
                "band": c.band,
                "certainty": round(min(c.certainty, ceiling), 3),
                "certainty_capped_by_evidence": c.certainty > ceiling,
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
    if state.get("evidence") is not None:
        payload["evidence"] = state["evidence"].as_dict()
    if state.get("life_expectancy"):
        payload["life_expectancy"] = state["life_expectancy"]
    if state.get("presentation"):
        payload["presentation"] = state["presentation"]
        payload.pop("answer", None)
    return payload
