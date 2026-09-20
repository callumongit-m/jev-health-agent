"""Scoring backends.

The whole Jev question set goes out in ONE call. Two implementations:

``JevBackend``         the real thing, via ``langchain-typesafe``.
``OpenRouterBackend``  the same model through OpenRouter's Decisions API, for
                       when TypeSafe's own console is invite-only.
``FakeBackend``        a seeded rule-based stand-in so the eval harness, the
                       graph and both adapters run with no API key. Deliberately
                       noisy -- a perfectly deterministic fake would make
                       pass^k meaningless.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Protocol

from health_agent.domain.conditions import (
    CONDITIONS,
    DATA_SUFFICIENCY,
    FACTORS,
)


@dataclass(frozen=True, slots=True)
class RawAnswer:
    """Backend-neutral answer. ``value`` is a probability for noul questions and
    a weighted level position for score questions."""

    kind: str  # "noul" | "score"
    value: float
    confidence: float = 1.0
    max_level: int = 1
    level_label: str = ""


@dataclass(frozen=True, slots=True)
class RawResult:
    answers: dict[str, RawAnswer]
    model: str
    request_id: str | None = None


class ScoringBackend(Protocol):
    def classify(self, state: dict[str, Any]) -> RawResult: ...

    @property
    def name(self) -> str: ...

    @property
    def noise_sigma(self) -> float:
        """Standard deviation of run-to-run variation on an unchanged input.
        Monotonicity checks derive their tolerance from this, so a backend that
        under-reports its own noise will produce spurious eval failures."""
        ...


def _build_questions() -> dict[str, Any]:
    from langchain_typesafe import Noul, Score

    questions: dict[str, Any] = {
        c.key: Noul(instructions=c.instructions) for c in CONDITIONS
    }
    questions[DATA_SUFFICIENCY.key] = Noul(instructions=DATA_SUFFICIENCY.instructions)
    for f in FACTORS:
        questions[f.key] = Score(instructions=f.instructions, criteria=list(f.levels))
    return questions


class JevBackend:
    """One classifier, built once, all 15 questions batched per call."""

    def __init__(self, *, model: str = "jev-latest", timeout: float = 30.0) -> None:
        from langchain_typesafe import TypeSafeClassifier

        if not os.getenv("TYPESAFE_API_KEY"):
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Export it, or run with "
                "--backend fake to use the offline stand-in."
            )
        self._classifier = TypeSafeClassifier(
            questions=_build_questions(), model=model, timeout=timeout
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"jev:{self._model}"

    @property
    def noise_sigma(self) -> float:
        # Measured, not guessed: scripts/measure_noise.py over three profiles
        # spanning the risk range, worst per-condition sd. See
        # evals/noise_profile.json. Re-measure when the model version moves.
        return 0.013

    def classify(self, state: dict[str, Any]) -> RawResult:
        response = self._classifier.invoke(state)
        answers: dict[str, RawAnswer] = {}

        for key, answer in response.nouls.items():
            answers[key] = RawAnswer(kind="noul", value=float(answer.noul))

        for key, answer in response.scores.items():
            legend = answer.legend or {}
            nearest = str(int(round(answer.score)))
            answers[key] = RawAnswer(
                kind="score",
                value=float(answer.score),
                confidence=float(answer.confidence),
                max_level=max(len(legend) - 1, 1),
                level_label=str(legend.get(nearest, "")),
            )

        return RawResult(
            answers=answers,
            model=response.model,
            request_id=response.request_id,
        )


# --------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------

#: Verified against the live API. OpenRouter's own reference documents
#: ".../api/v1/api/alpha/decisions", which 404s -- the doubled segment is a
#: docs generation artifact. The short form is the real one; the doubled form
#: is kept as a fallback in case they ever make the docs true.
_OPENROUTER_URLS = (
    "https://openrouter.ai/api/alpha/decisions",
    "https://openrouter.ai/api/v1/api/alpha/decisions",
)

#: OpenRouter prefixes floating "latest" aliases with a tilde. Without it the
#: API returns 400 "Model typesafe/jev-latest does not exist". Pin to
#: "typesafe/jev-1.13" instead if you want a fixed version.
DEFAULT_OPENROUTER_MODEL = "~typesafe/jev-latest"


def _questions_wire() -> dict[str, Any]:
    """The question set as raw JSON, matching the Decisions wire format."""
    questions: dict[str, Any] = {}
    for c in (*CONDITIONS, DATA_SUFFICIENCY):
        questions[c.key] = {"type": "noul", "instructions": c.instructions}
    for f in FACTORS:
        questions[f.key] = {
            "type": "score",
            "instructions": f.instructions,
            "criteria": list(f.levels),
        }
    return questions


class OpenRouterBackend:
    """Jev via OpenRouter's Decisions API.

    Same wire format as TypeSafe's /v1/systemone -- same question types, same
    answer shapes -- so this is a transport swap, not a different model.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        timeout: float = 30.0,
        url: str | None = None,
    ) -> None:
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it, or run with "
                "--backend fake to use the offline stand-in."
            )
        self._timeout = timeout
        self._model = os.getenv("OPENROUTER_JEV_MODEL", model)
        override = url or os.getenv("OPENROUTER_DECISIONS_URL")
        self._urls = (override,) if override else _OPENROUTER_URLS
        self._url: str | None = None  # learned on first success
        self._questions = _questions_wire()

    @property
    def name(self) -> str:
        return f"openrouter:{self._model}"

    @property
    def noise_sigma(self) -> float:
        # Same model as JevBackend, same measured figure.
        return 0.013

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        candidates = (self._url,) if self._url else self._urls
        last_error: Exception | None = None

        for url in candidates:
            try:
                response = httpx.post(
                    url, headers=headers, json=payload, timeout=self._timeout
                )
            except Exception as exc:  # network failure -- try the next path
                last_error = exc
                continue
            if response.status_code == 404 and len(candidates) > 1:
                continue  # wrong path variant; try the other
            response.raise_for_status()
            self._url = url
            return response.json()

        raise RuntimeError(
            f"OpenRouter Decisions API unreachable at {candidates}. "
            f"Set OPENROUTER_DECISIONS_URL if the path has moved. "
            f"Last error: {last_error}"
        )

    def classify(self, state: dict[str, Any]) -> RawResult:
        body = self._post(
            {"model": self._model, "state": state, "questions": self._questions}
        )

        answers: dict[str, RawAnswer] = {}
        for key, answer in (body.get("answers") or {}).items():
            kind = answer.get("type")
            if kind == "noul":
                answers[key] = RawAnswer(kind="noul", value=float(answer["noul"]))
            elif kind == "score":
                legend = answer.get("legend") or {}
                nearest = str(int(round(answer["score"])))
                answers[key] = RawAnswer(
                    kind="score",
                    value=float(answer["score"]),
                    confidence=float(answer.get("confidence", 1.0)),
                    max_level=max(len(legend) - 1, 1),
                    level_label=str(legend.get(nearest, "")),
                )

        return RawResult(
            answers=answers,
            model=body.get("model", self._model),
            request_id=body.get("id"),
        )


# --------------------------------------------------------------------------
# Offline stand-in
# --------------------------------------------------------------------------

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _blend(*parts: float | None) -> float:
    """Mean of the markers that are actually present. Unlike ``max`` this keeps
    every marker influential, so the scorer stays sensitive to each one."""
    present = [p for p in parts if p is not None]
    return sum(present) / len(present) if present else 0.0


def _ramp(value: float | None, low: float, high: float) -> float:
    """Map a value onto 0-1 across the [low, high] band. None -> neutral 0.0."""
    if value is None:
        return 0.0
    return _clamp((value - low) / (high - low))


class FakeBackend:
    """Rule-based scorer with bounded noise. Exists so every layer above it is
    testable offline; it makes no claim to clinical validity."""

    def __init__(self, *, seed: int | None = None, noise: float = 0.015) -> None:
        self._seed = seed
        self._noise = noise
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"fake(noise={self._noise})"

    @property
    def noise_sigma(self) -> float:
        return self._noise

    def _jitter(self, value: float) -> float:
        if self._noise <= 0:
            return value
        return _clamp(value + self._rng.gauss(0.0, self._noise))

    def classify(self, state: dict[str, Any]) -> RawResult:
        g = state.get
        bmi = g("bmi")
        age = g("age")
        smoking = g("smoking_status")
        cigs = g("cigarettes_per_day") or 0
        activity = g("moderate_activity_minutes_per_week")
        alcohol = g("alcohol_units_per_week")
        sleep_h = g("sleep_hours_avg")
        sleep_eff = g("sleep_efficiency_pct")
        family = {h.lower() for h in (g("family_history") or [])}

        adiposity = _ramp(bmi, 22.0, 38.0)
        age_risk = _ramp(age, 30.0, 75.0)
        glucose = _blend(
            _ramp(g("hba1c_mmol_mol"), 34.0, 70.0) if g("hba1c_mmol_mol") else None,
            _ramp(g("fasting_glucose_mmol_l"), 5.0, 9.0)
            if g("fasting_glucose_mmol_l")
            else None,
        )
        bp = _ramp(g("systolic_bp"), 110.0, 190.0)
        # Blended, not max(): a max() lets one marker mask every other, which
        # makes the model insensitive to the ones it is hiding.
        lipids = _blend(
            _ramp(g("ldl_mmol_l"), 2.0, 6.0) if g("ldl_mmol_l") else None,
            _ramp(g("triglycerides_mmol_l"), 1.0, 5.0)
            if g("triglycerides_mmol_l")
            else None,
            (1.0 - _ramp(g("hdl_mmol_l"), 0.8, 2.2)) if g("hdl_mmol_l") else None,
        )
        smoke = (
            _clamp(0.45 + _ramp(cigs, 0, 25) * 0.55)
            if smoking == "current"
            else 0.25 if smoking == "former" else 0.0
        )
        inactivity = 1.0 - _ramp(activity, 0.0, 200.0) if activity is not None else 0.4
        drink = _ramp(alcohol, 6.0, 50.0)
        liver = _ramp(g("alt_u_l"), 25.0, 120.0)
        kidney = 1.0 - _ramp(g("egfr"), 25.0, 100.0) if g("egfr") else 0.0
        poor_sleep = _blend(
            (1.0 - _ramp(sleep_h, 4.5, 7.5)) if sleep_h is not None else None,
            (1.0 - _ramp(sleep_eff, 70.0, 95.0)) if sleep_eff is not None else None,
        )
        diet = 1.0 - _ramp(g("diet_quality_self_rating"), 1.0, 5.0) if g(
            "diet_quality_self_rating"
        ) else 0.4
        stress = _ramp(g("perceived_stress_rating"), 1.0, 5.0)

        def fam(*needles: str) -> float:
            return 0.12 if any(n in h for h in family for n in needles) else 0.0

        raw_conditions = {
            "t2d_10yr": 0.60 * glucose + 0.25 * adiposity + 0.10 * inactivity
            + 0.10 * age_risk + fam("diabet"),
            "cvd_10yr": 0.30 * bp + 0.25 * lipids + 0.25 * smoke + 0.20 * age_risk
            + 0.10 * glucose + fam("heart", "cardiac", "stroke"),
            "hypertension": 0.65 * bp + 0.18 * adiposity + 0.12 * drink
            + 0.10 * age_risk,
            # Metabolic syndrome is three-of-five criteria: central adiposity,
            # raised triglycerides, low HDL, raised BP, raised fasting glucose.
            # Lipids therefore cover two of the five and must weigh accordingly;
            # under-weighting them made the scorer insensitive to triglycerides.
            "metabolic_syndrome": 0.40 * lipids + 0.22 * adiposity
            + 0.20 * glucose + 0.20 * bp,
            "sleep_apnoea": 0.55 * adiposity + 0.25 * poor_sleep + 0.10 * age_risk,
            "nafld": 0.45 * adiposity + 0.25 * liver + 0.20 * drink + 0.15 * glucose,
            "ckd": 0.55 * kidney + 0.25 * bp + 0.20 * glucose,
        }

        answers: dict[str, RawAnswer] = {
            key: RawAnswer(kind="noul", value=self._jitter(_clamp(value)))
            for key, value in raw_conditions.items()
        }

        known = sum(1 for k, v in state.items() if not k.startswith("_") and v is not None)
        answers[DATA_SUFFICIENCY.key] = RawAnswer(
            kind="noul", value=self._jitter(_ramp(known, 4.0, 14.0))
        )

        # (severity, is there actually evidence for this factor?)
        raw_factors: dict[str, tuple[float, bool]] = {
            "smoking_burden": (smoke, smoking is not None),
            "adiposity": (adiposity, bmi is not None),
            "activity_deficit": (
                inactivity, activity is not None or g("steps_daily_avg") is not None
            ),
            "sleep_debt": (poor_sleep, sleep_h is not None or sleep_eff is not None),
            "alcohol_burden": (drink, alcohol is not None),
            "diet_quality": (diet, g("diet_quality_self_rating") is not None),
            "stress_load": (stress, g("perceived_stress_rating") is not None),
        }
        for spec in FACTORS:
            severity, has_evidence = raw_factors.get(spec.key, (0.0, False))
            if not has_evidence:
                # No evidence must mean low confidence, not a confident guess.
                # Anything downstream that weights by confidence then stops
                # treating an absent field as a finding.
                severity, confidence = 0.4, 0.2
            else:
                confidence = round(0.55 + 0.4 * abs(severity - 0.5) * 2, 3)
            severity = _clamp(self._jitter(severity))
            max_level = len(spec.levels) - 1
            position = severity * max_level
            answers[spec.key] = RawAnswer(
                kind="score",
                value=position,
                confidence=confidence,
                max_level=max_level,
                level_label=spec.levels[int(round(position))],
            )

        return RawResult(answers=answers, model="fake-1.0", request_id=None)


def get_backend(kind: str = "auto", *, seed: int | None = None) -> ScoringBackend:
    """``auto`` prefers a direct TypeSafe key, falls back to OpenRouter, then
    to the offline fake. The first two are the same model."""
    if kind == "jev":
        return JevBackend()
    if kind == "openrouter":
        return OpenRouterBackend()
    if kind == "llm":
        from health_agent.scoring.llm_backend import LLMBackend

        return LLMBackend()
    if kind == "fake":
        return FakeBackend(seed=seed)
    if kind == "auto":
        if os.getenv("TYPESAFE_API_KEY"):
            return JevBackend()
        if os.getenv("OPENROUTER_API_KEY"):
            return OpenRouterBackend()
        return FakeBackend(seed=seed)
    raise ValueError(f"unknown backend: {kind!r}")
