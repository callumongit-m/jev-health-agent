"""An LLM standing in for Jev, so the two can be compared on the same suite.

This exists to answer one question honestly: does a calibrated decision model
actually earn its place, or could the calling model produce the same numbers
itself? Arguing about it is cheap; running both through the same monotonicity,
ordering and sensitivity checks is not.

The prompt asks for exactly the Jev question set in exactly the Jev answer
shape, so nothing but the estimator differs.
"""

from __future__ import annotations

import json
import os
from typing import Any

from health_agent.domain.conditions import CONDITIONS, DATA_SUFFICIENCY, FACTORS
from health_agent.scoring.backend import RawAnswer, RawResult


def _prompt() -> str:
    lines = [
        "You are a health risk estimator. Given a person's data, return "
        "calibrated probabilities and severity scores as JSON.",
        "",
        "Return ONLY a JSON object, no prose, with exactly these keys:",
        "",
        "Probabilities (a float 0-1, the probability the statement is true):",
    ]
    for c in (*CONDITIONS, DATA_SUFFICIENCY):
        lines.append(f'  "{c.key}": {c.instructions}')
    lines += ["", "Scores (an integer level index, plus your confidence 0-1):"]
    for f in FACTORS:
        levels = "; ".join(f"{i}={lvl}" for i, lvl in enumerate(f.levels))
        lines.append(f'  "{f.key}": {f.instructions} Levels: {levels}')
    lines += [
        "",
        'Format: {"t2d_10yr": 0.23, ..., "smoking_burden": {"level": 2, '
        '"confidence": 0.8}, ...}',
        "Be calibrated: a 20% should be right about one time in five.",
    ]
    return "\n".join(lines)


class LLMBackend:
    """Same questions, same answer shape, different estimator."""

    def __init__(self, *, model: str = "claude-haiku-4-5") -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        from langchain_anthropic import ChatAnthropic

        self._model = model
        self._llm = ChatAnthropic(model=model, temperature=0.0, max_tokens=1500)
        self._system = _prompt()
        self.calls = 0
        #: Responses that were not usable as asked. Counted rather than
        #: hidden -- "sometimes returns the wrong shape" is a real property
        #: of an estimator, not a detail of the parser.
        self.malformed = 0
        self.coerced = 0

    @property
    def name(self) -> str:
        return f"llm:{self._model}"

    @property
    def noise_sigma(self) -> float:
        # Measured, not assumed -- see scripts/compare_scorers.py.
        return float(os.getenv("LLM_NOISE_SIGMA", "0.05"))

    def _extract(self, text: str) -> dict[str, Any]:
        """Pull the JSON object out of whatever came back.

        Models wrap JSON in fences, add commentary, or append a second
        object. None of that is the estimator's fault in a way worth
        penalising, so it is recovered -- but counted.
        """
        cleaned = text.strip()
        for fence in ("```json", "```"):
            cleaned = cleaned.removeprefix(fence)
        cleaned = cleaned.removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        self.malformed += 1
        # take the first balanced top-level object
        start = cleaned.find("{")
        if start == -1:
            raise ValueError(f"no JSON object in response: {cleaned[:160]}")
        depth = 0
        for index, char in enumerate(cleaned[start:], start):
            depth += char == "{"
            depth -= char == "}"
            if depth == 0:
                try:
                    return json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError:
                    break
        raise ValueError(f"unparseable response: {cleaned[:160]}")

    def classify(self, state: dict[str, Any]) -> RawResult:
        self.calls += 1
        reply = self._llm.invoke(
            [
                ("system", self._system),
                ("user", json.dumps(state, default=str)),
            ]
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        data = self._extract(text)

        answers: dict[str, RawAnswer] = {}
        for c in (*CONDITIONS, DATA_SUFFICIENCY):
            value = data.get(c.key)
            if isinstance(value, dict):
                value = value.get("probability", value.get("value"))
            if isinstance(value, bool):
                # asked for a probability, given a boolean
                self.coerced += 1
                value = 0.9 if value else 0.1
            try:
                number = float(value)
            except (TypeError, ValueError):
                self.coerced += 1
                number = 0.0
            answers[c.key] = RawAnswer(kind="noul", value=max(0.0, min(1.0, number)))
        for f in FACTORS:
            raw = data.get(f.key)
            if isinstance(raw, dict):
                level = float(raw.get("level", raw.get("score", 0)) or 0)
                confidence = float(raw.get("confidence", 0.7) or 0.7)
            else:
                try:
                    level, confidence = float(raw or 0), 0.7
                except (TypeError, ValueError):
                    self.coerced += 1
                    level, confidence = 0.0, 0.5
            max_level = len(f.levels) - 1
            level = max(0.0, min(level, float(max_level)))
            answers[f.key] = RawAnswer(
                kind="score", value=level, confidence=confidence,
                max_level=max_level, level_label=f.levels[int(round(level))],
            )
        return RawResult(answers=answers, model=self._model, request_id=None)
