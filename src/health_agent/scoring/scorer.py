"""Turns a HealthProfile into a RiskAssessment via one backend call."""

from __future__ import annotations

import time

from health_agent.domain.conditions import (
    CONDITIONS,
    CONDITIONS_BY_KEY,
    DATA_SUFFICIENCY,
    FACTORS,
    FACTORS_BY_KEY,
)
from health_agent.domain.profile import HealthProfile
from health_agent.domain.results import ConditionRisk, FactorScore, RiskAssessment
from health_agent.scoring.backend import ScoringBackend, get_backend


class RiskScorer:
    def __init__(self, backend: ScoringBackend | None = None) -> None:
        self.backend = backend or get_backend()

    def score(self, profile: HealthProfile) -> RiskAssessment:
        state = profile.to_state()
        started = time.perf_counter()
        result = self.backend.classify(state)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        missing = (
            {c.key for c in CONDITIONS} | {f.key for f in FACTORS} | {DATA_SUFFICIENCY.key}
        ) - result.answers.keys()
        if missing:
            raise ValueError(f"backend omitted answers: {sorted(missing)}")

        conditions = [
            ConditionRisk(
                key=key,
                label=CONDITIONS_BY_KEY[key].label,
                probability=result.answers[key].value,
            )
            for key in CONDITIONS_BY_KEY
        ]

        factors = []
        for key, spec in FACTORS_BY_KEY.items():
            answer = result.answers[key]
            factors.append(
                FactorScore(
                    key=key,
                    label=spec.label,
                    score=answer.value,
                    max_level=answer.max_level or (len(spec.levels) - 1),
                    confidence=answer.confidence,
                    level_label=answer.level_label
                    or spec.levels[int(round(answer.value))],
                )
            )

        return RiskAssessment(
            conditions=conditions,
            factors=factors,
            data_sufficiency=result.answers[DATA_SUFFICIENCY.key].value,
            completeness=profile.completeness(),
            model=result.model,
            request_id=result.request_id,
            latency_ms=latency_ms,
        )
