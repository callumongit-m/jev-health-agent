"""Result shapes returned by the scoring layer and the agent."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from health_agent.domain.conditions import FACTORS_BY_KEY


class RiskBand(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @classmethod
    def from_probability(cls, p: float) -> "RiskBand":
        if p < 0.10:
            return cls.LOW
        if p < 0.25:
            return cls.MODERATE
        if p < 0.50:
            return cls.HIGH
        return cls.VERY_HIGH


class ConditionRisk(BaseModel):
    """One Jev Noul answer, interpreted."""

    key: str
    label: str
    probability: float = Field(ge=0.0, le=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def band(self) -> RiskBand:
        return RiskBand.from_probability(self.probability)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def certainty(self) -> float:
        """Noul answers carry no confidence field of their own, so derive it:
        a probability near 0.5 is the model declining to commit."""
        return round(abs(self.probability - 0.5) * 2, 3)


class FactorScore(BaseModel):
    """One Jev Score answer, interpreted."""

    key: str
    label: str
    score: float = Field(ge=0.0, description="probability-weighted level position")
    max_level: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    level_label: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity(self) -> float:
        """Score normalised to 0-1 so factors with different level counts compare."""
        return round(self.score / self.max_level, 3) if self.max_level else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def years_cost(self) -> float:
        """Life expectancy years attributable to this factor at its current
        severity, interpolated from the registry's published top-level cost."""
        spec = FACTORS_BY_KEY.get(self.key)
        if spec is None:
            return 0.0
        return round(self.severity * spec.max_years_cost, 2)


class RiskAssessment(BaseModel):
    """Everything the Jev layer produces. No LLM involved."""

    conditions: list[ConditionRisk]
    factors: list[FactorScore]
    data_sufficiency: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    model: str
    request_id: str | None = None
    latency_ms: float | None = None
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def condition(self, key: str) -> ConditionRisk:
        for c in self.conditions:
            if c.key == key:
                return c
        raise KeyError(key)

    def factor(self, key: str) -> FactorScore:
        for f in self.factors:
            if f.key == key:
                return f
        raise KeyError(key)

    def top_conditions(self, n: int = 3) -> list[ConditionRisk]:
        return sorted(self.conditions, key=lambda c: c.probability, reverse=True)[:n]

    def top_factors(self, n: int = 3) -> list[FactorScore]:
        """Highest-leverage modifiable factors -- ranked by years recoverable."""
        return sorted(self.factors, key=lambda f: f.years_cost, reverse=True)[:n]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_years_at_risk(self) -> float:
        """Sum of modifiable-factor costs. Input to the actuarial calculator,
        not a life expectancy in itself."""
        return round(sum(f.years_cost for f in self.factors), 2)
