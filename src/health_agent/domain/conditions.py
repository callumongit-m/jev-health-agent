"""Condition and factor registry -- the Jev question set, expressed as data.

Adding a condition is a single entry here; nothing else changes. Each condition
also declares ``worsens_with``, which the monotonicity eval uses to know which
profile field to perturb and in which direction, so the eval suite grows
automatically with the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["increase", "decrease"]


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """A Noul question: 'probability this person is on track for X'."""

    key: str
    label: str
    instructions: str
    #: profile field -> direction that should make this condition MORE likely
    worsens_with: dict[str, Direction] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """A Score question over ordered severity levels (index 0 = best)."""

    key: str
    label: str
    instructions: str
    #: ordered level descriptions, lowest severity first (2-10 entries)
    levels: tuple[str, ...]
    #: years of life expectancy lost at the TOP severity level, from published
    #: cohort literature; interpolated linearly by score position.
    max_years_cost: float


_HORIZON = "over the next 10 years, absent any change in their behaviour"


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        key="t2d_10yr",
        label="Type 2 diabetes",
        instructions=(
            f"This person will develop type 2 diabetes {_HORIZON}. "
            "Weigh HbA1c, fasting glucose, BMI, waist-related adiposity, physical "
            "activity, age and family history of diabetes."
        ),
        worsens_with={
            "hba1c_mmol_mol": "increase",
            "weight_kg": "increase",
            "fasting_glucose_mmol_l": "increase",
            "moderate_activity_minutes_per_week": "decrease",
        },
    ),
    ConditionSpec(
        key="cvd_10yr",
        label="Cardiovascular disease",
        instructions=(
            f"This person will have a major cardiovascular event -- heart attack "
            f"or stroke -- {_HORIZON}. Weigh age, sex, blood pressure, cholesterol "
            "ratio, smoking, diabetes markers and family history of heart disease."
        ),
        worsens_with={
            "systolic_bp": "increase",
            "ldl_mmol_l": "increase",
            "cigarettes_per_day": "increase",
            "hdl_mmol_l": "decrease",
        },
    ),
    ConditionSpec(
        key="hypertension",
        label="Hypertension",
        instructions=(
            f"This person will be diagnosed with hypertension {_HORIZON}, or "
            "already meets the threshold. Weigh measured blood pressure, BMI, "
            "alcohol intake, sodium-heavy diet signals, stress and age."
        ),
        worsens_with={
            "systolic_bp": "increase",
            "alcohol_units_per_week": "increase",
            "weight_kg": "increase",
        },
    ),
    ConditionSpec(
        key="metabolic_syndrome",
        label="Metabolic syndrome",
        instructions=(
            "This person currently meets the criteria for metabolic syndrome: at "
            "least three of central adiposity, raised triglycerides, low HDL, "
            "raised blood pressure, raised fasting glucose."
        ),
        worsens_with={
            "triglycerides_mmol_l": "increase",
            "weight_kg": "increase",
            "hdl_mmol_l": "decrease",
        },
    ),
    ConditionSpec(
        key="sleep_apnoea",
        label="Obstructive sleep apnoea",
        instructions=(
            "This person has undiagnosed obstructive sleep apnoea. Weigh BMI, "
            "neck-related adiposity, sleep efficiency, reported snoring or "
            "daytime sleepiness, sex, age and resting heart rate."
        ),
        worsens_with={"weight_kg": "increase", "sleep_efficiency_pct": "decrease"},
    ),
    ConditionSpec(
        key="nafld",
        label="Fatty liver disease",
        instructions=(
            "This person has metabolic-dysfunction-associated fatty liver disease. "
            "Weigh BMI, ALT, triglycerides, alcohol intake and glucose markers."
        ),
        worsens_with={
            "alt_u_l": "increase",
            "weight_kg": "increase",
            "alcohol_units_per_week": "increase",
        },
    ),
    ConditionSpec(
        key="ckd",
        label="Chronic kidney disease",
        instructions=(
            f"This person will develop stage 3 or worse chronic kidney disease "
            f"{_HORIZON}. Weigh eGFR, blood pressure, diabetes markers and age."
        ),
        worsens_with={"egfr": "decrease", "systolic_bp": "increase"},
    ),
)


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        key="smoking_burden",
        label="Smoking",
        instructions="How much cumulative harm is this person's smoking history doing?",
        levels=(
            "Never smoked, or quit more than fifteen years ago",
            "Former smoker, quit within the last fifteen years",
            "Current light smoker, under ten a day",
            "Current heavy smoker, ten a day or more",
        ),
        max_years_cost=10.0,
    ),
    FactorSpec(
        key="adiposity",
        label="Body composition",
        instructions="How far is this person's body composition from a healthy range?",
        levels=(
            "Healthy weight, BMI roughly 18.5 to 25",
            "Overweight, BMI roughly 25 to 30",
            "Obese, BMI roughly 30 to 35",
            "Severely obese, BMI above 35",
        ),
        max_years_cost=8.0,
    ),
    FactorSpec(
        key="activity_deficit",
        label="Physical activity",
        instructions="How far short of recommended physical activity is this person?",
        levels=(
            "Meets or exceeds 150 minutes of moderate activity weekly",
            "Somewhat short, roughly 75 to 150 minutes weekly",
            "Largely inactive, under 75 minutes weekly",
            "Sedentary, almost no purposeful movement",
        ),
        max_years_cost=5.0,
    ),
    FactorSpec(
        key="sleep_debt",
        label="Sleep",
        instructions="How impaired is this person's sleep?",
        levels=(
            "Consistently seven to nine hours of good quality sleep",
            "Mildly short or mildly disrupted sleep",
            "Persistently under six hours, or poor efficiency",
            "Severe chronic deprivation or badly fragmented sleep",
        ),
        max_years_cost=3.0,
    ),
    FactorSpec(
        key="alcohol_burden",
        label="Alcohol",
        instructions="How harmful is this person's alcohol intake?",
        levels=(
            "None, or well within fourteen units weekly",
            "Around the fourteen unit guideline",
            "Regularly above guidelines, roughly fourteen to thirty-five units",
            "Heavy intake, above thirty-five units weekly",
        ),
        max_years_cost=5.0,
    ),
    FactorSpec(
        key="diet_quality",
        label="Diet",
        instructions="How poor is this person's diet?",
        levels=(
            "Consistently varied and nutrient dense",
            "Reasonable with some gaps",
            "Poor, heavily processed or low in vegetables and fibre",
            "Very poor, largely ultra-processed",
        ),
        max_years_cost=4.0,
    ),
    FactorSpec(
        key="stress_load",
        label="Stress",
        instructions="How damaging is this person's chronic stress load?",
        levels=(
            "Low and well managed",
            "Noticeable but manageable",
            "High and sustained",
            "Severe, with signs of burnout or physiological strain",
        ),
        max_years_cost=2.5,
    ),
)


#: Meta question -- gates whether there is enough data to score at all.
DATA_SUFFICIENCY = ConditionSpec(
    key="data_sufficient",
    label="Data sufficiency",
    instructions=(
        "There is enough information here to estimate this person's long-term "
        "health risks with reasonable confidence. Answer false if the core "
        "picture -- age, body composition, and any metabolic or cardiovascular "
        "signal -- is missing or too thin to reason from."
    ),
)


CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}
FACTORS_BY_KEY = {f.key: f for f in FACTORS}
