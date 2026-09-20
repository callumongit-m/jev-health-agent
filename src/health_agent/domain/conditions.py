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
    #: One sentence a person can read. Shown next to the number, because a
    #: probability for something you cannot name is not information.
    plain: str = ""
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
    #: Profile fields this factor is actually about. If none are present the
    #: classifier is guessing, whatever confidence it reports -- measured
    #: against real Jev it returned 0.93 for diet on a profile containing
    #: only age and sex. See `evidence_floor` in results.py.
    evidence_fields: tuple[str, ...] = ()


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
        plain=(
            "Your body stops handling blood sugar properly. Common, largely preventable, and often silent for years before it is found."
        ),
        worsens_with={
            "hba1c_mmol_mol": "increase",
            "weight_kg": "increase",
            "waist_cm": "increase",
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
        plain=(
            "Disease of the heart and blood vessels -- the umbrella that heart attacks and strokes sit under."
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
        plain=(
            "Blood pressure high enough to damage arteries over time. It has almost no symptoms, so it is usually found by measuring rather than by feeling unwell."
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
        plain=(
            "A cluster that travels together -- weight around the middle, blood pressure, blood sugar and blood fats all drifting the wrong way at once."
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
        plain=(
            "Breathing repeatedly stops and restarts during sleep. It leaves people exhausted and raises blood pressure, and it treats well once found."
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
        plain=(
            "Fat building up in the liver, not caused by alcohol. Usually silent early on and often reversible."
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
        plain=(
            "Kidneys filtering less well than they should. Usually has no symptoms until it is advanced, which is why it is picked up on tests."
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
        evidence_fields=("smoking_status", "cigarettes_per_day", "years_smoked"),
    ),
    FactorSpec(
        key="adiposity",
        label="Body composition",
        instructions=(
            "How much excess body fat is this person carrying? If a "
            "waist-to-height ratio is given, weigh it above BMI -- BMI "
            "cannot distinguish muscle from fat, and reads a trained, "
            "muscular person as overweight when they are not. High activity "
            "with a normal waist points to lean mass, not adiposity."
        ),
        levels=(
            "Lean: waist-to-height under 0.5, or a BMI of 25-30 clearly "
            "explained by muscle in someone who trains regularly",
            "Mildly over: waist-to-height 0.5 to 0.55, or BMI 25-30 without "
            "substantial training",
            "Central adiposity: waist-to-height 0.55 to 0.6, or BMI 30-35",
            "Marked central adiposity: waist-to-height above 0.6, or BMI "
            "above 35",
        ),
        max_years_cost=8.0,
        evidence_fields=("waist_cm", "weight_kg", "height_cm"),
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
        evidence_fields=("moderate_activity_minutes_per_week", "steps_daily_avg"),
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
        evidence_fields=("sleep_hours_avg", "sleep_efficiency_pct"),
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
        evidence_fields=("alcohol_units_per_week",),
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
        evidence_fields=("diet_quality_self_rating", "eats_vegetables_daily"),
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
        evidence_fields=("perceived_stress_rating",),
    ),
)


# Safety screening, asked in the same batched call as everything else, so it
# costs no extra latency and no extra money.
#
# The keyword screen in safety.py is deterministic, free and instant, and it
# is also brittle: tested against paraphrases, it missed six real
# emergencies out of six, including "elephant sitting on my chest, left arm
# numb" -- the most recognisable description of a heart attack there is. The
# classifier got all ten cases right. Keyword matching stays as the floor
# because it works offline and cannot drift; this is the net above it.
ACUTE_SCREEN = ConditionSpec(
    key="acute_emergency",
    label="Acute emergency",
    instructions=(
        "This person is describing something that needs emergency medical "
        "care right now -- a heart attack, stroke, anaphylaxis, severe "
        "breathing difficulty, major bleeding, or a mental health crisis "
        "with risk to life. Judge what they are describing, not how calmly "
        "they describe it."
    ),
)

URGENT_SCREEN = ConditionSpec(
    key="needs_urgent_review",
    label="Needs urgent review",
    instructions=(
        "This person is describing symptoms that should be assessed by a "
        "clinician within days rather than left to see whether they pass. "
        "Answer false for ordinary aches, tiredness and everyday complaints."
    ),
)

#: Above this the assessment stops and routes to emergency care.
ACUTE_THRESHOLD = 0.5
#: Above this the report leads with getting seen, but still runs.
URGENT_THRESHOLD = 0.6

SCREENS: tuple[ConditionSpec, ...] = (ACUTE_SCREEN, URGENT_SCREEN)


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
