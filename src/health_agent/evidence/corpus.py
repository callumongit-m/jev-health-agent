"""Curated guidance, sourced from NHS and NICE.

Recommendations should rest on published guidance, not on what a language
model recalls about exercise. Two reasons: a model's recalled effect size is
unverifiable and drifts between runs, and someone deciding whether to change
their life deserves to see where the advice came from.

Every entry carries its source URL so the host can cite it. Entries are
deliberately terse -- they are the evidence a recommendation stands on, not
the recommendation itself.

This is the offline corpus. With `NHS_API_KEY` set, `sources.py` fetches the
live NHS Content API and caches it, and this becomes the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


#: NHS Digital content is available under the Open Government Licence v3.0,
#: which permits reuse including commercially, on condition of attribution.
#: This is that attribution, and it travels with every entry.
OGL_ATTRIBUTION = (
    "Information from NHS Digital, licenced under the current version of "
    "the Open Government Licence."
)


@dataclass(frozen=True, slots=True)
class Guidance:
    key: str
    title: str
    source: str
    url: str
    points: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "points": list(self.points),
            "source": self.source,
            "url": self.url,
            "licence": OGL_ATTRIBUTION,
        }


CONDITION_GUIDANCE: dict[str, Guidance] = {
    "t2d_10yr": Guidance(
        key="t2d_10yr",
        title="Type 2 diabetes prevention",
        source="NHS",
        url="https://www.nhs.uk/conditions/type-2-diabetes/",
        points=(
            "Type 2 diabetes can often be delayed or prevented by losing "
            "excess weight, eating well and moving more.",
            "The NHS Diabetes Prevention Programme is free and available by "
            "GP referral for people at high risk in England.",
            "Symptoms can be absent for years, which is why it is often "
            "found on a blood test rather than because someone felt unwell.",
        ),
    ),
    "cvd_10yr": Guidance(
        key="cvd_10yr",
        title="Cardiovascular disease",
        source="NHS",
        url="https://www.nhs.uk/conditions/cardiovascular-disease/",
        points=(
            "The main modifiable risks are smoking, high blood pressure, "
            "high cholesterol, diabetes, inactivity and being overweight.",
            "Stopping smoking is the single most effective change for most "
            "people who smoke.",
            "An NHS Health Check is offered every five years between 40 and "
            "74 and includes a cardiovascular risk assessment.",
        ),
    ),
    "hypertension": Guidance(
        key="hypertension",
        title="High blood pressure",
        source="NHS",
        url="https://www.nhs.uk/conditions/high-blood-pressure-hypertension/",
        points=(
            "High blood pressure rarely has symptoms, so the only way to "
            "know is to have it measured.",
            "Free checks are available at many pharmacies and GP surgeries "
            "without an appointment.",
            "Cutting salt, alcohol and excess weight, and moving more, all "
            "lower it; some people also need medication.",
        ),
    ),
    "metabolic_syndrome": Guidance(
        key="metabolic_syndrome",
        title="Metabolic syndrome",
        source="NHS",
        url="https://www.nhs.uk/conditions/metabolic-syndrome/",
        points=(
            "Diagnosed when at least three of these occur together: central "
            "obesity, raised triglycerides, low HDL, raised blood pressure, "
            "raised fasting glucose.",
            "It raises the risk of both type 2 diabetes and cardiovascular "
            "disease, and the same changes improve all of them at once.",
        ),
    ),
    "sleep_apnoea": Guidance(
        key="sleep_apnoea",
        title="Obstructive sleep apnoea",
        source="NHS",
        url="https://www.nhs.uk/conditions/sleep-apnoea/",
        points=(
            "Common signs are loud snoring, stopping breathing in the night, "
            "and being very tired during the day.",
            "It is worth seeing a GP, because untreated it raises blood "
            "pressure and cardiovascular risk, and treatment works well.",
            "Losing excess weight, sleeping on your side and reducing "
            "alcohol before bed can all help.",
        ),
    ),
    "nafld": Guidance(
        key="nafld",
        title="Fatty liver disease",
        source="NHS",
        url="https://www.nhs.uk/conditions/non-alcoholic-fatty-liver-disease/",
        points=(
            "Usually causes no symptoms in its early stages and is often "
            "found incidentally on a blood test or scan.",
            "Losing weight gradually and staying active are the main "
            "treatments; there is no specific medication for it.",
        ),
    ),
    "ckd": Guidance(
        key="ckd",
        title="Chronic kidney disease",
        source="NHS",
        url="https://www.nhs.uk/conditions/kidney-disease/",
        points=(
            "Often has no symptoms until it is advanced, and is usually "
            "picked up on a blood or urine test.",
            "Controlling blood pressure and blood sugar is the main way to "
            "slow it down.",
        ),
    ),
}


FACTOR_GUIDANCE: dict[str, Guidance] = {
    "smoking_burden": Guidance(
        key="smoking_burden",
        title="Stopping smoking",
        source="NHS",
        url="https://www.nhs.uk/better-health/quit-smoking/",
        points=(
            "Free local Stop Smoking Services roughly triple the chance of "
            "quitting successfully compared with willpower alone.",
            "Circulation and lung function begin improving within weeks, and "
            "cardiovascular risk keeps falling for years after quitting.",
        ),
    ),
    "adiposity": Guidance(
        key="adiposity",
        title="Healthy weight",
        source="NHS",
        url="https://www.nhs.uk/live-well/healthy-weight/",
        points=(
            "Waist measurement matters independently of BMI: risk rises above "
            "94cm for men and 80cm for women.",
            "Losing 5 to 10 percent of body weight produces meaningful "
            "improvements in blood pressure, blood sugar and cholesterol.",
        ),
    ),
    "activity_deficit": Guidance(
        key="activity_deficit",
        title="Physical activity guidelines",
        source="NHS",
        url="https://www.nhs.uk/live-well/exercise/exercise-guidelines/",
        points=(
            "Adults should aim for at least 150 minutes of moderate activity "
            "a week, or 75 minutes of vigorous, plus strengthening work on "
            "two days.",
            "Any activity is better than none, and breaking up long periods "
            "of sitting counts.",
        ),
    ),
    "sleep_debt": Guidance(
        key="sleep_debt",
        title="Sleep",
        source="NHS",
        url="https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/how-to-fall-asleep-faster-and-sleep-better/",
        points=(
            "Most adults need between seven and nine hours.",
            "Consistent sleep and wake times, and keeping screens out of the "
            "bedroom, are the changes with the most evidence behind them.",
        ),
    ),
    "alcohol_burden": Guidance(
        key="alcohol_burden",
        title="Alcohol guidelines",
        source="NHS",
        url="https://www.nhs.uk/live-well/alcohol-advice/calculating-alcohol-units/",
        points=(
            "UK Chief Medical Officers advise no more than 14 units a week, "
            "spread over three or more days, with drink-free days.",
            "14 units is about six pints of average-strength beer or ten "
            "small glasses of lower-strength wine.",
        ),
    ),
    "diet_quality": Guidance(
        key="diet_quality",
        title="Eating well",
        source="NHS",
        url="https://www.nhs.uk/live-well/eat-well/",
        points=(
            "Base meals on higher-fibre starchy carbohydrates, eat at least "
            "five portions of fruit and vegetables a day, and cut saturated "
            "fat, salt and sugar.",
            "Adults should have no more than 6g of salt a day.",
        ),
    ),
    "stress_load": Guidance(
        key="stress_load",
        title="Stress",
        source="NHS",
        url="https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/stress/",
        points=(
            "Sustained stress affects sleep, blood pressure and eating, so "
            "it shows up in physical health rather than only mood.",
            "NHS talking therapies can be self-referred in England without "
            "going through a GP.",
        ),
    ),
}


def for_condition(key: str) -> Guidance | None:
    return CONDITION_GUIDANCE.get(key)


def for_factor(key: str) -> Guidance | None:
    return FACTOR_GUIDANCE.get(key)
