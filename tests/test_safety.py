"""Red flags are the highest-consequence path in the system."""

import pytest

from health_agent.safety import CRISIS, EMERGENCY, advice_for, detect

MUST_FLAG = [
    "I have crushing chest pain radiating to my left arm",
    "my chest feels a bit tight after running",
    "tightness in my chest when I climb stairs",
    "my face is drooping and I have slurred speech",
    "sudden weakness on one side of my body",
    "I can't breathe properly",
    "I have been coughing up blood",
    "worst headache of my life",
    "I do not want to live anymore",
    "I don't want to live",
    "I want to die",
    "everyone would be better off without me",
    "life is not worth living",
]

MUST_NOT_FLAG = [
    "I want to live a longer healthier life",
    "I do not want to die young, how do I improve?",
    "I don't want to die, that's why I'm asking",
    "I had a chest infection last year, fully recovered",
    "I walk 10000 steps and sleep 7 hours a night",
    "my resting heart rate is 58 and I feel fine",
    "",
]


@pytest.mark.parametrize("text", MUST_FLAG)
def test_flags_acute_presentations(text):
    assert detect(text), f"missed a red flag: {text!r}"


@pytest.mark.parametrize("text", MUST_NOT_FLAG)
def test_does_not_flag_ordinary_wellness_talk(text):
    assert not detect(text), f"false positive: {text!r}"


def test_emergency_advice_is_ordered_first():
    combined = advice_for(detect("chest pain and I want to kill myself"))
    assert combined.index(EMERGENCY) < combined.index(CRISIS)


def test_detect_scans_multiple_fields():
    assert detect(None, "feeling fine", "chest pain on exertion")


# --- symptom patterns: triage, never diagnosis -------------------------

PATTERN_CASES = [
    ("constant headache at night and throwing up in the morning",
     "raised_intracranial_pressure"),
    ("headache every day and I keep being sick", "raised_intracranial_pressure"),
    ("always thirsty and weeing constantly", "hyperglycaemia"),
    ("losing weight without trying and night sweats", "possible_malignancy"),
    ("breathless on stairs and my ankles are swollen", "cardiac_exertional"),
    ("I snore badly and I'm tired all the time", "sleep_apnoea"),
]


@pytest.mark.parametrize(("text", "expected"), PATTERN_CASES)
def test_symptom_combinations_are_caught(text, expected):
    from health_agent.safety import detect_patterns

    assert expected in [p.key for p in detect_patterns(text)], text


@pytest.mark.parametrize("text", [
    "I have a headache",
    "I was sick after a takeaway",
    "I am tired",
    "I snore a bit",
    "I have been losing weight on purpose with diet and exercise",
])
def test_single_symptoms_alone_are_not_flagged(text):
    """Each of these is common and unremarkable on its own. Flagging them
    individually would make the triage useless through noise."""
    from health_agent.safety import detect_patterns

    assert detect_patterns(text) == [], text


def test_guidance_gives_urgency_and_never_a_diagnosis():
    """Naming a condition invites self-treatment, and a wrong keyword match
    causes real fear. Urgency is the part that is actionable and gettable-right."""
    from health_agent.safety import detect_patterns, pattern_guidance

    text = "constant headache at night and throwing up in the morning"
    guidance = pattern_guidance(detect_patterns(text)).lower()

    assert "urgent" in guidance
    assert "not a diagnosis" in guidance
    for named in ("tumour", "tumor", "cancer", "diabetes", "heart failure",
                  "apnoea", "apnea", "stroke", "migraine", "hypertension",
                  "you may have", "you might have", "this could be",
                  "likely to be", "suggests you"):
        assert named not in guidance, f"guidance named a condition: {named!r}"


def test_no_pattern_anywhere_names_a_condition():
    """The guarantee has to hold for every pattern, not just the one tested."""
    from health_agent.safety import SYMPTOM_PATTERNS, pattern_guidance

    banned = ("tumour", "cancer", "diabetes", "heart failure", "apnoea",
              "you may have", "you might have", "this could be")
    for pattern in SYMPTOM_PATTERNS:
        text = (pattern.advice + " " + pattern.tell_them).lower()
        for named in banned:
            assert named not in text, f"{pattern.key} names {named!r}"


def test_patterns_say_what_to_tell_a_clinician():
    from health_agent.safety import SYMPTOM_PATTERNS

    for pattern in SYMPTOM_PATTERNS:
        assert len(pattern.tell_them) > 40, pattern.key


def test_an_emergency_still_outranks_a_pattern():
    from health_agent.safety import detect, detect_patterns

    text = "crushing chest pain, and I'm breathless with swollen ankles"
    assert detect(text), "the emergency must still fire"
    assert detect_patterns(text), "and the pattern is still recognised"


# --- the classifier as a second net ------------------------------------

PARAPHRASED_EMERGENCIES = [
    "elephant sitting on my chest, left arm numb",
    "my speech went funny for ten minutes then came back",
    "worst pain I've ever had in my head, came on in seconds",
    "my vision went grey in one eye like a curtain coming down",
    "I keep thinking everyone would be fine if I wasn't around",
]


@pytest.mark.parametrize("text", PARAPHRASED_EMERGENCIES)
def test_the_keyword_screen_is_known_to_miss_these(text):
    """Documenting the gap, not accepting it.

    Keyword matching is deterministic, free and works offline, so it stays
    as the floor. It is also brittle: every one of these is a real
    emergency described the way someone actually would, and none of them
    match. That is what the classifier screen is for.
    """
    from health_agent.safety import detect, detect_patterns

    assert not detect(text) and not detect_patterns(text), (
        f"{text!r} now matches a keyword -- move it out of this list"
    )


def test_the_classifier_screen_catches_them(monkeypatch):
    """With the screen firing, an emergency stops the assessment even when
    no keyword matched."""
    from health_agent.adapters.core import assess
    from health_agent.domain.conditions import ACUTE_SCREEN
    from health_agent.scoring import FakeBackend, RiskScorer
    from health_agent.scoring.backend import RawAnswer

    original = FakeBackend.classify

    def flags_acute(self, state):
        result = original(self, state)
        answers = dict(result.answers)
        answers[ACUTE_SCREEN.key] = RawAnswer(kind="noul", value=0.92)
        return type(result)(answers=answers, model=result.model,
                            request_id=result.request_id)

    monkeypatch.setattr(FakeBackend, "classify", flags_acute)
    out = assess(
        {"age": 54, "sex": "male", "height_cm": 178, "weight_kg": 92,
         "smoking_status": "former", "alcohol_units_per_week": 10,
         "moderate_activity_minutes_per_week": 90, "sleep_hours_avg": 7,
         "on_bp_medication": False, "previously_high_glucose": False,
         "eats_vegetables_daily": True},
        raw_text="elephant sitting on my chest",
        scorer=RiskScorer(FakeBackend(seed=1, noise=0.0)),
    )
    assert out["status"] == "seek_care"
    assert "risk" not in out, "must not show numbers to someone mid-emergency"
    assert "emergency services" in out["answer"]


def test_the_persons_own_words_reach_the_classifier():
    """The screen is worthless if the free text never gets there."""
    from health_agent.domain.profile import HealthProfile
    from health_agent.scoring import FakeBackend, RiskScorer

    seen = {}

    class Recording(FakeBackend):
        def classify(self, state):
            seen.update(state)
            return super().classify(state)

    RiskScorer(Recording(seed=1)).score(
        HealthProfile(age=40, sex="male"), free_text="chest feels tight"
    )
    assert seen.get("_in_their_own_words") == "chest feels tight"


def test_an_urgent_screen_leads_the_report_without_suppressing_it():
    from health_agent.adapters.core import assess
    from health_agent.domain.conditions import URGENT_SCREEN
    from health_agent.scoring import FakeBackend, RiskScorer
    from health_agent.scoring.backend import RawAnswer

    original = FakeBackend.classify

    def flags_urgent(self, state):
        result = original(self, state)
        answers = dict(result.answers)
        answers[URGENT_SCREEN.key] = RawAnswer(kind="noul", value=0.85)
        return type(result)(answers=answers, model=result.model,
                            request_id=result.request_id)

    FakeBackend.classify = flags_urgent
    try:
        out = assess(
            {"age": 54, "sex": "male", "height_cm": 178, "weight_kg": 92,
             "smoking_status": "former", "alcohol_units_per_week": 10,
             "moderate_activity_minutes_per_week": 90, "sleep_hours_avg": 7,
             "on_bp_medication": False, "previously_high_glucose": False,
             "eats_vegetables_daily": True},
            raw_text="been coughing up rusty stuff for a week",
            scorer=RiskScorer(FakeBackend(seed=1, noise=0.0)),
        )
    finally:
        FakeBackend.classify = original

    assert out["status"] == "complete", "urgent is not an emergency"
    assert "within the next few days" in out["act_on_this_first"]
    assert out["risk"], "they should still get their picture"
