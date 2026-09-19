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
