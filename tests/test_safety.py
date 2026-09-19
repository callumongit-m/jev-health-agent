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
