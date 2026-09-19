"""Unit tolerance. Making people convert is friction, and where mistakes happen."""

import pytest

from health_agent.domain.units import (
    UnitError, normalise, parse_cholesterol, parse_glucose, parse_hba1c,
    parse_height, parse_waist, parse_weight,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5'11.5", 181.6), ("5'11.5\"", 181.6), ("5 ft 11.5 in", 181.6),
     ("5 foot 11", 180.3), ("181.6cm", 181.6), ("181.6 cm", 181.6),
     ("1.816m", 181.6), ("71.5in", 181.6), ("71.5 inches", 181.6),
     (181.6, 181.6), (1.816, 181.6), (71.5, 181.6)],
)
def test_height_accepts_anything_people_actually_say(raw, expected):
    assert parse_height(raw).value == pytest.approx(expected, abs=0.2)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("85kg", 85.0), ("85 kg", 85.0), ("85", 85.0),
     ("13 stone 4", 84.4), ("13st 4lb", 84.4), ("13.5 stone", 85.7),
     ("187lbs", 84.8), ("187 lb", 84.8), (187, 84.8)],
)
def test_weight_accepts_kg_pounds_and_stone(raw, expected):
    assert parse_weight(raw).value == pytest.approx(expected, abs=0.2)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("32in", 81.3), ("32 inches", 81.3), ('34"', 86.4),
     ("82cm", 82.0), (32, 81.3), (82, 82.0)],
)
def test_waist_accepts_inches_or_cm(raw, expected):
    assert parse_waist(raw).value == pytest.approx(expected, abs=0.2)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5.7%", 38.8), ("5.7 %", 38.8), ("39 mmol/mol", 39.0), (5.7, 38.8), (39, 39.0)],
)
def test_hba1c_accepts_both_scales(raw, expected):
    """IFCC and DCCT do not overlap, so this can never be ambiguous."""
    assert parse_hba1c(raw).value == pytest.approx(expected, abs=0.2)


def test_glucose_and_cholesterol_convert_from_mg_dl():
    assert parse_glucose("100 mg/dL").value == pytest.approx(5.55, abs=0.02)
    assert parse_cholesterol("200 mg/dl").value == pytest.approx(5.17, abs=0.02)
    assert parse_glucose("5.5 mmol/L").value == 5.5


def test_stated_units_are_not_flagged_as_guesses():
    """Only genuine inferences should be surfaced for confirmation."""
    for raw in ("181.6cm", "1.816m", "5'11", "85kg", "187lbs", "5.7%"):
        parser = parse_height if any(c in str(raw) for c in "cm'm") and "kg" not in str(raw) and "%" not in str(raw) and "lb" not in str(raw) else None
    assert parse_height("181.6cm").inferred is False
    assert parse_weight("85kg").inferred is False
    assert parse_hba1c("5.7%").inferred is False
    assert parse_height(180).inferred is True, "a bare number is a guess"


def test_implausible_values_raise_rather_than_guess():
    for bad in ("900", "3"):
        with pytest.raises(UnitError):
            parse_height(bad)
    with pytest.raises(UnitError):
        parse_weight("900")


def test_normalise_reports_guesses_and_problems():
    fields, conversions, problems = normalise(
        {"age": 21, "height_cm": "5'11", "weight_kg": 200, "hba1c_mmol_mol": "bananas"}
    )
    assert fields["age"] == 21, "untouched fields pass through"
    assert fields["height_cm"] == pytest.approx(180.3, abs=0.2)
    assert fields["weight_kg"] == pytest.approx(90.7, abs=0.2)
    assert "weight_kg" in conversions, "an inferred unit must be surfaced"
    assert problems and "hba1c" not in fields


def test_normalised_input_reaches_the_profile():
    from health_agent.domain.profile import HealthProfile

    fields, _, _ = normalise(
        {"age": 55, "height_cm": "5'10", "weight_kg": "14 stone 2", "waist_cm": "36in"}
    )
    profile = HealthProfile(**fields)
    assert profile.bmi == pytest.approx(28.4, abs=0.3)
    assert profile.waist_to_height == pytest.approx(0.514, abs=0.01)
