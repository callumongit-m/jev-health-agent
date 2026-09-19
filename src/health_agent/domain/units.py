"""Unit-tolerant parsing.

People know their height in feet and inches, their weight in stone, and their
cholesterol in whatever their GP wrote down. Making them convert first is
friction that loses them, and worse, is where they make mistakes.

So every measurement accepts whatever people actually say -- "5'11", "85kg",
"13 stone 4", "5.7%" -- and normalises to the one unit the model sees.
Ambiguity is resolved by plausibility, never silently guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CM_PER_INCH = 2.54
KG_PER_LB = 0.45359237
KG_PER_STONE = 6.35029318
#: Mass-per-volume to molar, for the markers reported both ways.
MGDL_PER_MMOL_GLUCOSE = 18.0182
MGDL_PER_MMOL_CHOL = 38.67


class UnitError(ValueError):
    """The value could not be read confidently. Ask rather than guess."""


@dataclass(frozen=True, slots=True)
class Parsed:
    value: float
    assumed_unit: str
    #: True when the unit was inferred from plausibility, not stated.
    inferred: bool = False


def _number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:[.,]\d+)?", text.replace(",", "."))
    return float(match.group()) if match else None


def parse_height(raw: str | float | int) -> Parsed:
    """cm, m, feet and inches, or a bare number resolved by plausibility."""
    if isinstance(raw, (int, float)):
        return _height_from_number(float(raw), stated=False)

    text = str(raw).strip().lower().replace("’", "'").replace("”", '"')

    # 5'11", 5 ft 11.5 in, 5 feet 11
    feet_inches = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:'|ft|feet|foot)\s*(\d+(?:\.\d+)?)?", text
    )
    if feet_inches:
        feet = float(feet_inches.group(1))
        inches = float(feet_inches.group(2) or 0)
        return Parsed(round((feet * 12 + inches) * CM_PER_INCH, 1), "cm")

    number = _number(text)
    if number is None:
        raise UnitError(f"could not read a height from {raw!r}")

    if re.search(r"cm\b|centimet", text):
        return Parsed(round(number, 1), "cm")
    if re.search(r"m\b|metre|meter", text) and "cm" not in text:
        return Parsed(round(number * 100, 1), "cm")
    if re.search(r'"|\d\s*in\b|\binch', text):
        return Parsed(round(number * CM_PER_INCH, 1), "cm")
    return _height_from_number(number, stated=False)


def _height_from_number(number: float, *, stated: bool) -> Parsed:
    """Bare numbers: 1.8 is metres, 71 is inches, 180 is centimetres."""
    if 1.2 <= number <= 2.3:
        return Parsed(round(number * 100, 1), "cm", inferred=not stated)
    if 48 <= number <= 90:
        return Parsed(round(number * CM_PER_INCH, 1), "cm", inferred=not stated)
    if 120 <= number <= 230:
        return Parsed(round(number, 1), "cm", inferred=not stated)
    raise UnitError(
        f"{number} is not a plausible height in any common unit -- "
        "please say cm, metres, or feet and inches"
    )


def parse_weight(raw: str | float | int) -> Parsed:
    """kg, lb, or stone (including '13 st 4 lb')."""
    if isinstance(raw, (int, float)):
        return _weight_from_number(float(raw))

    text = str(raw).strip().lower()

    stone = re.search(r"(\d+(?:\.\d+)?)\s*(?:st\b|stone)", text)
    if stone:
        total = float(stone.group(1)) * KG_PER_STONE
        pounds = re.search(r"(?:st\b|stone)\D{0,6}(\d+(?:\.\d+)?)", text)
        if pounds:
            total += float(pounds.group(1)) * KG_PER_LB
        return Parsed(round(total, 1), "kg")

    number = _number(text)
    if number is None:
        raise UnitError(f"could not read a weight from {raw!r}")

    if re.search(r"kgs?\b|kilo", text):
        return Parsed(round(number, 1), "kg")
    if re.search(r"lbs?\b|pound", text):
        return Parsed(round(number * KG_PER_LB, 1), "kg")
    return _weight_from_number(number)


def _weight_from_number(number: float) -> Parsed:
    # Overlapping ranges are real: 150 could be 150lb or 150kg. Pounds is far
    # more likely for someone writing a bare number that high, but this is
    # flagged as inferred so the caller can confirm.
    if 30 <= number <= 140:
        return Parsed(round(number, 1), "kg", inferred=True)
    if 140 < number <= 440:
        return Parsed(round(number * KG_PER_LB, 1), "kg", inferred=True)
    raise UnitError(
        f"{number} is not a plausible weight -- please say kg, lb or stone"
    )


def parse_waist(raw: str | float | int) -> Parsed:
    """cm or inches. A bare number under 60 is almost certainly inches."""
    text = str(raw).strip().lower()
    number = _number(text) if not isinstance(raw, (int, float)) else float(raw)
    if number is None:
        raise UnitError(f"could not read a waist measurement from {raw!r}")

    if re.search(r"cm\b|centimet", text):
        return Parsed(round(number, 1), "cm")
    if re.search(r'"|\d\s*in\b|\binch', text):
        return Parsed(round(number * CM_PER_INCH, 1), "cm")
    if 20 <= number <= 60:
        return Parsed(round(number * CM_PER_INCH, 1), "cm", inferred=True)
    if 60 < number <= 200:
        return Parsed(round(number, 1), "cm", inferred=True)
    raise UnitError(f"{number} is not a plausible waist measurement")


def parse_hba1c(raw: str | float | int) -> Parsed:
    """IFCC mmol/mol or DCCT %. The two scales do not overlap."""
    text = str(raw).strip().lower()
    number = _number(text) if not isinstance(raw, (int, float)) else float(raw)
    if number is None:
        raise UnitError(f"could not read an HbA1c from {raw!r}")

    if "%" in text or re.search(r"\bdcct\b", text):
        return Parsed(round((number - 2.15) * 10.929, 1), "mmol/mol")
    if re.search(r"mmol", text):
        return Parsed(round(number, 1), "mmol/mol")
    # 4-15 is the % scale; 20-200 is mmol/mol. They cannot be confused.
    if 3.5 <= number <= 18:
        return Parsed(round((number - 2.15) * 10.929, 1), "mmol/mol", inferred=True)
    if 20 <= number <= 200:
        return Parsed(round(number, 1), "mmol/mol", inferred=True)
    raise UnitError(f"{number} is not a plausible HbA1c")


def parse_glucose(raw: str | float | int) -> Parsed:
    """mmol/L or mg/dL."""
    text = str(raw).strip().lower()
    number = _number(text) if not isinstance(raw, (int, float)) else float(raw)
    if number is None:
        raise UnitError(f"could not read a glucose value from {raw!r}")

    if re.search(r"mg\s*/?\s*dl", text):
        return Parsed(round(number / MGDL_PER_MMOL_GLUCOSE, 2), "mmol/L")
    if re.search(r"mmol", text):
        return Parsed(round(number, 2), "mmol/L")
    if 2 <= number <= 30:
        return Parsed(round(number, 2), "mmol/L", inferred=True)
    if 40 <= number <= 600:
        return Parsed(round(number / MGDL_PER_MMOL_GLUCOSE, 2), "mmol/L", inferred=True)
    raise UnitError(f"{number} is not a plausible blood glucose")


def parse_cholesterol(raw: str | float | int) -> Parsed:
    """mmol/L or mg/dL. Used for total, HDL, LDL and triglycerides."""
    text = str(raw).strip().lower()
    number = _number(text) if not isinstance(raw, (int, float)) else float(raw)
    if number is None:
        raise UnitError(f"could not read a cholesterol value from {raw!r}")

    if re.search(r"mg\s*/?\s*dl", text):
        return Parsed(round(number / MGDL_PER_MMOL_CHOL, 2), "mmol/L")
    if re.search(r"mmol", text):
        return Parsed(round(number, 2), "mmol/L")
    if 0.2 <= number <= 20:
        return Parsed(round(number, 2), "mmol/L", inferred=True)
    if 20 < number <= 800:
        return Parsed(round(number / MGDL_PER_MMOL_CHOL, 2), "mmol/L", inferred=True)
    raise UnitError(f"{number} is not a plausible cholesterol value")


def parse_temperature_free(raw: str) -> float | None:  # pragma: no cover
    return None


#: field name -> parser, for bulk normalisation of a raw input dict.
PARSERS = {
    "height_cm": parse_height,
    "weight_kg": parse_weight,
    "waist_cm": parse_waist,
    "hba1c_mmol_mol": parse_hba1c,
    "fasting_glucose_mmol_l": parse_glucose,
    "total_cholesterol_mmol_l": parse_cholesterol,
    "hdl_mmol_l": parse_cholesterol,
    "ldl_mmol_l": parse_cholesterol,
    "triglycerides_mmol_l": parse_cholesterol,
}


def normalise(raw: dict) -> tuple[dict, dict[str, str], list[str]]:
    """Normalise a raw input dict.

    Returns (fields, conversions, problems). `conversions` records anything
    that was inferred rather than stated, so the agent can confirm it instead
    of quietly acting on a guess.
    """
    fields: dict = {}
    conversions: dict[str, str] = {}
    problems: list[str] = []

    for key, value in raw.items():
        parser = PARSERS.get(key)
        if parser is None or value is None:
            fields[key] = value
            continue
        try:
            parsed = parser(value)
        except UnitError as exc:
            problems.append(str(exc))
            continue
        fields[key] = parsed.value
        if parsed.inferred and str(value).strip() != str(parsed.value):
            conversions[key] = f"read {value!r} as {parsed.value} {parsed.assumed_unit}"
    return fields, conversions, problems
