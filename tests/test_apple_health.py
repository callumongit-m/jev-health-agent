"""Apple Health export parsing -- the free wearables path."""

import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from health_agent.domain.profile import FieldMeta, HealthProfile, Sex, Source
from health_agent.ingest.apple_health import merge_into, parse_export

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def _stamp(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S %z")


def _record(rtype: str, value, days_ago: float, unit: str = "", end_days=None) -> str:
    end = _stamp(end_days if end_days is not None else days_ago)
    return (
        f'<Record type="{rtype}" unit="{unit}" value="{value}" '
        f'startDate="{_stamp(days_ago)}" endDate="{end}"/>'
    )


def _export(records: list[str]) -> str:
    return f'<?xml version="1.0"?><HealthData>{"".join(records)}</HealthData>'


@pytest.fixture
def export_xml(tmp_path) -> Path:
    records = [
        _record("HKQuantityTypeIdentifierBodyMass", 86.4, 2, "kg"),
        _record("HKQuantityTypeIdentifierBodyMass", 91.0, 300, "kg"),  # outside window
        _record("HKQuantityTypeIdentifierHeight", 1.78, 5, "m"),
        _record("HKQuantityTypeIdentifierRestingHeartRate", 62, 1, "count/min"),
        _record("HKQuantityTypeIdentifierRestingHeartRate", 66, 3, "count/min"),
        _record("HKQuantityTypeIdentifierBloodPressureSystolic", 146, 4, "mmHg"),
        _record("HKQuantityTypeIdentifierBloodPressureDiastolic", 91, 4, "mmHg"),
        _record("HKQuantityTypeIdentifierBloodGlucose", 104.0, 6, "mg/dL"),
        # two days of steps, 3000 + 1500 and 4000
        _record("HKQuantityTypeIdentifierStepCount", 3000, 1, "count"),
        _record("HKQuantityTypeIdentifierStepCount", 1500, 1, "count"),
        _record("HKQuantityTypeIdentifierStepCount", 4000, 2, "count"),
        _record("HKQuantityTypeIdentifierAppleExerciseTime", 20, 1, "min"),
        _record("HKQuantityTypeIdentifierAppleExerciseTime", 10, 2, "min"),
        # one night: 7h asleep within 8h in bed
        f'<Record type="HKCategoryTypeIdentifierSleepAnalysis" '
        f'value="HKCategoryValueSleepAnalysisAsleepCore" '
        f'startDate="{_stamp(1.4)}" endDate="{_stamp(1.1)}"/>',
        f'<Record type="HKCategoryTypeIdentifierSleepAnalysis" '
        f'value="HKCategoryValueSleepAnalysisInBed" '
        f'startDate="{_stamp(1.45)}" endDate="{_stamp(1.1)}"/>',
    ]
    path = tmp_path / "export.xml"
    path.write_text(_export(records))
    return path


def test_parses_core_fields(export_xml):
    fields, _ = parse_export(export_xml, now=NOW)
    assert fields["weight_kg"] == 86.4, "should take the most recent weight"
    assert fields["height_cm"] == 178.0, "metres must convert to cm"
    assert fields["resting_hr"] == 64, "should average the samples"
    assert fields["systolic_bp"] == 146 and fields["diastolic_bp"] == 91


def test_ignores_samples_outside_the_window(export_xml):
    """A weight from a year ago is not this person's current picture."""
    fields, _ = parse_export(export_xml, now=NOW)
    assert fields["weight_kg"] != 91.0


def test_step_counts_are_summed_per_day_then_averaged(export_xml):
    """Apple writes many small step records a day; summing all of them would
    conflate a busy day with a long history."""
    fields, _ = parse_export(export_xml, now=NOW)
    assert fields["steps_daily_avg"] == 4250  # (4500 + 4000) / 2


def test_exercise_minutes_are_reported_weekly(export_xml):
    fields, _ = parse_export(export_xml, now=NOW)
    assert fields["moderate_activity_minutes_per_week"] == 105  # mean 15/day * 7


def test_glucose_converts_from_mg_dl(export_xml):
    fields, _ = parse_export(export_xml, now=NOW)
    assert fields["fasting_glucose_mmol_l"] == pytest.approx(5.77, abs=0.02)


def test_sleep_hours_and_efficiency(export_xml):
    fields, _ = parse_export(export_xml, now=NOW)
    assert fields["sleep_hours_avg"] == pytest.approx(7.2, abs=0.1)
    assert fields["sleep_efficiency_pct"] == pytest.approx(85.7, abs=1.0)


def test_output_validates_against_healthprofile(export_xml):
    fields, provenance = parse_export(export_xml, now=NOW)
    profile = HealthProfile(**fields, provenance=provenance)
    assert profile.bmi == pytest.approx(27.3, abs=0.1)
    assert all(m.source is Source.WEARABLE for m in profile.provenance.values())
    assert all(m.device == "apple_health_export" for m in profile.provenance.values())


def test_reads_a_zip_export(export_xml, tmp_path):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(export_xml, "apple_health_export/export.xml")
    assert parse_export(archive, now=NOW)[0]["weight_kg"] == 86.4


def test_zip_without_export_xml_is_an_error(tmp_path):
    archive = tmp_path / "wrong.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("readme.txt", "nope")
    with pytest.raises(ValueError, match="no export.xml"):
        parse_export(archive, now=NOW)


def test_merge_does_not_clobber_a_fresher_clinical_reading(export_xml):
    """A BP taken at the GP last week beats a watch reading from four days ago."""
    profile = HealthProfile(
        age=54, sex=Sex.MALE, systolic_bp=132, diastolic_bp=84,
        provenance={
            "systolic_bp": FieldMeta(
                source=Source.CLINICAL, observed_at=NOW - timedelta(days=1)
            )
        },
    )
    merged = merge_into(profile, export_xml, now=NOW)
    assert merged.systolic_bp == 132, "clinical reading was fresher"
    assert merged.diastolic_bp == 91, "but diastolic had no clinical provenance"
    assert merged.weight_kg == 86.4


def test_merge_fills_gaps_and_keeps_unrelated_fields(export_xml):
    profile = HealthProfile(age=54, sex=Sex.MALE, smoking_status="former")
    merged = merge_into(profile, export_xml, now=NOW)
    assert merged.age == 54 and merged.smoking_status == "former"
    assert merged.resting_hr == 64
