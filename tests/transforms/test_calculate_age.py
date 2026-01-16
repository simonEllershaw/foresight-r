"""Tests for the calculate_age transform."""

from datetime import datetime

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.transforms.calculate_age import calculate_age_fntr


@pytest.fixture
def sample_data_with_birth():
    """Sample MEDS data with birth records."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 2, 2, 2],
            "time": [
                datetime(2000, 1, 1, 0, 0, 0),  # Birth for subject 1
                datetime(2020, 6, 15, 10, 30, 0),  # ~20.5 years later
                datetime(2025, 3, 20, 14, 15, 0),  # ~25.2 years later
                datetime(2030, 1, 1, 0, 0, 0),  # Exactly 30 years later
                datetime(1990, 5, 10, 0, 0, 0),  # Birth for subject 2
                datetime(2010, 8, 25, 8, 0, 0),  # ~20.3 years later
                datetime(2015, 12, 31, 23, 59, 59),  # ~25.6 years later
            ],
            "code": [
                "Born",
                "Lab Test",
                "Admission",
                "Procedure",
                "Born",
                "Diagnosis",
                "Discharge",
            ],
            "prefix": [
                "Birth",
                "Laboratory",
                "Hospital",
                "Procedure",
                "Birth",
                "Clinical",
                "Hospital",
            ],
        }
    ).lazy()


@pytest.fixture
def sample_data_without_time():
    """Sample MEDS data where birth records have null time."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "time": [None, datetime(2020, 1, 1), datetime(2020, 1, 1)],
            "code": ["Born", "Lab Test", "Born"],
            "prefix": ["Birth", "Laboratory", "Birth"],
        }
    ).lazy()


def test_calculate_age_basic():
    """Test basic age calculation functionality."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2010, 1, 1),
                datetime(2020, 1, 1),
            ],
            "code": ["Born", "Event1", "Event2"],
            "prefix": ["Birth", "Other", "Other"],
        }
    ).lazy()

    fn = calculate_age_fntr(DictConfig({}))
    result = fn(test_df).collect()

    # Check columns exist
    assert "age_year" in result.columns
    assert "age_day" in result.columns

    # Birth row should have age 0
    assert result.filter(pl.col("code") == "Born")["age_day"][0] == 0
    assert result.filter(pl.col("code") == "Born")["age_year"][0] == 0

    # 10 years later (exactly on anniversary): age_year=10, age_day=0
    age_10y_year = result.filter(pl.col("code") == "Event1")["age_year"][0]
    age_10y_day = result.filter(pl.col("code") == "Event1")["age_day"][0]
    assert age_10y_year == 10
    assert age_10y_day == 0

    # 20 years later (exactly on anniversary): age_year=20, age_day=0
    age_20y_year = result.filter(pl.col("code") == "Event2")["age_year"][0]
    age_20y_day = result.filter(pl.col("code") == "Event2")["age_day"][0]
    assert age_20y_year == 20
    assert age_20y_day == 0


def test_calculate_age_multiple_subjects(sample_data_with_birth):
    """Test age calculation with multiple subjects."""
    fn = calculate_age_fntr(DictConfig({}))
    result = fn(sample_data_with_birth).collect()

    # Check all subjects have age columns
    assert result["age_year"].is_not_null().all()
    assert result["age_day"].is_not_null().all()

    # Birth rows should have age 0 for all subjects
    birth_rows = result.filter(pl.col("prefix") == "Birth")
    assert birth_rows["age_day"].to_list() == [0, 0]
    assert birth_rows["age_year"].to_list() == [0, 0]

    # Subject 1: Event at 2030-01-01 is exactly 30 years after 2000-01-01
    subj1_2030 = result.filter(
        (pl.col("subject_id") == 1) & (pl.col("time") == datetime(2030, 1, 1))
    )
    assert subj1_2030["age_year"][0] == 30
    # Exactly 30 years, so 0 remaining days
    assert subj1_2030["age_day"][0] == 0


def test_calculate_age_precision():
    """Test that age calculations work correctly with combined year and day representation."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0, 0),
                datetime(2000, 1, 2, 12, 0, 0),  # 1 day later
                datetime(2001, 1, 21, 0, 0, 0),  # 385 days later (1 year 20 days)
            ],
            "code": ["Born", "Event1", "Event2"],
            "prefix": ["Birth", "Other", "Other"],
        }
    ).lazy()

    fn = calculate_age_fntr(DictConfig({}))
    result = fn(test_df).collect()

    # 1 day old: age_year=0, age_day=1
    event1_row = result.filter(pl.col("code") == "Event1")
    assert event1_row["age_year"][0] == 0
    assert event1_row["age_day"][0] == 1

    # 385 days later: 1 year and 20 days
    event2_row = result.filter(pl.col("code") == "Event2")
    assert event2_row["age_year"][0] == 1
    assert event2_row["age_day"][0] == 20


def test_calculate_age_preserves_other_columns():
    """Test that the transform doesn't drop or modify other columns."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2000, 1, 1), datetime(2010, 1, 1)],
            "code": ["Born", "Event"],
            "prefix": ["Birth", "Other"],
            "numeric_value": [None, 42.0],
            "extra_col": ["A", "B"],
        }
    ).lazy()

    fn = calculate_age_fntr(DictConfig({}))
    result = fn(test_df).collect()

    # All original columns should be preserved
    assert "subject_id" in result.columns
    assert "time" in result.columns
    assert "code" in result.columns
    assert "prefix" in result.columns
    assert "numeric_value" in result.columns
    assert "extra_col" in result.columns

    # New columns added
    assert "age_year" in result.columns
    assert "age_day" in result.columns

    # Values preserved
    assert result["extra_col"].to_list() == ["A", "B"]
    assert result["numeric_value"][1] == 42.0


def test_calculate_age_year_decimal():
    """Test that age_year has decimal precision."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 7, 1),  # ~6 months later
            ],
            "code": ["Born", "Event"],
            "prefix": ["Birth", "Other"],
        }
    ).lazy()

    fn = calculate_age_fntr(DictConfig({}))
    result = fn(test_df).collect()

    event_row = result.filter(pl.col("code") == "Event")
    age_year = event_row["age_year"][0]
    age_day = event_row["age_day"][0]

    # ~181 days (6 months): age_year=0, age_day=181
    assert age_year == 0
    assert 180 <= age_day <= 182  # Account for exact day count


def test_calculate_age_handles_same_subject_multiple_events():
    """Test that age is calculated correctly when subject has many events."""
    times = [datetime(2000, 1, 1)] + [datetime(2000 + i, 1, 1) for i in range(1, 11)]
    test_df = pl.DataFrame(
        {
            "subject_id": [1] * 11,
            "time": times,
            "code": ["Born"] + [f"Event{i}" for i in range(1, 11)],
            "prefix": ["Birth"] + ["Other"] * 10,
        }
    ).lazy()

    fn = calculate_age_fntr(DictConfig({}))
    result = fn(test_df).collect()

    # Check that ages are monotonically increasing
    ages = result["age_year"].to_list()
    assert ages == sorted(ages)

    # Check specific values
    assert ages[0] == 0  # Birth
    for i in range(1, 11):
        expected_age = i
        assert ages[i] == expected_age


def test_calculate_age_custom_birth_code():
    """Test that birth_code parameter can be customized."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2010, 1, 1),
                datetime(2015, 6, 15),
            ],
            "code": ["DOB", "Event1", "Event2"],
            "prefix": ["Demographics", "Other", "Other"],
        }
    ).lazy()

    # Use custom birth_code
    fn = calculate_age_fntr(DictConfig({"birth_code": "DOB"}))
    result = fn(test_df).collect()

    # Birth row should have age 0
    assert result.filter(pl.col("code") == "DOB")["age_year"][0] == 0
    assert result.filter(pl.col("code") == "DOB")["age_day"][0] == 0

    # 10 years later
    age_10y = result.filter(pl.col("code") == "Event1")["age_year"][0]
    assert age_10y == 10

    # 15 years and ~166 days later
    age_15y = result.filter(pl.col("code") == "Event2")["age_year"][0]
    assert age_15y == 15


def test_calculate_age_default_birth_code():
    """Test that default birth_code is 'Born'."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2000, 1, 1), datetime(2010, 1, 1)],
            "code": ["Born", "Event"],
            "prefix": ["Birth", "Other"],
        }
    ).lazy()

    # Use default config (should use "Born" as birth_code)
    fn = calculate_age_fntr(DictConfig({}))
    result = fn(test_df).collect()

    assert result.filter(pl.col("code") == "Born")["age_year"][0] == 0
    assert result.filter(pl.col("code") == "Event")["age_year"][0] == 10
