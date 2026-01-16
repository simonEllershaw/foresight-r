#!/usr/bin/env python
"""Tests for the add_time_column transform."""

from datetime import datetime, time

import polars as pl
from omegaconf import DictConfig

from foresight_r.transforms.add_time_column import END_OF_DAY, add_time_column_fntr


def test_add_time_column_basic():
    """Test basic time extraction."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "time": [
                datetime(2020, 6, 15, 10, 30, 0),
                datetime(2020, 6, 15, 14, 45, 30),
                datetime(2021, 3, 10, 8, 15, 0),
            ],
            "code": ["Lab Test", "Medication", "Admission"],
        }
    ).lazy()

    fn = add_time_column_fntr(DictConfig({"null_end_of_day": False}))
    result = fn(df).collect()

    assert "time_of_day" in result.columns
    assert result["time_of_day"].to_list()[0] == time(10, 30, 0)
    assert result["time_of_day"].to_list()[1] == time(14, 45, 30)
    assert result["time_of_day"].to_list()[2] == time(8, 15, 0)


def test_add_time_column_with_end_of_day_null():
    """Test that END_OF_DAY is converted to null when null_end_of_day=True."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2020, 6, 15, 10, 30, 0),
                datetime(2020, 6, 15, 23, 59, 59, 999999),
                datetime(2021, 3, 10, 8, 15, 0),
            ],
            "code": ["Lab Test", "Discharge", "Admission"],
        }
    ).lazy()

    fn = add_time_column_fntr(DictConfig({"null_end_of_day": True}))
    result = fn(df).collect()

    assert result["time_of_day"].to_list()[0] == time(10, 30, 0)
    assert result["time_of_day"].to_list()[1] is None
    assert result["time_of_day"].to_list()[2] == time(8, 15, 0)


def test_add_time_column_with_end_of_day_kept():
    """Test that END_OF_DAY is kept when null_end_of_day=False."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [
                datetime(2020, 6, 15, 10, 30, 0),
                datetime(2020, 6, 15, 23, 59, 59, 999999),
            ],
            "code": ["Lab Test", "Discharge"],
        }
    ).lazy()

    fn = add_time_column_fntr(DictConfig({"null_end_of_day": False}))
    result = fn(df).collect()

    assert result["time_of_day"].to_list()[0] == time(10, 30, 0)
    assert result["time_of_day"].to_list()[1] == END_OF_DAY


def test_add_time_column_default_config():
    """Test that default config sets null_end_of_day to True."""
    df = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [datetime(2020, 6, 15, 23, 59, 59, 999999)],
            "code": ["Discharge"],
        }
    ).lazy()

    fn = add_time_column_fntr(DictConfig({}))
    result = fn(df).collect()

    assert result["time_of_day"].to_list()[0] is None


def test_add_time_column_hhmm_format():
    """Test that time_format='HH:MM' produces HH:MM string output."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "time": [
                datetime(2020, 6, 15, 10, 30, 45),
                datetime(2020, 6, 15, 14, 45, 30),
                datetime(2021, 3, 10, 8, 15, 0),
            ],
            "code": ["Lab Test", "Medication", "Admission"],
        }
    ).lazy()

    fn = add_time_column_fntr(
        DictConfig({"time_format": "HH:MM", "null_end_of_day": False})
    )
    result = fn(df).collect()

    assert result["time_of_day"].to_list()[0] == "10:30"
    assert result["time_of_day"].to_list()[1] == "14:45"
    assert result["time_of_day"].to_list()[2] == "08:15"


def test_add_time_column_hhmm_with_end_of_day():
    """Test that HH:MM format with null_end_of_day works correctly."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2020, 6, 15, 10, 30, 0),
                datetime(2020, 6, 15, 23, 59, 59, 999999),
                datetime(2021, 3, 10, 8, 15, 0),
            ],
            "code": ["Lab Test", "Discharge", "Admission"],
        }
    ).lazy()

    fn = add_time_column_fntr(
        DictConfig({"time_format": "HH:MM", "null_end_of_day": True})
    )
    result = fn(df).collect()

    assert result["time_of_day"].to_list()[0] == "10:30"
    assert result["time_of_day"].to_list()[1] is None
    assert result["time_of_day"].to_list()[2] == "08:15"
