"""Tests for the add_time_column transform."""

from datetime import datetime

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.meds.transforms.add_time_column import add_time_column_fntr


@pytest.fixture
def sample_data():
    """Sample MEDS data with various times."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [
                datetime(2020, 1, 1, 10, 30, 45),
                datetime(2020, 1, 1, 23, 59, 59, 999999),
                datetime(2020, 1, 2, 0, 0, 0),
                datetime(2020, 1, 3, 14, 15, 30),
                datetime(2020, 1, 4, 23, 59, 59, 999999),
            ],
            "code": ["Lab", "Admission", "Midnight", "Procedure", "Discharge"],
        }
    ).lazy()


class TestAddTimeColumnBasic:
    """Basic functionality tests."""

    def test_default_format(self, sample_data):
        """Test default HH:MM:SS format."""
        fn = add_time_column_fntr(DictConfig({}))
        result = fn(sample_data).collect()

        assert "time_of_day" in result.columns
        assert result["time_of_day"][0] == "10:30:45"
        assert result["time_of_day"][2] == "00:00:00"
        assert result["time_of_day"][3] == "14:15:30"

    def test_custom_format_hhmm(self, sample_data):
        """Test HH:MM format."""
        fn = add_time_column_fntr(DictConfig({"time_format": "%H:%M"}))
        result = fn(sample_data).collect()

        assert result["time_of_day"][0] == "10:30"
        assert result["time_of_day"][3] == "14:15"

    def test_custom_format_12hour(self, sample_data):
        """Test 12-hour format with AM/PM."""
        fn = add_time_column_fntr(DictConfig({"time_format": "%I:%M %p"}))
        result = fn(sample_data).collect()

        assert result["time_of_day"][0] == "10:30 AM"
        assert result["time_of_day"][3] == "02:15 PM"


class TestEndOfDayReplacement:
    """Tests for end-of-day replacement logic."""

    def test_replace_end_of_day_with_string(self, sample_data):
        """Test replacing 23:59:59.999999 with custom string."""
        cfg = DictConfig(
            {
                "replace_end_of_day": True,
                "end_of_day_fill_value": "Time Unknown",
            }
        )
        fn = add_time_column_fntr(cfg)
        result = fn(sample_data).collect()

        # End-of-day times should be replaced
        assert result["time_of_day"][1] == "Time Unknown"
        assert result["time_of_day"][4] == "Time Unknown"

        # Other times should be formatted normally
        assert result["time_of_day"][0] == "10:30:45"
        assert result["time_of_day"][2] == "00:00:00"

    def test_replace_end_of_day_with_null(self, sample_data):
        """Test replacing 23:59:59.999999 with null."""
        cfg = DictConfig(
            {
                "replace_end_of_day": True,
                "end_of_day_fill_value": None,
            }
        )
        fn = add_time_column_fntr(cfg)
        result = fn(sample_data).collect()

        # End-of-day times should be null
        assert result["time_of_day"][1] is None
        assert result["time_of_day"][4] is None

        # Other times should be formatted
        assert result["time_of_day"][0] == "10:30:45"

    def test_no_replacement_when_disabled(self, sample_data):
        """Test that end-of-day is NOT replaced when replace_end_of_day=False."""
        cfg = DictConfig(
            {
                "replace_end_of_day": False,
                "end_of_day_fill_value": "Should Not Appear",
            }
        )
        fn = add_time_column_fntr(cfg)
        result = fn(sample_data).collect()

        # End-of-day times should be formatted, not replaced
        assert result["time_of_day"][1] == "23:59:59"
        assert result["time_of_day"][4] == "23:59:59"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        empty_df = pl.DataFrame(
            {
                "subject_id": pl.Series([], dtype=pl.Int64),
                "time": pl.Series([], dtype=pl.Datetime),
                "code": pl.Series([], dtype=pl.Utf8),
            }
        ).lazy()

        fn = add_time_column_fntr(DictConfig({}))
        result = fn(empty_df).collect()

        assert "time_of_day" in result.columns
        assert len(result) == 0

    def test_midnight_times(self):
        """Test handling of midnight (00:00:00)."""
        df = pl.DataFrame(
            {
                "subject_id": [1],
                "time": [datetime(2020, 1, 1, 0, 0, 0)],
                "code": ["Event"],
            }
        ).lazy()

        fn = add_time_column_fntr(DictConfig({"time_format": "%H:%M:%S"}))
        result = fn(df).collect()

        assert result["time_of_day"][0] == "00:00:00"

    def test_near_end_of_day_not_replaced(self):
        """Test that times close to but not exactly end-of-day are not replaced."""
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [
                    datetime(2020, 1, 1, 23, 59, 59, 999998),  # 1 microsecond before
                    datetime(2020, 1, 1, 23, 59, 59, 0),  # No microseconds
                    datetime(2020, 1, 1, 23, 59, 58, 999999),  # 1 second before
                ],
                "code": ["A", "B", "C"],
            }
        ).lazy()

        cfg = DictConfig(
            {
                "replace_end_of_day": True,
                "end_of_day_fill_value": "Replaced",
            }
        )
        fn = add_time_column_fntr(cfg)
        result = fn(df).collect()

        # None of these should be replaced
        assert result["time_of_day"][0] != "Replaced"
        assert result["time_of_day"][1] != "Replaced"
        assert result["time_of_day"][2] != "Replaced"

    def test_preserves_other_columns(self, sample_data):
        """Test that original columns are preserved."""
        fn = add_time_column_fntr(DictConfig({}))
        result = fn(sample_data).collect()

        assert "subject_id" in result.columns
        assert "time" in result.columns
        assert "code" in result.columns
        assert result["subject_id"].to_list() == [1, 1, 1, 2, 2]

    def test_with_null_times(self):
        """Test handling of null datetime values."""
        df = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [datetime(2020, 1, 1, 10, 30, 0), None],
                "code": ["Event1", "Event2"],
            }
        ).lazy()

        fn = add_time_column_fntr(DictConfig({}))
        result = fn(df).collect()

        assert result["time_of_day"][0] == "10:30:00"
        assert result["time_of_day"][1] is None
