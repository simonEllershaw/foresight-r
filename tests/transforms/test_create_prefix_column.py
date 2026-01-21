"""Tests for create_prefix_column stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.transforms.create_prefix_column import create_prefix_column_fntr


@pytest.fixture
def sample_data_with_prefixes():
    """Sample data with codes that have prefixes."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "time": [1, 2, 3, 4],
            "code": [
                "LAB//glucose//fasting",
                "ADMISSION//emergency",
                "MEDICATION//aspirin//100mg",
                "DIAGNOSIS//diabetes",
            ],
        }
    ).lazy()


@pytest.fixture
def sample_data_without_prefixes():
    """Sample data with codes that have no prefixes."""
    return pl.DataFrame(
        {"subject_id": [1, 2], "time": [1, 2], "code": ["glucose", "aspirin"]}
    ).lazy()


def test_create_prefix_column_with_prefixes(sample_data_with_prefixes):
    """Test that codes with '//' are split correctly into prefix and code."""
    fn = create_prefix_column_fntr(DictConfig({}))
    result = fn(sample_data_with_prefixes).collect()

    assert result["prefix"].to_list() == [
        "LAB",
        "ADMISSION",
        "MEDICATION",
        "DIAGNOSIS",
    ]
    assert result["code"].to_list() == [
        "glucose//fasting",
        "emergency",
        "aspirin//100mg",
        "diabetes",
    ]


def test_create_prefix_column_without_prefixes(sample_data_without_prefixes):
    """Test handling of codes without '//' delimiter.

    Current behavior: entire code becomes prefix, code becomes empty string.
    """
    fn = create_prefix_column_fntr(DictConfig({}))
    result = fn(sample_data_without_prefixes).collect()

    assert result["prefix"].to_list() == ["glucose", "aspirin"]
    assert result["code"].to_list() == ["", ""]


def test_create_prefix_column_preserves_other_columns():
    """Test that transformation preserves all other columns."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [100, 200],
            "code": ["LAB//glucose", "MEDICATION//aspirin"],
            "numeric_value": [5.5, 10.0],
        }
    ).lazy()

    fn = create_prefix_column_fntr(DictConfig({}))
    result = fn(test_df).collect()

    assert "subject_id" in result.columns
    assert "time" in result.columns
    assert "numeric_value" in result.columns
    assert result["subject_id"].to_list() == [1, 2]
    assert result["time"].to_list() == [100, 200]
    assert result["numeric_value"].to_list() == [5.5, 10.0]


def test_create_prefix_column_multiple_delimiters():
    """Test that only first '//' is used as delimiter, rest stay in code."""
    test_df = pl.DataFrame(
        {"subject_id": [1], "time": [1], "code": ["LAB//glucose//fasting//venous"]}
    ).lazy()

    fn = create_prefix_column_fntr(DictConfig({}))
    result = fn(test_df).collect()

    assert result["prefix"][0] == "LAB"
    assert result["code"][0] == "glucose//fasting//venous"


def test_create_prefix_column_empty_dataframe():
    """Test that transformation handles empty DataFrames."""
    test_df = pl.DataFrame(
        {
            "subject_id": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Int64),
            "code": pl.Series([], dtype=pl.String),
        }
    ).lazy()

    fn = create_prefix_column_fntr(DictConfig({}))
    result = fn(test_df).collect()

    assert len(result) == 0
    assert "prefix" in result.columns
    assert "code" in result.columns
