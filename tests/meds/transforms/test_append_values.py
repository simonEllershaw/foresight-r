"""Tests for append_values stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.meds.transforms.append_values import append_values_fntr


@pytest.fixture
def sample_data():
    """Sample data with codes, numeric_value, and text_value."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4],
            "time": [1, 2, 3, 4, 5],
            "code": ["glucose", "hemoglobin", "diagnosis", "weight", "status"],
            "numeric_value": [32.0, 37.62, None, 75.5, None],
            "text_value": ["ml", "g/dL", "confirmed", None, None],
        }
    ).lazy()


@pytest.fixture
def default_config():
    """Default configuration."""
    return DictConfig({"join_char": "//"})


def test_append_values_all_present(sample_data, default_config):
    """Test appending with all values present."""
    fn = append_values_fntr(default_config)
    result = fn(sample_data).collect()

    # glucose: 32.0 -> "32" (integer), text_value: "ml"
    assert result["code"][0] == "glucose//32//ml"
    # hemoglobin: 37.62 -> "37.6" (3sf), text_value: "g/dL"
    assert result["code"][1] == "hemoglobin//37.6//g/dL"


def test_append_values_text_only(sample_data, default_config):
    """Test appending with only text_value present."""
    fn = append_values_fntr(default_config)
    result = fn(sample_data).collect()

    # diagnosis: no numeric_value, text_value: "confirmed"
    assert result["code"][2] == "diagnosis//confirmed"


def test_append_values_numeric_only(sample_data, default_config):
    """Test appending with only numeric_value present."""
    fn = append_values_fntr(default_config)
    result = fn(sample_data).collect()

    # weight: 75.5 -> "75.5", no text_value
    assert result["code"][3] == "weight//75.5"


def test_append_values_neither(sample_data, default_config):
    """Test that code is unchanged when neither value is present."""
    fn = append_values_fntr(default_config)
    result = fn(sample_data).collect()

    # status: no numeric_value, no text_value
    assert result["code"][4] == "status"


def test_append_values_custom_join_char():
    """Test with custom join character."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [1],
            "code": ["glucose"],
            "numeric_value": [32.0],
            "text_value": ["ml"],
        }
    ).lazy()

    cfg = DictConfig({"join_char": " | "})
    fn = append_values_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"][0] == "glucose | 32 | ml"


def test_append_values_preserves_other_columns(sample_data, default_config):
    """Test that transformation preserves all other columns."""
    fn = append_values_fntr(default_config)
    result = fn(sample_data).collect()

    assert "subject_id" in result.columns
    assert "time" in result.columns
    assert result["subject_id"].to_list() == [1, 1, 2, 3, 4]
    assert result["time"].to_list() == [1, 2, 3, 4, 5]


def test_append_values_empty_dataframe(default_config):
    """Test that transformation handles empty DataFrames."""
    test_df = pl.DataFrame(
        {
            "subject_id": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Int64),
            "code": pl.Series([], dtype=pl.String),
            "numeric_value": pl.Series([], dtype=pl.Float64),
            "text_value": pl.Series([], dtype=pl.String),
        }
    ).lazy()

    fn = append_values_fntr(default_config)
    result = fn(test_df).collect()

    assert len(result) == 0
    assert "code" in result.columns


def test_append_values_null_code(default_config):
    """Test handling of null code values."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["glucose", None],
            "numeric_value": [32.0, 45.0],
            "text_value": ["ml", "units"],
        }
    ).lazy()

    fn = append_values_fntr(default_config)
    result = fn(test_df).collect()

    assert result["code"][0] == "glucose//32//ml"
    assert result["code"][1] is None


def test_append_values_integer_formatting():
    """Test that integer-like floats are formatted without decimal."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["a", "b", "c"],
            "numeric_value": [1.0, 100.0, 0.0],
            "text_value": [None, None, None],
        }
    ).lazy()

    cfg = DictConfig({"join_char": "//"})
    fn = append_values_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["a//1", "b//100", "c//0"]


def test_append_values_3sf_formatting():
    """Test 3 significant figure formatting."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": ["a", "b", "c", "d"],
            "numeric_value": [1.234, 12.34, 123.4, 1234.0],
            "text_value": [None, None, None, None],
        }
    ).lazy()

    cfg = DictConfig({"join_char": "//"})
    fn = append_values_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["a//1.23", "b//12.3", "c//123", "d//1234"]


def test_append_values_custom_sig_figs():
    """Test custom significant figures."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["a", "b"],
            "numeric_value": [1.23456, 9.87654],
            "text_value": [None, None],
        }
    ).lazy()

    # Test with 2 sig figs
    cfg = DictConfig({"join_char": "//", "sig_figs": 2})
    fn = append_values_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["a//1.2", "b//9.9"]

    # Test with 5 sig figs
    cfg2 = DictConfig({"join_char": "//", "sig_figs": 5})
    fn2 = append_values_fntr(cfg2)
    result2 = fn2(test_df).collect()

    assert result2["code"].to_list() == ["a//1.2346", "b//9.8765"]
