"""Tests for filter_null_values stage."""

import polars as pl
from omegaconf import DictConfig

from foresight_r.meds.transforms.filter_null_values import (
    filter_null_values_fntr,
)


def test_filter_nulls_for_given_prefixes():
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": [
                "LAB//glucose",
                "LAB//glucose",
                "MEDICATION//aspirin",
                "LAB//glucose",
            ],
            "numeric_value": [None, 10.0, None, None],
            "text_value": [None, None, None, "Positive"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["LAB"]})
    fn = filter_null_values_fntr(cfg)
    result = fn(df).collect()

    # Row 1: LAB//glucose, numeric_value=None, text_value=None -> should be removed
    # Row 2: LAB//glucose, numeric_value=10.0, text_value=None -> should be kept
    # Row 3: MEDICATION//aspirin, numeric_value=None, text_value=None -> should be kept (code doesn't start with LAB)
    # Row 4: LAB//glucose, numeric_value=None, text_value="Positive" -> should be kept

    assert result.shape[0] == 3
    assert result["subject_id"].to_list() == [2, 3, 4]


def test_filter_nulls_when_no_prefixes_provided():
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": ["A//1", "B//2", "C//3", "D//4"],
            "numeric_value": [None, 10.0, None, None],
            "text_value": [None, None, "High", None],
        }
    ).lazy()

    cfg = DictConfig({})
    fn = filter_null_values_fntr(cfg)
    result = fn(df).collect()

    # Rows with both nulls should be removed
    # Row 1 and 4 have both nulls.
    assert result.shape[0] == 2
    assert result["subject_id"].to_list() == [2, 3]


def test_filter_nulls_without_text_value_column():
    df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["LAB//glucose", "LAB//glucose"],
            "numeric_value": [None, 10.0],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["LAB"]})
    fn = filter_null_values_fntr(cfg)
    result = fn(df).collect()

    # Row 1 should be removed
    assert result.shape[0] == 1
    assert result["subject_id"].to_list() == [2]


def test_preserve_other_columns_and_empty_dataframe():
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [100, 200],
            "code": ["X//1", "Y//2"],
            "numeric_value": [None, None],
            "text_value": ["Foo", None],
        }
    ).lazy()

    fn = filter_null_values_fntr(DictConfig({"prefixes": ["Y"]}))
    result = fn(test_df).collect()

    # Ensure columns preserved and correct removal
    assert "subject_id" in result.columns
    assert "time" in result.columns
    assert "text_value" in result.columns
    assert result.shape[0] == 1

    # Test empty dataframe handled
    empty_df = pl.DataFrame(
        {
            "subject_id": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Int64),
            "code": pl.Series([], dtype=pl.Utf8),
            "numeric_value": pl.Series([], dtype=pl.Float64),
            "text_value": pl.Series([], dtype=pl.Utf8),
        }
    ).lazy()

    empty_fn = filter_null_values_fntr(DictConfig({}))
    empty_result = empty_fn(empty_df).collect()
    assert len(empty_result) == 0
