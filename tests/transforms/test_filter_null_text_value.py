"""Tests for filter_null_text_value stage."""

import polars as pl
from omegaconf import DictConfig

from foresight_r.transforms.filter_null_text_value import (
    filter_null_text_value_fntr,
)


def test_filter_nulls_for_given_prefixes():
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "prefix": ["LAB", "MEDICATION", "LAB"],
            "text_value": [None, "Positive", "Negative"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["LAB"]})
    fn = filter_null_text_value_fntr(cfg)
    result = fn(df).collect()

    # Row with prefix LAB and text_value None should be removed; others kept
    assert result.shape[0] == 2
    assert result["prefix"].to_list() == ["MEDICATION", "LAB"]


def test_filter_nulls_when_no_prefixes_provided():
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "prefix": ["A", "B", "C"],
            "text_value": ["High", None, None],
        }
    ).lazy()

    cfg = DictConfig({})
    fn = filter_null_text_value_fntr(cfg)
    result = fn(df).collect()

    # All rows with null text_value should be removed
    assert result.shape[0] == 1
    assert result["text_value"].to_list() == ["High"]


def test_preserve_other_columns_and_empty_dataframe():
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [100, 200],
            "prefix": ["X", "Y"],
            "text_value": ["Foo", None],
        }
    ).lazy()

    fn = filter_null_text_value_fntr(DictConfig({"prefixes": ["Y"]}))
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
            "prefix": pl.Series([], dtype=pl.Utf8),
            "text_value": pl.Series([], dtype=pl.Utf8),
        }
    ).lazy()

    empty_fn = filter_null_text_value_fntr(DictConfig({}))
    empty_result = empty_fn(empty_df).collect()
    assert len(empty_result) == 0
