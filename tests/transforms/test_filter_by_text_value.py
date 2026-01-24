"""Tests for filter_by_text_value stage."""

import polars as pl
from omegaconf import DictConfig

from foresight_r.transforms.filter_by_text_value import (
    filter_by_text_value_fntr,
)


def test_filter_keeps_matching_prefix_and_text_value():
    """Rows with configured prefix and matching text_value are kept."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": [
                "TRANSFER//ward",
                "TRANSFER//icu",
                "TRANSFER//ward",
                "LAB//glucose",
            ],
            "text_value": ["ADMIT", "DISCHARGE", "TRANSFER", "HIGH"],
        }
    ).lazy()

    cfg = DictConfig({"prefix_to_text_values": {"TRANSFER": ["ADMIT", "TRANSFER"]}})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    # Row 1 (TRANSFER + ADMIT), Row 3 (TRANSFER + TRANSFER), Row 4 (LAB - not filtered)
    # Row 2 removed (TRANSFER + DISCHARGE, not in allowed values)
    assert result.shape[0] == 3
    assert result["subject_id"].to_list() == [1, 3, 4]


def test_filter_keeps_rows_with_non_configured_prefixes():
    """Rows with prefixes NOT in the config are always kept."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "time": [1, 2, 3, 4, 5],
            "code": [
                "TRANSFER//ward",
                "LAB//glucose",
                "VITAL//hr",
                "MEDICATION//aspirin",
                "TRANSFER//icu",
            ],
            "text_value": ["ADMIT", "HIGH", "NORMAL", "TAKEN", "CANCELLED"],
        }
    ).lazy()

    cfg = DictConfig({"prefix_to_text_values": {"TRANSFER": ["ADMIT"]}})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    # Row 1 (TRANSFER + ADMIT - kept), Row 5 (TRANSFER + CANCELLED - removed)
    # Rows 2, 3, 4 (LAB, VITAL, MEDICATION - not in config, always kept)
    assert result.shape[0] == 4
    assert result["subject_id"].to_list() == [1, 2, 3, 4]


def test_filter_multiple_prefixes():
    """Multiple prefixes can be configured."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "time": [1, 2, 3, 4, 5, 6],
            "code": [
                "TRANSFER//ward",
                "HOSPITAL//admission",
                "TRANSFER//icu",
                "HOSPITAL//discharge",
                "LAB//glucose",
                "HOSPITAL//er",
            ],
            "text_value": ["ADMIT", "IN", "DISCHARGE", "OUT", "HIGH", "VISIT"],
        }
    ).lazy()

    cfg = DictConfig(
        {"prefix_to_text_values": {"TRANSFER": ["ADMIT"], "HOSPITAL": ["IN", "OUT"]}}
    )
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    # Row 1 (TRANSFER + ADMIT - kept)
    # Row 2 (HOSPITAL + IN - kept)
    # Row 3 (TRANSFER + DISCHARGE - removed)
    # Row 4 (HOSPITAL + OUT - kept)
    # Row 5 (LAB - not configured, kept)
    # Row 6 (HOSPITAL + VISIT - removed)
    assert result.shape[0] == 4
    assert result["subject_id"].to_list() == [1, 2, 4, 5]


def test_filter_no_config_keeps_all():
    """No config provided means all rows are kept."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["LAB//glucose", "MEDICATION//aspirin", "VITAL//hr"],
            "text_value": ["High", "Taken", "Normal"],
        }
    ).lazy()

    cfg = DictConfig({})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 3


def test_filter_empty_dict_keeps_all():
    """Empty prefix_to_text_values dict means all rows are kept."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["LAB//glucose", "MEDICATION//aspirin"],
            "text_value": ["High", "Taken"],
        }
    ).lazy()

    cfg = DictConfig({"prefix_to_text_values": {}})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 2


def test_filter_all_configured_prefix_rows_removed():
    """If all rows with configured prefix have non-matching text_value, they are all removed."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["TRANSFER//ward", "TRANSFER//icu", "LAB//glucose"],
            "text_value": ["UNKNOWN", "CANCELLED", "HIGH"],
        }
    ).lazy()

    cfg = DictConfig({"prefix_to_text_values": {"TRANSFER": ["ADMIT", "DISCHARGE"]}})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    # Both TRANSFER rows removed (text_values don't match)
    # LAB row kept (not in config)
    assert result.shape[0] == 1
    assert result["subject_id"].to_list() == [3]


def test_filter_preserves_all_columns():
    """All columns are preserved after filtering."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [100, 200],
            "code": ["TRANSFER//ward", "MEDICATION//aspirin"],
            "text_value": ["ADMIT", "Taken"],
            "numeric_value": [None, 10.0],
            "extra_col": ["a", "b"],
        }
    ).lazy()

    cfg = DictConfig({"prefix_to_text_values": {"TRANSFER": ["ADMIT"]}})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    # Both rows kept (TRANSFER matches, MEDICATION not in config)
    assert result.shape[0] == 2
    assert "subject_id" in result.columns
    assert "time" in result.columns
    assert "code" in result.columns
    assert "text_value" in result.columns
    assert "numeric_value" in result.columns
    assert "extra_col" in result.columns


def test_filter_empty_dataframe():
    """Empty dataframe returns empty dataframe."""
    df = pl.DataFrame(
        {
            "subject_id": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Int64),
            "code": pl.Series([], dtype=pl.Utf8),
            "text_value": pl.Series([], dtype=pl.Utf8),
        }
    ).lazy()

    cfg = DictConfig({"prefix_to_text_values": {"TRANSFER": ["ADMIT"]}})
    fn = filter_by_text_value_fntr(cfg)
    result = fn(df).collect()

    assert len(result) == 0
