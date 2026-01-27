"""Tests for replace_code_with_text_value_by_prefix stage."""

import polars as pl
from omegaconf import DictConfig

from foresight_r.transforms.replace_code_with_text_value_by_prefix import (
    replace_code_with_text_value_by_prefix_fntr,
)


def test_replace_matching_prefix():
    """Rows with matching prefix have code replaced by text_value."""
    df = pl.DataFrame(
        {
            "code": ["TRANSFER//ward", "TRANSFER//icu", "LAB//glucose"],
            "prefix": ["TRANSFER", "TRANSFER", "LAB"],
            "text_value": ["ADMIT", "DISCHARGE", "HIGH"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["TRANSFER"]})
    fn = replace_code_with_text_value_by_prefix_fntr(cfg)
    result = fn(df).collect()

    assert result["code"].to_list() == ["ADMIT", "DISCHARGE", "LAB//glucose"]


def test_no_replacement_if_no_match():
    """Rows with non-matching prefixes keep original code."""
    df = pl.DataFrame(
        {
            "code": ["LAB//glucose", "VITAL//hr"],
            "prefix": ["LAB", "VITAL"],
            "text_value": ["HIGH", "100"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["TRANSFER"]})
    fn = replace_code_with_text_value_by_prefix_fntr(cfg)
    result = fn(df).collect()

    assert result["code"].to_list() == ["LAB//glucose", "VITAL//hr"]


def test_empty_config():
    """Empty prefixes list results in no changes."""
    df = pl.DataFrame(
        {
            "code": ["TRANSFER//ward", "LAB//glucose"],
            "prefix": ["TRANSFER", "LAB"],
            "text_value": ["ADMIT", "HIGH"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": []})
    fn = replace_code_with_text_value_by_prefix_fntr(cfg)
    result = fn(df).collect()

    assert result["code"].to_list() == ["TRANSFER//ward", "LAB//glucose"]


def test_multiple_prefixes():
    """Multiple prefixes can be configured."""
    df = pl.DataFrame(
        {
            "code": ["TRANSFER//ward", "HOSPITAL//admission", "LAB//glucose"],
            "prefix": ["TRANSFER", "HOSPITAL", "LAB"],
            "text_value": ["ADMIT", "IN", "HIGH"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["TRANSFER", "HOSPITAL"]})
    fn = replace_code_with_text_value_by_prefix_fntr(cfg)
    result = fn(df).collect()

    assert result["code"].to_list() == ["ADMIT", "IN", "LAB//glucose"]


def test_preserves_columns():
    """Other columns are preserved."""
    df = pl.DataFrame(
        {
            "code": ["TRANSFER//ward"],
            "prefix": ["TRANSFER"],
            "text_value": ["ADMIT"],
            "other": [123],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["TRANSFER"]})
    fn = replace_code_with_text_value_by_prefix_fntr(cfg)
    result = fn(df).collect()

    assert "other" in result.columns
    assert result["other"][0] == 123
