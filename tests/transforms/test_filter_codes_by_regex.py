"""Tests for filter_codes_by_regex stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.transforms.filter_codes_by_regex import filter_codes_by_regex_fntr


def test_filter_single_pattern():
    """Rows matching a single regex pattern are kept."""
    df = pl.DataFrame(
        {
            "code": ["MATCH", "NOPE", "PARTIAL_MATCH_SUFFIX"],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["MATCH"]})
    fn = filter_codes_by_regex_fntr(cfg)
    result = fn(df).collect()

    # Matches "MATCH" and "PARTIAL_MATCH_SUFFIX"
    assert result.shape[0] == 2
    assert sorted(result["val"].to_list()) == [1, 3]


def test_filter_multiple_patterns():
    """Rows matching ANY of the multiple regex patterns are kept."""
    df = pl.DataFrame(
        {
            "code": ["A_123", "B_456", "C_789", "D_000"],
            "val": [1, 2, 3, 4],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["A_.*", "B_.*"]})
    fn = filter_codes_by_regex_fntr(cfg)
    result = fn(df).collect()

    # Matches A_123 and B_456
    assert result.shape[0] == 2
    assert sorted(result["val"].to_list()) == [1, 2]


def test_filter_no_matches():
    """Rows matching NONE of the patterns are dropped."""
    df = pl.DataFrame(
        {
            "code": ["X", "Y", "Z"],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["A", "B"]})
    fn = filter_codes_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 0


def test_filter_custom_column():
    """Filter works on a specified column other than 'code'."""
    df = pl.DataFrame(
        {
            "code": ["IGNORE", "IGNORE"],
            "target": ["KEEP_ME", "DROP_ME"],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["KEEP"], "column": "target"})
    fn = filter_codes_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 1
    assert result["target"][0] == "KEEP_ME"


def test_filter_empty_config_raises_error():
    """ValueError is raised if patterns list is empty."""
    cfg = DictConfig({"patterns": []})
    with pytest.raises(ValueError, match="Must provide at least one pattern"):
        filter_codes_by_regex_fntr(cfg)
