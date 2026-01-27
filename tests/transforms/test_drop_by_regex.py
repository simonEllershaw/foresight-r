"""Tests for drop_by_regex stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.transforms.drop_by_regex import drop_by_regex_fntr


def test_drop_single_pattern():
    """Rows matching a single regex pattern are DROPPED."""
    df = pl.DataFrame(
        {
            "code": ["MATCH", "NOPE", "PARTIAL_MATCH_SUFFIX"],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["MATCH"]})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    # Matches "MATCH" and "PARTIAL_MATCH_SUFFIX" -> DROPPED
    # "NOPE" -> KEPT
    assert result.shape[0] == 1
    assert result["val"][0] == 2


def test_drop_multiple_patterns():
    """Rows matching ANY of the multiple regex patterns are DROPPED."""
    df = pl.DataFrame(
        {
            "code": ["A_123", "B_456", "C_789", "D_000"],
            "val": [1, 2, 3, 4],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["A_.*", "B_.*"]})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    # Matches A_123 and B_456 -> DROPPED
    # C_789 and D_000 -> KEPT
    assert result.shape[0] == 2
    assert sorted(result["val"].to_list()) == [3, 4]


def test_drop_no_matches():
    """Rows matching NONE of the patterns are KEPT."""
    df = pl.DataFrame(
        {
            "code": ["X", "Y", "Z"],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["A", "B"]})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 3


def test_drop_custom_column():
    """Filter works on a specified column other than 'code'."""
    df = pl.DataFrame(
        {
            "code": ["IGNORE", "IGNORE"],
            "target": ["KEEP_ME", "DROP_ME"],
        }
    ).lazy()

    cfg = DictConfig({"patterns": ["DROP"], "column": "target"})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 1
    assert result["target"][0] == "KEEP_ME"


def test_drop_empty_config_raises_error():
    """ValueError is raised if patterns list is empty."""
    cfg = DictConfig({"patterns": []})
    with pytest.raises(ValueError, match="Must provide at least one pattern"):
        drop_by_regex_fntr(cfg)
