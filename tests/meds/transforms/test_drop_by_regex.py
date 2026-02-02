"""Tests for drop_by_regex stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.meds.transforms.drop_by_regex import drop_by_regex_fntr


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


def test_drop_with_except_regex():
    """Rows matching regex are DROPPED unless they also match except_regex."""
    df = pl.DataFrame(
        {
            "code": [
                "DIAGNOSIS_RELATED_GROUPS//123",  # Match regex, no exception -> DROPPED
                "DIAGNOSIS_RELATED_GROUPS//HCFA/456",  # Match regex AND exception -> KEPT
                "OTHER//789",  # No match -> KEPT
            ],
            "val": [1, 2, 3],
        }
    ).lazy()

    patterns = [
        {
            "regex": "^DIAGNOSIS_RELATED_GROUPS//.*",
            "except_regex": "^DIAGNOSIS_RELATED_GROUPS//HCFA.*",
        }
    ]
    cfg = DictConfig({"patterns": patterns})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 2
    assert sorted(result["val"].to_list()) == [2, 3]


def test_drop_mixed_patterns():
    """Support mixing string patterns and dictionary patterns."""
    df = pl.DataFrame(
        {
            "code": [
                "A_DROP",  # String match -> DROPPED
                "B_DROP_EXCEPT_ME",  # Dict match + exception -> KEPT
                "B_DROP_THIS",  # Dict match -> DROPPED
                "C_SAFE",  # No match -> KEPT
            ],
            "val": [1, 2, 3, 4],
        }
    ).lazy()

    patterns = [
        "A_.*",  # String pattern
        {
            "regex": "B_.*",
            "except_regex": ".*EXCEPT_ME",
        },
    ]
    cfg = DictConfig({"patterns": patterns})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 2
    assert sorted(result["val"].to_list()) == [2, 4]


def test_invalid_dict_config_raises_error():
    """ValueError is raised if dictionary missing 'regex' key."""
    df = pl.DataFrame({"code": ["A", "B"], "val": [1, 2]}).lazy()

    cfg = DictConfig({"patterns": [{"except_regex": "foo"}]})  # Missing regex
    with pytest.raises(ValueError, match="must have 'regex' key"):
        fn = drop_by_regex_fntr(cfg)
        fn(df).collect()


def test_dict_pattern_without_except_regex():
    """Dict with only 'regex' behaves like a string pattern."""
    df = pl.DataFrame(
        {
            "code": ["A_DROP", "B_KEEP"],
            "val": [1, 2],
        }
    ).lazy()

    patterns = [{"regex": "A_.*"}]  # No except_regex, just match and drop
    cfg = DictConfig({"patterns": patterns})
    fn = drop_by_regex_fntr(cfg)
    result = fn(df).collect()

    assert result.shape[0] == 1
    assert result["val"][0] == 2
