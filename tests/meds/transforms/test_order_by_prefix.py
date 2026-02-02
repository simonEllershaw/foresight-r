"""Tests for order_by_prefix stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.meds.transforms.order_events import order_events


def test_order_basic():
    """Rows within same subject/time are ordered by prefix priority."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [100, 100, 100],
            "code": [
                "MEDICATION//aspirin",
                "LAB//glucose",
                "ADMISSION//emergency",
            ],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["ADMISSION", "LAB", "MEDICATION"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # Order should be: ADMISSION, LAB, MEDICATION (all at same time)
    assert result["val"].to_list() == [3, 2, 1]
    assert result["code"][0] == "ADMISSION//emergency"
    assert result["code"][1] == "LAB//glucose"
    assert result["code"][2] == "MEDICATION//aspirin"


def test_order_preserves_subject_time_sorting():
    """Subject and time ordering is preserved, prefix ordering is within each time."""
    # Input is pre-sorted by subject_id, time (as in the real pipeline)
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "time": [100, 100, 200, 200],
            "code": [
                "LAB//test",  # subject 1, time 100
                "ADMISSION//er",  # subject 1, time 100
                "LAB//test2",  # subject 2, time 200
                "ADMISSION//er2",  # subject 2, time 200
            ],
            "numeric_value": [None, None, None, None],
            "text_value": [None, None, None, None],
            "val": [1, 2, 3, 4],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["ADMISSION", "LAB"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # Subject 1 comes first, then subject 2 (stable sort preserves this)
    # Within each subject/time: ADMISSION before LAB
    assert result["subject_id"].to_list() == [1, 1, 2, 2]
    assert result["val"].to_list() == [2, 1, 4, 3]  # ADMISSION first in each group


def test_order_across_times():
    """Times are preserved in order, prefix sorting within same time."""
    # Input is pre-sorted by subject_id, time (as in the real pipeline)
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "time": [100, 100, 200, 200],
            "code": [
                "LAB//d",  # time 100
                "ADMISSION//b",  # time 100
                "LAB//a",  # time 200
                "ADMISSION//c",  # time 200
            ],
            "numeric_value": [None, None, None, None],
            "text_value": [None, None, None, None],
            "val": [4, 2, 1, 3],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["ADMISSION", "LAB"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # Time ordering is preserved, prefix priority applied within each time group
    assert result["time"].to_list() == [100, 100, 200, 200]
    assert result["val"].to_list() == [2, 4, 3, 1]


def test_order_unmatched_last():
    """Rows not matching any prefix come last within their time group."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [100, 100, 100],
            "code": [
                "OTHER//unknown",
                "LAB//glucose",
                "RANDOM//thing",
            ],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["LAB"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # LAB comes first, then unmatched rows
    assert result["val"][0] == 2
    assert result["code"][0] == "LAB//glucose"


def test_order_multiple_prefixes():
    """Multiple prefixes are ordered correctly within same time."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "time": [100, 100, 100, 100],
            "code": [
                "Z_LAST//x",
                "B_SECOND//y",
                "A_FIRST//z",
                "C_THIRD//w",
            ],
            "numeric_value": [None, None, None, None],
            "text_value": [None, None, None, None],
            "val": [1, 2, 3, 4],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["A_FIRST", "B_SECOND", "C_THIRD", "Z_LAST"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    assert result["val"].to_list() == [3, 2, 4, 1]


def test_order_custom_column():
    """Order works on a specified column other than 'code'."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [100, 100],
            "code": ["ignore", "ignore"],
            "numeric_value": [None, None],
            "text_value": [None, None],
            "prefix": ["B_PREFIX", "A_PREFIX"],
            "val": [1, 2],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["A_PREFIX", "B_PREFIX"], "column": "prefix"})
    fn = order_events(cfg)
    result = fn(df).collect()

    assert result["val"].to_list() == [2, 1]


def test_order_empty_prefixes_raises_error():
    """ValueError is raised if prefixes list is empty."""
    cfg = DictConfig({"prefixes": []})
    with pytest.raises(ValueError, match="Must provide at least one prefix"):
        order_events(cfg)


def test_order_preserves_columns():
    """All original columns are preserved after ordering."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [100, 100, 100],
            "code": ["C//z", "A//x", "B//y"],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "extra": ["a", "b", "c"],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["A", "B", "C"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    assert result.columns == [
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "text_value",
        "extra",
    ]
    assert result["code"].to_list() == ["A//x", "B//y", "C//z"]
    assert result["extra"].to_list() == ["b", "c", "a"]


def test_order_starts_with_not_contains():
    """Only rows starting with prefix match, not containing."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [100, 100, 100],
            "code": [
                "LAB//test",  # Starts with LAB
                "XLAB//other",  # Contains LAB but doesn't start with it
                "LABORATORY//thing",  # Starts with LAB
            ],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["LAB"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # LAB// and LABORATORY// both start with "LAB" so they come first
    # XLAB// doesn't start with LAB so it comes last
    assert result["val"][-1] == 2  # XLAB is last


def test_order_prefix_priority():
    """Earlier prefixes in list have higher priority (come first)."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [100, 100, 100],
            "code": [
                "LOW_PRIORITY//x",
                "HIGH_PRIORITY//y",
                "MED_PRIORITY//z",
            ],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["HIGH_PRIORITY", "MED_PRIORITY", "LOW_PRIORITY"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    assert result["val"].to_list() == [2, 3, 1]


def test_order_with_all_unmatched():
    """All rows unmatched preserves relative order within same time."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [100, 100, 100],
            "code": ["X//1", "Y//2", "Z//3"],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "val": [1, 2, 3],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["A", "B", "C"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # None match, order should be preserved (stable sort)
    assert result["val"].to_list() == [1, 2, 3]


def test_order_by_numeric_and_text_value():
    """Rows with same prefix/time are ordered by numeric_value then text_value."""
    df = pl.DataFrame(
        {
            "subject_id": [1] * 5,
            "time": [100] * 5,
            "code": ["A//x"] * 5,
            "numeric_value": [2.0, 1.0, 1.0, None, 1.0],
            "text_value": ["b", "b", "a", "c", "b"],
            "val": [1, 2, 3, 4, 5],
        }
    ).lazy()

    # Expected order:
    # 1. numeric_value=1.0, text_value="a" -> val=3
    # 2. numeric_value=1.0, text_value="b" -> val=2
    # 3. numeric_value=1.0, text_value="b" -> val=5 (tie-break stable)
    # 4. numeric_value=2.0, text_value="b" -> val=1
    # 5. numeric_value=None, text_value="c" -> val=4 (nulls last by default)

    cfg = DictConfig({"prefixes": ["A"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    assert result["val"].to_list() == [3, 2, 5, 1, 4]


def test_order_by_code():
    """Rows with same prefix/time are ordered by code before numeric_value."""
    df = pl.DataFrame(
        {
            "subject_id": [1] * 2,
            "time": [100] * 2,
            "code": ["A//z", "A//a"],
            "numeric_value": [1.0, 1.0],
            "text_value": [None, None],
            "val": [1, 2],
        }
    ).lazy()

    cfg = DictConfig({"prefixes": ["A"]})
    fn = order_events(cfg)
    result = fn(df).collect()

    # Code A//a should come before A//z
    assert result["val"].to_list() == [2, 1]
