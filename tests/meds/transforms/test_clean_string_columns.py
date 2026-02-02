"""Tests for clean_string_columns stage."""

import polars as pl
import pytest
from omegaconf import DictConfig

from foresight_r.meds.transforms.clean_string_columns import clean_string_columns_fntr


@pytest.fixture
def sample_data():
    """Sample data with codes that need cleaning."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4],
            "time": [1, 2, 3, 4, 5],
            "code": [
                "LAB//glucose",
                "ADMISSION//UNK",
                "MEDICATION  //  aspirin",
                "DIAGNOSIS//___",
                "LAB//glucose//UNK",
            ],
        }
    ).lazy()


@pytest.fixture
def default_config():
    """Default configuration matching extract_MIMIC.yaml."""
    return DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "//", "replacement": " "},
                {"pattern": "___|UNK", "replacement": "?"},
                {"pattern": r"(\s){2,}", "replacement": " "},
            ],
        }
    )


def test_clean_string_columns_default_patterns(sample_data, default_config):
    """Test that default patterns clean codes correctly."""
    fn = clean_string_columns_fntr(default_config)
    result = fn(sample_data).collect()

    assert result["code"].to_list() == [
        "LAB glucose",
        "ADMISSION ?",
        "MEDICATION aspirin",
        "DIAGNOSIS ?",
        "LAB glucose ?",
    ]


def test_clean_string_columns_double_slash_replacement(default_config):
    """Test that '//' is replaced with space."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["A//B", "C//D//E", "F//G//H//I"],
        }
    ).lazy()

    fn = clean_string_columns_fntr(default_config)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["A B", "C D E", "F G H I"]


def test_clean_string_columns_unk_replacement(default_config):
    """Test that UNK and ___ are replaced with ?."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["UNK", "___", "value//UNK//___"],
        }
    ).lazy()

    fn = clean_string_columns_fntr(default_config)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["?", "?", "value ? ?"]


def test_clean_string_columns_multiple_spaces(default_config):
    """Test that multiple spaces are collapsed to single space."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["A  B", "C   D    E", "F     G"],
        }
    ).lazy()

    fn = clean_string_columns_fntr(default_config)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["A B", "C D E", "F G"]


def test_clean_string_columns_custom_column():
    """Test cleaning a different column."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["unchanged", "unchanged"],
            "description": ["value//UNK", "test  //  data"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "description",
            "patterns": [
                {"pattern": "//", "replacement": " "},
                {"pattern": "UNK", "replacement": "?"},
                {"pattern": r"(\s){2,}", "replacement": " "},
            ],
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["unchanged", "unchanged"]
    assert result["description"].to_list() == ["value ?", "test data"]


def test_clean_string_columns_custom_patterns():
    """Test with custom replacement patterns."""
    test_df = pl.DataFrame(
        {"subject_id": [1, 2], "time": [1, 2], "code": ["UPPER_CASE", "lower-case"]}
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "_", "replacement": " "},
                {"pattern": "-", "replacement": "_"},
            ],
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["UPPER CASE", "lower_case"]


def test_clean_string_columns_preserves_other_columns(sample_data, default_config):
    """Test that transformation preserves all other columns."""
    fn = clean_string_columns_fntr(default_config)
    result = fn(sample_data).collect()

    assert "subject_id" in result.columns
    assert "time" in result.columns
    assert result["subject_id"].to_list() == [1, 1, 2, 3, 4]
    assert result["time"].to_list() == [1, 2, 3, 4, 5]


def test_clean_string_columns_empty_dataframe(default_config):
    """Test that transformation handles empty DataFrames."""
    test_df = pl.DataFrame(
        {
            "subject_id": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Int64),
            "code": pl.Series([], dtype=pl.String),
        }
    ).lazy()

    fn = clean_string_columns_fntr(default_config)
    result = fn(test_df).collect()

    assert len(result) == 0
    assert "code" in result.columns


def test_clean_string_columns_pattern_order_matters():
    """Test that patterns are applied in sequence."""
    test_df = pl.DataFrame({"subject_id": [1], "time": [1], "code": ["A//B//C"]}).lazy()

    # First replace '//' with space, then collapse spaces
    cfg1 = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "//", "replacement": " "},
                {"pattern": r"(\s){2,}", "replacement": " "},
            ],
        }
    )

    fn1 = clean_string_columns_fntr(cfg1)
    result1 = fn1(test_df).collect()

    # Just replace '//' with multiple spaces
    cfg2 = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "//", "replacement": "  "},
            ],
        }
    )

    fn2 = clean_string_columns_fntr(cfg2)
    result2 = fn2(test_df).collect()

    assert result1["code"][0] == "A B C"
    assert result2["code"][0] == "A  B  C"


def test_clean_string_columns_no_patterns_raises_error():
    """Test that missing patterns and strip_whitespace raises ValueError."""
    cfg = DictConfig({"column": "code", "patterns": []})

    with pytest.raises(
        ValueError,
        match="At least one pattern, strip_whitespace, to_titlecase, or to_uppercase",
    ):
        clean_string_columns_fntr(cfg)


def test_clean_string_columns_null_values(default_config):
    """Test handling of null values."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["LAB//glucose", None, "ADMISSION//UNK"],
        }
    ).lazy()

    fn = clean_string_columns_fntr(default_config)
    result = fn(test_df).collect()

    assert result["code"][0] == "LAB glucose"
    assert result["code"][1] is None
    assert result["code"][2] == "ADMISSION ?"


def test_clean_string_columns_regex_patterns():
    """Test that regex patterns work correctly."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["test123", "value456xyz", "abc789def"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": r"\d+", "replacement": "#"},
            ],
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["test#", "value#xyz", "abc#def"]


def test_clean_string_columns_strip_whitespace():
    """Test that strip_whitespace removes leading/trailing whitespace."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": [" leading", "trailing ", " both ", "none"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [],
            "strip_whitespace": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["leading", "trailing", "both", "none"]


def test_clean_string_columns_strip_whitespace_with_patterns():
    """Test strip_whitespace combined with patterns."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["Hospital Transfer// Emergency Department", " LAB//glucose "],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "//", "replacement": " "},
                {"pattern": r"(\s){2,}", "replacement": " "},
            ],
            "strip_whitespace": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == [
        "Hospital Transfer Emergency Department",
        "LAB glucose",
    ]


def test_clean_string_columns_strip_whitespace_false_preserves_spaces():
    """Test that strip_whitespace=False preserves leading/trailing whitespace."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [1],
            "code": [" spaced "],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [{"pattern": "x", "replacement": "y"}],  # Dummy pattern
            "strip_whitespace": False,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"][0] == " spaced "


def test_clean_string_columns_to_titlecase():
    """Test that to_titlecase converts strings to title case."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": ["UPPER CASE", "lower case", "mixed CASE string", "Already Title"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [],
            "to_titlecase": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == [
        "Upper Case",
        "Lower Case",
        "Mixed Case String",
        "Already Title",
    ]


def test_clean_string_columns_to_titlecase_with_patterns():
    """Test to_titlecase combined with patterns."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["LAB//GLUCOSE", "ADMISSION//EMERGENCY ROOM"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "//", "replacement": " "},
            ],
            "to_titlecase": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["Lab Glucose", "Admission Emergency Room"]


def test_clean_string_columns_to_titlecase_with_null():
    """Test that to_titlecase handles null values."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["UPPER", None, "lower"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [],
            "to_titlecase": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"][0] == "Upper"
    assert result["code"][1] is None
    assert result["code"][2] == "Lower"


def test_clean_string_columns_to_uppercase():
    """Test that to_uppercase converts strings to uppercase."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "time": [1, 2, 3, 4],
            "code": ["lower case", "UPPER CASE", "Mixed Case String", "already upper"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [],
            "to_uppercase": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == [
        "LOWER CASE",
        "UPPER CASE",
        "MIXED CASE STRING",
        "ALREADY UPPER",
    ]


def test_clean_string_columns_to_uppercase_with_patterns():
    """Test to_uppercase combined with patterns."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["lab//glucose", "admission//emergency room"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [
                {"pattern": "//", "replacement": " "},
            ],
            "to_uppercase": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["LAB GLUCOSE", "ADMISSION EMERGENCY ROOM"]


def test_clean_string_columns_to_uppercase_with_null():
    """Test that to_uppercase handles null values."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [1, 2, 3],
            "code": ["lower", None, "UPPER"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": "code",
            "patterns": [],
            "to_uppercase": True,
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"][0] == "LOWER"
    assert result["code"][1] is None
    assert result["code"][2] == "UPPER"
