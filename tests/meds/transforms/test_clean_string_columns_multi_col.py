import polars as pl
from omegaconf import DictConfig

from foresight_r.meds.transforms.clean_string_columns import clean_string_columns_fntr


def test_clean_string_columns_multiple_columns():
    """Test cleaning multiple columns."""
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [1, 2],
            "code": ["unchanged", "unchanged"],
            "col1": ["value//1", "value//2"],
            "col2": ["test//A", "test//B"],
        }
    ).lazy()

    cfg = DictConfig(
        {
            "column": ["col1", "col2"],
            "patterns": [
                {"pattern": "//", "replacement": " "},
            ],
        }
    )

    fn = clean_string_columns_fntr(cfg)
    result = fn(test_df).collect()

    assert result["code"].to_list() == ["unchanged", "unchanged"]
    assert result["col1"].to_list() == ["value 1", "value 2"]
    assert result["col2"].to_list() == ["test A", "test B"]
