"""Test script for create_prefix_column stage."""

import polars as pl
from omegaconf import DictConfig
from foresight_r.transforms.create_prefix_column import create_prefix_column_fntr


def test_create_prefix_column():
    """Test that code splitting works correctly."""
    # Create test data
    test_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "time": [1, 2, 3, 4],
            "code": [
                "LAB//glucose//fasting",
                "ADMISSION//emergency",
                "MEDICATION//aspirin//100mg",
                "DIAGNOSIS//diabetes",
            ],
        }
    ).lazy()

    # Get the transformation function
    fn = create_prefix_column_fntr(DictConfig({}))

    # Apply transformation
    result = fn(test_df).collect()

    # Check results
    print("Original and transformed data:")
    print(result.select("code", "category"))

    # Verify
    assert result["category"].to_list() == [
        "LAB",
        "ADMISSION",
        "MEDICATION",
        "DIAGNOSIS",
    ]
    assert result["code"].to_list() == [
        "glucose//fasting",
        "emergency",
        "aspirin//100mg",
        "diabetes",
    ]

    print("\n✅ All tests passed!")


def test_codes_without_delimiter():
    """Test handling of codes without '//' delimiter."""
    test_df = pl.DataFrame(
        {"subject_id": [1, 2], "time": [1, 2], "code": ["glucose", "aspirin"]}
    ).lazy()

    fn = create_prefix_column_fntr(DictConfig({}))
    result = fn(test_df).collect()

    print("\nCodes without delimiter:")
    print(result.select("code", "category"))

    # Current behavior: entire code becomes category, code becomes empty
    assert result["category"].to_list() == ["glucose", "aspirin"]
    assert result["code"].to_list() == ["", ""]

    print("\n✅ Test passed - codes without delimiter handled")


if __name__ == "__main__":
    test_create_prefix_column()
    test_codes_without_delimiter()
