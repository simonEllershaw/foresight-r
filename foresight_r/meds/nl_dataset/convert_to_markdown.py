#!/usr/bin/env python
"""Utility functions for converting MEDS data to markdown format."""

import polars as pl


def _changed(col: str, fill_null: bool = True) -> pl.Expr:
    """Detect when a column value changes from the previous row."""
    return (pl.col(col) != pl.col(col).shift(1)).fill_null(fill_null)


def patient_to_text(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format.

    Expects columns: time, code, prefix, age_year, age_day, time_of_day.
    Data should be sorted by time.

    Args:
        patient_df: DataFrame containing a single patient's events with required columns.

    Returns:
        Markdown-formatted string representing the patient's EHR.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "time": [datetime(2020, 1, 1, 10, 30), datetime(2020, 1, 1, 14, 0)],
        ...     "code": ["Lab Result 1", "Medication A"],
        ...     "prefix": ["Lab", "Medication"],
        ...     "age_year": [30, 30],
        ...     "age_day": [5, 5],
        ...     "time_of_day": ["10:30", "14:00"],
        ... })
        >>> text = _patient_to_text(df)
        >>> text.startswith("# Patient's Electronic Health Record")
        True
    """
    # Fill null age values with -1 for comparison, then handle in formatting
    df = patient_df.with_columns(
        [
            pl.col("age_year").fill_null(-1).alias("age_year"),
            pl.col("age_day").fill_null(-1).alias("age_day"),
            pl.col("time_of_day").fill_null("Time Unspecified").alias("time_of_day"),
        ]
    )

    age_changed = _changed("age_year") | _changed("age_day")
    time_changed = _changed("time_of_day")
    prefix_changed = _changed("prefix")
    section_changed = age_changed | time_changed

    # Build the formatted output
    df = df.with_columns(
        [
            pl.concat_str(
                [
                    # Age header (## Age X years Y days)
                    pl.when(age_changed)
                    .then(
                        pl.format(
                            "## Age {} years {} days\n",
                            pl.col("age_year"),
                            pl.col("age_day"),
                        )
                    )
                    .otherwise(pl.lit("")),
                    # Time header (### HH:MM or ### Time Unspecified)
                    pl.when(section_changed)
                    .then(pl.format("### {}\n", pl.col("time_of_day")))
                    .otherwise(pl.lit("")),
                    # Category header (#### Category)
                    pl.when(section_changed | prefix_changed)
                    .then(pl.format("#### {}\n", pl.col("prefix")))
                    .otherwise(pl.lit("")),
                    # Code content
                    pl.format("{}\n", pl.col("code")),
                ]
            ).alias("full_line")
        ]
    )

    lines = "".join(df["full_line"].to_list()).rstrip()
    return f"# Patient's Electronic Health Record\n\n{lines}"
