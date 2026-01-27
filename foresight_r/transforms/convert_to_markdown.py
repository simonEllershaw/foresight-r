#!/usr/bin/env python
"""Custom MAP stage that converts MEDS data to markdown format per subject."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def _changed(col: str, default: bool = True) -> pl.Expr:
    """Detect when a column value changes from the previous row."""
    return (pl.col(col) != pl.col(col).shift(1)).fill_null(default)


def _patient_to_text(patient_df: pl.DataFrame) -> str:
    """Convert a patient's MEDS data to free text EHR format.

    Expects columns: time, code, prefix, age_year, age_day, time_of_day.
    Data should be sorted by time.
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
                    # Age header (## Age X years Y days) - skip if age is -1 (was null)
                    pl.when(age_changed & (pl.col("age_year") >= 0))
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


def convert_to_markdown_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that converts MEDS data to markdown per subject.

    Args:
        stage_cfg: Configuration for the stage (unused for this stage)

    Returns:
        Function that transforms a LazyFrame by grouping by subject and
        producing markdown text for each subject.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [
        ...         datetime(2020, 1, 1, 10, 30),
        ...         datetime(2020, 1, 1, 10, 30),
        ...         datetime(2020, 1, 1, 14, 0),
        ...     ],
        ...     "code": ["Lab Result 1", "Lab Result 2", "Medication A"],
        ...     "prefix": ["Lab", "Lab", "Medication"],
        ...     "age_year": [30, 30, 30],
        ...     "age_day": [5, 5, 5],
        ...     "time_of_day": ["10:30", "10:30", "14:00"],
        ... }).lazy()
        >>> fn = convert_to_markdown_fntr(DictConfig({}))
        >>> result = fn(df).collect()
        >>> print(result["text"][0][:50])
        # Patient's Electronic Health Record
        <BLANKLINE>
        ## Age
    """

    def convert_to_markdown_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Convert MEDS data to markdown grouped by subject_id."""
        # Collect to DataFrame for groupby operation with custom agg
        collected_df = df.collect()

        results = []
        # Use partition_by since data is pre-sorted by subject_id, time
        for group_df in collected_df.partition_by("subject_id", maintain_order=True):
            subject_id = group_df["subject_id"][0]
            markdown_text = _patient_to_text(group_df)
            results.append({"subject_id": subject_id, "text": markdown_text})

        if not results:
            # Return empty frame with expected schema
            return pl.DataFrame({"subject_id": [], "text": []}).lazy()

        return pl.DataFrame(results).lazy()

    return convert_to_markdown_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the convert_to_markdown stage over MEDS data shards."""
    map_over(cfg, compute_fn=convert_to_markdown_fntr)


if __name__ == "__main__":
    main()
