#!/usr/bin/env python
"""Custom MAP stage that calculates patient age in years and days."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def calculate_age_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that calculates patient age from birth date.

    Finds the birth date (row with code matching birth_code) for each subject and calculates
    age_year (complete calendar years) and age_day (days since last birthday).
    Properly handles leap years by using calendar-based year calculation rather than
    simple day division. For example, someone born Jan 1, 2000 on Jan 1, 2030 will be
    exactly 30 years 0 days, not 30 years 8 days.

    Args:
        stage_cfg: Configuration containing:
            - birth_code: Code value identifying birth events (default: "Born")

    Returns:
        Function that transforms a LazyFrame by adding age columns

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [
        ...         datetime(2000, 1, 1),
        ...         datetime(2020, 6, 15),
        ...         datetime(2025, 3, 20),
        ...         datetime(1990, 5, 10),
        ...         datetime(2010, 8, 25),
        ...     ],
        ...     "code": ["Born", "Lab Test", "Admission", "Born", "Diagnosis"],
        ...     "prefix": ["Birth", "Lab", "Hospital", "Birth", "Clinical"],
        ... }).lazy()
        >>> fn = calculate_age_fntr(DictConfig({}))
        >>> result = fn(df).collect()
        >>> result.select("subject_id", "code", "age_year", "age_day")
        shape: (5, 4)
        ┌────────────┬───────────┬──────────┬─────────┐
        │ subject_id ┆ code      ┆ age_year ┆ age_day │
        │ ---        ┆ ---       ┆ ---      ┆ ---     │
        │ i64        ┆ str       ┆ i64      ┆ i64     │
        ╞════════════╪═══════════╪══════════╪═════════╡
        │ 1          ┆ Born      ┆ 0        ┆ 0       │
        │ 1          ┆ Lab Test  ┆ 20       ┆166      │
        │ 1          ┆ Admission ┆ 25       ┆ 79      │
        │ 2          ┆ Born      ┆ 0        ┆ 0       │
        │ 2          ┆ Diagnosis ┆ 20       ┆107      │
        └────────────┴───────────┴──────────┴─────────┘
    """
    birth_code = stage_cfg.get("birth_code", "MEDS_BIRTH")

    def calculate_age_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate age using calendar years (age_year) and days since last birthday (age_day)."""
        # Extract birth time for each subject using window function
        # This finds the time value where code matches birth_code for each subject_id
        birth_time = (
            pl.when(pl.col("code") == birth_code)
            .then(pl.col("time"))
            .otherwise(None)
            .max()
            .over("subject_id")
        )

        # Calculate years by comparing dates directly (handles leap years properly)
        # Get year, month, day components
        current_year = pl.col("time").dt.year()
        current_month = pl.col("time").dt.month()
        current_day = pl.col("time").dt.day()

        birth_year = birth_time.dt.year()
        birth_month = birth_time.dt.month()
        birth_day = birth_time.dt.day()

        # Calculate age in years (years difference, minus 1 if birthday hasn't occurred yet this year)
        age_years = (
            current_year
            - birth_year
            - pl.when(
                (current_month < birth_month)
                | ((current_month == birth_month) & (current_day < birth_day))
            )
            .then(1)
            .otherwise(0)
        )

        # Calculate the "last birthday" date by adding age_years to birth date
        # Then calculate days since that date
        last_birthday = birth_time.dt.offset_by(pl.format("{}y", age_years))
        age_days = (pl.col("time") - last_birthday).dt.total_days()

        return df.with_columns(
            [
                age_years.cast(pl.Int64).alias("age_year"),
                age_days.cast(pl.Int64).alias("age_day"),
            ]
        )

    return calculate_age_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the calculate_age stage over MEDS data shards."""
    map_over(cfg, compute_fn=calculate_age_fntr)


if __name__ == "__main__":
    main()
