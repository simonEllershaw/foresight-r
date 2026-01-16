#!/usr/bin/env python
"""Custom MAP stage that extracts time-of-day information from datetime column."""

from collections.abc import Callable
from datetime import time

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over

# Sentinel value representing end of day
END_OF_DAY = time(23, 59, 59, 999999)


def add_time_column_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that extracts time-of-day from datetime column.

    Extracts the time component from the datetime 'time' column into a new 'time_of_day' column.
    Optionally filters out END_OF_DAY sentinel values (23:59:59.999999) and sets them as null.

    Args:
        stage_cfg: Configuration containing:
            - null_end_of_day: If True, replaces END_OF_DAY times with null (default: True)
            - time_format: Format string for time output. Options: "HH:MM:SS" (default), "HH:MM"

    Returns:
        Function that transforms a LazyFrame by adding time_of_day column

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2],
        ...     "time": [
        ...         datetime(2020, 6, 15, 10, 30, 0),
        ...         datetime(2020, 6, 15, 14, 45, 30),
        ...         datetime(2020, 6, 15, 23, 59, 59, 999999),
        ...         datetime(2021, 3, 10, 8, 15, 0),
        ...     ],
        ...     "code": ["Lab Test", "Medication", "Discharge", "Admission"],
        ... }).lazy()
        >>> fn = add_time_column_fntr(DictConfig({"null_end_of_day": True}))
        >>> result = fn(df).collect()
        >>> result.select("code", "time_of_day")
        shape: (4, 2)
        ┌────────────┬─────────────┐
        │ code       ┆ time_of_day │
        │ ---        ┆ ---         │
        │ str        ┆ time        │
        ╞════════════╪═════════════╡
        │ Lab Test   ┆ 10:30:00    │
        │ Medication ┆ 14:45:30    │
        │ Discharge  ┆ null        │
        │ Admission  ┆ 08:15:00    │
        └────────────┴─────────────┘
    """
    null_end_of_day = stage_cfg.get("null_end_of_day", True)
    time_format = stage_cfg.get("time_format", "HH:MM:SS")

    def add_time_column_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Extract time-of-day from datetime column."""
        # Extract time component
        time_expr = pl.col("time").dt.time()

        # Optionally replace END_OF_DAY with null
        if null_end_of_day:
            time_expr = pl.when(time_expr == END_OF_DAY).then(None).otherwise(time_expr)

        # Format time according to time_format
        if time_format == "HH:MM":
            # Convert to string and truncate to HH:MM
            time_expr = time_expr.cast(pl.String).str.slice(0, 5)
        # else: keep as time object (HH:MM:SS format by default)

        return df.with_columns(time_expr.alias("time_of_day"))

    return add_time_column_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the add_time_column stage over MEDS data shards."""
    map_over(cfg, compute_fn=add_time_column_fntr)


if __name__ == "__main__":
    main()
