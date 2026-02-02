#!/usr/bin/env python
"""Custom MAP stage that adds a formatted time-of-day column."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over

END_OF_DAY = pl.time(23, 59, 59, 999999)


def add_time_column_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that adds a formatted time-of-day column.

    Extracts the time component from the datetime 'time' column and formats it
    as a string. Optionally replaces end-of-day times (23:59:59.999999) with a
    configurable fill value.

    Args:
        stage_cfg: Configuration containing:
            - time_format: strftime format string (default: "%H:%M:%S")
            - replace_end_of_day: Whether to replace 23:59:59.999999 (default: False)
            - end_of_day_fill_value: Replacement value for end-of-day (default: None)

    Returns:
        Function that transforms a LazyFrame by adding a time_of_day column

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [
        ...         datetime(2020, 1, 1, 10, 30, 45),
        ...         datetime(2020, 1, 1, 23, 59, 59, 999999),
        ...         datetime(2020, 1, 2, 14, 15, 0),
        ...     ],
        ...     "code": ["Lab", "Admission", "Discharge"],
        ... }).lazy()
        >>> cfg = DictConfig({"time_format": "%H:%M", "replace_end_of_day": True, "end_of_day_fill_value": "Unknown"})
        >>> fn = add_time_column_fntr(cfg)
        >>> result = fn(df).collect()
        >>> result.select("code", "time_of_day")
        shape: (3, 2)
        ┌───────────┬─────────────┐
        │ code      ┆ time_of_day │
        │ ---       ┆ ---         │
        │ str       ┆ str         │
        ╞═══════════╪═════════════╡
        │ Lab       ┆ 10:30       │
        │ Admission ┆ Unknown     │
        │ Discharge ┆ 14:15       │
        └───────────┴─────────────┘
    """
    time_format = stage_cfg.get("time_format", "%H:%M:%S")
    replace_end_of_day = stage_cfg.get("replace_end_of_day", False)
    end_of_day_fill_value = stage_cfg.get("end_of_day_fill_value", None)

    def add_time_column_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Add formatted time-of-day column to the dataframe."""
        time_col = pl.col("time").dt.time()
        # Format time by combining with epoch date, then strftime
        formatted_time = pl.col("time").dt.time().dt.to_string(time_format)

        if replace_end_of_day:
            time_of_day_expr = (
                pl.when(time_col == END_OF_DAY)
                .then(pl.lit(end_of_day_fill_value))
                .otherwise(formatted_time)
                .alias("time_of_day")
            )
        else:
            time_of_day_expr = formatted_time.alias("time_of_day")

        return df.with_columns(time_of_day_expr)

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
