#!/usr/bin/env python
"""Custom MAP stage that orders rows by prefix priority."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def order_events(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that orders rows by prefix priority.

    Rows are ordered based on which prefix they start with from the priority list.
    Prefixes earlier in the list have higher priority (lower sort value).
    Rows that don't match any prefix are given the lowest priority.

    Args:
        stage_cfg: Configuration containing:
            - prefixes: List of prefix strings in priority order (first = highest priority).
            - column: Name of the column to match against (default: "code").

    Returns:
        Function that orders the DataFrame by prefix priority.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "time": [1, 1, 1, 1],
        ...     "code": ["MEDICATION//aspirin", "LAB//glucose", "ADMISSION//emergency", "OTHER//unknown"]
        ... }).lazy()
        >>> cfg = DictConfig({"prefixes": ["ADMISSION", "LAB", "MEDICATION"]})
        >>> fn = order_events(cfg)
        >>> result = fn(df).collect()
        >>> result.select("code")
        shape: (4, 1)
        ┌─────────────────────┐
        │ code                │
        │ ---                 │
        │ str                 │
        ╞═════════════════════╡
        │ ADMISSION//emergency│
        │ LAB//glucose        │
        │ MEDICATION//aspirin │
        │ OTHER//unknown      │
        └─────────────────────┘
    """
    prefixes = stage_cfg.get("prefixes", [])
    column = stage_cfg.get("column", "code")

    if not prefixes:
        raise ValueError("Must provide at least one prefix in 'prefixes'")

    # Create a mapping from prefix to priority (0 = highest priority)
    # Use len(prefixes) as the default priority for non-matching rows
    default_priority = len(prefixes)

    def order_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Build a list of expressions that return priority for matching prefixes
        # Use coalesce to pick the first non-null (i.e., first matching prefix)
        priority_exprs = [
            pl.when(pl.col(column).str.starts_with(prefix))
            .then(pl.lit(idx))
            .otherwise(pl.lit(None))
            for idx, prefix in enumerate(prefixes)
        ]
        # Add default priority for non-matching rows
        priority_exprs.append(pl.lit(default_priority))

        return (
            df.with_columns(pl.coalesce(priority_exprs).alias("_prefix_priority"))
            .sort(
                [
                    "subject_id",
                    "time",
                    "_prefix_priority",
                    "code",
                    "numeric_value",
                    "text_value",
                ],
                nulls_last=True,
            )
            .drop("_prefix_priority")
        )

    return order_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the order_by_prefix stage over MEDS data shards."""
    map_over(cfg, compute_fn=order_events)


if __name__ == "__main__":
    main()
