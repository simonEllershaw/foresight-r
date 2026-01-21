#!/usr/bin/env python
"""Custom MAP stage that creates a prefix column from composite codes."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def create_prefix_column_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that creates a prefix column from composite codes.

    Args:
        stage_cfg: Configuration for the stage (unused for this stage)
        code_metadata: Code metadata (unused for this stage)
        code_modifiers: Code modifier columns (unused for this stage)

    Returns:
        Function that transforms a LazyFrame by splitting codes

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [1, 2, 3],
        ...     "code": ["LAB//glucose", "ADMISSION//emergency", "MEDICATION//aspirin"]
        ... }).lazy()
        >>> fn = create_prefix_column_fntr(DictConfig({}))
        >>> result = fn(df).collect()
        >>> result.select("code", "category")
        shape: (3, 2)
        ┌───────────┬────────────┐
        │ code      ┆ category   │
        │ ---       ┆ ---        │
        │ str       ┆ str        │
        ╞═══════════╪════════════╡
        │ glucose   ┆ LAB        │
        │ emergency ┆ ADMISSION  │
        │ aspirin   ┆ MEDICATION │
        └───────────┴────────────┘
    """

    def create_prefix_column_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Create prefix column by splitting code on first '//' delimiter."""
        # Split code on first "//"
        code_parts = pl.col("code").str.split("//")

        return df.with_columns(
            [
                # First element becomes prefix
                code_parts.list.first().alias("category"),
                # Join remaining elements back with "//" as the new code
                code_parts.list.slice(1).list.join("//").alias("code"),
            ]
        )

    return create_prefix_column_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the create_prefix_column stage over MEDS data shards."""
    map_over(cfg, compute_fn=create_prefix_column_fntr)


if __name__ == "__main__":
    main()
