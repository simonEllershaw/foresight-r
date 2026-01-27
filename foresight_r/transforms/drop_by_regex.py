#!/usr/bin/env python
"""Custom MAP stage that drops rows based on regex matches."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def drop_by_regex_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that drops rows based on regex matches.

    Args:
        stage_cfg: Configuration containing:
            - patterns: List[str] of regex patterns to match against for dropping.
            - column: Name of the column to filter (default: "code").

    Returns:
        Function that filters the DataFrame.
    """
    patterns = stage_cfg.get("patterns", [])
    column = stage_cfg.get("column", "code")

    if not patterns:
        raise ValueError("Must provide at least one pattern in 'patterns'")

    # Join patterns with OR operator
    combined_pattern = "|".join(patterns)

    def filter_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Keep rows that do NOT match the pattern
        return df.filter(~pl.col(column).str.contains(combined_pattern))

    return filter_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the drop_by_regex stage over MEDS data shards."""
    map_over(cfg, compute_fn=drop_by_regex_fntr)


if __name__ == "__main__":
    main()
