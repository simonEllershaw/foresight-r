#!/usr/bin/env python
"""Custom MAP stage that filters rows based on regex matches."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_codes_by_regex_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that filters rows based on regex matches.

    Args:
        stage_cfg: Configuration containing:
            - patterns: List[str] of regex patterns to match against.
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
        return df.filter(pl.col(column).str.contains(combined_pattern))

    return filter_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the filter_codes_by_regex stage over MEDS data shards."""
    map_over(cfg, compute_fn=filter_codes_by_regex_fntr)


if __name__ == "__main__":
    main()
