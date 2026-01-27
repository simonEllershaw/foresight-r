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
            - patterns: List of patterns. Each pattern can be:
                - A string: rows matching this regex are dropped.
                - A dict with 'regex' and optional 'except_regex': rows matching
                  'regex' are dropped UNLESS they also match 'except_regex'.
            - column: Name of the column to filter (default: "code").

    Returns:
        Function that filters the DataFrame.
    """
    patterns = stage_cfg.get("patterns", [])
    column = stage_cfg.get("column", "code")

    if not patterns:
        raise ValueError("Must provide at least one pattern in 'patterns'")

    def filter_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Build a combined "drop condition" from all patterns
        drop_conditions = []

        for pattern in patterns:
            if isinstance(pattern, str):
                # Simple string pattern: drop if matches
                drop_conditions.append(pl.col(column).str.contains(pattern))
            elif hasattr(pattern, "get") or isinstance(pattern, dict):
                # Dictionary pattern with regex and optional except_regex
                regex = pattern.get("regex")
                except_regex = pattern.get("except_regex")

                if not regex:
                    raise ValueError(
                        f"Dictionary pattern must have 'regex' key, got: {pattern}"
                    )

                if except_regex:
                    # Drop if matches regex AND does NOT match except_regex
                    condition = pl.col(column).str.contains(regex) & ~pl.col(
                        column
                    ).str.contains(except_regex)
                else:
                    # No exception, just drop if matches regex
                    condition = pl.col(column).str.contains(regex)

                drop_conditions.append(condition)
            else:
                raise ValueError(f"Unknown pattern type: {type(pattern)}")

        # Combine all conditions with OR: drop if ANY pattern matches
        combined_drop = drop_conditions[0]
        for cond in drop_conditions[1:]:
            combined_drop = combined_drop | cond

        # Keep rows that do NOT match the combined drop condition
        return df.filter(~combined_drop)

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
