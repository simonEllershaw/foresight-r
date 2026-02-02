#!/usr/bin/env python
"""Custom MAP stage that replaces code with text_value based on prefix.

This transform replaces the 'code' column value with the 'text_value' column value
for rows where the 'code' prefix matches one of the configured prefixes.
"""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def replace_code_with_text_value_by_prefix_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that replaces code with text_value by prefix.

    Config keys:
        - prefixes: list[str] - list of code prefixes to match.
          Rows with these prefixes will have their 'code' column replaced
          by the value in 'text_value'.

    Examples:
        >>> import polars as pl
        >>> from omegaconf import DictConfig
        >>> df = pl.DataFrame({
        ...     "code": ["TRANSFER//ward", "LAB//glucose", "TRANSFER//icu"],
        ...     "text_value": ["ADMIT", "HIGH", "DISCHARGE"],
        ... }).lazy()
        >>> fn = replace_code_with_text_value_by_prefix_fntr(DictConfig({
        ...     "prefixes": ["TRANSFER"]
        ... }))
        >>> fn(df).collect().select("code").to_series().to_list()
        ['ADMIT', 'LAB//glucose', 'DISCHARGE']
    """
    prefixes = stage_cfg.get("prefixes", [])

    def replace_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        if not prefixes:
            return df

        # condition: prefix column is in the configured prefixes
        condition = pl.col("prefix").is_in(prefixes)

        return df.with_columns(
            pl.when(condition)
            .then(pl.col("text_value"))
            .otherwise(pl.col("code"))
            .alias("code")
        )

    return replace_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the replace_code_with_text_value_by_prefix stage over MEDS data shards."""
    map_over(cfg, compute_fn=replace_code_with_text_value_by_prefix_fntr)


if __name__ == "__main__":
    main()
