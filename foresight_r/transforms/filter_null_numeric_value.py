#!/usr/bin/env python
"""Custom MAP stage that filters rows with null numeric_value for given prefixes.

If `prefixes` is provided in the stage config, rows whose `prefix` is in
that list will be filtered to remove those with null `numeric_value`. Rows
with prefixes not in the list are left unchanged. If `prefixes` is empty
or absent, the transform removes rows where `numeric_value` is null for
all rows.
"""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_null_numeric_value_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that filters out rows with null `numeric_value`.

    Config keys:
        - prefixes: Optional[list[str]] - if provided, only rows whose
          `prefix` is in this list will be subject to null filtering. Other
          rows are left unchanged.

    Examples:
        >>> import polars as pl
        >>> from omegaconf import DictConfig
        >>> df = pl.DataFrame({
        ...     "subject_id": [1,2],
        ...     "time": [1,2],
        ...     "prefix": ["LAB","MEDICATION"],
        ...     "numeric_value": [None, 5.0]
        ... }).lazy()
        >>> fn = filter_null_numeric_value_fntr(DictConfig({"prefixes":["LAB"]}))
        >>> fn(df).collect().shape
        (1, 4)
    """

    prefixes = stage_cfg.get("prefixes", None)

    def filter_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Ensure numeric_value column exists without forcing full schema resolution
        schema_names = df.collect_schema().names()
        if "numeric_value" not in schema_names:
            raise ValueError(
                "numeric_value column required for filter_null_numeric_value transform"
            )

        if prefixes:
            # Keep rows where prefix is not in prefixes OR numeric_value is not null
            keep_expr = (~pl.col("prefix").is_in(prefixes)) | (
                ~pl.col("numeric_value").is_null()
            )
        else:
            # No prefixes provided -> filter out rows with null numeric_value for all rows
            keep_expr = ~pl.col("numeric_value").is_null()

        return df.filter(keep_expr)

    return filter_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the filter_null_numeric_value stage over MEDS data shards."""
    map_over(cfg, compute_fn=filter_null_numeric_value_fntr)


if __name__ == "__main__":
    main()
