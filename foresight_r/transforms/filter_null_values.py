#!/usr/bin/env python
"""Custom MAP stage that filters rows where both numeric_value and text_value are null.

If `prefixes` is provided in the stage config, rows whose `prefix` is in
that list will be filtered to remove those where both `numeric_value`
and `text_value` (if column exists) are null.
Rows with prefixes not in the list are left unchanged.
If `prefixes` is empty or absent, the transform removes rows where values
are null for all rows.
"""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_null_values_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that filters out rows with null values.

    Config keys:
        - prefixes: Optional[list[str]] - if provided, only rows whose
          `prefix` is in this list will be subject to null filtering. Other
          rows are left unchanged.

    Examples:
        >>> import polars as pl
        >>> from omegaconf import DictConfig
        >>> df = pl.DataFrame({
        ...     "subject_id": [1,2,3],
        ...     "time": [1,2,3],
        ...     "prefix": ["LAB","LAB","MEDICATION"],
        ...     "numeric_value": [None, 10.0, None],
        ...     "text_value": [None, None, None]
        ... }).lazy()
        >>> fn = filter_null_values_fntr(DictConfig({"prefixes":["LAB"]}))
        >>> # Row 1 (LAB, None, None) filtered.
        >>> # Row 2 (LAB, 10.0, None) kept.
        >>> # Row 3 (MED, None, None) kept (prefix not in filter list).
        >>> fn(df).collect().shape
        (2, 5)
    """

    prefixes = stage_cfg.get("prefixes", None)

    def filter_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        schema_names = df.collect_schema().names()

        # numeric_value is expected in MEDS, but we should be robust
        has_numeric = "numeric_value" in schema_names
        has_text = "text_value" in schema_names

        if not has_numeric and not has_text:
            # Nothing to filter on
            return df

        if has_numeric and has_text:
            is_null_expr = (
                pl.col("numeric_value").is_null() & pl.col("text_value").is_null()
            )
        elif has_numeric:
            is_null_expr = pl.col("numeric_value").is_null()
        else:
            is_null_expr = pl.col("text_value").is_null()

        if prefixes:
            # Keep rows where prefix is not in prefixes OR it's not a null row
            keep_expr = (~pl.col("prefix").is_in(prefixes)) | (~is_null_expr)
        else:
            # No prefixes provided -> filter out null rows for all
            keep_expr = ~is_null_expr

        return df.filter(keep_expr)

    return filter_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the filter_null_values stage over MEDS data shards."""
    map_over(cfg, compute_fn=filter_null_values_fntr)


if __name__ == "__main__":
    main()
