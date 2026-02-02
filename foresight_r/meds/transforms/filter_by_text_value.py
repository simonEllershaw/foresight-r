#!/usr/bin/env python
"""Custom MAP stage that filters rows by prefix and text_value.

This transform keeps rows where the code's prefix matches a config key
and the text_value exactly matches one of the allowed values for that prefix.
"""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_by_text_value_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that filters rows by prefix and text_value.

    For prefixes specified in the config:
    - Only keeps rows where text_value matches one of the allowed values

    For prefixes NOT in the config:
    - All rows are kept (no filtering)

    Config keys:
        - prefix_to_text_values: dict[str, list[str]] - mapping of code prefixes to
          allowed text_value values. Only rows with these prefixes are filtered.

    Examples:
        >>> import polars as pl
        >>> from omegaconf import DictConfig
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 4],
        ...     "time": [1, 2, 3, 4],
        ...     "code": ["TRANSFER//ward", "TRANSFER//icu", "TRANSFER//ward", "LAB//glucose"],
        ...     "text_value": ["ADMIT", "DISCHARGE", "TRANSFER", "HIGH"],
        ... }).lazy()
        >>> fn = filter_by_text_value_fntr(DictConfig({
        ...     "prefix_to_text_values": {"TRANSFER": ["ADMIT", "TRANSFER"]}
        ... }))
        >>> # Rows 1, 3 (TRANSFER with ADMIT/TRANSFER) and 4 (LAB - not filtered) kept
        >>> # Row 2 removed (TRANSFER prefix but DISCHARGE not in allowed values)
        >>> fn(df).collect().shape
        (3, 4)
        >>> fn(df).collect()["subject_id"].to_list()
        [1, 3, 4]
    """
    prefix_to_text_values = stage_cfg.get("prefix_to_text_values", {})

    def filter_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        if not prefix_to_text_values:
            # No filter config provided -> return df unchanged
            return df

        # Extract prefix from code (everything before the first //)
        prefix_col = pl.col("code").str.split("//").list.first()

        # Build filter expression:
        # - Keep row if prefix is NOT in config (not filtered)
        # - OR if prefix IS in config AND text_value matches allowed values
        filter_exprs = []
        for prefix, allowed_values in prefix_to_text_values.items():
            allowed_set = (
                set(allowed_values)
                if not isinstance(allowed_values, set)
                else allowed_values
            )
            # For this prefix: keep if text_value is in allowed set
            expr = (prefix_col == prefix) & (
                pl.col("text_value").is_in(list(allowed_set))
            )
            filter_exprs.append(expr)

        # Check if prefix is NOT in any of the configured prefixes (keep those rows)
        all_prefixes = list(prefix_to_text_values.keys())
        prefix_not_in_config = ~prefix_col.is_in(all_prefixes)

        # Combine: keep if (prefix not in config) OR (prefix in config AND text_value matches)
        combined_expr = prefix_not_in_config
        for expr in filter_exprs:
            combined_expr = combined_expr | expr

        return df.filter(combined_expr)

    return filter_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the filter_by_text_value stage over MEDS data shards."""
    map_over(cfg, compute_fn=filter_by_text_value_fntr)


if __name__ == "__main__":
    main()
