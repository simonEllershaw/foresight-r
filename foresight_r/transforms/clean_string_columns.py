#!/usr/bin/env python
"""Custom MAP stage that cleans code strings with configurable patterns."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def clean_string_columns_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that cleans string columns using configurable replacement patterns.

    Args:
        stage_cfg: Configuration containing:
            - column: Column name(s) to clean (default: "code")
            - patterns: List of replacement patterns, each with 'pattern' and 'replacement' keys
            - strip_whitespace: Whether to strip leading/trailing whitespace (default: False)
            - to_titlecase: Whether to convert to title case (default: False)
            - to_uppercase: Whether to convert to uppercase (default: False)

    Returns:
        Function that transforms a LazyFrame by cleaning the specified column
    """
    columns = stage_cfg.get("column", "code")
    if isinstance(columns, str):
        columns = [columns]

    patterns = stage_cfg.get("patterns", [])
    strip_whitespace = stage_cfg.get("strip_whitespace", False)
    to_titlecase = stage_cfg.get("to_titlecase", False)
    to_uppercase = stage_cfg.get("to_uppercase", False)

    if not patterns and not strip_whitespace and not to_titlecase and not to_uppercase:
        raise ValueError(
            "At least one pattern, strip_whitespace, to_titlecase, or to_uppercase must be specified"
        )

    def clean_string_columns_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean the specified column using the configured replacement patterns."""
        cleaned_exprs = []

        for col_name in columns:
            # Start with the column
            cleaned_col = pl.col(col_name)

            # Strip leading/trailing whitespace if enabled
            if strip_whitespace:
                cleaned_col = cleaned_col.str.strip_chars()

            # Apply each replacement pattern in sequence
            for pattern_cfg in patterns:
                pattern = pattern_cfg["pattern"]
                replacement = pattern_cfg["replacement"]
                cleaned_col = cleaned_col.str.replace_all(pattern, replacement)

            # Convert to title case if enabled
            if to_titlecase:
                cleaned_col = cleaned_col.str.to_titlecase()

            # Convert to uppercase if enabled
            if to_uppercase:
                cleaned_col = cleaned_col.str.to_uppercase()

            cleaned_exprs.append(cleaned_col.alias(col_name))

        return df.with_columns(cleaned_exprs)

    return clean_string_columns_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the clean_string_columns stage over MEDS data shards."""
    map_over(cfg, compute_fn=clean_string_columns_fntr)


if __name__ == "__main__":
    main()
