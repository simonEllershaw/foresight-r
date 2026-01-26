#!/usr/bin/env python
"""Custom MAP stage that cleans code strings with configurable patterns."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def clean_code_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that cleans code strings using configurable replacement patterns.

    Args:
        stage_cfg: Configuration containing:
            - column: Column name to clean (default: "code")
            - patterns: List of replacement patterns, each with 'pattern' and 'replacement' keys
            - strip_whitespace: Whether to strip leading/trailing whitespace (default: False)
            - to_titlecase: Whether to convert to title case (default: False)
            - to_uppercase: Whether to convert to uppercase (default: False)

    Returns:
        Function that transforms a LazyFrame by cleaning the specified column

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [1, 2, 3],
        ...     "code": ["LAB//glucose", "ADMIT//UNK", "MED  //  aspirin"]
        ... }).lazy()
        >>> cfg = DictConfig({
        ...     "column": "code",
        ...     "patterns": [
        ...         {"pattern": "//", "replacement": " "},
        ...         {"pattern": "UNK", "replacement": "?"},
        ...         {"pattern": r"(\\s){2,}", "replacement": " "}
        ...     ]
        ... })
        >>> fn = clean_code_fntr(cfg)
        >>> result = fn(df).collect()
        >>> result.select("code")
        shape: (3, 1)
        ┌──────────────┐
        │ code         │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ LAB glucose  │
        │ ADMIT ?      │
        │ MED aspirin  │
        └──────────────┘
    """
    column = stage_cfg.get("column", "code")
    patterns = stage_cfg.get("patterns", [])
    strip_whitespace = stage_cfg.get("strip_whitespace", False)
    to_titlecase = stage_cfg.get("to_titlecase", False)
    to_uppercase = stage_cfg.get("to_uppercase", False)

    if not patterns and not strip_whitespace and not to_titlecase and not to_uppercase:
        raise ValueError(
            "At least one pattern, strip_whitespace, to_titlecase, or to_uppercase must be specified"
        )

    def clean_code_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean the specified column using the configured replacement patterns."""
        # Start with the column
        cleaned_col = pl.col(column)

        # Apply each replacement pattern in sequence
        for pattern_cfg in patterns:
            pattern = pattern_cfg["pattern"]
            replacement = pattern_cfg["replacement"]
            cleaned_col = cleaned_col.str.replace_all(pattern, replacement)

        # Strip leading/trailing whitespace if enabled
        if strip_whitespace:
            cleaned_col = cleaned_col.str.strip_chars()

        # Convert to title case if enabled
        if to_titlecase:
            cleaned_col = cleaned_col.str.to_titlecase()

        # Convert to uppercase if enabled
        if to_uppercase:
            cleaned_col = cleaned_col.str.to_uppercase()

        return df.with_columns([cleaned_col.alias(column)])

    return clean_code_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the clean_code stage over MEDS data shards."""
    map_over(cfg, compute_fn=clean_code_fntr)


if __name__ == "__main__":
    main()
