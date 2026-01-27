#!/usr/bin/env python
"""Custom MAP stage that appends numeric and text values to the code column."""

from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def _format_numeric_expr(col: str = "numeric_value", sig_figs: int = 3) -> pl.Expr:
    """Create a Polars expression that formats numeric values.

    Integers if .0, otherwise rounded to specified significant figures.

    Args:
        col: Column name containing numeric values
        sig_figs: Number of significant figures (default: 3)

    Returns:
        Polars expression that produces formatted string or null
    """
    num = pl.col(col)

    # Check if original value is an integer (don't round at all)
    is_original_integer = num == num.floor()

    # Round non-integers to sig figs
    rounded = num.round_sig_figs(sig_figs)

    # Check if rounded result is also an integer (e.g., 123.4 -> 123.0)
    is_rounded_integer = rounded == rounded.floor()

    # Format appropriately
    formatted = (
        pl.when(is_original_integer)
        .then(num.cast(pl.Int64).cast(pl.String))
        .when(is_rounded_integer)
        .then(rounded.cast(pl.Int64).cast(pl.String))
        .otherwise(rounded.cast(pl.String))
    )

    return pl.when(num.is_null()).then(pl.lit(None)).otherwise(formatted)


def append_values_fntr(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that appends numeric and text values to the code column.

    Pure Polars implementation using round_sig_figs for 3sf formatting.

    Args:
        stage_cfg: Configuration containing:
            - column: Column name to modify (default: "code")
            - join_char: Character(s) to join values with (default: "//")

    Returns:
        Function that transforms a LazyFrame by appending values to the code column

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [1, 2, 3],
        ...     "code": ["glucose", "hemoglobin", "diagnosis"],
        ...     "numeric_value": [32.0, 37.62, None],
        ...     "text_value": ["ml", "g/dL", "confirmed"]
        ... }).lazy()
        >>> cfg = DictConfig({"join_char": "//"})
        >>> fn = append_values_fntr(cfg)
        >>> result = fn(df).collect()
        >>> result.select("code")
        shape: (3, 1)
        ┌─────────────────────────┐
        │ code                    │
        │ ---                     │
        │ str                     │
        ╞═════════════════════════╡
        │ glucose//32//ml         │
        │ hemoglobin//37.6//g/dL  │
        │ diagnosis//confirmed    │
        └─────────────────────────┘
    """
    column = stage_cfg.get("column", "code")
    join_char = stage_cfg.get("join_char", "//")
    sig_figs = stage_cfg.get("sig_figs", 3)

    def append_values_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Append numeric and text values to the code column using pure Polars."""
        # Format numeric value as string (or null)
        formatted_numeric = _format_numeric_expr("numeric_value", sig_figs=sig_figs)

        # Build the result by concatenating non-null parts with join_char
        result = (
            pl.when(
                formatted_numeric.is_not_null() & pl.col("text_value").is_not_null()
            )
            .then(
                pl.concat_str(
                    [pl.col(column), formatted_numeric, pl.col("text_value")],
                    separator=join_char,
                )
            )
            .when(formatted_numeric.is_not_null())
            .then(
                pl.concat_str([pl.col(column), formatted_numeric], separator=join_char)
            )
            .when(pl.col("text_value").is_not_null())
            .then(
                pl.concat_str(
                    [pl.col(column), pl.col("text_value")], separator=join_char
                )
            )
            .otherwise(pl.col(column))
        )

        return df.with_columns(result.alias(column))

    return append_values_fn


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """Run the append_values stage over MEDS data shards."""
    map_over(cfg, compute_fn=append_values_fntr)


if __name__ == "__main__":
    main()
