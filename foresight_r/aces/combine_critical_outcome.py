"""Combine critical outcome tasks using OR logic.
ACES config not currently expressive enough to combine tasks with different prediction times into a single label.
(At least I can't find a way to do it!)

Creates a combined 'critical_outcome' task from:
- critical_outcome_icu_12h
- critical_outcome_in_hospital_mortality

The combined boolean_value is True if either source task is True.
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def combine_critical_outcomes(
    icu_dir: Path, mortality_dir: Path, output_dir: Path
) -> None:
    """Combine critical outcome tasks using OR logic on boolean_value.

    Args:
        icu_dir: Path to the ICU 12h task directory.
        mortality_dir: Path to the in-hospital mortality task directory.
        output_dir: Path to the combined output directory.
    """

    if not icu_dir.exists():
        print(f"Error: ICU directory not found: {icu_dir}")
        sys.exit(1)
    if not mortality_dir.exists():
        print(f"Error: Mortality directory not found: {mortality_dir}")
        sys.exit(1)

    # Discover all shards from ICU task
    shards: list[tuple[Path, Path, Path]] = []
    for split_dir in icu_dir.iterdir():
        if split_dir.is_dir() and not split_dir.name.startswith("."):
            for shard_file in split_dir.iterdir():
                if shard_file.suffix == ".parquet":
                    mortality_shard = mortality_dir / split_dir.name / shard_file.name
                    output_shard = output_dir / split_dir.name / shard_file.name
                    if mortality_shard.exists():
                        shards.append((shard_file, mortality_shard, output_shard))
                    else:
                        print(f"Warning: Missing mortality shard: {mortality_shard}")

    if not shards:
        print("No matching shards found.")
        sys.exit(1)

    print(f"Combining {len(shards)} shards...")

    for icu_path, mortality_path, output_path in shards:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read both parquet files with lazy evaluation
        icu_df = pl.scan_parquet(icu_path)
        mortality_df = pl.scan_parquet(mortality_path)

        # Join on subject_id and prediction_time, then OR the boolean values
        combined = (
            icu_df.join(
                mortality_df.select(
                    ["subject_id", "prediction_time", "boolean_value"]
                ).rename({"boolean_value": "mortality_value"}),
                on=["subject_id", "prediction_time"],
                how="inner",
            )
            .with_columns(
                (pl.col("boolean_value") | pl.col("mortality_value")).alias(
                    "boolean_value"
                )
            )
            .drop("mortality_value")
        )

        # Collect and write
        combined.collect().write_parquet(output_path)

    print(f"✓ Combined critical outcome saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine critical outcome ACES tasks using OR logic."
    )
    parser.add_argument(
        "icu_dir",
        type=Path,
        help="Path to the ICU 12h task directory.",
    )
    parser.add_argument(
        "mortality_dir",
        type=Path,
        help="Path to the in-hospital mortality task directory.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the combined output directory.",
    )
    args = parser.parse_args()

    if not args.icu_dir.exists():
        print(f"Error: ICU directory does not exist: {args.icu_dir}")
        sys.exit(1)
    if not args.mortality_dir.exists():
        print(f"Error: Mortality directory does not exist: {args.mortality_dir}")
        sys.exit(1)

    combine_critical_outcomes(args.icu_dir, args.mortality_dir, args.output_dir)


if __name__ == "__main__":
    main()
