"""Validate prediction time consistency across ACES tasks.

All ACES task definitions share the same trigger point (ED arrival), so for each subject_id
and shard, the prediction_time values should be identical across tasks.
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def validate_prediction_times(aces_output_dir: Path) -> bool:
    """Validate that prediction times are consistent across all ACES tasks.

    Args:
        aces_output_dir: Path to the ACES output directory.

    Returns:
        True if validation passes, False otherwise.
    """
    # Discover task directories
    tasks = sorted(
        [
            t
            for t in aces_output_dir.iterdir()
            if t.is_dir() and not t.name.startswith(".")
        ]
    )

    if len(tasks) < 2:
        print(
            f"Found {len(tasks)} task(s). Need at least 2 tasks to validate consistency."
        )
        return True

    print(
        f"Validating prediction times across {len(tasks)} tasks: {[t.name for t in tasks]}"
    )

    # Collect all shard paths from the first task as reference
    reference_task = tasks[0]
    reference_shards: dict[str, Path] = {}

    for split_dir in reference_task.iterdir():
        if split_dir.is_dir() and not split_dir.name.startswith("."):
            for shard_file in split_dir.iterdir():
                if shard_file.suffix == ".parquet":
                    shard_key = f"{split_dir.name}/{shard_file.name}"
                    reference_shards[shard_key] = shard_file

    if not reference_shards:
        print("No parquet files found in reference task.")
        return False

    # Validate each shard across all tasks
    validation_passed = True

    for shard_key, ref_shard_path in sorted(reference_shards.items()):
        # Load reference prediction times
        ref_df = pl.read_parquet(ref_shard_path).select(
            ["subject_id", "prediction_time"]
        )
        ref_df = ref_df.sort(["subject_id", "prediction_time"])

        # Compare with other tasks
        for task in tasks[1:]:
            task_shard_path = task / shard_key
            if not task_shard_path.exists():
                print(f"  MISSING: {task.name}/{shard_key}")
                validation_passed = False
                continue

            task_df = pl.read_parquet(task_shard_path).select(
                ["subject_id", "prediction_time"]
            )
            task_df = task_df.sort(["subject_id", "prediction_time"])

            # Compare DataFrames
            if not ref_df.equals(task_df):
                validation_passed = False
                print(
                    f"  MISMATCH: {shard_key} differs between {reference_task.name} and {task.name}"
                )

                # Show details about the mismatch
                if len(ref_df) != len(task_df):
                    print(f"    Row count: {len(ref_df)} vs {len(task_df)}")
                else:
                    # Find differing rows
                    diff = ref_df.join(
                        task_df,
                        on=["subject_id", "prediction_time"],
                        how="anti",
                    )
                    if len(diff) > 0:
                        print(
                            f"    {len(diff)} prediction times in {reference_task.name} not in {task.name}"
                        )
                        print(diff.head(5))

    if validation_passed:
        print(
            f"✓ Validation passed: All {len(reference_shards)} shards have consistent prediction times"
        )
    else:
        print("✗ Validation failed: Prediction times are inconsistent across tasks")

    return validation_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prediction time consistency across ACES tasks."
    )
    parser.add_argument(
        "aces_output_dir",
        type=Path,
        help="Path to the ACES output directory.",
    )
    args = parser.parse_args()

    if not args.aces_output_dir.exists():
        print(f"Error: Directory does not exist: {args.aces_output_dir}")
        sys.exit(1)

    if not validate_prediction_times(args.aces_output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
