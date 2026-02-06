#!/usr/bin/env python
"""Create natural language dataset from MEDS data with ACES labels.

Combines MEDS events with ACES labels, applying transforms and converting to markdown
text truncated at each prediction time. Output is Parquet format compatible with
HuggingFace datasets.

Usage:
    MEDS_DIR=data/mimic-iv-meds/data \
    LABELS_DIR=data/mimic-iv-aces-labels \
    OUTPUT_DIR=data/mimic-iv-nl-dataset \
    uv run create-nl-dataset
"""

import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from foresight_r.meds.transforms.add_time_column import add_time_column_fntr
from foresight_r.meds.transforms.append_values import append_values_fntr
from foresight_r.meds.transforms.calculate_age import calculate_age_fntr
from foresight_r.meds.transforms.clean_string_columns import clean_string_columns_fntr
from foresight_r.meds.nl_dataset.convert_to_markdown import patient_to_text
from foresight_r.meds.transforms.create_prefix_column import create_prefix_column_fntr
from foresight_r.meds.transforms.drop_by_regex import drop_by_regex_fntr
from foresight_r.meds.transforms.replace_code_with_text_value_by_prefix import (
    replace_code_with_text_value_by_prefix_fntr,
)

logger = logging.getLogger(__name__)


def build_transform_pipeline(cfg: DictConfig) -> list:
    """Build the transform pipeline from Hydra config.

    Args:
        cfg: Hydra config containing stage_configs

    Returns:
        List of transform functions to apply in order.
    """
    stage_configs = cfg.stage_configs

    return [
        drop_by_regex_fntr(stage_configs.drop_age_events),
        append_values_fntr(stage_configs.append_values),
        create_prefix_column_fntr(stage_configs.create_prefix_column),
        replace_code_with_text_value_by_prefix_fntr(
            stage_configs.replace_icd_code_with_text_description
        ),
        clean_string_columns_fntr(stage_configs.clean_string_columns),
        calculate_age_fntr(stage_configs.calculate_age),
        add_time_column_fntr(stage_configs.add_time_column),
    ]


def apply_transforms(df: pl.LazyFrame, transforms: list) -> pl.LazyFrame:
    """Apply a list of transforms to a LazyFrame.

    Args:
        df: Input LazyFrame
        transforms: List of transform functions

    Returns:
        Transformed LazyFrame
    """
    for transform in transforms:
        df = transform(df)
    return df


def process_shard(
    meds_path: Path,
    labels_path: Path,
    output_path: Path,
    transforms: list,
) -> int:
    """Process a single shard, creating NL dataset entries.

    Args:
        meds_path: Path to MEDS parquet file
        labels_path: Path to ACES labels parquet file
        output_path: Path to write output parquet file
        transforms: List of transform functions to apply

    Returns:
        Number of rows written
    """
    # Load MEDS data and apply transforms
    meds_df = pl.scan_parquet(meds_path)
    transformed_df = apply_transforms(meds_df, transforms).collect()

    # Load labels
    labels_df = pl.read_parquet(labels_path)

    if len(labels_df) == 0:
        logger.warning(f"No labels found in {labels_path}")
        return 0

    results = []

    # Group transformed MEDS by subject_id for efficient lookup
    subject_groups = {
        subj_id: group_df
        for group_df in transformed_df.partition_by("subject_id", maintain_order=True)
        for subj_id in [group_df["subject_id"][0]]
    }

    # Process each (subject_id, prediction_time) pair
    for row in labels_df.iter_rows(named=True):
        subject_id = row["subject_id"]
        prediction_time = row["prediction_time"]

        # Get subject's data
        subject_df = subject_groups.get(subject_id)
        if subject_df is None:
            logger.warning(f"No MEDS data found for subject_id {subject_id}")
            continue

        # Filter to events up to and including prediction_time
        truncated_df = subject_df.filter(pl.col("time") <= prediction_time)

        if len(truncated_df) == 0:
            logger.warning(
                f"No events before prediction_time {prediction_time} for subject_id {subject_id}"
            )
            continue

        # Convert to markdown text
        text = patient_to_text(truncated_df)

        # Build output row with all label columns
        result_row = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "text": text,
            "boolean_value": row["boolean_value"],
            "integer_value": row["integer_value"],
            "float_value": row["float_value"],
            "categorical_value": row["categorical_value"],
        }
        results.append(result_row)

    if not results:
        logger.warning(f"No results generated for {labels_path}")
        return 0

    # Create output dataframe and write
    output_df = pl.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_path)

    return len(output_df)


def discover_shards(labels_dir: Path) -> list[tuple[str, str, str]]:
    """Discover all task/split/shard combinations from labels directory.

    Args:
        labels_dir: Root directory containing ACES labels

    Returns:
        List of (task_name, split_name, shard_name) tuples
    """
    shards = []
    for task_dir in labels_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name

        for split_dir in task_dir.iterdir():
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name

            for parquet_file in split_dir.glob("*.parquet"):
                shard_name = parquet_file.stem
                shards.append((task_name, split_name, shard_name))

    return shards


def create_nl_dataset(cfg: DictConfig) -> None:
    """Create natural language dataset from MEDS data with ACES labels.

    Args:
        cfg: Hydra configuration
    """
    meds_dir = Path(cfg.meds_dir)
    labels_dir = Path(cfg.labels_dir)
    output_dir = Path(cfg.output_dir)

    # Check if already done
    done_fp = output_dir / ".done"
    if done_fp.is_file() and not cfg.do_overwrite:
        logger.info(
            f"NL dataset creation already complete as {done_fp} exists and "
            f"do_overwrite={cfg.do_overwrite}. Returning."
        )
        return

    logger.info(f"MEDS directory: {meds_dir}")
    logger.info(f"Labels directory: {labels_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Build transform pipeline from config
    transforms = build_transform_pipeline(cfg)

    # Discover all shards to process
    shards = discover_shards(labels_dir)
    logger.info(f"Found {len(shards)} shards to process")

    total_rows = 0
    for task_name, split_name, shard_name in shards:
        # Construct paths
        meds_path = meds_dir / split_name / f"{shard_name}.parquet"
        labels_path = labels_dir / task_name / split_name / f"{shard_name}.parquet"
        output_path = output_dir / task_name / split_name / f"{shard_name}.parquet"

        if not meds_path.exists():
            logger.warning(f"MEDS file not found: {meds_path}")
            continue

        logger.info(f"Processing {task_name}/{split_name}/{shard_name}")
        rows = process_shard(meds_path, labels_path, output_path, transforms)
        total_rows += rows
        logger.info(f"  Wrote {rows} rows to {output_path}")

    # Mark as done
    done_fp.parent.mkdir(parents=True, exist_ok=True)
    done_fp.write_text(f"Total rows: {total_rows}")

    logger.info(f"Complete! Total rows written: {total_rows}")


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../config/meds"),
    config_name="create_nl_dataset",
)
def main(cfg: DictConfig) -> None:
    """CLI entry point using Hydra."""
    create_nl_dataset(cfg)


if __name__ == "__main__":
    main()
