#!/usr/bin/env python
"""Aggregate markdown parquet shards and save as a HuggingFace dataset.

This is a finalization script that runs after the MEDS markdown pipeline.
"""

import argparse
from pathlib import Path

import polars as pl
from datasets import Dataset
from loguru import logger


def save_as_hf_dataset(input_dir: Path, output_dir: Path) -> None:
    """Aggregate all parquet shards and save as a HuggingFace dataset.

    Args:
        input_dir: Directory containing input parquet files (the convert_to_markdown stage output)
        output_dir: Directory to save the HuggingFace dataset
    """
    logger.info(f"Reading parquet files from: {input_dir}")

    # Find all parquet files recursively (handles train/test splits)
    parquet_files = list(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Read and concatenate all parquet files
    dfs = []
    for pf in parquet_files:
        df = pl.read_parquet(pf)
        dfs.append(df)
        logger.info(f"  Loaded {pf.relative_to(input_dir)}: {len(df)} rows")

    combined_df = pl.concat(dfs)
    logger.info(f"Combined dataset: {len(combined_df)} total rows")

    # Convert to HuggingFace dataset
    # Select only the columns we need for the final dataset
    columns_to_keep = ["subject_id", "text"]
    available_columns = [c for c in columns_to_keep if c in combined_df.columns]

    if "text" not in available_columns:
        raise ValueError(
            f"Expected 'text' column not found. Available columns: {combined_df.columns}"
        )

    output_df = combined_df.select(available_columns)

    # Convert to pandas for HuggingFace compatibility
    pandas_df = output_df.to_pandas()
    hf_dataset = Dataset.from_pandas(pandas_df)

    # Save the dataset
    hf_output_path = output_dir / "hf_dataset"
    logger.info(f"Saving HuggingFace dataset to: {hf_output_path}")
    hf_dataset.save_to_disk(str(hf_output_path))

    # Also save a sample markdown file for inspection
    if len(hf_dataset) > 0:
        sample_path = output_dir / "sample_output.md"
        sample_text = hf_dataset[0]["text"]
        sample_path.write_text(sample_text)
        logger.info(f"Saved sample markdown to: {sample_path}")

    logger.info(f"Successfully saved {len(hf_dataset)} subjects to HuggingFace dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate markdown parquet shards and save as HuggingFace dataset"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the convert_to_markdown stage output",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the HuggingFace dataset",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_as_hf_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
