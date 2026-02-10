"""Generic shard processor for model inference pipelines."""

import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from foresight_r.models.utils import (
    create_prompt,
    discover_shards,
    load_task_description,
)

logger = logging.getLogger(__name__)


def process_shard(
    dataset_dir: Path,
    task_name: str,
    split: str,
    shard_name: str,
    output_path: Path,
    tokenizer,
    cfg: DictConfig,
    batch_fn: Callable[[list[str]], dict],
    post_process_fn: Callable[[Dataset], Dataset] | None = None,
) -> int:
    """Process a single shard with a custom batch function.

    Args:
        dataset_dir: Root dataset directory.
        task_name: Name of the task.
        split: Split name.
        shard_name: Shard name (without extension).
        output_path: Path to save outputs.
        tokenizer: Loaded tokenizer.
        cfg: Hydra configuration.
        batch_fn: Function(prompts) -> dict of new columns to add.
        post_process_fn: Optional function(dataset) -> dataset for post-processing.

    Returns:
        Number of rows processed.
    """
    # Load dataset shard
    input_path = dataset_dir / task_name / split / f"{shard_name}.parquet"
    dataset = load_dataset("parquet", data_files=str(input_path), split="train")

    if cfg.max_samples:
        dataset = dataset.take(cfg.max_samples)

    if len(dataset) == 0:
        logger.warning(f"Empty shard: {input_path}")
        return 0

    # Load task description
    task_description = load_task_description(task_name)

    # Create prompts
    def _generate_prompt_and_length(text: str) -> dict:
        prompt = create_prompt(
            text,
            task_description=task_description,
            tokenizer=tokenizer,
            **cfg.tokenization,
        )
        return {"prompt": prompt, "prompt_length": len(prompt)}

    dataset = dataset.map(
        _generate_prompt_and_length,
        input_columns="text",
        desc=f"Creating prompts for {task_name} shard {shard_name}",
        keep_in_memory=True,
        num_proc=os.cpu_count(),
    )

    # Sort by prompt length to minimise padding during batch processing
    dataset = dataset.sort("prompt_length", reverse=True)

    # Apply batch function
    dataset = dataset.map(
        batch_fn,
        batched=True,
        batch_size=cfg.batch_size,
        input_columns="prompt",
        desc=f"Processing {task_name} shard {shard_name}",
        load_from_cache_file=False,
        new_fingerprint=f"process_{task_name}_{shard_name}_{split}",
    )

    # Apply post-processing if provided
    if post_process_fn is not None:
        dataset = post_process_fn(dataset)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)

    return len(dataset)


def process_shards_with_model(
    cfg: DictConfig,
    tokenizer,
    batch_fn: Callable[[list[str]], dict],
    post_process_fn: Callable[[Dataset], Dataset] | None = None,
) -> int:
    """Process all shards with a custom batch function.

    Args:
        cfg: Hydra config with dataset_dir, split, output_dir, tasks, etc.
        tokenizer: Loaded tokenizer.
        batch_fn: Function(prompts) -> dict of new columns.
        post_process_fn: Optional function(dataset) -> dataset.

    Returns:
        Total rows processed.
    """
    dataset_dir = Path(cfg.dataset_dir)
    output_dir = Path(cfg.output_dir)
    split = cfg.split

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / split / timestamp

    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Split: {split}")
    logger.info(f"Output directory: {run_output_dir}")

    # Discover shards
    shards = discover_shards(dataset_dir, split)

    # Filter by specific tasks if configured
    if cfg.tasks is not None:
        allowed_tasks = set(cfg.tasks)
        shards = [(t, s) for t, s in shards if t in allowed_tasks]

    logger.info(f"Found {len(shards)} shards to process")

    # Process each shard
    total_rows = 0
    for task_name, shard_name in shards:
        try:
            # Check if task info is available before processing
            try:
                load_task_description(task_name)
            except FileNotFoundError:
                logger.warning(f"Skipping {task_name}: Config not found")
                continue

            logger.info(f"Processing {task_name}/{split}/{shard_name}")
            output_path = run_output_dir / task_name / f"{shard_name}.parquet"

            rows = process_shard(
                dataset_dir,
                task_name,
                split,
                shard_name,
                output_path,
                tokenizer,
                cfg,
                batch_fn,
                post_process_fn,
            )
            total_rows += rows
            logger.info(f"  Wrote {rows} rows to {output_path}")
        except Exception as e:
            logger.error(f"Error processing {task_name}/{shard_name}: {e}")
            continue

    logger.info(f"Complete! Total rows processed: {total_rows}")
    return total_rows
