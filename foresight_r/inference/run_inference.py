#!/usr/bin/env python
"""Run LLM inference on natural language EHR datasets.

Usage:
    MODEL_PATH=models/your-model \
    DATASET_DIR=data/mimic-iv-nl-dataset \
    SPLIT=tuning \
    OUTPUT_DIR=outputs/inference \
    uv run run-llm-inference
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import hydra
import polars as pl
import torch
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from foresight_r.inference.prompt_template import create_prompt, load_task_description

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    model_dtype: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokeniser from local path.

    Args:
        model_path: Path to local HuggingFace model weights.
        device: Device to load model onto.
        load_in_4bit: Enable 4-bit quantization (requires CUDA).
        load_in_8bit: Enable 8-bit quantization (requires CUDA).
        model_dtype: Override data type (float16, bfloat16, float32).

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"  # Required for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine best dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if model_dtype in dtype_map:
        resolved_dtype = dtype_map[model_dtype]
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        resolved_dtype = torch.bfloat16
    elif device.type in ("cuda", "mps"):
        resolved_dtype = torch.float16
    else:
        resolved_dtype = torch.float32

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        if device.type != "cuda":
            logger.warning(
                "Quantization requires CUDA. Falling back to standard loading."
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(
                f"Using {'4-bit' if load_in_4bit else '8-bit'} quantization "
                f"with compute dtype {resolved_dtype}"
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=resolved_dtype,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=quantization_config,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer


def parse_output(text: str) -> dict:
    """Parse LLM output to extract answer, probability, explanation and thinking.

    Expected format (Thinking Mode):
        <think>...</think>
        Explanation: ...
        Answer: Yes/No
        Probability: 0-100%

    Expected format (Non-Thinking Mode):
        Explanation: ...
        Answer: Yes/No
        Probability: 0-100%

    Args:
        text: Raw model output text.

    Returns:
        Dictionary with 'parsed_answer', 'parsed_probability', 'parsed_explanation', and 'parsed_thinking'.
    """
    result: dict[str, Any] = {
        "parsed_answer": None,
        "parsed_probability": None,
        "parsed_thinking": None,
        "parsed_explanation": None,
    }

    # Extract thinking (only from <think> tags)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        result["parsed_thinking"] = think_match.group(1).strip()

    # Extract explanation
    expl_match = re.search(
        r"Explanation:\s*(.*?)(?=Answer:|Probability:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if expl_match:
        result["parsed_explanation"] = expl_match.group(1).strip()

    # Extract answer (Yes/No)
    answer_match = re.search(r"Answer:\s*(Yes|No)", text, re.IGNORECASE)
    if answer_match:
        result["parsed_answer"] = answer_match.group(1).lower() == "yes"

    # Extract probability (0-100)
    prob_match = re.search(r"Probability:\s*(\d+)", text)
    if prob_match:
        prob = int(prob_match.group(1))
        result["parsed_probability"] = max(0, min(100, prob))

    return result


def run_batch_inference(
    prompts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: dict,
) -> list[str]:
    """Run inference on a batch of prompts.

    Args:
        prompts: List of prompt strings.
        model: Loaded model.
        tokenizer: Loaded tokeniser.
        generation_config: Dictionary of generation parameters.

    Returns:
        List of generated text outputs.
    """
    # Tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(
        model.device
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    # Decode outputs
    input_ids_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, input_ids_len:]

    # Use batch_decode for better performance
    generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return [text.strip() for text in generated_texts]


def discover_shards(dataset_dir: Path, split: str) -> list[tuple[str, str]]:
    """Discover all shards for a given split across all tasks.

    Args:
        dataset_dir: Root directory of NL dataset.
        split: Split name (train, tuning, held_out).

    Returns:
        List of (task_name, shard_name) tuples.
    """
    shards = []
    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue

        split_dir = task_dir / split
        if not split_dir.exists():
            continue

        for shard_file in split_dir.glob("*.parquet"):
            shards.append((task_dir.name, shard_file.stem))

    return sorted(shards)


def process_shard(
    dataset_dir: Path,
    task_name: str,
    split: str,
    shard_name: str,
    output_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: DictConfig,
) -> int:
    """Process a single shard of the dataset.

    Args:
        dataset_dir: Root (absolute) dataset directory.
        task_name: Name of the task.
        split: Split name.
        shard_name: Shard name (without extension).
        output_path: Path to save outputs.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        cfg: Hydra configuration.

    Returns:
        Number of rows processed.
    """
    # Load dataset shard
    input_path = dataset_dir / task_name / split / f"{shard_name}.parquet"
    df = pl.read_parquet(input_path, n_rows=cfg.max_samples)

    if len(df) == 0:
        logger.warning(f"Empty shard: {input_path}")
        return 0

    # Load task description
    task_description = load_task_description(task_name)

    # Generate prompts with truncation
    prompts = [
        create_prompt(
            text,
            task_description,
            tokenizer=tokenizer,
            max_new_tokens=cfg.generation.max_new_tokens,
            max_length=cfg.tokenization.max_length,
            enable_thinking=cfg.tokenization.enable_thinking,
        )
        for text in tqdm(df["text"].to_list())
    ]

    # Run inference in batches
    all_outputs = []
    batch_size = cfg.batch_size

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(prompts), batch_size), total=num_batches, desc="Batches"
    ):
        batch_prompts = prompts[i : i + batch_size]

        batch_outputs = run_batch_inference(
            batch_prompts,
            model,
            tokenizer,
            generation_config=cfg.generation,
        )
        all_outputs.extend(batch_outputs)

    # Parse outputs
    parsed_results = [parse_output(output) for output in all_outputs]

    # Create results DataFrame
    results_df = pl.DataFrame(parsed_results)

    # Add input prompt and raw output
    results_df = results_df.with_columns(
        [pl.Series("full_prompt", prompts), pl.Series("raw_output", all_outputs)]
    )

    # Combine with original dataframe
    result_df = pl.concat([df, results_df], how="horizontal")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)

    return len(result_df)


def run_inference(cfg: DictConfig) -> None:
    """Run LLM inference on NL datasets.

    Args:
        cfg: Hydra configuration.
    """
    dataset_dir = Path(cfg.dataset_dir)
    output_dir = Path(cfg.output_dir)
    split = cfg.split

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / split / timestamp

    logger.info(f"Model path: {cfg.model_path}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Split: {split}")
    logger.info(f"Output directory: {run_output_dir}")

    # Load model
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_path,
        device,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        model_dtype=cfg.model_dtype,
    )

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
                model,
                tokenizer,
                cfg,
            )
            total_rows += rows
            logger.info(f"  Wrote {rows} rows to {output_path}")
        except Exception as e:
            logger.error(f"Error processing {task_name}/{shard_name}: {e}")
            continue

    logger.info(f"Complete! Total rows processed: {total_rows}")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="run_inference",
)
def main(cfg: DictConfig) -> None:
    """CLI entry point using Hydra."""
    run_inference(cfg)


if __name__ == "__main__":
    main()
