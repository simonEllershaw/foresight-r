#!/usr/bin/env python
"""Run zero-shot LLM inference on natural language EHR datasets.

Usage:
    uv run run-llm-inference \
        dataset_dir=data/mimic-iv-nl-dataset \
        split=tuning \
        output_dir=outputs/zero_shot/inference
"""

import logging
import re
from typing import Any

import hydra
import torch
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from foresight_r.models.utils import (
    load_model_and_tokenizer,
    process_shards_with_model,
)

logger = logging.getLogger(__name__)


def parse_output(text: str) -> dict:
    """Parse LLM output to extract answer, probability, explanation and thinking.

    Expected format:
        <think>...</think>  (optional)
        Explanation: ...
        Answer: Yes/No
        Probability: 0-100%

    Args:
        text: Raw model output text.

    Returns:
        Dictionary with parsed fields.
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


def run_inference(cfg: DictConfig) -> None:
    """Run zero-shot LLM inference on NL datasets.

    Args:
        cfg: Hydra configuration.
    """
    logger.info(f"Model path: {cfg.model.pretrained_model_name_or_path}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_class=AutoModelForCausalLM,
        model_config=cfg.model,
    )

    # Define batch function for generation
    def batch_fn(prompts: list[str]) -> dict:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, **cfg.generation)

        input_ids_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_ids_len:]

        return {
            "model_output": tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        }

    # Define post-processing to parse outputs
    def post_process_fn(dataset: Dataset) -> Dataset:
        return dataset.map(
            parse_output,
            input_columns="model_output",
            desc="Parsing outputs",
            keep_in_memory=True,
        )

    # Process all shards
    process_shards_with_model(cfg, tokenizer, batch_fn, post_process_fn)


@hydra.main(
    version_base=None,
    config_path="../../../config/models/zero_shot",
    config_name="run_inference",
)
def main(cfg: DictConfig) -> None:
    """CLI entry point using Hydra."""
    run_inference(cfg)


if __name__ == "__main__":
    main()
