#!/usr/bin/env python
"""Prompt template generation for LLM inference on EHR data."""

import logging
import re
from pathlib import Path

import yaml
from transformers import PreTrainedTokenizer

# Path to ACES benchmark configs
ACES_CONFIG_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "config"
    / "aces"
    / "mimic4ed-benchmark"
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """# Patient's Electronic Health Record
{ehr}

# Task
{task_description}

# Response Format
You must respond with the following format. The final answer is either yes or no.

Explanation: Brief explanation of your answer
Answer: Yes/No
Probability: 0-100
"""

# Regex to match age headers: ## Age X years Y days
AGE_HEADER_PATTERN = re.compile(r"^## Age \d+ years \d+ days$", re.MULTILINE)


def load_task_description(task_name: str, config_dir: Path | None = None) -> str:
    """Load task description from ACES YAML config.

    Args:
        task_name: Name of the task (e.g., 'hospitalisation', 'ed_reattendance')
        config_dir: Optional path to config directory. Defaults to mimic4ed-benchmark.

    Returns:
        Task description string from the YAML metadata.

    Raises:
        FileNotFoundError: If task config file doesn't exist.
        KeyError: If description not found in config.
    """
    if config_dir is None:
        config_dir = ACES_CONFIG_DIR

    config_path = config_dir / f"{task_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config["metadata"]["description"]


def ehr_text_to_prompt(
    ehr_text: str,
    task_description: str,
    tokenizer,
    enable_thinking: bool = False,
) -> str:
    """Convert EHR text to a formatted prompt.

    Args:
        ehr_text: The patient's electronic health record in markdown format.
        task_description: Description of the prediction task.
        tokenizer: Optional tokenizer for truncation and chat templating.
        enable_thinking: Whether to enable thinking mode.

    Returns:
        Formatted prompt string ready for LLM input.
    """
    formatted_content = PROMPT_TEMPLATE.format(
        ehr=ehr_text,
        task_description=task_description,
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": formatted_content}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return formatted_content


def truncate_ehr(
    ehr_text: str,
    tokenizer,
    max_prompt_length: int,
    task_description: str,
    enable_thinking: bool = False,
) -> str:
    """Truncate EHR text at age header boundaries to fit within token limit.

    Args:
        ehr_text: The patient's EHR in markdown format.
        tokenizer: HuggingFace tokenizer for counting tokens.
        max_prompt_length: Maximum total sequence length (context window).
        task_description: Task description.
        enable_thinking: Whether to use the thinking prompt template (ignored).

    Returns:
        Truncated EHR text that fits within the token limit.
    """
    # Calculate lengths based on the actual full prompt
    # 1. Check if the full text already fits
    full_prompt = ehr_text_to_prompt(
        ehr_text,
        task_description,
        tokenizer,
        enable_thinking,
    )
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    if len(full_ids) <= max_prompt_length:
        return ehr_text

    # 2. Derive available tokens correctly
    ehr_tokens = tokenizer.encode(ehr_text, add_special_tokens=False)
    num_tokens_to_drop = len(full_ids) - max_prompt_length
    available_tokens = len(ehr_tokens) - num_tokens_to_drop

    if available_tokens <= 0:
        # This means even dropping ALL EHR tokens isn't enough (overhead > max_len)
        overhead_estimate = len(full_ids) - len(ehr_tokens)
        raise ValueError(
            f"Available tokens ({available_tokens}) <= 0. "
            f"max_prompt_length={max_prompt_length}, "
            f"estimated_overhead={overhead_estimate}, "
            f"ehr_tokens={len(ehr_tokens)}"
        )

    # 3. Truncate using regex
    ehr_suffix = tokenizer.decode(
        ehr_tokens[-available_tokens:], skip_special_tokens=True
    )

    match = AGE_HEADER_PATTERN.search(ehr_suffix)
    if match:
        return ehr_suffix[match.start() :]

    # Last resort: truncate tokens from the end
    logger.warning(
        f"Could not find a compatible header to truncate on. "
        f"EHR tokens: {len(ehr_tokens)}, Available tokens: {available_tokens}. "
        f"Max prompt length: {max_prompt_length}. Truncating from end."
    )
    return ehr_suffix


def create_prompt(
    ehr_text: str,
    task_description: str,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int | None = None,
    enable_thinking: bool = False,
) -> str:
    """Create a formatted prompt from EHR text and task description.

    Args:
        ehr_text: The patient's electronic health record in markdown format.
        task_description: Description of the prediction task.
        tokenizer: Optional tokenizer for truncation and chat templating.
        max_prompt_length: Maximum total sequence length (context window).
        enable_thinking: Whether to enable thinking mode.

    Returns:
        Formatted prompt string ready for LLM input.
    """
    if max_prompt_length is not None:
        ehr_text = truncate_ehr(
            ehr_text,
            tokenizer,
            max_prompt_length,
            task_description,
            enable_thinking=enable_thinking,
        )

    return ehr_text_to_prompt(
        ehr_text,
        task_description,
        tokenizer,
        enable_thinking,
    )
