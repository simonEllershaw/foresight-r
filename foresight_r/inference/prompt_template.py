#!/usr/bin/env python
"""Prompt template generation for LLM inference on EHR data."""

import logging
import re
from pathlib import Path

import yaml

# Path to ACES benchmark configs
ACES_CONFIG_DIR = (
    Path(__file__).parent.parent / "aces" / "config" / "mimic4ed-benchmark"
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
    max_length: int,
    max_new_tokens: int,
    task_description: str,
    enable_thinking: bool = False,
    header_delimiter: str = "##",
) -> str:
    """Truncate EHR text at age header boundaries to fit within token limit.

    Args:
        ehr_text: The patient's EHR in markdown format.
        tokenizer: HuggingFace tokenizer for counting tokens.
        max_length: Maximum total sequence length (context window).
        max_new_tokens: Maximum tokens reserved for generation.
        task_description: Task description.
        enable_thinking: Whether to use the thinking prompt template (ignored).

    Returns:
        Truncated EHR text that fits within the token limit.
    """
    # Calculate token overhead from prompt template (excluding EHR)
    overhead_tokens = len(
        ehr_text_to_prompt(
            "",
            task_description,
            tokenizer,
            enable_thinking,
        )
    )
    available_tokens = max_length - max_new_tokens - overhead_tokens

    if available_tokens < 0:
        raise ValueError(
            f"Available tokens ({available_tokens}) < 0. "
            f"max_length={max_length}, max_new_tokens={max_new_tokens}, "
            f"overhead_tokens={overhead_tokens}"
        )

    # Check if truncation is needed
    ehr_tokens = tokenizer.encode(ehr_text, add_special_tokens=False)
    if len(ehr_tokens) <= available_tokens:
        return ehr_text

    # We truncate on Age headers (as keeps a valid sequence)
    # These uniquely start with "##" and are on a new line
    header_ids = tokenizer.encode(header_delimiter, add_special_tokens=False)
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    # We want to find the first cut point `idx` such that `len(ehr_tokens) - idx <= available_tokens`.
    # This corresponds to searching for the header sequence near the start of the token list.
    num_header_tokens = len(header_ids)
    required_len_to_remove = len(ehr_tokens) - available_tokens

    for i in range(required_len_to_remove, len(ehr_tokens) - num_header_tokens + 1):
        # Check for "##" match at the start of a line (safe to do i-1 as required_len_to_remove > 0)
        if (
            ehr_tokens[i : i + num_header_tokens] == header_ids
            and ehr_tokens[i - 1] == newline_id
        ):
            return tokenizer.decode(ehr_tokens[i:], skip_special_tokens=True)

    # Last resort: truncate tokens from the end of the equivalent "start tokens"
    # (i.e. keep last N tokens)
    logger.warning(
        f"Could not find a compatible header to truncate on. "
        f"EHR tokens: {len(ehr_tokens)}, Available tokens: {available_tokens}. "
        "Truncating from end."
    )
    return tokenizer.decode(ehr_tokens[-available_tokens:], skip_special_tokens=True)


def create_prompt(
    ehr_text: str,
    task_description: str,
    tokenizer=None,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    enable_thinking: bool = False,
) -> str:
    """Create a formatted prompt from EHR text and task description.

    Args:
        ehr_text: The patient's electronic health record in markdown format.
        task_description: Description of the prediction task.
        tokenizer: Optional tokenizer for truncation and chat templating.
        max_length: Maximum total sequence length (context window).
        max_new_tokens: Maximum tokens reserved for generation.
        enable_thinking: Whether to enable thinking mode.

    Returns:
        Formatted prompt string ready for LLM input.
    """
    if max_length is not None and max_new_tokens is not None and tokenizer is not None:
        ehr_text = truncate_ehr(
            ehr_text,
            tokenizer,
            max_length,
            max_new_tokens,
            task_description,
            enable_thinking=enable_thinking,
        )

    return ehr_text_to_prompt(
        ehr_text,
        task_description,
        tokenizer,
        enable_thinking,
    )
