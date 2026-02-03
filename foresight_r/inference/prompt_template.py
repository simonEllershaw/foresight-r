#!/usr/bin/env python
"""Prompt template generation for LLM inference on EHR data."""

import re
from pathlib import Path

import yaml

# Path to ACES benchmark configs
ACES_CONFIG_DIR = (
    Path(__file__).parent.parent / "aces" / "config" / "mimic4ed-benchmark"
)

PROMPT_TEMPLATE = """{ehr}

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


def truncate_ehr(
    ehr_text: str,
    tokenizer,
    max_length: int,
    max_new_tokens: int,
    task_description: str,
    enable_thinking: bool = False,
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
    formatted_prompt = PROMPT_TEMPLATE.format(ehr="", task_description=task_description)

    # If chat template is available, approximate overhead using apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": formatted_prompt}]
        overhead_tokens = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    else:
        overhead_tokens = len(
            tokenizer.encode(formatted_prompt, add_special_tokens=False)
        )

    available_tokens = max_length - max_new_tokens - overhead_tokens

    # Check if truncation is needed
    ehr_tokens = tokenizer.encode(ehr_text, add_special_tokens=False)
    if len(ehr_tokens) <= available_tokens:
        return ehr_text

    # Find all age header positions
    matches = list(AGE_HEADER_PATTERN.finditer(ehr_text))
    if len(matches) <= 1:
        # No age headers to split on, fall back to token truncation
        truncated_tokens = ehr_tokens[-available_tokens:]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    # Get section boundaries (start positions of each age section)
    section_starts = [m.start() for m in matches]

    # Try removing sections from the beginning until we fit
    for i in range(1, len(section_starts)):
        truncated_text = ehr_text[section_starts[i] :]
        truncated_tokens = tokenizer.encode(truncated_text, add_special_tokens=False)
        if len(truncated_tokens) <= available_tokens:
            return truncated_text

    # If still too long, use the last section only
    last_section = ehr_text[section_starts[-1] :]
    last_tokens = tokenizer.encode(last_section, add_special_tokens=False)
    if len(last_tokens) <= available_tokens:
        return last_section

    # Last resort: truncate tokens from the last section
    return tokenizer.decode(last_tokens[-available_tokens:], skip_special_tokens=True)


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
