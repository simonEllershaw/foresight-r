"""Shared utilities for model training and inference."""

from foresight_r.models.utils.device import get_device
from foresight_r.models.utils.model_loading import load_model_and_tokenizer
from foresight_r.models.utils.dataset import discover_shards
from foresight_r.models.utils.prompt import (
    create_prompt,
    load_task_description,
    ehr_text_to_prompt,
    truncate_ehr,
)

__all__ = [
    "get_device",
    "load_model_and_tokenizer",
    "discover_shards",
    "create_prompt",
    "load_task_description",
    "ehr_text_to_prompt",
    "truncate_ehr",
]
