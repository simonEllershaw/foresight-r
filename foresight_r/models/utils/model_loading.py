"""Model and tokeniser loading utilities."""

import logging
from typing import Type

import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device for inference.

    Returns:
        torch.device: CUDA if available, then MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_class: Type[PreTrainedModel],
    model_config: DictConfig,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load model and tokeniser from local path.

    Args:
        model_path: Path to local HuggingFace model weights.
        device: Device to load model onto.
        model_class: Model class to use (AutoModelForCausalLM or AutoModel).
        load_in_4bit: Enable 4-bit quantisation (requires CUDA).
        load_in_8bit: Enable 8-bit quantisation (requires CUDA).
        model_dtype: Override data type (float16, bfloat16, float32).

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model from {model_config.pretrained_model_name_or_path}")
    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name_or_path
    )
    tokenizer.padding_side = "left"  # Required for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device()

    # Determine best dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if model_config.dtype in dtype_map:
        resolved_dtype = dtype_map[model_config.dtype]
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        resolved_dtype = torch.bfloat16
    elif device.type in ("cuda", "mps"):
        resolved_dtype = torch.float16
    else:
        resolved_dtype = torch.float32

    # Configure quantisation if requested
    quantization_config = None
    if model_config.load_in_4bit or model_config.load_in_8bit:
        if device.type != "cuda":
            logger.warning(
                "Quantisation requires CUDA. Falling back to standard loading."
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=model_config.load_in_4bit,
                load_in_8bit=model_config.load_in_8bit,
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(
                f"Using {'4-bit' if model_config.load_in_4bit else '8-bit'} quantisation "
                f"with compute dtype {resolved_dtype}"
            )

    model = model_class.from_pretrained(
        model_config.pretrained_model_name_or_path,
        dtype=resolved_dtype,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=quantization_config,
    )

    if model_config.compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer
