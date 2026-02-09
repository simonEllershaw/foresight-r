"""Device detection utilities."""

import torch


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
