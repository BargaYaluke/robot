"""Random seed helpers for reproducible INR experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible INR image fitting experiments.

    This controls the most common randomness sources used in this project:
    Python random, NumPy, PyTorch CPU operations, and PyTorch CUDA operations.
    cuDNN is also configured for deterministic behavior when possible.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN makes repeated INR runs easier to compare. It can be
    # slower than benchmark mode, but reproducibility is more useful here.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
