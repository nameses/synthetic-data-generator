"""Utilities methods for a project"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set *all* library RNGs for reproducibility.

    Args:
        seed: Arbitrary integer used to seed *random*, *NumPy* and *PyTorch*
            (CPU & CUDA) generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
