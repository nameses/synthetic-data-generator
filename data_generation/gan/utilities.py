"""Utilities methods for GAN training."""

import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm


def amp_autocast():
    """Context manager for automatic mixed precision (AMP) in PyTorch."""
    return torch.amp.autocast("cuda", enabled=True)


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


def lin_sn(inp: int, out: int) -> nn.Linear:
    """Return a spectrally‑normalized ``nn.Linear`` layer."""
    return spectral_norm(nn.Linear(inp, out))


class CriticScheduler:
    """
    Holds the smoothed W‐distance and current n_critic, and
    updates both via a three‐level hysteresis rule.
    """

    def __init__(
        self,
        initial_n_critic: int,
        alpha: float = 0.9,
        lower_threshold: float = 0.015,
        upper_threshold: float = 0.04,
    ):
        self.initial_n_critic = initial_n_critic
        self.alpha = alpha
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        self.w_smooth = None

    def update_smooth_wassertain_distance(self, mean_w: float) -> None:
        """Update the smoothed W‐distance and n_critic."""
        # update exponential‐smoothed W
        if self.w_smooth is None:
            self.w_smooth = mean_w
        else:
            self.w_smooth = self.alpha * self.w_smooth + (1 - self.alpha) * mean_w

    def get_updated_n_critic(self, current_n_critic: int) -> int:
        """Get the updated n_critic."""
        if (
            self.w_smooth < self.lower_threshold
            and current_n_critic < self.initial_n_critic
        ):
            current_n_critic += 1
        elif self.w_smooth > self.upper_threshold and current_n_critic > 1:
            current_n_critic -= 1

        return current_n_critic
