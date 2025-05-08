"""Variational Autoencoder utilities"""

import torch
from torch.nn import functional


def sliced_wasserstein(
    x: torch.Tensor, y: torch.Tensor, n_proj: int = 128
) -> torch.Tensor:
    """Wasserstein distance between two distributions using sliced Wasserstein distance."""
    p = functional.normalize(torch.randn(x.size(1), n_proj, device=x.device), dim=0)
    return ((x @ p).sort(0).values - (y @ p).sort(0).values).abs().mean()


def batch_corr(m: torch.Tensor) -> torch.Tensor:
    """Batch correlation matrix"""
    m = m - m.mean(0, keepdim=True)
    m = m / (m.std(0, keepdim=True) + 1e-6)
    return (m.t() @ m) / m.size(0)
