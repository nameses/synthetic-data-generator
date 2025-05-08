"""Models for GAN Neural Network(Generator and Discriminator)"""

from __future__ import annotations

import math
from typing import List
import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
from torch.utils.data import DataLoader, TensorDataset

from data_generation.gan.dataclasses.training import GanConfig
from data_generation.gan.utilities import lin_sn


class MinibatchStd(nn.Module):
    """Append per‑batch standard deviation as an additional feature column."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Append minibatch std to input tensor"""
        # Compute std across batch for every feature, then average → scalar
        std = x.std(0, keepdim=True).mean().expand(x.size(0), 1)
        # Concatenate scalar std to every sample
        return torch.cat([x, std], dim=1)


class _Generator(nn.Module):
    """Conditional generator that supports numeric and categorical outputs."""

    def __init__(
        self,
        cfg: GanConfig,
        num_out: int,
        cat_dims: List[int],
        total_steps: int,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.total_steps = total_steps
        # registering step buffer
        self.step = torch.zeros(1)

        hid = cfg.model.hidden_g
        # Backbone comprises 3 fully‑connected layers + MinibatchStd
        self.backbone = nn.Sequential(
            nn.Linear(cfg.model.latent_dim + sum(cat_dims), hid),
            nn.LeakyReLU(0.2),
            nn.Linear(hid, hid),
            nn.LeakyReLU(0.2),
            MinibatchStd(),
            nn.Linear(hid + 1, hid),
            nn.LeakyReLU(0.2),
        )

        # Output heads
        self.num_head = nn.Linear(hid, num_out) if num_out else None
        self.cat_heads = nn.ModuleList(nn.Linear(hid, k) for k in cat_dims)

        if num_out and self.cfg.loss.regularization.use_bias_correction:
            bias = nn.Parameter(torch.zeros(num_out))
            scale = nn.Parameter(torch.ones(num_out))
            self.bias_scale = (bias, scale)
        else:
            self.bias_scale = None

    def forward(
        self, z: torch.Tensor, cond_vec: torch.Tensor, hard: bool = False
    ) -> torch.Tensor:
        """Forward pass. Generate a synthetic batch

        Args:
            z: Latent noise of shape [batch, latent_dim].
            cond_vec: Optional conditional vector (labels) concatenated to *z*
            hard: If True, return hard categorical samples (one‑hot)
            instead of gumbel‑softmax probabilities.
        """
        h = self.backbone(torch.cat([z, cond_vec], 1))

        out: List[torch.Tensor] = []
        if self.num_head is not None:
            out.append(self.num_head(h))

        # Cosine‑annealed Gumbel temperature
        tau = self.cfg.loss.temperature.tau_end + 0.5 * (
            self.cfg.loss.temperature.tau_start - self.cfg.loss.temperature.tau_end
        ) * (1 + math.cos(math.pi * self.step.item() / self.total_steps))

        for head in self.cat_heads:
            logits = head(h)
            out.append(gumbel_softmax(logits, tau=tau, hard=hard))

        # Increase global step counter
        self.step += 1

        y = torch.cat(out, 1)

        # Affine post‑hoc correction for continuous outputs if enabled
        if self.bias_scale is not None:
            bias, scale = self.bias_scale
            k = self.num_head.out_features  # number of numeric columns
            cont = y[:, :k] * scale + bias
            y = torch.cat([cont, y[:, k:]], 1)
        return y


class _Discriminator(nn.Module):
    """MLP discriminator with spectral normalization on each layer."""

    def __init__(self, loader: DataLoader, cond_dim: int, cfg: GanConfig):
        super().__init__()
        assert isinstance(loader.dataset, TensorDataset), "Expected a TensorDataset"
        inp_dim = loader.dataset.tensors[0].shape[1]

        hid = cfg.model.hidden_d
        self.net = nn.Sequential(
            lin_sn(inp_dim + cond_dim, hid),
            nn.LeakyReLU(0.2),
            lin_sn(hid, hid),
            nn.LeakyReLU(0.2),
            lin_sn(hid, hid // 2),
            nn.LeakyReLU(0.2),
            lin_sn(hid // 2, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Return real/fake logits for batch *(x, c)*."""
        return self.net(torch.cat([x, c], 1)).view(-1)

    @torch.no_grad()
    def feature_map(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Return activations before the last linear layer (feature matching)"""
        return self.net[:-1](torch.cat([x, c], dim=1))
