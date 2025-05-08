"""Variational Autoencoder pipeline dataclasses."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from common import DataSchema
from .training import VaeConfig

if TYPE_CHECKING:
    from vae.models import Encoder, Decoder


@dataclass
class DataPipeline:
    """All things related to preparing and loading the real dataset."""

    training_df: np.ndarray
    validation_df: np.ndarray

    @classmethod
    def from_schema(cls, schema: DataSchema, cfg: VaeConfig) -> "DataPipeline":
        """Converts a DataSchema into a DataPipeline."""

        idx = np.arange(len(schema.real_df))
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - cfg.training.valid_split))
        training_df, validation_df = idx[:split], idx[split:]

        return cls(training_df=training_df, validation_df=validation_df)


@dataclass
class ModelContainer:
    """Container for the VAE model and its components."""

    encoder: Encoder
    decoder: Decoder
    optimizer: torch.optim.Optimizer


@dataclass
class TrainingDataContainer:
    """Container for the training data and its properties."""

    tf_num: dict
    tf_dt: dict
    cat_maps: dict
    real_corr: torch.Tensor
    num_dim: int
    cat_dims: list[int]


@dataclass
class TrainingState:
    """All of your metrics, early-stop flags, and best-model snapshots."""

    no_improvements: int = 0
    best: int | float | bool = 1e9
    history_ep: list[int] = field(default_factory=list)
    history_swd: list[float] = field(default_factory=list)
    history_beta: list[float] = field(default_factory=list)
