"""Moudle for VAE-based data generation."""

from .utilities import sliced_wasserstein, batch_corr
from .models import Encoder, Decoder
from .pipeline import VAE
from .dataclasses import pipeline as datapipe, training as datatrain

from .dataclasses.pipeline import (
    TrainingDataContainer,
    ModelContainer,
)
from .dataclasses.training import VaeConfig, TrainingConfig, ModelConfig

__all__ = [
    "sliced_wasserstein",
    "batch_corr",
    "Encoder",
    "Decoder",
    "VAE",
    "TrainingDataContainer",
    "ModelContainer",
    "VaeConfig",
    "TrainingConfig",
    "ModelConfig",
]
