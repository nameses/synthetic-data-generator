"""Module for Generative Adversarial Networks (GANs) in data generation."""

from .utilities import amp_autocast, lin_sn, CriticScheduler
from .models import MinibatchStd, _Generator, _Discriminator
from .pipeline import GAN
from .dataclasses import pipeline as datapipe, training as datatrain

from .dataclasses.pipeline import (
    DataSchema,
    DataPipeline,
    ModelContainer,
    Optimizers,
    TrainingState,
)
from .dataclasses.training import (
    GanConfig,
    TrainingConfig,
    SchedulerConfig,
    LossConfig,
    EmaConfig,
    FeatureMatchingConfig,
    RegularizationConfig,
    DeltaConfig,
    TemperatureConfig,
    LossGPConfig,
    OptimizerConfig,
    ModelConfig,
    CovarianceConfig,
)

__all__ = [
    "GAN",
    "amp_autocast",
    "lin_sn",
    "CriticScheduler",
    "MinibatchStd",
    "_Generator",
    "_Discriminator",
    "DataSchema",
    "DataPipeline",
    "ModelContainer",
    "Optimizers",
    "TrainingState",
    "GanConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "LossConfig",
    "EmaConfig",
    "FeatureMatchingConfig",
    "RegularizationConfig",
    "DeltaConfig",
    "TemperatureConfig",
    "LossGPConfig",
    "OptimizerConfig",
    "ModelConfig",
    "CovarianceConfig",
]
