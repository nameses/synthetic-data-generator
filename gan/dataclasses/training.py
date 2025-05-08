"""Dataclasses with settings for GAN model and architecture parameters."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:
    """Config for generator and discriminator architectures."""

    latent_dim: int = 128
    hidden_g: int = 512
    hidden_d: int = 384


@dataclass(slots=True)
class TrainingConfig:
    """Config for training parameters."""

    batch_size: int = 256
    epochs: int = 500
    seed: int = 42
    verbose: bool = False


@dataclass(slots=True)
class OptimizerConfig:
    """Config for optimizer parameters."""

    g_lr: float = 1e-4
    d_lr: float = 2e-4
    lr_min_ratio: float = 0.05
    n_critic_initial: int = 4


@dataclass(slots=True)
class SchedulerConfig:
    """Config for learning rate scheduler parameters."""

    patience: int = 320
    val_interval: int = 5


@dataclass(slots=True)
class LossGPConfig:
    """Config for gradient penalty parameters."""

    gp_weight: float = 2.5
    drift_epsilon: float = 5e-4


@dataclass(slots=True)
class TemperatureConfig:
    """Config for Gumbel-Softmax temperature parameters."""

    tau_start: float = 2.5
    tau_end: float = 0.25


@dataclass(slots=True)
class DeltaConfig:
    """Config for delta loss parameters."""

    use_delta_loss: bool = False
    delta_w: float = 5.0
    delta_warmup: int = 100


@dataclass(slots=True)
class RegularizationConfig:
    """Config for regularization parameters."""

    use_bias_correction: bool = False
    use_hinge: bool = False


@dataclass(slots=True)
class FeatureMatchingConfig:
    """Config for feature matching parameters."""

    use_fm_loss: bool = False
    fm_w: float = 1.0


@dataclass(slots=True)
class CovarianceConfig:
    """Config for covariance regularization parameters."""

    use_cov_loss: bool = False
    cov_w: float = 1.0


@dataclass(slots=True)
class EmaConfig:
    """Config for Exponential Moving Average parameters."""

    beta: float = 0.999


@dataclass(slots=True)
class LossConfig:
    """Config for loss function parameters."""

    gp: LossGPConfig = field(default_factory=LossGPConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    feature_matching: FeatureMatchingConfig = field(
        default_factory=FeatureMatchingConfig
    )
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)


@dataclass(slots=True)
class GanConfig:
    """Config for GAN training and architecture parameters."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
