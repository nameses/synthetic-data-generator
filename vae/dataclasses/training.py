"""Define dataclasses for VAE training configuration."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:
    """VAE architecture / latent‐space settings."""

    latent_dim: int = 128
    kl_max: float = 1.0
    n_cycles: int = 4
    beta_warmup: int = 30


@dataclass(slots=True)
class TrainingConfig:
    """Training loop and optimizer settings."""

    epochs: int = 400
    batch_size: int = 512
    valid_split: float = 0.1
    seed: int = 43
    lr: float = 2e-4
    weight_decay: float = 1e-4
    verbose: bool = False


@dataclass(slots=True)
class VaeConfig:
    """Variational Autoencoder configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
