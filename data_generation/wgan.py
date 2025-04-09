# data_generation/wgan.py
import time
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Dict, Optional, List, Union, Tuple
from models.enums import DataType
from models.field_metadata import FieldMetadata
from faker import Faker

logger = logging.getLogger(__name__)


class GANConfig:
    """Configuration class for WGAN parameters"""

    def __init__(
            self,
            latent_dim: int = 256,
            batch_size: int = 64,
            n_critic: int = 5,
            gp_weight: float = 5.0,
            g_lr: float = 1e-5,
            d_lr: float = 1e-5,
            g_betas: tuple = (0.5, 0.9),
            d_betas: tuple = (0.5, 0.9),
            patience: int = 20,
            clip_value: float = 0.1,
            use_mixed_precision: bool = True,
            spectral_norm: bool = True,
            noise_level: float = 0.1,
            minibatch_std: bool = True,
            feature_matching: bool = True,
            max_epochs: int = 200,
            normalization_method: str = 'quantile',
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_betas = g_betas
        self.d_betas = d_betas
        self.patience = patience
        self.clip_value = clip_value
        self.use_mixed_precision = use_mixed_precision
        self.spectral_norm = spectral_norm
        self.noise_level = noise_level
        self.minibatch_std = minibatch_std
        self.feature_matching = feature_matching
        self.max_epochs = max_epochs
        self.normalization_method = normalization_method


class MinibatchStdDev(nn.Module):
    """Minibatch Standard Deviation Layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size < 2:
            return x

        # Calculate std over batch
        std = torch.std(x, dim=0, keepdim=True)
        mean_std = torch.mean(std, dim=1, keepdim=True)

        # Expand and concatenate
        mean_std = mean_std.expand(x.size(0), -1)
        return torch.cat([x, mean_std], dim=1)


class WGANGenerator(nn.Module):
    def __init__(self, config: GANConfig, output_dim: int, num_categories: int = 0):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim + num_categories

        layers = []
        hidden_dims = [256, 512, 256]

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if config.spectral_norm:
            layers[-1] = nn.utils.spectral_norm(layers[-1])
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if config.spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, noise, labels=None):
        if labels is not None:
            noise = torch.cat([noise, labels], dim=1)
        return self.net(noise)


class WGANDiscriminator(nn.Module):
    def __init__(self, config: GANConfig, input_dim: int, num_categories: int = 0):
        super().__init__()
        self.config = config
        input_dim = input_dim + num_categories

        layers = []
        hidden_dims = [256, 512, 256]

        # Input layer
        if config.minibatch_std:
            input_dim += 1  # For minibatch std

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if config.spectral_norm:
            layers[-1] = nn.utils.spectral_norm(layers[-1])
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if config.spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.2))

        # Output layers
        self.feature_layer = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.validity_layer = nn.Linear(hidden_dims[-1], 1)
        if config.spectral_norm:
            self.validity_layer = nn.utils.spectral_norm(self.validity_layer)

        if num_categories > 0:
            self.aux_layer = nn.Linear(hidden_dims[-1], num_categories)
        else:
            self.aux_layer = None

        self.minibatch_std = MinibatchStdDev() if config.minibatch_std else None
        self.net = nn.Sequential(*layers)

    def forward(self, x, labels=None):
        if labels is not None:
            x = torch.cat([x, labels], dim=1)

        if self.minibatch_std is not None:
            x = self.minibatch_std(x)

        features = self.net(x)
        features = self.feature_layer(features)

        validity = self.validity_layer(features)
        aux = self.aux_layer(features) if self.aux_layer is not None else None

        return validity, aux, features


class WGAN:
    def __init__(self, real_data: pd.DataFrame, metadata: Dict[str, FieldMetadata], config: GANConfig):
        self.real_data = real_data
        self.metadata = metadata
        self.config = config
        self.faker = Faker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize history tracking
        self.history = {
            'g_loss': [], 'd_loss': [], 'wasserstein': [], 'gp': [],
            'feature_matching': [], 'aux_loss': [], 'aux_g_loss': [],
            'd_grad': [], 'g_grad': [], 'lr_g': [], 'lr_d': [], 'gp_weight': []
        }

        # Identify column types
        self.numerical_cols = [col for col, meta in metadata.items()
                               if meta.data_type in [DataType.INTEGER, DataType.DECIMAL]]
        self.categorical_cols = [col for col, meta in metadata.items()
                                 if meta.data_type == DataType.CATEGORICAL]
        self.boolean_cols = [col for col, meta in metadata.items()
                             if meta.data_type == DataType.BOOLEAN]

        # Store original ranges
        self.num_ranges = {
            col: (meta.min_value, meta.max_value)
            for col, meta in metadata.items()
            if col in self.numerical_cols and meta.min_value is not None and meta.max_value is not None
        }

        # Store boolean distributions
        self.bool_distributions = {
            col: real_data[col].mean()
            for col in self.boolean_cols if col in real_data.columns
        }

        # Preprocess data and initialize models
        self.preprocess_data()
        self.init_models()

    def preprocess_data(self):
        """Preprocess data with proper normalization"""
        # Numerical pipeline
        if self.config.normalization_method == 'quantile':
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=1000)),
                ('scaler', StandardScaler())
            ])
        else:  # Default to standard scaling
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

        # Categorical pipeline
        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create column transformer
        transformers = []
        if self.numerical_cols:
            transformers.append(('num', num_pipeline, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', cat_pipeline, self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers, remainder='drop')

        # Fit and transform
        self.processed_data = self.preprocessor.fit_transform(self.real_data)

        # Get dimensions
        self.num_numerical = len(self.numerical_cols)
        self.num_categorical = 0
        if 'cat' in self.preprocessor.named_transformers_:
            self.num_categorical = \
            self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out().shape[0]

        # Create tensors
        self.X_num = torch.FloatTensor(self.processed_data[:, :self.num_numerical])
        self.X_cat = torch.FloatTensor(
            self.processed_data[:, self.num_numerical:]) if self.num_categorical > 0 else None

        # Create dataloader
        dataset = TensorDataset(self.X_num, self.X_cat) if self.X_cat is not None else TensorDataset(self.X_num)
        self.loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

    def init_models(self):
        """Initialize models with proper configuration"""
        self.generator = WGANGenerator(
            config=self.config,
            output_dim=self.num_numerical,
            num_categories=self.num_categorical
        ).to(self.device)

        self.discriminator = WGANDiscriminator(
            config=self.config,
            input_dim=self.num_numerical,
            num_categories=self.num_categorical
        ).to(self.device)

        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.g_lr,
            betas=self.config.g_betas
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.d_lr,
            betas=self.config.d_betas
        )

        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_g, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_d, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_mixed_precision)

    def gradient_penalty(self, real_data, fake_data, labels=None):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)

        # Interpolate between real and fake data
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # Discriminator output
        if labels is not None:
            disc_interpolates = self.discriminator(interpolates, labels)[0]
        else:
            disc_interpolates = self.discriminator(interpolates)[0]

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty * self.config.gp_weight

    def train_step(self, real_num, real_cat, epoch, batch_idx):
        """Ultra-stable WGAN training step"""
        # 1. CUDA initialization and input validation
        torch.cuda.empty_cache()
        torch.zeros(1).to(self.device)  # Ensure CUDA context

        if not torch.isfinite(real_num).all():
            logger.error("Non-finite values in real data")
            return None

        real_num = real_num.to(self.device)
        real_cat = real_cat.to(self.device) if real_cat is not None else None
        batch_size = real_num.size(0)

        # 2. Conservative noise generation
        noise = torch.randn(batch_size, self.config.latent_dim,
                            device=self.device) * 0.05  # Reduced noise scale

        # 3. Discriminator Update - Simplified
        self.opt_d.zero_grad(set_to_none=True)

        # Generator with strict output constraints
        with torch.no_grad():
            fake_num = torch.clamp(self.generator(noise, real_cat), -3.0, 3.0)

        # Basic discriminator forward
        d_real = self.discriminator(real_num, real_cat)[0]
        d_fake = self.discriminator(fake_num, real_cat)[0]

        # Stabilized loss calculation
        wasserstein_loss = torch.clamp(d_fake.mean() - d_real.mean(), -1.0, 1.0)
        gp = self._ultra_safe_gradient_penalty(real_num, fake_num, real_cat)
        d_loss = wasserstein_loss + gp * min(2.0, self.config.gp_weight * 0.1)  # Reduced GP

        # Manual gradient handling
        d_loss.backward()

        # Extreme gradient clipping
        d_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            max_norm=0.1,  # Very conservative
            norm_type=2.0
        )

        if not torch.isfinite(d_grad_norm) or d_grad_norm > 100:
            logger.warning(f"Discriminator gradient failure: {d_grad_norm}")
            self.opt_d.zero_grad(set_to_none=True)
            return None

        self.opt_d.step()

        # 4. Generator Update - Conservative
        g_metrics = {'g_loss': 0.0, 'g_grad': 0.0}

        if batch_idx % max(5, self.config.n_critic) == 0:  # Fewer updates
            self.opt_g.zero_grad(set_to_none=True)

            new_noise = torch.randn(batch_size, self.config.latent_dim,
                                    device=self.device) * 0.05

            fake_num = self.generator(new_noise, real_cat)
            fake_num = torch.clamp(fake_num, -3.0, 3.0)
            d_fake = self.discriminator(fake_num, real_cat)[0]

            # Very conservative generator loss
            g_loss = -d_fake.mean() * 0.2  # Reduced impact
            g_loss.backward()

            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=0.5,
                norm_type=2.0
            )

            if not torch.isfinite(g_grad_norm):
                logger.warning("Generator gradient failure")
                self.opt_g.zero_grad(set_to_none=True)
            else:
                self.opt_g.step()
                g_metrics = {
                    'g_loss': g_loss.item(),
                    'g_grad': g_grad_norm.item()
                }

        return {
            'd_loss': d_loss.item(),
            'wasserstein': wasserstein_loss.item(),
            'gp': gp.item(),
            'd_grad': d_grad_norm.item(),
            **g_metrics,
            'lr_g': self.opt_g.param_groups[0]['lr'],
            'lr_d': self.opt_d.param_groups[0]['lr']
        }

    def _ultra_safe_gradient_penalty(self, real_data, fake_data, labels=None):
        """Bulletproof gradient penalty calculation"""
        batch_size = real_data.size(0)

        # Conservative interpolation
        alpha = torch.rand(batch_size, 1, device=self.device, requires_grad=False)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        # Simple discriminator output
        if labels is not None:
            d_interpolates = self.discriminator(interpolates, labels)[0]
        else:
            d_interpolates = self.discriminator(interpolates)[0]

        # Manual gradient calculation
        gradients = torch.autograd.grad(
            outputs=d_interpolates.sum(),  # Sum reduces instability
            inputs=interpolates,
            create_graph=False,  # Safer
            retain_graph=True,
            only_inputs=True
        )[0]

        # Extreme gradient stabilization
        gradients = gradients.view(batch_size, -1)
        gradients = torch.clamp(gradients, -1e2, 1e2)  # Hard clipping
        grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)

        # Conservative penalty
        penalty = ((grad_norm - 1.0).clamp(-10.0, 10.0) ** 2).mean()
        return torch.clamp(penalty, 0.0, 5.0)

    def train_loop(self, epochs=None):
        """Robust training loop with advanced recovery mechanisms"""
        # Initialize CUDA and models
        torch.cuda.empty_cache()
        torch.zeros(1).to(self.device)

        if epochs is None:
            epochs = self.config.max_epochs

        # Training header with config summary
        logger.info("\n=== Stabilized WGAN Training ===")
        logger.info(f"Device: {self.device} | Batch size: {self.config.batch_size}")
        logger.info(f"GP Weight: {self.config.gp_weight} | LR: {self.config.g_lr}/{self.config.d_lr}")
        logger.info("-" * 60)

        self._best_loss = float('inf')
        self._no_improve = 0

        for epoch in range(epochs):
            epoch_metrics = {
                'g_loss': [], 'd_loss': [],
                'wasserstein': [], 'gp': [],
                'g_grad': [], 'd_grad': []
            }
            valid_batches = 0

            self.generator.train()
            self.discriminator.train()

            for batch_idx, data in enumerate(self.loader):
                try:
                    # Get batch data with validation
                    real_num = data[0].to(self.device)
                    real_cat = data[1].to(self.device) if len(data) > 1 else None

                    if real_num.size(0) < 2:
                        continue

                    # Training step with recovery
                    metrics = self.train_step(real_num, real_cat, epoch, batch_idx)
                    if metrics is None:
                        self._recover_from_failure()
                        continue

                    # Track metrics
                    valid_batches += 1
                    for k, v in metrics.items():
                        if k in epoch_metrics:
                            epoch_metrics[k].append(v)

                    # Enhanced logging
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | "
                            f"G: {metrics['g_loss']:.3f} (grad:{metrics['g_grad']:.1f}) | "
                            f"D: {metrics['d_loss']:.3f} (grad:{metrics['d_grad']:.1f}) | "
                            f"W: {-metrics['wasserstein']:.3f} | GP: {metrics['gp']:.2f}"
                        )

                except Exception as e:
                    logger.error(f"Batch failed: {str(e)}")
                    self._recover_from_failure()
                    continue

            # Epoch validation
            if valid_batches == 0:
                logger.error("Epoch failed - no valid batches")
                break

            # Calculate epoch averages
            epoch_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}

            # Update history and learning rates
            for k, v in epoch_avg.items():
                self.history[k].append(v)

            self._update_learning_rates(epoch_avg['g_loss'], epoch_avg['d_loss'])

            # Early stopping and progress
            if self._check_early_stopping(epoch_avg, epoch):
                break

            # Periodic evaluation
            if (epoch + 1) % 5 == 0:
                self._evaluate_progress(epoch + 1)

        return self.history

    def _recover_from_failure(self):
        """Simplified recovery without scaler update"""
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)

        # Reduce learning rates
        for param_group in self.opt_g.param_groups:
            param_group['lr'] *= 0.9
        for param_group in self.opt_d.param_groups:
            param_group['lr'] *= 0.9

        torch.cuda.empty_cache()

    def _update_learning_rates(self, g_loss, d_loss):
        """Adaptive learning rate adjustment"""
        # Generator LR update
        if g_loss < 0.1:  # Too small loss
            for param_group in self.opt_g.param_groups:
                param_group['lr'] *= 0.98

        # Discriminator LR update
        if d_loss > 5.0:  # Too large loss
            for param_group in self.opt_d.param_groups:
                param_group['lr'] *= 0.9
        elif d_loss < 0.5:  # Too small loss
            for param_group in self.opt_d.param_groups:
                param_group['lr'] *= 1.02  # Slight increase

    def _check_early_stopping(self, metrics, epoch):
        """Enhanced early stopping criteria with proper state tracking"""
        # Initialize tracking variables if they don't exist
        if not hasattr(self, '_best_loss'):
            self._best_loss = float('inf')
        if not hasattr(self, '_no_improve'):
            self._no_improve = 0

        # Check for NaN/inf first
        if any(not math.isfinite(v) for v in metrics.values()):
            logger.error("Non-finite metrics detected, stopping training")
            return True

        # Check for loss divergence
        if metrics['d_loss'] > 100 or metrics['g_loss'] > 100:
            logger.error("Loss divergence detected (D: %.2f, G: %.2f), stopping",
                         metrics['d_loss'], metrics['g_loss'])
            return True

        # Normal early stopping logic
        if metrics['g_loss'] < self._best_loss * 0.99:  # 1% improvement
            logger.info("Improvement detected (%.4f -> %.4f)",
                        self._best_loss, metrics['g_loss'])
            self._best_loss = metrics['g_loss']
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self.config.patience:
                logger.info("Early stopping at epoch %d - no improvement for %d epochs",
                            epoch + 1, self._no_improve)
                return True

        # Additional stopping criteria
        if -metrics['wasserstein'] < 0.001:  # Wasserstein distance too small
            logger.info("Stopping - Wasserstein distance converged (%.4f)",
                        -metrics['wasserstein'])
            return True

        return False

    def generate(self, n_samples):
        """Generate synthetic samples"""
        self.generator.eval()

        with torch.no_grad():
            # Generate in batches
            batch_size = min(512, n_samples)
            synthetic_num = []

            for i in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - i)

                # Generate noise
                noise = torch.randn(current_batch, self.config.latent_dim, device=self.device)

                # Sample categorical data if needed
                if self.num_categorical > 0 and self.X_cat is not None:
                    idx = torch.randint(0, len(self.X_cat), (current_batch,))
                    cat_samples = self.X_cat[idx].to(self.device)
                else:
                    cat_samples = None

                # Generate numerical data
                batch_num = self.generator(noise, cat_samples)
                synthetic_num.append(batch_num.cpu().numpy())

            # Combine batches
            synthetic_num = np.vstack(synthetic_num)

            # Inverse transform numerical data
            if 'num' in self.preprocessor.named_transformers_:
                synthetic_num = np.clip(synthetic_num, -5, 5)  # Prevent extreme values
                inverted_num = self.preprocessor.named_transformers_['num'].inverse_transform(synthetic_num)

                # Apply range constraints
                for i, col in enumerate(self.numerical_cols):
                    if col in self.num_ranges:
                        min_val, max_val = self.num_ranges[col]
                        if min_val is not None:
                            inverted_num[:, i] = np.maximum(inverted_num[:, i], min_val)
                        if max_val is not None:
                            inverted_num[:, i] = np.minimum(inverted_num[:, i], max_val)

                synthetic_df = pd.DataFrame(inverted_num, columns=self.numerical_cols)
            else:
                synthetic_df = pd.DataFrame(columns=self.numerical_cols)

            # Add categorical data
            if self.num_categorical > 0 and 'cat' in self.preprocessor.named_transformers_:
                cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']

                for i, col in enumerate(self.categorical_cols):
                    # Sample categories based on original distribution
                    categories = cat_encoder.categories_[i]
                    value_counts = self.real_data[col].value_counts(normalize=True)
                    probs = [value_counts.get(cat, 0) for cat in categories]
                    synthetic_df[col] = np.random.choice(categories, size=n_samples, p=probs)

            # Add boolean columns
            for col in self.boolean_cols:
                if col in self.bool_distributions:
                    synthetic_df[col] = np.random.random(n_samples) < self.bool_distributions[col]

            # Add other field types
            for col, meta in self.metadata.items():
                if col in synthetic_df.columns:
                    continue

                if meta.data_type == DataType.DATE_TIME:
                    synthetic_df[col] = self._generate_datetime_column(col, n_samples)
                elif meta.data_type == DataType.STRING:
                    synthetic_df[col] = self._generate_string_column(meta, n_samples)

            return synthetic_df

    def _evaluate_progress(self, epoch):
        """Evaluation method that matches your training loop"""
        try:
            # Generate sample data
            with torch.no_grad():
                samples = self.generate(100)

            # Log basic statistics
            logger.info(f"\nEpoch {epoch} Evaluation:")

            # Numerical columns
            num_cols = [col for col in samples.columns
                        if col in self.numerical_cols][:3]  # First 3 cols
            for col in num_cols:
                logger.info(
                    f"{col:<15}: mean={samples[col].mean():>8.3f} | "
                    f"std={samples[col].std():>7.3f} | "
                    f"min={samples[col].min():>7.3f} | "
                    f"max={samples[col].max():>7.3f}"
                )

            # Categorical columns
            cat_cols = [col for col in samples.columns
                        if col in self.categorical_cols][:2]  # First 2 cols
            for col in cat_cols:
                counts = samples[col].value_counts(normalize=True)
                logger.info(f"{col:<15}: " + " | ".join(
                    f"{k}: {v:.2%}" for k, v in counts.iloc[:3].items()
                ) + ("..." if len(counts) > 3 else ""))

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

    def _evaluate_generator(self, epoch):
        """Evaluate generator quality"""
        samples = self.generate(100)
        logger.info(f"Epoch {epoch} sample stats:")
        for col in samples.columns[:3]:  # Show first 3 columns
            logger.info(f"{col}: mean={samples[col].mean():.4f}, std={samples[col].std():.4f}")

    def _generate_datetime_column(self, col, n_samples):
        """Generate datetime column"""
        if col in self.real_data.columns:
            min_date = self.real_data[col].min()
            max_date = self.real_data[col].max()
            delta = max_date - min_date
            return [min_date + delta * np.random.random() for _ in range(n_samples)]
        else:
            return pd.date_range('2020-01-01', periods=n_samples, freq='D')

    def _generate_string_column(self, meta, n_samples):
        """Generate string column"""
        if meta.fake_strategy:
            if meta.string_format:
                return [getattr(self.faker, meta.fake_strategy)(meta.string_format) for _ in range(n_samples)]
            return [getattr(self.faker, meta.fake_strategy)() for _ in range(n_samples)]
        elif meta.custom_faker:
            return [meta.custom_faker() for _ in range(n_samples)]
        return [self.faker.email() for _ in range(n_samples)]