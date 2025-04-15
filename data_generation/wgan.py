import logging
import math
from datetime import datetime
from random import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Dict
from models.enums import DataType
from models.field_metadata import FieldMetadata
from faker import Faker

logger = logging.getLogger(__name__)


class GANConfig:
    def __init__(
            self,
            latent_dim: int = 256,
            batch_size: int = 64,
            n_critic: int = 15,
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
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        batch_size, num_features = x.shape
        if batch_size < 2:  # Can't compute std with single sample
            zeros = torch.zeros(batch_size, 1, device=x.device)
            return torch.cat([x, zeros], dim=1)

        # Calculate std over batch (along dimension 0)
        std = torch.std(x, dim=0, unbiased=False, keepdim=True)

        # Calculate mean of std for each sample
        mean_std = torch.mean(std, dim=1, keepdim=True)

        # Expand and concatenate
        return torch.cat([x, mean_std.expand(batch_size, 1)], dim=1)


class WGANGenerator(nn.Module):
    def __init__(self, config: GANConfig, num_numerical: int, num_categorical: int = 0, num_nan: int = 0):
        super().__init__()
        self.config = config
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.num_nan = num_nan

        # Input dimension includes latent dim + categorical features
        input_dim = config.latent_dim + num_categorical

        # Main network architecture
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(1024),
            # Output includes numerical values and NaN indicators
            nn.Linear(1024, num_numerical + num_nan)
        )

    def forward(self, noise, labels=None):
        # Concatenate noise with labels if provided
        if labels is not None:
            noise = torch.cat([noise, labels], dim=1)

        # Forward pass through network
        output = self.main(noise)

        # Split into values and NaN indicators
        values = output[:, :self.num_numerical]
        nan_probs = torch.sigmoid(output[:, self.num_numerical:]) if self.num_nan > 0 else None

        # Apply clamping to values
        values = torch.clamp(values, -3.0, 3.0)

        return values, nan_probs


# class WGANGenerator(nn.Module):
#     def __init__(self, config: GANConfig, output_dim: int, num_categories: int = 0):
#         super().__init__()
#         self.config = config
#         input_dim = config.latent_dim + num_categories
#
#         layers = []
#         hidden_dims = [256, 512, 256]
#
#         # Input layer
#         layers.append(nn.Linear(input_dim, hidden_dims[0]))
#         if config.spectral_norm:
#             layers[-1] = nn.utils.spectral_norm(layers[-1])
#         layers.append(nn.LeakyReLU(0.2))
#
#         # Hidden layers
#         for i in range(1, len(hidden_dims)):
#             layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
#             if config.spectral_norm:
#                 layers[-1] = nn.utils.spectral_norm(layers[-1])
#             layers.append(nn.LayerNorm(hidden_dims[i]))
#             layers.append(nn.LeakyReLU(0.2))
#
#         # Output layer
#         layers.append(nn.Linear(hidden_dims[-1], output_dim))
#
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, noise, labels=None):
#         if labels is not None:
#             noise = torch.cat([noise, labels], dim=1)
#         return self.net(noise)


class WGANDiscriminator(nn.Module):
    def __init__(self, config: GANConfig, num_numerical: int, num_categorical: int = 0, num_nan: int = 0):
        super().__init__()
        self.config = config
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.num_nan = num_nan

        # Base input dimension (without minibatch_std)
        self.base_input_dim = num_numerical + num_categorical + num_nan

        # Minibatch std dev adds 1 feature if enabled
        self.minibatch_std = MinibatchStdDev() if config.minibatch_std else None
        self.input_dim = self.base_input_dim + (1 if config.minibatch_std else 0)

        print(f"Initializing discriminator:")
        print(f"- Numerical features: {num_numerical}")
        print(f"- Categorical features: {num_categorical}")
        print(f"- NaN masks: {num_nan}")
        print(f"- Base input dim: {self.base_input_dim}")
        print(f"- Final input dim: {self.input_dim} (minibatch_std: {config.minibatch_std})")

        # Network architecture
        hidden_dims = [256, 512, 256]

        layers = []
        current_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if config.spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.LayerNorm(h_dim))
            current_dim = h_dim

        self.net = nn.Sequential(*layers)

        # Output layers
        self.feature_layer = nn.Linear(current_dim, current_dim)
        self.validity_layer = nn.Linear(current_dim, 1)
        if config.spectral_norm:
            self.validity_layer = nn.utils.spectral_norm(self.validity_layer)

        if num_categorical > 0:
            self.aux_layer = nn.Linear(current_dim, num_categorical)
        else:
            self.aux_layer = None

    def forward(self, x):
        # Verify input dimensions match base_input_dim
        if x.shape[1] != self.base_input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.base_input_dim}, got {x.shape[1]}. "
                f"Numerical: {self.num_numerical}, Categorical: {self.num_categorical}, NaN: {self.num_nan}"
            )

        # Apply minibatch std if enabled
        if self.minibatch_std is not None:
            x = self.minibatch_std(x)
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"After minibatch_std, expected {self.input_dim}, got {x.shape[1]}"
                )

        # Forward pass
        features = self.net(x)
        features = self.feature_layer(features)
        validity = self.validity_layer(features)
        aux = self.aux_layer(features) if self.aux_layer is not None else None

        return validity, aux, features


class WGAN:
    def __init__(self, real_data: pd.DataFrame, metadata: Dict[str, FieldMetadata], config: GANConfig):
        self.datetime_ranges = {}
        self.real_data = real_data
        self.metadata = metadata
        self.config = config
        self.faker = Faker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dimensions
        self.num_numerical = 0
        self.num_categorical = 0
        self.num_nan = 0

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
        self.preprocess_data(real_data_length=len(real_data))
        self.init_models()

    def preprocess_data(self, real_data_length):
        # Ensure numerical columns are actually numeric
        for col in self.numerical_cols:
            if col in self.real_data.columns:
                self.real_data[col] = pd.to_numeric(self.real_data[col], errors='coerce')
                if self.real_data[col].isna().all():
                    self.real_data[col] = 0

        # Track which columns can have NaNs
        self.nan_columns = [col for col, meta in self.metadata.items()
                            if meta.allow_nans and col in self.real_data.columns]

        # Create NaN indicator features (1 if value is NaN, 0 otherwise)
        self.nan_mask = self.real_data[self.nan_columns].isna().astype(float)

        self.datetime_cols = {}
        for col, meta in self.metadata.items():
            if meta.data_type == DataType.DATETIME and col in self.real_data.columns:
                try:
                    # First convert to datetime objects
                    self.real_data[col] = pd.to_datetime(
                        self.real_data[col],
                        format=meta.datetime_format,
                        errors='coerce'
                    )

                    # Then convert to numerical representation
                    if meta.datetime_type == 'date':
                        self.real_data[f'_num_{col}'] = self.real_data[col].apply(
                            lambda x: x.toordinal() if pd.notna(x) else np.nan
                        )
                    elif meta.datetime_type == 'time':
                        self.real_data[f'_num_{col}'] = self.real_data[col].apply(
                            lambda x: (x.hour * 3600 + x.minute * 60 + x.second)
                            if pd.notna(x) else np.nan
                        )
                    else:  # datetime
                        self.real_data[f'_num_{col}'] = self.real_data[col].apply(
                            lambda x: x.timestamp() if pd.notna(x) else np.nan
                        )

                    # Store metadata for reconstruction
                    valid_values = self.real_data[f'_num_{col}'][self.real_data[f'_num_{col}'].notna()]
                    if len(valid_values) > 0:
                        self.datetime_cols[col] = {
                            'type': meta.datetime_type,
                            'format': meta.datetime_format,
                            'min': valid_values.min(),
                            'max': valid_values.max()
                        }
                        self.numerical_cols.append(f'_num_{col}')
                    else:
                        logger.warning(f"No valid datetime values found for {col}")

                    if meta.data_type == DataType.DATETIME:
                        self.datetime_ranges[col] = {
                            'min': self.real_data[f'_num_{col}'].min(),
                            'max': self.real_data[f'_num_{col}'].max()
                        }

                except Exception as e:
                    logger.error(f"Error processing datetime column {col}: {str(e)}")
                    continue

        # Numerical pipeline
        if self.config.normalization_method == 'quantile':
            self.num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                ('robust', RobustScaler()),
                ('quantile', QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=min(1000, real_data_length // 10),
                    subsample=100000)),
                ('clip', FunctionTransformer(
                    func=lambda x: np.clip(x, -5, 5)))
            ])
        else:
            self.num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

        # Process numerical data
        numerical_data = self.num_pipeline.fit_transform(self.real_data[self.numerical_cols])
        self.num_numerical = numerical_data.shape[1]

        # Process categorical data
        self.categorical_encoders = {}
        categorical_data = []
        for col in self.categorical_cols:
            if col in self.real_data.columns:
                encoder = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                encoded = encoder.fit_transform(self.real_data[[col]])
                categorical_data.append(encoded)
                self.categorical_encoders[col] = encoder

        # Combine categorical data if exists
        if categorical_data:
            categorical_data = np.hstack(categorical_data)
            self.num_categorical = categorical_data.shape[1]
        else:
            self.num_categorical = 0

        # Process NaN mask
        self.num_nan = len(self.nan_columns) if self.nan_columns else 0

        # Verify dimensions
        print(f"Numerical data shape: {numerical_data.shape}")
        if self.num_categorical > 0:
            print(f"Categorical data shape: {categorical_data.shape}")
        if hasattr(self, 'nan_mask'):
            print(f"NaN mask shape: {self.nan_mask.shape}")

        self.X_num = torch.FloatTensor(numerical_data.astype(np.float32))
        if self.num_categorical > 0:
            self.X_cat = torch.FloatTensor(categorical_data.astype(np.float32))
        if hasattr(self, 'nan_mask'):
            self.X_nan = torch.FloatTensor(self.nan_mask.values.astype(np.float32))

        # Update the actual counts based on the processed data
        self.num_numerical = numerical_data.shape[1]
        if self.num_categorical > 0:
            self.num_categorical = categorical_data.shape[1]
        self.num_nan = len(self.nan_columns) if hasattr(self, 'nan_columns') else 0

        # Verify total dimensions match
        total_dim = self.num_numerical + self.num_categorical + self.num_nan
        print(f"Total input dimension: {total_dim}")
        print(f"Numerical cols: {len(self.numerical_cols)}")
        print(f"Categorical cols: {len(self.categorical_cols)}")
        print(f"NaN cols: {len(self.nan_columns) if hasattr(self, 'nan_columns') else 0}")

        # Create dataloader
        tensors = [t for t in [self.X_num, self.X_cat, self.X_nan] if t is not None]
        self.loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

    def init_models(self):
        # Make sure all dimensions are properly calculated
        self.num_numerical = len(self.numerical_cols)
        # Recalculate categorical dimension
        self.num_categorical = 0
        if hasattr(self, 'X_cat') and self.X_cat is not None:
            self.num_categorical = self.X_cat.shape[1]
        # Recalculate nan dimension
        self.num_nan = 0
        if hasattr(self, 'nan_mask') and self.nan_mask is not None:
            self.num_nan = self.nan_mask.shape[1]

        print(f"Initializing models with dimensions:")
        print(f"- Numerical features: {self.num_numerical}")
        print(f"- Categorical features: {self.num_categorical}")
        print(f"- NaN masks: {self.num_nan}")

        # Now initialize the models with the correct dimensions
        self.generator = WGANGenerator(
            config=self.config,
            num_numerical=self.num_numerical,
            num_categorical=self.num_categorical,
            num_nan=self.num_nan
        ).to(self.device)

        self.discriminator = WGANDiscriminator(
            config=self.config,
            num_numerical=self.num_numerical,
            num_categorical=self.num_categorical,
            num_nan=self.num_nan
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

    def train_step(self, real_num, real_cat, real_nan, epoch, epochs, batch_idx):
        # Move data to device
        real_num = real_num.to(self.device)
        real_cat = real_cat.to(self.device) if real_cat is not None else None
        real_nan = real_nan.to(self.device) if real_nan is not None else None
        batch_size = real_num.size(0)

        # Generate noise
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)

        # Generate fake data
        fake_num, fake_nan = self.generator(noise, real_cat)

        # Combine inputs for discriminator
        def combine_inputs(num, cat, nan):
            inputs = [num]
            if self.num_categorical > 0 and cat is not None:
                inputs.append(cat)
            if self.num_nan > 0:
                nan_tensor = nan if nan is not None else torch.zeros(batch_size, self.num_nan, device=self.device)
                inputs.append(nan_tensor)
            return torch.cat(inputs, dim=1)

        real_combined = combine_inputs(real_num, real_cat, real_nan)
        fake_combined = combine_inputs(fake_num.detach(), real_cat, fake_nan.detach() if fake_nan is not None else None)

        # Verify dimensions
        # print(f"Training step dimensions:")
        # print(f"- Real combined: {real_combined.shape} (expected: {self.discriminator.base_input_dim})")
        # print(f"- Fake combined: {fake_combined.shape} (expected: {self.discriminator.base_input_dim})")

        # Discriminator update
        self.opt_d.zero_grad()

        # Forward pass
        d_real = self.discriminator(real_combined)
        d_fake = self.discriminator(fake_combined)

        # Calculate losses
        wasserstein_loss = torch.clamp(d_fake[0].mean() - d_real[0].mean(), -1.0, 1.0)
        gp = self.gradient_penalty(real_combined, fake_combined)
        d_loss = wasserstein_loss + gp * self.config.gp_weight

        # Backpropagate
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.1)
        self.opt_d.step()

        # Generator update (every n_critic steps)
        g_metrics = {'g_loss': 0.0, 'g_grad': 0.0}
        if batch_idx % max(5, self.config.n_critic) == 0:
            self.opt_g.zero_grad(set_to_none=True)

            new_noise = torch.randn(batch_size, self.config.latent_dim, device=self.device) * 0.05
            fake_num, _ = self.generator(new_noise, real_cat)
            fake_combined = combine_inputs(fake_num, real_cat, None)

            d_fake = self.discriminator(fake_combined)[0]
            dist_loss = self.distribution_matching_loss(real_num, fake_num) * 10.0

            current_epoch_ratio = min(1.0, epoch / 50)
            g_loss = (-d_fake.mean() + current_epoch_ratio * dist_loss * 5.0)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            self.opt_g.step()

            g_metrics = {
                'g_loss': g_loss.item(),
                'g_grad': torch.norm(
                    torch.stack([p.grad.norm() for p in self.generator.parameters() if p.grad is not None])).item()
            }

        return {
            'd_loss': d_loss.item(),
            'wasserstein': wasserstein_loss.item(),
            'gp': gp.item(),
            'd_grad': torch.norm(
                torch.stack([p.grad.norm() for p in self.discriminator.parameters() if p.grad is not None])).item(),
            **g_metrics,
            'lr_g': self.opt_g.param_groups[0]['lr'],
            'lr_d': self.opt_d.param_groups[0]['lr']
        }

    def distribution_matching_loss(self, real_data, fake_data):
        # moment matching (mean, std)
        moment_loss = F.mse_loss(fake_data.mean(dim=0), real_data.mean(dim=0)) + \
                      F.mse_loss(fake_data.std(dim=0), real_data.std(dim=0))

        # quantile matching (captures full distribution shape)
        quantiles = torch.linspace(0.1, 0.9, 5).to(real_data.device)
        real_quantiles = torch.quantile(real_data, quantiles, dim=0)
        fake_quantiles = torch.quantile(fake_data, quantiles, dim=0)
        quantile_loss = F.mse_loss(fake_quantiles, real_quantiles)

        # histogram matching (fine-grained distribution)
        bins = 20
        hist_loss = 0
        for i in range(real_data.shape[1]):
            real_hist = torch.histc(real_data[:, i], bins=bins, min=-3, max=3)
            fake_hist = torch.histc(fake_data[:, i], bins=bins, min=-3, max=3)
            hist_loss += F.kl_div(
                F.log_softmax(fake_hist, dim=0),
                F.softmax(real_hist, dim=0),
                reduction='batchmean'
            )

        return moment_loss + quantile_loss + (hist_loss / real_data.shape[1])

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)

        # Conservative interpolation
        alpha = torch.rand(batch_size, 1, device=self.device, requires_grad=False)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        # Simple discriminator output
        d_interpolates = self.discriminator(interpolates)[0]

        # Manual gradient calculation
        gradients = torch.autograd.grad(
            outputs=d_interpolates.sum(),
            inputs=interpolates,
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]

        # gradient stabilization
        gradients = gradients.view(batch_size, -1)
        gradients = torch.clamp(gradients, -1e2, 1e2)
        grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)

        penalty = ((grad_norm - 1.0).clamp(-10.0, 10.0) ** 2).mean()
        return torch.clamp(penalty, 0.0, 5.0)

    def train_loop(self, epochs=None):
        # initialize CUDA and models
        torch.cuda.empty_cache()
        torch.zeros(1).to(self.device)

        if epochs is None:
            epochs = self.config.max_epochs

        # training header config summary
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
                    real_cat = data[1].to(self.device) if len(data) > 1 and self.num_categorical > 0 else None
                    real_nan = data[2].to(self.device) if len(data) > 2 and self.num_nan > 0 else None

                    # Dimension verification
                    expected_dim = self.num_numerical
                    if real_cat is not None:
                        expected_dim += self.num_categorical
                    if real_nan is not None:
                        expected_dim += self.num_nan

                    if real_num.shape[1] != self.num_numerical:
                        raise ValueError(
                            f"Numerical dimension mismatch. Expected {self.num_numerical}, got {real_num.shape[1]}")

                    if real_num.size(0) < 2:
                        continue

                    # Training step with recovery
                    metrics = self.train_step(real_num, real_cat, real_nan, epoch, epochs, batch_idx)
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
            if (epoch + 1) % 2 == 0 or epoch == 0:
                self._evaluate_progress(epoch + 1)

                # Wasserstein distance for first 5 numerical columns
                with torch.no_grad():
                    samples = self.generate(min(1000, len(self.real_data)))

                    logger.info("\n=== Distribution Metrics ===")
                    for col in self.numerical_cols[:5]:
                        if col in self.real_data.columns and col in samples.columns:
                            w_dist = self._calculate_wasserstein(
                                self.real_data[col],
                                samples[col]
                            )
                            logger.info(f"Wasserstein {col}: {w_dist:.4f}")
                    logger.info("=" * 30)

        return self.history

    def _calculate_wasserstein(self, real, synthetic):
        # Convert to numpy if needed
        if isinstance(real, pd.Series):
            real = real.values
        if isinstance(synthetic, pd.Series):
            synthetic = synthetic.values

        # Sort the values
        real_sorted = np.sort(real)
        synth_sorted = np.sort(synthetic)

        # Calculate the distance
        n = len(real_sorted)
        m = len(synth_sorted)
        all_sorted = np.sort(np.concatenate([real_sorted, synth_sorted]))

        # Compute the cumulative distributions
        cdf_real = np.searchsorted(real_sorted, all_sorted, side='right') / n
        cdf_synth = np.searchsorted(synth_sorted, all_sorted, side='right') / m

        # Integrate the absolute difference
        return np.trapz(np.abs(cdf_real - cdf_synth), all_sorted)

    def calibrate_numeric_columns(self, synthetic_df):
        for col in self.numerical_cols:
            if col not in self.real_data.columns:
                continue

            if col.startswith('_num_'):  # Skip temporary columns
                continue

            # Handle duplicate values by adding tiny noise
            synth_vals = synthetic_df[col].values.copy()
            if len(np.unique(synth_vals)) < len(synth_vals):
                noise = np.random.normal(0, 1e-10 * synth_vals.std(), size=len(synth_vals))
                synth_vals = synth_vals + noise

            # Calculate quantiles with epsilon padding
            epsilon = 1e-10
            quantiles = np.linspace(epsilon, 1 - epsilon, 100)

            try:
                orig_quantiles = np.quantile(self.real_data[col], quantiles)
                synth_quantiles = np.quantile(synth_vals, quantiles)

                # Create robust mapping function
                valid_mask = ~np.isnan(orig_quantiles) & ~np.isnan(synth_quantiles)
                if valid_mask.sum() > 1:  # Need at least 2 points for interpolation
                    mapping = interp1d(
                        synth_quantiles[valid_mask],
                        orig_quantiles[valid_mask],
                        bounds_error=False,
                        fill_value="extrapolate",
                        assume_sorted=True
                    )
                    synthetic_df[col] = mapping(synth_vals)
            except Exception as e:
                logger.warning(f"Calibration failed for {col}: {str(e)}")
                continue

            # Apply decimal places if specified in metadata
            if col in self.metadata:
                if self.metadata[col].data_type == DataType.INTEGER:
                    synthetic_df[col] = synthetic_df[col].round().astype(int)
                elif self.metadata[col].decimal_places is not None:
                    synthetic_df[col] = synthetic_df[col].round(self.metadata[col].decimal_places)

            # Handle boolean columns that were numeric in original data
            if col in self.boolean_cols and col in self.real_data.columns:
                if self.real_data[col].dtype.kind in ['i', 'b']:
                    synthetic_df[col] = synthetic_df[col].astype(self.real_data[col].dtype)

            # Ensure constraints are still met
            if col in self.num_ranges:
                min_val, max_val = self.num_ranges[col]
                synthetic_df[col] = synthetic_df[col].clip(min_val, max_val)

        return synthetic_df

    def _recover_from_failure(self):
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)

        # Reduce learning rates
        for param_group in self.opt_g.param_groups:
            param_group['lr'] *= 0.9
        for param_group in self.opt_d.param_groups:
            param_group['lr'] *= 0.9

        torch.cuda.empty_cache()

    def _update_learning_rates(self, g_loss, d_loss):
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
        # Initialize tracking variables if they don't exist
        if not hasattr(self, '_best_loss'):
            self._best_loss = float('inf')
        if not hasattr(self, '_no_improve'):
            self._no_improve = 0

        # Check for NaN/inf first
        if any(not math.isfinite(v) for v in metrics.values()):
            logger.error("Non-finite metrics detected, stopping training")
            return True

        if epoch < 20:
            return False

        # Check for loss divergence
        if metrics['d_loss'] > 100 or metrics['g_loss'] > 100:
            logger.error("Loss divergence detected (D: %.2f, G: %.2f), stopping",
                         metrics['d_loss'], metrics['g_loss'])
            return True

        # Normal early stopping logic
        if metrics['g_loss'] < self._best_loss * 0.95:  # 1% improvement
            logger.info("Improvement detected (%.4f -> %.4f)", self._best_loss, metrics['g_loss'])
            self._best_loss = metrics['g_loss']
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self.config.patience:
                logger.info("Early stopping at epoch %d - no improvement for %d epochs", epoch + 1, self._no_improve)
                return True

        # Additional stopping criteria
        if -metrics['wasserstein'] < 0.001:  # Wasserstein distance too small
            logger.info("Stopping - Wasserstein distance converged (%.4f)",
                        -metrics['wasserstein'])
            return True

        return False

    def generate(self, n_samples):
        self.generator.eval()

        with torch.no_grad():
            batch_size = min(512, n_samples)
            synthetic_num = []
            synthetic_nan = []

            for i in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - i)
                noise = torch.randn(current_batch, self.config.latent_dim, device=self.device)

                if self.num_categorical > 0 and self.X_cat is not None:
                    idx = torch.randint(0, len(self.X_cat), (current_batch,))
                    cat_samples = self.X_cat[idx].to(self.device)
                else:
                    cat_samples = None

                batch_num, batch_nan = self.generator(noise, cat_samples)
                synthetic_num.append(batch_num.cpu().numpy())
                if batch_nan is not None:
                    synthetic_nan.append(batch_nan.cpu().numpy())

            synthetic_num = np.vstack(synthetic_num)
            synthetic_nan = np.vstack(synthetic_nan) if synthetic_nan else None

            # Create DataFrame with numerical columns
            synthetic_df = pd.DataFrame(synthetic_num, columns=self.numerical_cols)

            # Handle categorical columns
            for col in self.categorical_cols:
                if col in self.categorical_encoders:
                    encoder = self.categorical_encoders[col]['onehot']
                    categories = encoder.categories_[0]

                    # Get probabilities from real data
                    real_counts = self.real_data[col].value_counts(normalize=True)
                    probs = real_counts.reindex(categories, fill_value=0).values

                    # Ensure probabilities sum to 1
                    probs = probs / probs.sum()

                    synthetic_df[col] = np.random.choice(categories, size=n_samples, p=probs)

            # Handle boolean columns
            for col in self.boolean_cols:
                if col in self.bool_distributions:
                    prob = self.bool_distributions[col]
                    synthetic_df[col] = np.random.random(n_samples) < prob

            # Handle string columns with faker
            for col, meta in self.metadata.items():
                if meta.data_type == DataType.STRING and meta.faker_method:
                    synthetic_df[col] = [meta.faker_method(**meta.faker_args) for _ in range(n_samples)]

            if synthetic_nan is not None and len(self.nan_columns) > 0:
                for i, col in enumerate(self.nan_columns):
                    if col in synthetic_df.columns:
                        nan_prob = self.metadata[col].nan_probability if col in self.metadata else 0.1
                        nan_mask = np.random.random(n_samples) < nan_prob
                        synthetic_df.loc[nan_mask, col] = np.nan

            synthetic_df = self.calibrate_numeric_columns(synthetic_df)

            # for col in self.datetime_cols:
            #     num_col = f'_num_{col}'
            #     if num_col in synthetic_df.columns:
            #         # Clip values to the observed range
            #         meta = self.datetime_cols[col]
            #         synthetic_df[num_col] = np.clip(
            #             synthetic_df[num_col],
            #             meta['min'],
            #             meta['max']
            #         )

            # Reconstruct datetime columns

            for col, meta in self.datetime_cols.items():
                num_col = f'_num_{col}'

                if num_col in synthetic_df.columns:
                    # Scale back to original range
                    synthetic_df[num_col] = (
                            (synthetic_df[num_col] - (-5)) / (5 - (-5)) *  # Assuming data was normalized to [-5,5]
                            (meta['max'] - meta['min']) + meta['min']
                    )

                    # Convert to datetime objects
                    values = []
                    for num_val in synthetic_df[num_col]:
                        if pd.isna(num_val):
                            values.append(None)
                            continue
                        try:
                            num_val = float(num_val)
                            if meta['type'] == 'date':
                                ordinal = int(np.clip(num_val, 1, 3652059))  # Clamp to valid ordinal range
                                dt = datetime.fromordinal(ordinal)
                            elif meta['type'] == 'time':
                                seconds = int(np.clip(num_val, 0, 86399))  # Clamp to 0-86399 seconds
                                dt = datetime(2000, 1, 1, seconds // 3600 % 24, (seconds % 3600) // 60, seconds % 60)
                            else:  # datetime
                                timestamp = np.clip(num_val, 0, 4102444800)  # Clamp to 1970-2100
                                dt = datetime.fromtimestamp(timestamp)
                            values.append(dt.strftime(meta['format']))
                        except Exception as e:
                            logger.warning(f"Error reconstructing datetime for {col}: {str(e)}")
                            values.append(None)

                    synthetic_df[col] = values
                    synthetic_df.drop(num_col, axis=1, inplace=True)

            for col, meta in self.metadata.items():
                if col not in synthetic_df.columns:
                    if meta.data_type == DataType.STRING and meta.faker_method:
                        synthetic_df[col] = [meta.faker_method(**meta.faker_args) for _ in range(n_samples)]
                    elif meta.data_type == DataType.DATETIME:
                        synthetic_df[col] = self._generate_datetime_column(col, n_samples)
                    else:
                        synthetic_df[col] = None

            return synthetic_df

    def _apply_nans(self, series, meta):
        """Apply NaN values to a series based on metadata"""
        if not meta or not meta.allow_nans or meta.nan_probability <= 0:
            return series

        mask = np.random.random(len(series)) < meta.nan_probability
        if series.dtype.kind in ['i', 'f']:
            series[mask] = np.nan
        elif series.dtype.kind == 'b':
            series[mask] = None
        else:
            series[mask] = None
        return series

    def _evaluate_progress(self, epoch):
        try:
            with torch.no_grad():
                samples = self.generate(100)

            # Log basic statistics
            logger.info(f"\nEpoch {epoch} Evaluation:")

            # Numerical columns
            num_cols = [col for col in samples.columns
                        if col in self.numerical_cols][:3]
            for col in num_cols:
                logger.info(
                    f"{col:<15}: mean={samples[col].mean():>8.3f} | "
                    f"std={samples[col].std():>7.3f} | "
                    f"min={samples[col].min():>7.3f} | "
                    f"max={samples[col].max():>7.3f}"
                )

            # Categorical columns
            cat_cols = [col for col in samples.columns
                        if col in self.categorical_cols][:2]
            for col in cat_cols:
                counts = samples[col].value_counts(normalize=True)
                logger.info(f"{col:<15}: " + " | ".join(
                    f"{k}: {v:.2%}" for k, v in counts.iloc[:3].items()
                ) + ("..." if len(counts) > 3 else ""))

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

    def _generate_datetime_column(self, col, n_samples):
        meta = self.metadata.get(col)
        if not meta or not meta.datetime_format:
            return [None] * n_samples

        try:
            # Generate dates based on the format
            if meta.datetime_min and meta.datetime_max:
                try:
                    start = datetime.strptime(meta.datetime_min, meta.datetime_format)
                    end = datetime.strptime(meta.datetime_max, meta.datetime_format)
                except ValueError:
                    # Fallback if format doesn't match
                    start = datetime(1995, 1, 1)
                    end = datetime(2023, 12, 31)
            else:
                start = datetime(1995, 1, 1)
                end = datetime(2023, 12, 31)

            dates = []
            for _ in range(n_samples):
                try:
                    random_date = start + (end - start) * random()
                    formatted = random_date.strftime(meta.datetime_format)
                    dates.append(formatted)
                except:
                    dates.append(None)

            # Apply NaN values if specified
            if meta.allow_nans:
                mask = np.random.random(n_samples) < meta.nan_probability
                dates = [None if mask[i] else d for i, d in enumerate(dates)]

            return dates
        except Exception as e:
            logger.error(f"Error generating datetime for {col}: {str(e)}")
            return [None] * n_samples

    def _generate_string_column(self, column_name: str, meta: FieldMetadata, n_samples: int):
        """Generate synthetic string data using a Faker method."""
        if meta.faker_method:
            return [meta.faker_method(**meta.faker_args) for _ in range(n_samples)]
        else:
            logger.error("Faker method not specified for column: %s", column_name)
            return ['' for _ in range(n_samples)]
