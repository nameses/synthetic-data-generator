# data_generation/wgan.py
import time

import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from typing import Dict, Optional, List, Union, Tuple
from models.enums import DataType
from models.field_metadata import FieldMetadata
from faker import Faker
import logging
import math
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANConfig:
    """Configuration class for WGAN parameters"""

    def __init__(
            self,
            latent_dim: int = 512,
            batch_size: int = 128,
            n_critic: int = 1,
            gp_weight: float = 10.0,
            phys_weight: float = 0.1,
            g_lr: float = 1e-4,
            d_lr: float = 1e-4,
            g_betas: tuple = (0.5, 0.9),
            d_betas: tuple = (0.5, 0.9),
            patience: int = 25,
            clip_value: float = 0.5,
            use_mixed_precision: bool = True,
            spectral_norm: bool = True,
            residual_blocks: bool = True,
            noise_level: float = 0.05,
            minibatch_std: bool = True,
            feature_matching: bool = True,
            num_epochs_per_stage: int = 10,
            max_epochs: int = 500,
            scheduler_type: str = 'cosine',
            normalization_method: str = 'power',
            constraint_relaxation: float = 0.9,
            progressive_training: bool = True,
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.phys_weight = phys_weight
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_betas = g_betas
        self.d_betas = d_betas
        self.patience = patience
        self.clip_value = clip_value
        self.use_mixed_precision = use_mixed_precision
        self.spectral_norm = spectral_norm
        self.residual_blocks = residual_blocks
        self.progressive_training = progressive_training
        self.noise_level = noise_level
        self.minibatch_std = minibatch_std
        self.feature_matching = feature_matching
        self.num_epochs_per_stage = num_epochs_per_stage
        self.max_epochs = max_epochs
        self.scheduler_type = scheduler_type
        self.normalization_method = normalization_method
        self.constraint_relaxation = constraint_relaxation


def apply_spectral_norm(module, apply_spectral_norm=True):
    """Apply spectral norm to a single module"""
    return nn.utils.spectral_norm(module) if apply_spectral_norm else module


class MinibatchStdDev(nn.Module):
    """Minibatch Standard Deviation Layer for the Discriminator"""

    def __init__(self, averaging='all'):
        super().__init__()
        self.averaging = averaging

    def forward(self, x):
        # [N, C, ...] Input shape
        batch_size = x.shape[0]
        if batch_size < 2:  # Skip if batch is too small
            return x

        # [N, 1, ...] Standard deviation across batch dimension
        std = torch.std(x, dim=0, unbiased=False)

        # Average across feature channels and spatial dimensions
        mean_std = torch.mean(std)
        mean_std = mean_std.expand(batch_size, 1)

        # Append as a new feature map
        return torch.cat([x, mean_std], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, config: GANConfig):
        super().__init__()
        self.config = config

        # First linear layer with spectral norm
        self.linear1 = apply_spectral_norm(
            nn.Linear(in_features, out_features),
            config.spectral_norm
        )
        self.norm1 = nn.LayerNorm(out_features)
        self.activation = nn.LeakyReLU(0.2)

        # Second linear layer with spectral norm
        self.linear2 = apply_spectral_norm(
            nn.Linear(out_features, out_features),
            config.spectral_norm
        )
        self.norm2 = nn.LayerNorm(out_features)

        # Shortcut connection
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.norm2(out)

        # Use addition operator (+) instead of in-place add (+=)
        out = out + self.shortcut(identity)
        out = self.activation(out)

        return out


class WGANGenerator(nn.Module):
    def __init__(self, config: GANConfig, output_dim: int, num_categories: int = 0):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.output_dim = output_dim

        # Input dimension includes latent vector and optional category labels
        in_features = config.latent_dim + num_categories

        # Input layer - smaller initial size for more stable training
        self.input_layer = nn.Sequential(
            apply_spectral_norm(nn.Linear(in_features, 128), config.spectral_norm),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)  # Add dropout for regularization
        )

        # Middle layers with different architectures
        if config.residual_blocks:
            self.middle_layers = nn.Sequential(
                ResidualBlock(128, 256, config),
                ResidualBlock(256, 512, config),
                ResidualBlock(512, 512, config)  # Added extra depth
            )
        else:
            self.middle_layers = nn.Sequential(
                apply_spectral_norm(nn.Linear(128, 256), config.spectral_norm),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                apply_spectral_norm(nn.Linear(256, 512), config.spectral_norm),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )

        # Output layers - gradually reduce size
        self.output_layers = nn.Sequential(
            apply_spectral_norm(nn.Linear(512, 256), config.spectral_norm),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(256, output_dim)
        )

        # Initialize weights for better convergence
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, noise, labels=None):
        # Add small noise to prevent mode collapse
        if self.training and self.config.noise_level > 0:
            noise = noise + torch.randn_like(noise) * self.config.noise_level

        if labels is not None:
            # Add small noise to categorical labels as well
            if self.training and self.config.noise_level > 0:
                label_noise = torch.randn_like(labels) * (self.config.noise_level * 0.1)
                labels = labels + label_noise
            noise = torch.cat([noise, labels], dim=1)

        x = self.input_layer(noise)
        x = self.middle_layers(x)
        return self.output_layers(x)


class WGANDiscriminator(nn.Module):
    def __init__(self, config: GANConfig, input_dim: int, num_categories: int = 0):
        super().__init__()
        self.config = config
        self.feature_matching = config.feature_matching

        # Add input normalization
        self.input_norm = nn.LayerNorm(input_dim + num_categories)

        # Input dimension includes input vector and optional category labels
        in_features = input_dim + num_categories

        # Feature extraction layers
        self.features = nn.Sequential(
            apply_spectral_norm(nn.Linear(in_features, 256), config.spectral_norm),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            apply_spectral_norm(nn.Linear(256, 256), config.spectral_norm),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        # Optional minibatch standard deviation layer
        self.use_minibatch_std = config.minibatch_std
        if self.use_minibatch_std:
            self.minibatch_std = MinibatchStdDev()
            self.post_minibatch = nn.Sequential(
                apply_spectral_norm(nn.Linear(256 + 1, 128), config.spectral_norm),  # +1 for minibatch std
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )
        else:
            self.post_features = nn.Sequential(
                apply_spectral_norm(nn.Linear(256, 128), config.spectral_norm),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )

        # Output layer
        self.output_layer = apply_spectral_norm(nn.Linear(128, 1), config.spectral_norm)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, labels=None):
        if labels is not None:
            inputs = torch.cat([inputs, labels], dim=1)

        inputs = self.input_norm(inputs)

        features = self.features(inputs)

        if self.use_minibatch_std:
            features = self.minibatch_std(features)
            features = self.post_minibatch(features)
        else:
            features = self.post_features(features)

        output = self.output_layer(features)

        if self.feature_matching and self.training:
            return output, features
        else:
            return output


class WGAN:
    def __init__(self, real_data: pd.DataFrame, metadata: Dict[str, FieldMetadata], config: GANConfig):
        self.real_data = real_data
        self.metadata = metadata
        self.config = config
        self.faker = Faker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {'g_loss': [], 'd_loss': [], 'grad_norms': [], 'phys_loss': []}

        # Enable anomaly detection during development
        # torch.autograd.set_detect_anomaly(True)

        # Identify column types
        self.numerical_cols = [col for col, meta in metadata.items() if
                               meta.data_type in [DataType.INTEGER, DataType.DECIMAL]]
        self.categorical_cols = [col for col, meta in metadata.items() if
                                 meta.data_type == DataType.CATEGORICAL]
        self.boolean_cols = [col for col, meta in metadata.items() if
                             meta.data_type == DataType.BOOLEAN]

        # Store original ranges and expected distributions
        self.num_ranges = {
            col: (meta.min_value, meta.max_value)
            for col, meta in metadata.items()
            if col in self.numerical_cols and meta.min_value is not None and meta.max_value is not None
        }

        # Store boolean column distributions
        self.bool_distributions = {}
        for col in self.boolean_cols:
            if col in real_data.columns:
                true_prop = real_data[col].mean()
                self.bool_distributions[col] = true_prop

        self.preprocess_data()
        self.init_models()

    def preprocess_data(self):
        """Enhanced preprocessing for better distribution preservation"""
        # Choose normalization method based on config
        if self.config.normalization_method == 'power':
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),  # Handle missing values
                ('scaler', sklearn.preprocessing.RobustScaler()),  # Less sensitive to outliers
                ('power', PowerTransformer(method='yeo-johnson'))  # Keep but with safeguards
            ])
        elif self.config.normalization_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            num_pipeline = Pipeline([
                ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=1000)),
                ('robust', RobustScaler())
            ])
        else:  # Default to quantile
            num_pipeline = Pipeline([
                ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=1000)),
                ('scaler', StandardScaler())
            ])

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

        # Fit and transform the data
        self.processed_data = self.preprocessor.fit_transform(self.real_data)

        # Get dimensions
        self.num_numerical = len(self.numerical_cols)
        self.num_categorical = 0
        if 'cat' in self.preprocessor.named_transformers_:
            self.num_categorical = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out().shape[0]

        # Separate numerical and categorical data
        if self.num_numerical > 0:
            self.X_num = torch.FloatTensor(self.processed_data[:, :self.num_numerical])
        else:
            self.X_num = torch.FloatTensor(0)

        if self.num_categorical > 0:
            self.X_cat = torch.FloatTensor(
                self.processed_data[:, self.num_numerical:])
        else:
            self.X_cat = None

        # Create dataset and loader
        dataset = TensorDataset(self.X_num, self.X_cat) if self.X_cat is not None else TensorDataset(self.X_num)
        self.loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        # Calculate mean and std for each numerical column to use in physics constraints
        self.column_stats = {}
        for i, col in enumerate(self.numerical_cols):
            if self.num_numerical > 0:
                col_data = self.X_num[:, i].numpy()
                self.column_stats[col] = {
                    'mean': float(np.mean(col_data)),
                    'std': float(np.std(col_data)),
                    'min': float(np.min(col_data)),
                    'max': float(np.max(col_data)),
                    'median': float(np.median(col_data)),
                    'q1': float(np.percentile(col_data, 25)),
                    'q3': float(np.percentile(col_data, 75))
                }

        logger.info(
            f"Preprocessed data with {self.num_numerical} numerical features and {self.num_categorical} categorical features")

        if self.numerical_cols:
            self.numerical_cols = [col for col, meta in self.metadata.items()
                                   if meta.data_type in [DataType.INTEGER, DataType.DECIMAL]]
            logger.info(f"Numerical columns: {self.numerical_cols}")

        if self.categorical_cols:
            self.categorical_cols = [col for col, meta in self.metadata.items()
                                     if meta.data_type == DataType.CATEGORICAL]
            logger.info(f"Categorical columns: {self.categorical_cols}")

    def init_models(self):
        """Initialize models with improved architecture"""
        self.generator = WGANGenerator(
            config=self.config,
            output_dim=self.num_numerical,
            num_categories=self.num_categorical
        )

        self.discriminator = WGANDiscriminator(
            config=self.config,
            input_dim=self.num_numerical,
            num_categories=self.num_categorical
        )

        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Setup optimizers with lower learning rates
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

        # Setup learning rate schedulers
        if self.config.scheduler_type == 'cosine':
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_g, T_max=self.config.max_epochs, eta_min=self.config.g_lr * 0.1
            )
            self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_d, T_max=self.config.max_epochs, eta_min=self.config.d_lr * 0.1
            )
        elif self.config.scheduler_type == 'reduce':
            self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_g, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
            )
            self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_d, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
            )
        else:  # Default to StepLR
            self.scheduler_g = optim.lr_scheduler.StepLR(self.opt_g, step_size=30, gamma=0.5)
            self.scheduler_d = optim.lr_scheduler.StepLR(self.opt_d, step_size=30, gamma=0.5)

        # Mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_mixed_precision)

        # Check generator parameters
        logger.info("Generator parameter checks:")
        for name, param in self.generator.named_parameters():
            logger.info(f"{name}: requires_grad={param.requires_grad}")

        logger.info(f"Models initialized on {self.device}")

    def apply_physics_constraints(self, synthetic):
        """Apply constraints without any inplace operations"""
        # Start with a clone to avoid inplace operations
        result = synthetic.clone()

        # Create a dictionary for easier reference
        synthetic_dict = {col: result[:, i].clone() for i, col in enumerate(self.numerical_cols)}

        # Apply constraints column by column with debugging
        for i, col in enumerate(self.numerical_cols):
            if col not in self.num_ranges:
                continue

            min_val, max_val = self.num_ranges[col]
            col_data = result[:, i]

            # Debug before applying constraints
            if torch.isnan(col_data).any():
                logger.warning(f"NaN detected in {col} before constraints")

            # Soft constraints with gradient flow
            if min_val is not None:
                result[:, i] = col_data + F.softplus(min_val - col_data, beta=1)
            if max_val is not None:
                result[:, i] = col_data - F.softplus(col_data - max_val, beta=1)

            # Debug after applying constraints
            if torch.isnan(result[:, i]).any():
                logger.warning(f"NaN detected in {col} after constraints")

        # Apply relational constraints
        for col, meta in self.metadata.items():
            if col not in self.numerical_cols:
                continue

            col_idx = self.numerical_cols.index(col)

            # Process each constraint defined in metadata
            for constraint in getattr(meta, 'constraints', []):
                other_col = constraint.get('other_column')
                if other_col not in synthetic_dict:
                    continue

                other_idx = self.numerical_cols.index(other_col)

                if constraint['type'] == 'greater_than':
                    margin = constraint.get('margin', 0.0)
                    # Get the current values
                    current_val = synthetic_dict[col]
                    other_val = synthetic_dict[other_col]

                    # Calculate violation amount
                    violation = F.relu(other_val - current_val + margin)

                    # Apply correction with gradual adjustment to avoid sharp changes
                    adjustment = violation * 0.9  # Apply 90% of the needed adjustment
                    corrected = current_val + adjustment

                    # Update tensors
                    result[:, col_idx] = corrected
                    synthetic_dict[col] = corrected

                # Handle less_than constraint
                elif constraint['type'] == 'less_than':
                    margin = constraint.get('margin', 0.0)
                    # Get the current values
                    current_val = synthetic_dict[col]
                    other_val = synthetic_dict[other_col]

                    # Calculate violation amount
                    violation = F.relu(current_val - other_val + margin)

                    # Apply correction with gradual adjustment
                    adjustment = violation * 0.9  # Apply 90% of the needed adjustment
                    corrected = current_val - adjustment

                    # Update tensors
                    result[:, col_idx] = corrected
                    synthetic_dict[col] = corrected

                # Handle min_ratio constraint
                elif constraint['type'] == 'min_ratio':
                    ratio = constraint.get('ratio', 1.0)
                    # Ensure col >= other_col * ratio
                    current_val = synthetic_dict[col]
                    other_val = synthetic_dict[other_col]

                    # Target minimum value
                    min_target = other_val * ratio

                    # Calculate violation
                    violation = F.relu(min_target - current_val)

                    # Apply correction
                    corrected = current_val + violation

                    # Update tensors
                    result[:, col_idx] = corrected
                    synthetic_dict[col] = corrected

                # Handle max_ratio constraint
                elif constraint['type'] == 'max_ratio':
                    ratio = constraint.get('ratio', 1.0)
                    # Ensure col <= other_col * ratio
                    current_val = synthetic_dict[col]
                    other_val = synthetic_dict[other_col]

                    # Target maximum value
                    max_target = other_val * ratio

                    # Calculate violation
                    violation = F.relu(current_val - max_target)

                    # Apply correction
                    corrected = current_val - violation

                    # Update tensors
                    result[:, col_idx] = corrected
                    synthetic_dict[col] = corrected

        return result

    def calculate_physics_loss(self, synthetic):
        """Enhanced physics-based loss calculation"""
        loss = torch.tensor(0.0, device=self.device)

        # Dictionary for easier access
        synthetic_dict = {col: synthetic[:, i] for i, col in enumerate(self.numerical_cols)}

        # Range violations penalties - softer approach
        for i, col in enumerate(self.numerical_cols):
            if col in self.num_ranges:
                min_val, max_val = self.num_ranges[col]
                if min_val is not None:
                    loss += torch.sigmoid((min_val - synthetic[:, i]) * 5).mean()
                if max_val is not None:
                    loss += torch.sigmoid((synthetic[:, i] - max_val) * 5).mean()

        # Distribution matching penalties
        for i, col in enumerate(self.numerical_cols):
            if col in self.column_stats:
                stats = self.column_stats[col]
                # Encourage values to stay within statistical norms
                mean_dev = torch.abs(synthetic[:, i].mean() - stats['mean'])
                std_dev = torch.abs(synthetic[:, i].std() - stats['std'])
                loss += mean_dev * 2.0  # Penalize mean deviation
                loss += std_dev * 2.0  # Penalize std deviation

        # Relational constraint penalties
        for col, meta in self.metadata.items():
            if col not in self.numerical_cols:
                continue

            for constraint in getattr(meta, 'constraints', []):
                other_col = constraint.get('other_column')
                if other_col not in synthetic_dict:
                    continue

                # Handle greater_than constraint
                if constraint['type'] == 'greater_than':
                    margin = constraint.get('margin', 0)
                    violation = F.relu(synthetic_dict[other_col] - synthetic_dict[col] + margin)
                    loss += (violation ** 2).mean() * 5  # Stronger quadratic penalty

                # Handle less_than constraint
                elif constraint['type'] == 'less_than':
                    margin = constraint.get('margin', 0)
                    violation = F.relu(synthetic_dict[col] - synthetic_dict[other_col] + margin)
                    loss += (violation ** 2).mean() * 5  # Stronger quadratic penalty

                # Handle min_ratio constraint
                elif constraint['type'] == 'min_ratio':
                    ratio = constraint.get('ratio', 1.0)
                    # Ensure col >= other_col * ratio
                    violation = F.relu(synthetic_dict[other_col] * ratio - synthetic_dict[col])
                    loss += (violation ** 2).mean() * 5  # Stronger quadratic penalty

                # Handle max_ratio constraint
                elif constraint['type'] == 'max_ratio':
                    ratio = constraint.get('ratio', 1.0)
                    # Ensure col <= other_col * ratio
                    violation = F.relu(synthetic_dict[col] - synthetic_dict[other_col] * ratio)
                    loss += (violation ** 2).mean() * 5  # Stronger quadratic penalty

        # Apply configurable weight
        return loss * self.config.phys_weight

    def wasserstein_loss(self, real_pred, fake_pred):
        """Standard Wasserstein loss"""
        return fake_pred.mean() - real_pred.mean()

    def gradient_penalty(self, real_data, fake_data, labels=None):
        """Improved gradient penalty calculation"""
        batch_size = real_data.size(0)

        # Create random interpolation factors for each sample in the batch
        alpha = torch.rand(batch_size, 1, device=self.device)

        # Interpolate between real and fake data
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # Calculate discriminator output for interpolated data
        if labels is not None:
            disc_interpolates = self.discriminator(interpolates, labels)
        else:
            disc_interpolates = self.discriminator(interpolates)

        # Extract the output if the discriminator returns a tuple
        if isinstance(disc_interpolates, tuple):
            disc_interpolates = disc_interpolates[0]

        # Create gradient outputs
        grad_outputs = torch.ones_like(disc_interpolates, device=self.device, requires_grad=False)

        # Calculate gradients with respect to inputs
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Reshape and calculate gradient norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Calculate and return penalty
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty * self.config.gp_weight

    def feature_matching_loss(self, real_features, fake_features):
        """Feature matching loss to help prevent mode collapse"""
        # Calculate mean across batch dimension for each feature
        real_mean = real_features.mean(0)
        fake_mean = fake_features.mean(0)

        # Mean squared error between feature means
        return F.mse_loss(fake_mean, real_mean) * 10.0  # Scale the loss appropriately

    def train_step(self, real_num, real_cat, epoch, progress):
        """Single training step with improved stability measures"""
        batch_size = real_num.size(0)

        # Move data to device
        real_num = real_num.to(self.device)
        if real_cat is not None:
            real_cat = real_cat.to(self.device)

        # Calculate global step (total batches processed across all epochs)
        global_step = epoch * len(self.loader) + progress

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.opt_d.zero_grad()

        # Get random noise and generate fake data
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_num = self.generator(noise, real_cat)

        #fake_num = self.apply_physics_constraints(fake_num)

        # Add small noise to real and fake data for stability
        real_num_noise = real_num + torch.randn_like(real_num) * 0.01
        fake_num_noise = fake_num + torch.randn_like(fake_num) * 0.01

        # Calculate discriminator outputs
        if self.config.feature_matching:
            d_real, real_features = self.discriminator(real_num_noise, real_cat)
            d_fake, fake_features = self.discriminator(fake_num_noise.detach(), real_cat)
        else:
            d_real = self.discriminator(real_num_noise, real_cat)
            d_fake = self.discriminator(fake_num_noise.detach(), real_cat)

        # Wasserstein discriminator loss
        loss_d = self.wasserstein_loss(d_real, d_fake)

        # Gradient penalty
        gp = self.gradient_penalty(real_num_noise, fake_num_noise.detach(), real_cat)

        # Total discriminator loss
        d_loss = loss_d + gp

        # Backward and optimize
        d_loss.backward()

        # Gradient clipping for discriminator
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)

        d_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            max_norm=self.config.clip_value
        )
        self.opt_d.step()

        # ---------------------
        # Train Generator
        # ---------------------
        # Determine if this step should update the generator (every n_critic steps)
        generator_update = (global_step % self.config.n_critic == 0)

        # Initialize generator metrics
        g_loss = torch.tensor(0.0, device=self.device)
        phys_loss = torch.tensor(0.0, device=self.device)
        total_g_loss = torch.tensor(0.0, device=self.device)
        g_grad_norm = torch.tensor(0.0, device=self.device)
        fm_loss = torch.tensor(0.0, device=self.device)

        if generator_update:
            self.opt_g.zero_grad()

            # Generate new fake data for generator update
            noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_num = self.generator(noise, real_cat)

            if torch.isnan(fake_num).any():
                logger.error("NaN detected in generator output!")
                logger.error(f"NaN positions: {torch.isnan(fake_num).nonzero()}")
                fake_num = torch.nan_to_num(fake_num, nan=0.0)

            # Apply constraints - this creates a new tensor
            #fake_num = self.apply_physics_constraints(fake_num)

            # Apply physics-based constraints and calculate physics loss
            #phys_loss = self.calculate_physics_loss(fake_num)

            # Get discriminator predictions for generator loss
            if self.config.feature_matching:
                g_fake, fake_features = self.discriminator(fake_num, real_cat)
                _, real_features = self.discriminator(real_num, real_cat)
                # Add feature matching loss
                fm_loss = self.feature_matching_loss(real_features, fake_features)
            else:
                g_fake = self.discriminator(fake_num, real_cat)
                fm_loss = torch.tensor(0.0, device=self.device)

            # Standard generator loss (minimize -D(G(z)))
            g_loss = -g_fake.mean()

            # Total generator loss with physics constraints and feature matching
            total_g_loss = g_loss + phys_loss + fm_loss

            # Check for gradients
            if total_g_loss.grad_fn is None:
                logger.error("Generator has no gradients! Check architecture")
                logger.error(f"g_fake grad_fn: {g_fake.grad_fn}")
                logger.error(f"fake_num grad_fn: {fake_num.grad_fn}")
                logger.warning(f"g_loss: {g_loss.item():.4f}, phys_loss: {phys_loss.item():.4f}, fm_loss: {fm_loss.item():.4f}")
                logger.warning(f"fake_num requires_grad: {fake_num.requires_grad}")
                logger.warning(f"generator parameters require grad: {any(p.requires_grad for p in self.generator.parameters())}")

            # Backward and optimize
            total_g_loss.backward()

            # Gradient clipping for generator
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)

            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=self.config.clip_value
            )
            self.opt_g.step()

        return {
            'd_loss': d_loss.item(),
            'g_loss': total_g_loss.item(),
            'phys_loss': phys_loss.item() if isinstance(phys_loss, torch.Tensor) else 0.0,
            'wasserstein': loss_d.item(),
            'gp': gp.item(),
            'feature_matching': fm_loss.item() if isinstance(fm_loss, torch.Tensor) else 0.0,
            'd_grad': d_grad_norm.item(),
            'g_grad': g_grad_norm.item() if generator_update else 0.0
        }

    def progressive_train(self, epochs_per_stage=None):
        """Progressive training approach with curriculum learning"""
        if epochs_per_stage is None:
            epochs_per_stage = self.config.num_epochs_per_stage

        total_stages = 3
        logger.info(f"Starting progressive training with {total_stages} stages, {epochs_per_stage} epochs per stage")

        # Store original config values
        original_phys_weight = self.config.phys_weight

        # First stage: Train with higher noise and lower physics weight
        self.config.phys_weight = original_phys_weight * 0.5
        self.config.noise_level = 0.1
        logger.info(f"Stage 1: Higher noise (0.1), reduced physics weight ({self.config.phys_weight:.4f})")
        stage1_history = self.train(epochs_per_stage)

        # Second stage: Medium noise, normal physics weight
        self.config.phys_weight = original_phys_weight
        self.config.noise_level = 0.05
        logger.info(f"Stage 2: Medium noise (0.05), normal physics weight ({self.config.phys_weight:.4f})")
        stage2_history = self.train(epochs_per_stage)

        # Third stage: Low noise, high physics weight for refinement
        self.config.phys_weight = original_phys_weight * 1.5
        self.config.noise_level = 0.02
        logger.info(f"Stage 3: Low noise (0.02), increased physics weight ({self.config.phys_weight:.4f})")
        stage3_history = self.train(epochs_per_stage)

        # Combine histories
        combined_history = {}
        for key in stage1_history:
            combined_history[key] = stage1_history[key] + stage2_history[key] + stage3_history[key]

        # Reset config to original values
        self.config.phys_weight = original_phys_weight
        self.config.noise_level = 0.05

        return combined_history

    def train(self, epochs=None):
        """Optimized training loop with configurable parameters"""
        torch.autograd.set_detect_anomaly(True)
        try:
            if epochs is None:
                epochs = self.config.max_epochs

            # Calculate total batches per epoch
            total_batches = len(self.loader)
            logger.info(f"Starting training with {total_batches} batches per epoch")

            # Training metrics tracking
            metrics_history = {
                'g_loss': [],
                'd_loss': [],
                'phys_loss': [],
                'wasserstein': [],
                'gp': [],
                'feature_matching': [],
                'd_grad': [],
                'g_grad': []
            }

            # Early stopping variables
            best_loss = float('inf')
            no_improve_epochs = 0

            for epoch in range(epochs):
                epoch_metrics = {key: 0.0 for key in metrics_history}
                epoch_batches = 0
                start_time = time.time()
                generator_updates = 0  # Track generator updates

                # Set models to training mode
                self.generator.train()
                self.discriminator.train()

                for batch_idx, data in enumerate(self.loader):
                    # Data handling - either tuple of (num, cat) or just num
                    if len(data) == 2:
                        real_num, real_cat = data
                    else:
                        real_num = data[0]
                        real_cat = None

                    # Skip small batches
                    if real_num.size(0) < 2:
                        continue

                    # Execute training step
                    step_metrics = self.train_step(real_num, real_cat, epoch, batch_idx)

                    # Check if generator was updated (non-zero g_grad indicates an update)
                    if step_metrics['g_grad'] > 0:
                        generator_updates += 1

                    # Update epoch metrics
                    for key in step_metrics:
                        epoch_metrics[key] += step_metrics[key]
                    epoch_batches += 1

                    # Logging (reduced frequency)
                    if batch_idx % 20 == 0 or batch_idx == total_batches - 1:
                        # Format loss info
                        log_str = (
                            f"Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{total_batches} | "
                            f"G Loss: {step_metrics['g_loss']:.4f} | "
                            f"D Loss: {step_metrics['d_loss']:.4f} | "
                            f"Phys Loss: {step_metrics['phys_loss']:.4f} | "
                            f"G Grad: {step_metrics['g_grad']:.2f} | "
                            f"D Grad: {step_metrics['d_grad']:.2f} | "
                            f"W-dist: {-step_metrics['wasserstein']:.4f}"
                        )
                        logger.info(log_str)

                    # Check for NaN values and stop if needed
                    if any(torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))
                           for k, v in step_metrics.items() if k.endswith('loss')):
                        logger.warning("NaN or Inf detected in loss values. Stopping training.")
                        return metrics_history

                # Epoch completion - calculate averages
                if epoch_batches > 0:
                    for key in epoch_metrics:
                        epoch_metrics[key] /= epoch_batches
                        metrics_history[key].append(epoch_metrics[key])

                # Calculate epoch time
                epoch_time = time.time() - start_time

                # Logging for epoch completion
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} Complete | "
                    f"Time: {epoch_time:.2f}s | "
                    f"Avg G Loss: {epoch_metrics['g_loss']:.4f} | "
                    f"Avg D Loss: {epoch_metrics['d_loss']:.4f} | "
                    f"Avg Phys Loss: {epoch_metrics['phys_loss']:.4f} | "
                    f"Avg W-dist: {-epoch_metrics['wasserstein']:.4f} | "
                    f"Generator Updates: {generator_updates}/{epoch_batches} ({generator_updates / epoch_batches * 100:.1f}%)"
                )

                # Evaluate generator quality
                if (epoch + 1) % 5 == 0:
                    self._evaluate_generator(epoch + 1)

                # Learning rate scheduling
                if self.config.scheduler_type == 'cosine':
                    self.scheduler_g.step()
                    self.scheduler_d.step()

                    # Log new learning rates
                    new_lr_g = self.scheduler_g.get_last_lr()[0]
                    new_lr_d = self.scheduler_d.get_last_lr()[0]
                    logger.info(f"Learning rates: G={new_lr_g:.2e}, D={new_lr_d:.2e}")

                elif self.config.scheduler_type == 'reduce':
                    # Use average losses to step schedulers
                    self.scheduler_g.step(epoch_metrics['g_loss'])
                    self.scheduler_d.step(epoch_metrics['d_loss'])

                else:  # Step scheduler
                    self.scheduler_g.step()
                    self.scheduler_d.step()

                # Early stopping check
                current_loss = epoch_metrics['g_loss'] + epoch_metrics['phys_loss']
                if current_loss < best_loss - 0.01:  # Improvement threshold
                    best_loss = current_loss
                    no_improve_epochs = 0
                    # Save best model
                    # self._save_checkpoint('best_model.pt')
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= self.config.patience:
                        logger.info(f"Early stopping after {epoch + 1} epochs without improvement")
                        break

                # Check other stopping conditions
                if self._check_early_stopping(metrics_history, epoch):
                    break

            return metrics_history
        finally:
            # Disable anomaly detection when done
            torch.autograd.set_detect_anomaly(False)

    def _evaluate_generator(self, epoch):
        """Evaluate generator quality with some basic metrics"""
        # Generate a small batch of samples
        n_samples = 500
        self.generator.eval()

        with torch.no_grad():
            noise = torch.randn(n_samples, self.config.latent_dim, device=self.device)

            # Sample random categorical data if needed
            if self.num_categorical > 0 and self.X_cat is not None:
                idx = torch.randint(0, len(self.X_cat), (n_samples,))
                cat_samples = self.X_cat[idx].to(self.device)
            else:
                cat_samples = None

            # Generate samples
            fake_num = self.generator(noise, cat_samples)
            fake_num = self.apply_physics_constraints(fake_num)

            # Move to CPU for evaluation
            fake_num = fake_num.cpu().numpy()

            # Calculate some basic stats for a few columns
            for i, col in enumerate(self.numerical_cols[:min(3, len(self.numerical_cols))]):
                col_data = fake_num[:, i]
                logger.info(
                    f"Column {col}: "
                    f"Mean={np.mean(col_data):.4f}, "
                    f"Std={np.std(col_data):.4f}, "
                    f"Min={np.min(col_data):.4f}, "
                    f"Max={np.max(col_data):.4f}"
                )

        self.generator.train()

    def _check_early_stopping(self, history, epoch):
        """Check for early stopping conditions"""
        if epoch < self.config.patience:
            return False

        # Check for NaN/inf
        if np.isnan(history['g_loss'][-1]) or np.isinf(history['g_loss'][-1]):
            logger.warning("NaN/Inf detected, stopping training")
            return True

        # Check loss divergence - stop if losses are too high
        if history['d_loss'][-1] < -200 or history['g_loss'][-1] > 200:
            logger.warning("Loss divergence detected, stopping training")
            return True

        # Check for plateaued physics loss
        if epoch > self.config.patience:
            recent_phys = history['phys_loss'][-self.config.patience:]
            if all(abs(recent_phys[i] - recent_phys[i - 1]) < 0.001 for i in range(1, len(recent_phys))):
                logger.info("Physics loss plateaued, stopping training")
                return True

        return False

    def generate(self, n_samples):
        """Generate synthetic samples with enhanced post-processing"""
        self.generator.eval()

        with torch.no_grad():
            # Process in smaller batches for memory efficiency
            batch_size = min(n_samples, 512)
            all_synthetic_num = []

            # Generate batches
            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)

                # Generate noise with a bit of structure to prevent mode collapse
                noise = torch.randn(current_batch_size, self.config.latent_dim, device=self.device)

                # Add some structure to noise (optional)
                if current_batch_size > 1:
                    # Add slight correlation to help with diversity
                    corr_factor = 0.1
                    shared_noise = torch.randn(1, self.config.latent_dim, device=self.device)
                    noise = noise * (1 - corr_factor) + shared_noise * corr_factor

                # Sample categorical data if needed
                if self.num_categorical > 0 and self.X_cat is not None:
                    # Sample with replacement from real categorical data
                    idx = torch.randint(0, len(self.X_cat), (current_batch_size,))
                    cat_samples = self.X_cat[idx].to(self.device)
                else:
                    cat_samples = None

                # Generate synthetic numerical data
                synthetic_num = self.generator(noise, cat_samples)

                # Apply constraints
                synthetic_num = self.apply_physics_constraints(synthetic_num)

                # Add to collection
                all_synthetic_num.append(synthetic_num.cpu().numpy())

            # Combine all batches
            synthetic_num = np.vstack(all_synthetic_num)

            # After generating synthetic_num, add validation
            logger.info(f"Generated synthetic data shape: {synthetic_num.shape}")
            logger.info(f"Column means: {np.nanmean(synthetic_num, axis=0)}")

            # Create dataframe with numerical columns
            if 'num' in self.preprocessor.named_transformers_:
                # Inverse transform numerical data
                num_pipeline = self.preprocessor.named_transformers_['num']

                # Clip extreme values before inverse transform
                synthetic_num = np.clip(synthetic_num, -5, 5)

                try:
                    inverted_num_data = num_pipeline.inverse_transform(synthetic_num)

                    # Handle any remaining invalid values
                    inverted_num_data = np.nan_to_num(inverted_num_data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Apply final range constraints
                    for i, col in enumerate(self.numerical_cols):
                        if col in self.num_ranges:
                            min_val, max_val = self.num_ranges[col]
                            if min_val is not None:
                                inverted_num_data[:, i] = np.maximum(inverted_num_data[:, i], min_val)
                            if max_val is not None:
                                inverted_num_data[:, i] = np.minimum(inverted_num_data[:, i], max_val)

                    synthetic_df = pd.DataFrame(inverted_num_data, columns=self.numerical_cols)
                except Exception as e:
                    logger.error(f"Inverse transform failed: {str(e)}")
                    logger.error(f"Input stats - min: {np.min(synthetic_num)}, max: {np.max(synthetic_num)}")
                    logger.error(f"NaN count: {np.isnan(synthetic_num).sum()}")
                    raise

                # Check for NaN/inf values
                # if np.isnan(inverted_num_data).any():
                #    logger.warning("NaN values detected in inverted data")
                # if np.isinf(inverted_num_data).any():
                #    logger.warning("Inf values detected in inverted data")

                #synthetic_df = pd.DataFrame(inverted_num_data, columns=self.numerical_cols)
            else:
                synthetic_df = pd.DataFrame(columns=self.numerical_cols)

            # Handle categorical data
            if self.num_categorical > 0 and self.X_cat is not None and 'cat' in self.preprocessor.named_transformers_:
                # Sample categorical data using frequency-based approach
                synthetic_cat = self._generate_categorical_data(n_samples)

                for col, values in synthetic_cat.items():
                    synthetic_df[col] = values

            # Add validation checks (Post-generation validation)
            for col in self.numerical_cols:
                if col in synthetic_df.columns:
                    nan_count = synthetic_df[col].isna().sum()
                    if nan_count > 0:
                        logger.warning(f"Column {col} has {nan_count} NaN values - filling with median")
                        synthetic_df[col] = synthetic_df[col].fillna(synthetic_df[col].median())

            # Add boolean columns with proper distribution
            for col in self.boolean_cols:
                if col in self.bool_distributions:
                    true_prob = self.bool_distributions[col]
                    synthetic_df[col] = np.random.random(n_samples) < true_prob

            # Add other fields like dates and strings
            for col, meta in self.metadata.items():
                if col in synthetic_df.columns:
                    continue

                if meta.data_type == DataType.DATE_TIME:
                    synthetic_df[col] = self._generate_datetime_column(col, n_samples)
                elif meta.data_type == DataType.STRING:
                    synthetic_df[col] = self._generate_string_column(meta, n_samples)

            # Apply final consistency checks
            synthetic_df = self._post_process_constraints(synthetic_df)

            return synthetic_df

    def _generate_categorical_data(self, n_samples):
        """Generate categorical data preserving distributions from original data"""
        synthetic_cat = {}

        if 'cat' not in self.preprocessor.named_transformers_:
            return synthetic_cat

        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']

        # Process each categorical column
        for i, col in enumerate(self.categorical_cols):
            # Get original categories and their frequencies
            categories = cat_encoder.categories_[i]

            # Count occurrences in original data
            col_idx = self.real_data.columns.get_loc(col)
            value_counts = self.real_data.iloc[:, col_idx].value_counts(normalize=True)

            # Generate values based on frequencies
            synthetic_values = np.random.choice(
                categories,
                size=n_samples,
                p=[value_counts.get(cat, 0) for cat in categories]
            )

            synthetic_cat[col] = synthetic_values

        return synthetic_cat

    def _generate_datetime_column(self, col, n_samples):
        """Generate datetime column with realistic distribution"""
        if col not in self.real_data.columns:
            # Default date range if column not in original data
            return pd.date_range(
                start='2020-01-01',
                end='2023-01-01',
                periods=n_samples
            )

        # Get min and max dates from original data
        min_date = self.real_data[col].min()
        max_date = self.real_data[col].max()

        # Calculate date range in seconds
        min_ts = min_date.timestamp()
        max_ts = max_date.timestamp()

        # Generate timestamps with slight preference for recent dates
        alpha, beta = 2, 5  # Beta distribution shape parameters
        rand_values = np.random.beta(alpha, beta, n_samples)
        timestamps = min_ts + rand_values * (max_ts - min_ts)

        # Convert to datetime
        return pd.to_datetime(timestamps, unit='s')

    def _generate_string_column(self, meta, n_samples):
        """Generate string column based on metadata"""
        if meta.fake_strategy:
            # Use specified faker strategy
            if meta.string_format:
                return [getattr(self.faker, meta.fake_strategy)(meta.string_format) for _ in range(n_samples)]
            else:
                return [getattr(self.faker, meta.fake_strategy)() for _ in range(n_samples)]
        elif meta.custom_faker:
            # Use custom faker function
            return [meta.custom_faker() for _ in range(n_samples)]
        else:
            # Default to email if no strategy specified
            return [self.faker.email() for _ in range(n_samples)]

    def _post_process_constraints(self, df):
        """Apply final post-processing to ensure constraints are met"""
        # Apply range constraints
        for col, (min_val, max_val) in self.num_ranges.items():
            if col in df.columns:
                if min_val is not None:
                    df[col] = np.maximum(df[col], min_val)
                if max_val is not None:
                    df[col] = np.minimum(df[col], max_val)

        # Apply relational constraints
        for col, meta in self.metadata.items():
            if col not in df.columns:
                continue

            for constraint in getattr(meta, 'constraints', []):
                other_col = constraint.get('other_column')
                if other_col not in df.columns:
                    continue

                if constraint['type'] == 'greater_than':
                    margin = constraint.get('margin', 0)
                    violation_mask = df[col] <= df[other_col] - margin
                    if violation_mask.any():
                        # Fix violations by adjusting the column value
                        df.loc[violation_mask, col] = df.loc[violation_mask, other_col] - margin + 0.01

                elif constraint['type'] == 'less_than':
                    margin = constraint.get('margin', 0)
                    violation_mask = df[col] >= df[other_col] + margin
                    if violation_mask.any():
                        # Fix violations by adjusting the column value
                        df.loc[violation_mask, col] = df.loc[violation_mask, other_col] + margin - 0.01

                elif constraint['type'] == 'min_ratio':
                    ratio = constraint.get('ratio', 1.0)
                    violation_mask = df[col] < df[other_col] * ratio
                    if violation_mask.any():
                        # Fix violations
                        df.loc[violation_mask, col] = df.loc[violation_mask, other_col] * ratio * 1.01

                elif constraint['type'] == 'max_ratio':
                    ratio = constraint.get('ratio', 1.0)
                    violation_mask = df[col] > df[other_col] * ratio
                    if violation_mask.any():
                        # Fix violations
                        df.loc[violation_mask, col] = df.loc[violation_mask, other_col] * ratio * 0.99

        return df