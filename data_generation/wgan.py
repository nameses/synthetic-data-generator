# data_generation/wgan.py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, Optional, List
from models.enums import DataType
from models.field_metadata import FieldMetadata
from faker import Faker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GANConfig:
    """Configuration class for WGAN parameters"""
    def __init__(
            self,
            latent_dim: int = 256,
            batch_size: int = 128,
            n_critic: int = 5,
            gp_weight: float = 10.0,
            phys_weight: float = 1.0,
            g_lr: float = 5e-5,
            d_lr: float = 5e-5,
            g_betas: tuple = (0.5, 0.9),
            d_betas: tuple = (0.5, 0.9),
            patience: int = 25,
            clip_value: float = 0.5,
            use_mixed_precision: bool = True,
            spectral_norm: bool = True,
            residual_blocks: bool = True
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


def apply_spectral_norm(module, apply_spectral_norm=True):
    """Apply spectral norm to a single module"""
    return nn.utils.spectral_norm(module) if apply_spectral_norm else module


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

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.linear2(out)
        out = self.norm2(out)

        out = out + self.shortcut(identity)
        out = self.activation(out)

        return out


class EnhancedWGANGenerator(nn.Module):
    def __init__(self, config: GANConfig, output_dim: int, num_categories: int = 0):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.output_dim = output_dim

        # Input dimension includes latent vector and optional category labels
        in_features = config.latent_dim + num_categories

        # Input layer
        self.input_layer = nn.Sequential(
            apply_spectral_norm(nn.Linear(in_features, 256), config.spectral_norm),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2)
        )

        # Middle layers (either residual or standard)
        if config.residual_blocks:
            self.middle_layers = nn.Sequential(
                ResidualBlock(256, 512, config),
                ResidualBlock(512, 1024, config)
            )
        else:
            self.middle_layers = nn.Sequential(
                apply_spectral_norm(nn.Linear(256, 512), config.spectral_norm),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                apply_spectral_norm(nn.Linear(512, 1024), config.spectral_norm),
                nn.LayerNorm(1024),
                nn.LeakyReLU(0.2)
            )

        # Output layers
        self.output_layers = nn.Sequential(
            apply_spectral_norm(nn.Linear(1024, 512), config.spectral_norm),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim)  # No spectral norm on final output layer
        )

    def forward(self, noise, labels=None):
        if labels is not None:
            noise = torch.cat([noise, labels], dim=1)

        x = self.input_layer(noise)
        x = self.middle_layers(x)
        return self.output_layers(x)


class EnhancedWGANDiscriminator(nn.Module):
    def __init__(self, config: GANConfig, input_dim: int, num_categories: int = 0):
        super().__init__()
        self.config = config

        # Input dimension includes input vector and optional category labels
        in_features = input_dim + num_categories

        # Sequential model with proper spectral normalization
        self.main = nn.Sequential(
            apply_spectral_norm(nn.Linear(in_features, 512), config.spectral_norm),
            nn.LeakyReLU(0.2),
            apply_spectral_norm(nn.Linear(512, 512), config.spectral_norm),
            nn.LeakyReLU(0.2),
            apply_spectral_norm(nn.Linear(512, 256), config.spectral_norm),
            nn.LeakyReLU(0.2),
            apply_spectral_norm(nn.Linear(256, 1), config.spectral_norm)
        )

    def forward(self, inputs, labels=None):
        if labels is not None:
            inputs = torch.cat([inputs, labels], dim=1)
        return self.main(inputs)


class PhysicsInformedWGAN:
    def __init__(self, real_data: pd.DataFrame, metadata: Dict[str, FieldMetadata], config: GANConfig):
        self.real_data = real_data
        self.metadata = metadata
        self.config = config
        self.faker = Faker()

        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

        # Identify column types
        self.numerical_cols = [col for col, meta in metadata.items() if
                               meta.data_type in [DataType.INTEGER, DataType.DECIMAL]]
        self.categorical_cols = [col for col, meta in metadata.items() if
                                 meta.data_type == DataType.CATEGORICAL]

        # Store original ranges
        self.num_ranges = {
            col: (meta.min_value, meta.max_value)
            for col, meta in metadata.items()
            if col in self.numerical_cols and meta.min_value is not None and meta.max_value is not None
        }

        self.preprocess_data()
        self.init_models()

    def preprocess_data(self):
        """Preprocess data for WGAN training"""
        num_pipeline = Pipeline([
            ('quantile', QuantileTransformer(output_distribution='normal')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.numerical_cols),
            ('cat', cat_pipeline, self.categorical_cols)
        ], remainder='drop')

        self.processed_data = self.preprocessor.fit_transform(self.real_data)
        self.num_numerical = len(self.numerical_cols)
        self.num_categorical = \
            self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out().shape[0]

        self.X_num = torch.FloatTensor(self.processed_data[:, :self.num_numerical])
        self.X_cat = torch.FloatTensor(
            self.processed_data[:, self.num_numerical:]) if self.num_categorical > 0 else None

        dataset = TensorDataset(self.X_num, self.X_cat) if self.X_cat is not None else TensorDataset(self.X_num)
        self.loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def init_models(self):
        """Initialize models with proper weight initialization"""

        def weights_init(m):
            if isinstance(m, nn.Linear):
                # Use more stable initialization with smaller values
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.generator = EnhancedWGANGenerator(
            config=self.config,
            output_dim=self.num_numerical,
            num_categories=self.num_categorical
        ).apply(weights_init)

        self.discriminator = EnhancedWGANDiscriminator(
            config=self.config,
            input_dim=self.num_numerical,
            num_categories=self.num_categorical
        ).apply(weights_init)

        # Use separate optimizers with reduced learning rates for stability
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.g_lr * 0.1,  # Reduce learning rate for stability
            betas=self.config.g_betas
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.d_lr * 0.1,  # Reduce learning rate for stability
            betas=self.config.d_betas
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_mixed_precision)

    def apply_physics_constraints(self, synthetic):
        """Apply constraints without any inplace operations"""
        constrained = synthetic.clone()

        # Create a new tensor that we'll build up with constrained values
        result = synthetic.clone()

        # Range constraints
        for i, col in enumerate(self.numerical_cols):
            if col in self.num_ranges:
                min_val, max_val = self.num_ranges[col]
                result[:, i] = torch.clamp(constrained[:, i], min_val, max_val)

        # Relational constraints from metadata
        synthetic_dict = {col: constrained[:, i] for i, col in enumerate(self.numerical_cols)}

        for col, meta in self.metadata.items():
            if col not in self.numerical_cols:
                continue

            for constraint in getattr(meta, 'constraints', []):
                other_col = constraint.get('other_column')
                if other_col not in synthetic_dict:
                    continue

                col_idx = self.numerical_cols.index(col)
                other_idx = self.numerical_cols.index(other_col)

                if constraint['type'] == 'greater_than':
                    margin = constraint.get('margin', 0)
                    new_col = torch.maximum(
                        constrained[:, col_idx],
                        synthetic_dict[other_col] - margin
                    )
                    result[:, col_idx] = new_col
                elif constraint['type'] == 'less_than':
                    margin = constraint.get('margin', 0)
                    new_col = torch.minimum(
                        constrained[:, col_idx],
                        synthetic_dict[other_col] + margin
                    )
                    result[:, col_idx] = new_col

        return result

    def calculate_physics_loss(self, synthetic):
        """Calculate physics loss without inplace ops"""
        loss = torch.tensor(0.0, device=self.device)

        # Range violations
        for i, col in enumerate(self.numerical_cols):
            meta = self.metadata.get(col)
            if meta and meta.data_type in [DataType.INTEGER, DataType.DECIMAL]:
                if meta.min_value is not None:
                    loss += F.relu(meta.min_value - synthetic[:, i]).mean()
                if meta.max_value is not None:
                    loss += F.relu(synthetic[:, i] - meta.max_value).mean()

        # Relational constraints
        synthetic_dict = {col: synthetic[:, i] for i, col in enumerate(self.numerical_cols)}

        for col, meta in self.metadata.items():
            if col not in self.numerical_cols:
                continue

            for constraint in getattr(meta, 'constraints', []):
                other_col = constraint.get('other_column')
                if other_col not in synthetic_dict:
                    continue

                if constraint['type'] == 'greater_than':
                    margin = constraint.get('margin', 0)
                    loss += F.relu(
                        synthetic_dict[other_col] - synthetic_dict[col] + margin
                    ).mean()
                elif constraint['type'] == 'less_than':
                    margin = constraint.get('margin', 0)
                    loss += F.relu(
                        synthetic_dict[col] - synthetic_dict[other_col] + margin
                    ).mean()

        return loss * self.config.phys_weight  # Apply configured weight

    def train(self, epochs):
        """Optimized training loop with configurable parameters"""
        # Calculate total batches per epoch
        total_batches = len(self.loader)
        logger.info(f"Starting training with {total_batches} batches per epoch")

        # Training metrics tracking
        history = {
            'g_loss': [],
            'd_loss': [],
            'grad_norms': []
        }

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_grad_norms = []
            start_time = time.time()

            for batch_idx, data in enumerate(self.loader):
                # Data to device - handle either tuple of (num, cat) or just num
                if len(data) == 2:
                    real_num, real_cat = data
                    real_num = real_num.to(self.device, non_blocking=True)
                    real_cat = real_cat.to(self.device, non_blocking=True)
                else:
                    real_num = data[0].to(self.device, non_blocking=True)
                    real_cat = None

                batch_size = real_num.size(0)

                # Skip small batches
                if batch_size < 2:
                    continue

                # ---------------------
                # Train Discriminator
                # ---------------------
                d_loss_accum = 0.0
                # Reduced critic iterations for stability
                critic_iters = min(2, self.config.n_critic)

                for _ in range(critic_iters):
                    self.opt_d.zero_grad(set_to_none=True)

                    # Generate fake data with added noise for stability
                    with torch.no_grad():
                        # Smaller noise scale and add small epsilon for numerical stability
                        noise = torch.randn(batch_size, self.config.latent_dim,
                                            device=self.device, dtype=torch.float32) * 0.5 + 1e-8
                        fake_num = self.generator(noise, real_cat)
                        fake_num = self.apply_physics_constraints(fake_num)

                    # Add small noise to real data for stability
                    real_with_noise = real_num + torch.randn_like(real_num) * 1e-4

                    # Calculate Wasserstein distance
                    d_real = self.discriminator(real_with_noise, real_cat)
                    d_fake = self.discriminator(fake_num, real_cat)

                    # Calculate simple WGAN loss without mixed precision for stability
                    loss_d = d_fake.mean() - d_real.mean()

                    # Compute gradient penalty with more stable implementation
                    alpha = torch.rand(batch_size, 1, device=self.device, dtype=torch.float32)
                    interpolates = alpha * real_with_noise + (1 - alpha) * fake_num.detach()
                    interpolates.requires_grad_(True)

                    d_interpolates = self.discriminator(interpolates, real_cat)

                    # Create gradient for penalty calculation
                    grad_outputs = torch.ones_like(d_interpolates)
                    gradients = torch.autograd.grad(
                        outputs=d_interpolates,
                        inputs=interpolates,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True
                    )[0]

                    # Calculate gradient penalty with small epsilon to avoid sqrt(0)
                    gradient_norm = gradients.reshape(batch_size, -1).norm(2, dim=1) + 1e-10
                    gradient_penalty = ((gradient_norm - 1) ** 2).mean() * self.config.gp_weight

                    # Add gradient penalty to discriminator loss
                    loss_d = loss_d + gradient_penalty

                    # Standard backward pass without mixed precision
                    loss_d.backward()

                    # Gradient clipping to avoid exploding gradients
                    d_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        max_norm=self.config.clip_value
                    )

                    # Apply gradients
                    self.opt_d.step()
                    d_loss_accum += loss_d.item()

                # -----------------
                # Train Generator
                # -----------------
                self.opt_g.zero_grad(set_to_none=True)

                # Generate fake data with noise for stability
                noise = torch.randn(batch_size, self.config.latent_dim,
                                    device=self.device, dtype=torch.float32) * 0.5 + 1e-8
                fake_num = self.generator(noise, real_cat)
                fake_num = self.apply_physics_constraints(fake_num)

                # Calculate simple generator loss without mixed precision
                loss_g = -self.discriminator(fake_num, real_cat).mean()

                # Add physics loss if enabled
                if self.config.phys_weight > 0:
                    phys_loss = self.calculate_physics_loss(fake_num)
                    loss_g = loss_g + phys_loss

                # Standard backward pass without mixed precision
                loss_g.backward()

                # Gradient clipping for generator
                g_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    max_norm=self.config.clip_value
                )

                # Apply gradients
                self.opt_g.step()

                # Update metrics
                avg_d_loss = d_loss_accum / critic_iters
                epoch_g_loss += loss_g.item()
                epoch_d_loss += avg_d_loss
                epoch_grad_norms.append((g_grad_norm.item(), d_grad_norm.item()))

                # Logging (reduced frequency)
                if batch_idx % 20 == 0:
                    avg_grad_norms = np.mean(epoch_grad_norms[-min(20, len(epoch_grad_norms)):], axis=0)
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{total_batches} | "
                        f"G Loss: {loss_g.item():.4f} | D Loss: {avg_d_loss:.4f} | "
                        f"G Grad: {avg_grad_norms[0]:.2f} | D Grad: {avg_grad_norms[1]:.2f}"
                    )

                # Check for NaN values and stop if needed
                if torch.isnan(loss_g) or torch.isnan(loss_d):
                    logger.warning("NaN detected in loss values. Stopping training.")
                    return history

            # Epoch completion
            epoch_time = time.time() - start_time
            avg_g_loss = epoch_g_loss / total_batches
            avg_d_loss = epoch_d_loss / total_batches
            avg_grad_norms = np.mean(epoch_grad_norms, axis=0) if epoch_grad_norms else (0, 0)

            history['g_loss'].append(avg_g_loss)
            history['d_loss'].append(avg_d_loss)
            history['grad_norms'].append(avg_grad_norms)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} Complete | "
                f"Time: {epoch_time:.2f}s | "
                f"Avg G Loss: {avg_g_loss:.4f} | "
                f"Avg D Loss: {avg_d_loss:.4f} | "
                f"Avg Grad Norms: G={avg_grad_norms[0]:.2f}, D={avg_grad_norms[1]:.2f}"
            )

            # Learning rate scheduling (slower decay)
            if (epoch + 1) % 10 == 0:
                self._adjust_learning_rate(epoch)

            # Early stopping check
            if self._check_early_stopping(history, epoch):
                break

        return history

    def _adjust_learning_rate(self, epoch):
        """Adjust learning rates more gradually"""
        for param_group in self.opt_g.param_groups:
            param_group['lr'] = self.config.g_lr * 0.1 * (0.95 ** (epoch // 10))
        for param_group in self.opt_d.param_groups:
            param_group['lr'] = self.config.d_lr * 0.1 * (0.95 ** (epoch // 10))
        logger.info(
            f"Learning rates adjusted to G: {self.opt_g.param_groups[0]['lr']:.2e}, D: {self.opt_d.param_groups[0]['lr']:.2e}")

    def _check_early_stopping(self, history, epoch):
        """Check for early stopping conditions based on configuration"""
        if epoch < self.config.patience:
            return False

        # Check for NaN/inf
        if np.isnan(history['g_loss'][-1]) or np.isinf(history['g_loss'][-1]):
            logger.warning("NaN/Inf detected, stopping training")
            return True

        # Check loss divergence
        if (history['d_loss'][-1] < -50) or (history['g_loss'][-1] > 50):
            logger.warning("Loss divergence detected, stopping training")
            return True

        # Check for plateau
        if epoch > self.config.patience:
            recent_g = history['g_loss'][-self.config.patience:]
            if (max(recent_g) - min(recent_g)) < 0.01 * np.mean(recent_g):
                logger.info("Generator loss plateaued, stopping training")
                return True

        return False

    def generate(self, n_samples):
        """Generate synthetic samples with proper post-processing"""
        with torch.no_grad():
            batch_size = min(n_samples, 512)  # Process in smaller batches for memory efficiency
            all_synthetic_num = []

            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)

                noise = torch.randn(current_batch_size, self.config.latent_dim, device=self.device) * 0.5

                cat_samples = None
                if self.num_categorical > 0:
                    idx = torch.randint(0, len(self.X_cat), (current_batch_size,))
                    cat_samples = self.X_cat[idx].to(self.device)

                synthetic_num = self.generator(noise, cat_samples)
                synthetic_num = self.apply_physics_constraints(synthetic_num)
                all_synthetic_num.append(synthetic_num.cpu().numpy())

            # Combine all batches
            synthetic_num = np.vstack(all_synthetic_num)

            # Create the dataframe with numerical columns
            # First inverse transform numerical data
            num_pipeline = self.preprocessor.named_transformers_['num']
            inverted_num_data = num_pipeline.inverse_transform(synthetic_num)
            synthetic_df = pd.DataFrame(inverted_num_data, columns=self.numerical_cols)

            # Handle categorical data
            if self.num_categorical > 0 and self.X_cat is not None:
                # Sample categorical data
                cat_indices = np.random.randint(0, len(self.X_cat), n_samples)
                cat_data = self.X_cat[cat_indices].cpu().numpy()

                # Get the original categories for each categorical column
                cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = cat_encoder.get_feature_names_out(self.categorical_cols)

                start_idx = 0
                for i, col in enumerate(self.categorical_cols):
                    # Get the one-hot encoded values for this column
                    col_indices = [j for j, name in enumerate(cat_features) if name.startswith(f"{col}_")]
                    n_categories = len(col_indices)
                    one_hot_values = cat_data[:, start_idx:start_idx + n_categories]
                    start_idx += n_categories

                    # Get the categories from the encoder
                    categories = cat_encoder.categories_[i]

                    # Assign the most likely category based on the one-hot values
                    cat_idx = np.argmax(one_hot_values, axis=1)
                    synthetic_df[col] = [categories[idx] for idx in cat_idx]

            # Add other fields
            for col, meta in self.metadata.items():
                if col in synthetic_df.columns:
                    continue

                if meta.data_type == DataType.BOOLEAN:
                    synthetic_df[col] = np.random.choice([True, False], size=n_samples)
                elif meta.data_type == DataType.DATE_TIME:
                    synthetic_df[col] = pd.date_range(
                        start=self.real_data[col].min(),
                        end=self.real_data[col].max(),
                        periods=n_samples
                    )
                elif meta.data_type == DataType.STRING:
                    if meta.custom_faker:
                        synthetic_df[col] = [meta.custom_faker() for _ in range(n_samples)]
                    elif meta.fake_strategy:
                        if meta.string_format:
                            synthetic_df[col] = [getattr(self.faker, meta.fake_strategy)(meta.string_format) for _ in
                                                 range(n_samples)]
                        else:
                            synthetic_df[col] = [getattr(self.faker, meta.fake_strategy)() for _ in range(n_samples)]
                    else:
                        synthetic_df[col] = [self.faker.text() for _ in range(n_samples)]

            return synthetic_df