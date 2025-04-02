import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import numpy._core.multiarray
import time
from tqdm import tqdm  # Add this import at the top

from models.enums import DataType
from models.field_metadata import FieldMetadata

# Initialize CUDA properly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.init()
    # Create initial context by allocating a small tensor
    _ = torch.empty(1, device=device)
torch.backends.cudnn.benchmark = True

# Allowlist necessary globals for safe loading
torch.serialization.add_safe_globals([np._core.multiarray.scalar])


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_size, cat_feature_sizes=None):
        super().__init__()
        self.input_dim = input_dim
        self.cat_feature_sizes = cat_feature_sizes if cat_feature_sizes else []

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_size),
            *[self._make_residual_block(hidden_size) for _ in range(2)]
        ).to(device)
        self.main.apply(init_weights)

        self.cont_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_dim - sum(self.cat_feature_sizes)),
            nn.Tanh()
        ).to(device)
        self.cont_out.apply(init_weights)

        if self.cat_feature_sizes:
            self.cat_outs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_size, size),
                    nn.Softmax(dim=1)
                ).to(device) for size in self.cat_feature_sizes
            ])
            for cat_out in self.cat_outs:
                cat_out.apply(init_weights)

    def _make_residual_block(self, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(device)
        x = self.main(x)
        cont_output = self.cont_out(x)
        if self.cat_feature_sizes:
            cat_outputs = [cat_out(x) for cat_out in self.cat_outs]
            return torch.cat([cont_output] + cat_outputs, dim=1)
        return cont_output


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

        self.model = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_size)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_size),
            *[self._make_residual_block(hidden_size) for _ in range(2)],
            spectral_norm(nn.Linear(hidden_size, 1))
        ).to(device)
        self.model.apply(init_weights)

    def _make_residual_block(self, hidden_size):
        return nn.Sequential(
            spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_size),
            spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(device)
        return self.model(x)


def generate_synthetic_data_wgan(
        df: pd.DataFrame,
        metadata: Dict[str, FieldMetadata],
        synthetic_size: int = None,
        batch_size: int = 512,
        hidden_size: int = 512,
        epochs: int = 500,
        lr: float = 0.0001,
        critic_updates: int = 5,
        gp_weight: float = 10.0,
        temperature: float = 1.0,
        patience: int = 20,
        checkpoint_dir: str = "wgan_checkpoints",
        verbose: bool = True
) -> pd.DataFrame:
    """Complete WGAN implementation with enhanced stability and error handling"""

    # Setup logging and device
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    def log(message, level=logging.INFO):
        if verbose: logger.log(level, message)

    # 1. Enhanced Data Preprocessing
    log("Preprocessing data with robust transformations...")
    processed_df, column_info, cat_feature_sizes, training_cols = preprocess_data_enhanced(df, metadata)
    input_dim = len(training_cols)
    log(f"Processed data shape: {processed_df.shape}, Input dimension: {input_dim}")

    # 2. Model Initialization
    generator = Generator(input_dim, hidden_size, cat_feature_sizes).to(device)
    critic = Critic(input_dim, hidden_size).to(device)
    log(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    log(f"Critic parameters: {sum(p.numel() for p in critic.parameters())}")

    # 3. Optimizers and Gradient Handling
    generator_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    critic_optim = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # 4. Data Loading with Dimension Validation
    tensor_data = torch.tensor(processed_df.values, dtype=torch.float32)
    if len(tensor_data.shape) == 1:
        tensor_data = tensor_data.unsqueeze(1)
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    log(f"Data loader initialized with {len(data_loader)} batches")

    # 5. Training Loop with Enhanced Stability
    best_loss = float('inf')
    patience_counter = 0
    min_epochs = 50  # Minimum epochs before early stopping checks

    log(f"Starting training for {epochs} epochs...")
    for epoch in tqdm(range(epochs), disable=not verbose):
        epoch_gen_loss = 0
        epoch_crit_loss = 0

        # Dynamic temperature scheduling
        current_temp = max(0.5, temperature * (1 - epoch / (epochs * 1.2)))

        for batch_idx, (real_batch,) in enumerate(data_loader):
            real_data = real_batch.to(device, non_blocking=True)

            # Validate dimensions before processing
            if real_data.shape[1] != input_dim:
                log(f"Dimension mismatch in batch {batch_idx}: expected {input_dim}, got {real_data.shape[1]}",
                    logging.WARNING)
                continue

            # Train Critic with Gradient Penalty
            critic_optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Generate synthetic data
                noise = torch.randn(real_data.size(0), input_dim, device=device) * current_temp
                fake_data = generator(noise)

                # Gradient Penalty
                alpha = torch.rand(real_data.size(0), 1, device=device)
                interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
                crit_interpolates = critic(interpolates)

                gradients = torch.autograd.grad(
                    outputs=crit_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(crit_interpolates),
                    create_graph=True,
                    retain_graph=True)[0]

                gradient_penalty = ((gradients.norm(2, dim=1) - 1).pow(2).mean() * gp_weight)
                crit_loss = -(torch.mean(critic(real_data)) - torch.mean(critic(fake_data))) + gradient_penalty

            scaler.scale(crit_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            scaler.step(critic_optim)
            scaler.update()
            epoch_crit_loss += crit_loss.item()

            # Train Generator (less frequently)
            if batch_idx % critic_updates == 0:
                generator_optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    noise = torch.randn(real_data.size(0), input_dim, device=device) * current_temp
                    gen_loss = -torch.mean(critic(generator(noise)))

                scaler.scale(gen_loss).backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                scaler.step(generator_optim)
                scaler.update()
                epoch_gen_loss += gen_loss.item()

        # Epoch statistics and early stopping
        avg_gen_loss = epoch_gen_loss / len(data_loader)
        avg_crit_loss = epoch_crit_loss / len(data_loader)

        if epoch > min_epochs:
            current_loss = avg_crit_loss + avg_gen_loss
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                torch.save({
                    'generator': generator.state_dict(),
                    'critic': critic.state_dict(),
                    'column_info': column_info,
                    'training_cols': training_cols
                }, f"{checkpoint_dir}/best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log(f"Early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0:
            log(f"Epoch {epoch}: Critic Loss {avg_crit_loss:.4f}, Generator Loss {avg_gen_loss:.4f}")
            torch.cuda.empty_cache()

    # 6. Enhanced Synthetic Data Generation
    log("Generating final synthetic samples...")
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    synthetic_numerical = []
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for _ in range(0, synthetic_size, batch_size):
            # Mixed noise sampling for better diversity
            gauss = torch.randn(min(batch_size, synthetic_size - len(synthetic_numerical)),
                                input_dim // 2, device=device)
            uniform = torch.rand(min(batch_size, synthetic_size - len(synthetic_numerical)),
                                 input_dim - input_dim // 2, device=device) * 2 - 1
            noise = torch.cat([gauss, uniform], dim=1) * temperature

            synthetic_batch = generator(noise)
            if synthetic_batch.shape[1] != input_dim:
                log("Dimension mismatch in generated samples", logging.WARNING)
                continue

            synthetic_numerical.append(synthetic_batch.cpu().numpy())

    # 7. Robust Post-Processing
    log("Applying final post-processing...")
    result_df = post_process_enhanced(
        np.concatenate(synthetic_numerical)[:synthetic_size],
        checkpoint['column_info'],
        metadata,
        synthetic_size
    )

    # Ensure all columns are present
    for col in metadata:
        if col not in result_df:
            if metadata[col].data_type == DataType.STRING:
                result_df[col] = [metadata[col].get_string_generator()() for _ in range(synthetic_size)]
            else:
                result_df[col] = get_default_values(metadata[col].data_type, synthetic_size)

    log(f"Synthetic generation complete. Final shape: {result_df.shape}")
    return result_df


def post_process_enhanced(synthetic_data, column_info, metadata, size):
    """Robust post-processing with error handling"""
    result_df = pd.DataFrame()
    current_col = 0

    for col_name, meta in metadata.items():
        try:
            if col_name not in column_info:
                # Handle missing columns
                if meta.data_type == DataType.STRING:
                    result_df[col_name] = [meta.get_string_generator()() for _ in range(size)]
                else:
                    result_df[col_name] = get_default_values(meta.data_type, size)
                continue

            info = column_info[col_name]

            if info['type'] == 'numerical':
                # Add controlled noise
                noise = np.random.normal(0, 0.05, size=size)
                scaled = synthetic_data[:, current_col] * info['scale'] + info['min']
                result_df[col_name] = np.clip(scaled + noise, info['min'] - info['scale'], info['max'] + info['scale'])
                current_col += 1

            elif info['type'] == 'categorical':
                # Handle one-hot encoded columns
                probs = synthetic_data[:, current_col:current_col + len(info['values'])]
                result_df[col_name] = [info['values'][i] for i in np.argmax(probs, axis=1)]
                current_col += len(info['values'])

            elif info['type'] == 'boolean':
                result_df[col_name] = synthetic_data[:, current_col] > 0.5
                current_col += 1

            elif info['type'] == 'datetime':
                # Add random time component
                day_seconds = 24 * 60 * 60
                timestamps = synthetic_data[:, current_col] * (info['max_ts'] - info['min_ts']) + info['min_ts']
                timestamps += np.random.uniform(0, day_seconds, size=size)
                result_df[col_name] = pd.to_datetime(timestamps, unit='s')
                current_col += 1

        except Exception as e:
            print(f"Error post-processing column {col_name}: {str(e)}")
            result_df[col_name] = get_default_values(meta.data_type, size)
            continue

    return result_df


def preprocess_data_enhanced(df: pd.DataFrame, metadata: Dict[str, FieldMetadata]):
    """Robust preprocessing with proper dimension handling"""
    processed_df = pd.DataFrame()
    column_info = {}
    cat_feature_sizes = []
    training_cols = []

    for col_name, meta in metadata.items():
        if col_name not in df.columns:
            continue

        try:
            if meta.data_type in [DataType.INTEGER, DataType.DECIMAL]:
                # Numerical columns
                col_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
                if len(col_data) == 0:
                    processed_df[col_name] = 0.0
                    column_info[col_name] = {'type': 'numerical', 'min': 0, 'max': 1}
                else:
                    p5, p95 = col_data.quantile([0.05, 0.95])
                    scale = max(p95 - p5, 1e-8)
                    processed_df[col_name] = (df[col_name].astype(float) - p5) / scale
                    column_info[col_name] = {
                        'type': 'numerical',
                        'min': float(p5),
                        'max': float(p95),
                        'scale': float(scale)
                    }
                training_cols.append(col_name)

            elif meta.data_type == DataType.CATEGORICAL:
                # One-hot encoding for categoricals
                unique_values = df[col_name].astype(str).unique()
                for val in unique_values:
                    new_col = f"{col_name}_{val}"
                    processed_df[new_col] = (df[col_name].astype(str) == val).astype(float)
                    training_cols.append(new_col)

                column_info[col_name] = {
                    'type': 'categorical',
                    'values': unique_values.tolist()
                }
                cat_feature_sizes.append(len(unique_values))

            elif meta.data_type == DataType.BOOLEAN:
                processed_df[col_name] = df[col_name].astype(float)
                column_info[col_name] = {'type': 'boolean'}
                training_cols.append(col_name)

            elif meta.data_type == DataType.DATE_TIME:
                # Safe datetime handling
                dt_series = pd.to_datetime(df[col_name], errors='coerce')
                valid_mask = dt_series.notna()
                if valid_mask.any():
                    timestamps = dt_series[valid_mask].astype(np.int64) // 10 ** 9
                    min_ts, max_ts = timestamps.min(), timestamps.max()
                    processed_df[col_name] = np.zeros(len(df))
                    processed_df.loc[valid_mask, col_name] = (timestamps - min_ts) / (max_ts - min_ts + 1e-8)
                    column_info[col_name] = {
                        'type': 'datetime',
                        'min_ts': float(min_ts),
                        'max_ts': float(max_ts)
                    }
                    training_cols.append(col_name)

        except Exception as e:
            print(f"Error processing {col_name}: {str(e)}")
            continue

    # Ensure consistent dimensions
    processed_df = processed_df.fillna(0).astype(np.float32)
    return processed_df, column_info, cat_feature_sizes, training_cols


def get_default_values(data_type: DataType, size: int):
    if data_type in [DataType.INTEGER, DataType.DECIMAL]:
        return np.zeros(size)
    elif data_type == DataType.CATEGORICAL:
        return np.array([''] * size)
    elif data_type == DataType.BOOLEAN:
        return np.random.choice([True, False], size)
    elif data_type == DataType.DATE_TIME:
        return pd.to_datetime([datetime.now()] * size)
    elif data_type == DataType.STRING:
        return np.array([''] * size)
    return np.array([None] * size)
