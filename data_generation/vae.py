# Enhanced VAE Model with Normalizing Flows, Attention Mechanisms, and Improved Training
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp


@dataclass
class VAEConfig:
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    architecture: str = "residual"  # Options: "mlp", "residual", "attention"
    use_latent_dropout: bool = True
    dropout_rate: float = 0.1
    batch_size: int = 256
    lr: float = 1e-3
    lr_schedule: str = "cyclic"  # Options: "constant", "cyclic", "reduce_on_plateau"
    epochs: int = 150
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 80
    lambda_mmd: float = 0.1
    lambda_corr: float = 1.0  # Increased from 0.1
    use_normalizing_flows: bool = True
    flow_layers: int = 2
    use_mdn: bool = True  # Use Mixture Density Networks for outputs
    n_mdn_components: int = 10
    use_attention: bool = True
    eval_every: int = 10
    seed: int = 42
    verbose: bool = True
    save_best: bool = True


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch, dim]
        q = self.query(x).unsqueeze(1)  # [batch, 1, dim]
        k = self.key(x).unsqueeze(1)    # [batch, 1, dim]
        v = self.value(x).unsqueeze(1)  # [batch, 1, dim]

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [batch, 1, 1]
        attn_scores = torch.softmax(attn_weights, dim=-1)            # [batch, 1, 1]

        attn_output = torch.bmm(attn_scores, v).squeeze(1)           # [batch, dim]

        return self.out(attn_output) + x  # residual connection



class ResBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.block(x) + x)


class NormalizingFlow(nn.Module):
    """Simple normalizing flow using planar transformations"""

    def __init__(self, dim, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([PlanarFlow(dim) for _ in range(n_layers)])

    def forward(self, z):
        log_det_sum = 0
        for flow in self.layers:
            z, log_det = flow(z)
            log_det_sum += log_det
        return z, log_det_sum


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        # Ensure u·w > -1 for invertibility
        wTu = (self.w @ self.u).unsqueeze(0)
        m_wTu = -1 + torch.log(1 + torch.exp(wTu))
        u_hat = self.u + (m_wTu - wTu) * self.w / (self.w @ self.w + 1e-8)

        # Forward transformation
        zwT = z @ self.w
        f_z = z + u_hat.unsqueeze(0) * torch.tanh(zwT.unsqueeze(1) + self.b)

        # Log determinant
        psi = (1 - torch.tanh(zwT + self.b) ** 2).unsqueeze(1) * self.w
        log_det = torch.log(torch.abs(1 + psi @ u_hat) + 1e-8)

        return f_z, log_det


class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.output_dim = output_dim

        # Each component has mean, variance, and weight
        self.out_layer = nn.Linear(input_dim, n_components * (2 * output_dim + 1))

    def forward(self, x):
        params = self.out_layer(x)

        # Split into mixture components
        mix_params = params.view(-1, self.n_components, 2 * self.output_dim + 1)

        # Extract parameters
        logits = mix_params[:, :, 0]  # shape: [batch_size, n_components]
        means = mix_params[:, :, 1:self.output_dim + 1]  # shape: [batch_size, n_components, output_dim]
        log_vars = mix_params[:, :, self.output_dim + 1:]  # shape: [batch_size, n_components, output_dim]

        # Apply softmax to get mixture weights
        weights = F.softmax(logits, dim=1)

        return weights, means, log_vars

    def sample(self, x):
        weights, means, log_vars = self.forward(x)

        # Sample component indices based on weights
        batch_size = x.size(0)
        component_indices = torch.multinomial(weights, 1).squeeze(1)  # [batch_size]

        # Extract means and variances for the selected components
        batch_indices = torch.arange(batch_size, device=x.device)
        selected_means = means[batch_indices, component_indices]  # [batch_size, output_dim]
        selected_log_vars = log_vars[batch_indices, component_indices]  # [batch_size, output_dim]

        # Sample from the selected Gaussians
        eps = torch.randn_like(selected_means)
        samples = selected_means + eps * torch.exp(0.5 * selected_log_vars)

        return samples

    def log_prob(self, x, target):
        weights, means, log_vars = self.forward(x)

        # Calculate log prob for each component
        vars = torch.exp(log_vars)
        log_probs = -0.5 * ((target.unsqueeze(1) - means) ** 2 / (vars + 1e-8))
        log_probs = log_probs - 0.5 * (log_vars + np.log(2 * np.pi))
        log_probs = log_probs.sum(-1)  # Sum over dimensions

        # Weighted sum of component log probs
        weighted_log_probs = torch.log(weights + 1e-8) + log_probs
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)

        return mixture_log_prob


class VAEPipeline:
    def __init__(self, df: pd.DataFrame, meta: Dict[str, object], cfg: VAEConfig):
        self.cfg = cfg
        self.meta = meta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(cfg.seed)

        self.num_cols = [k for k, v in meta.items() if v.data_type.value in ("int", "decimal")]
        self.cat_cols = [k for k, v in meta.items() if v.data_type.value in ("categorical", "boolean")]

        # Setup scalers for numerical columns
        self.scalers = {}
        for col in self.num_cols:
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
            df[col] = self.scalers[col].fit_transform(df[[col]])

        self.cat_maps = {
            col: {val: idx for idx, val in enumerate(sorted(df[col].astype(str).unique()))}
            for col in self.cat_cols
        }
        self.inverse_cat_maps = {
            col: {idx: val for val, idx in mapping.items()}
            for col, mapping in self.cat_maps.items()
        }

        cont_data = df[self.num_cols].astype(float).values
        cat_data = np.stack([
            df[col].astype(str).map(self.cat_maps[col]).fillna(0).astype(int).values
            for col in self.cat_cols
        ], axis=1) if self.cat_cols else np.zeros((len(df), 0))

        self.X_tensor = torch.tensor(np.hstack(
            [cont_data, pd.get_dummies(df[self.cat_cols]).values if self.cat_cols else np.zeros((len(df), 0))]),
                                     dtype=torch.float32)
        self.cat_tensor = torch.tensor(cat_data, dtype=torch.long)
        self.cont_tensor = torch.tensor(cont_data, dtype=torch.float32)

        # Store original data statistics
        self.original_corr = np.corrcoef(cont_data.T) if cont_data.shape[1] > 1 else np.array([[1.0]])
        self.original_mean = np.mean(cont_data, axis=0)
        self.original_std = np.std(cont_data, axis=0)

        # Initialize model
        self.model = EnhancedVAE(
            input_dim=self.X_tensor.shape[1],
            cont_dim=len(self.num_cols),
            cat_sizes=[len(self.cat_maps[c]) for c in self.cat_cols] if self.cat_cols else [],
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            architecture=cfg.architecture,
            use_latent_dropout=cfg.use_latent_dropout,
            dropout_rate=cfg.dropout_rate,
            use_flows=cfg.use_normalizing_flows,
            flow_layers=cfg.flow_layers,
            use_mdn=cfg.use_mdn,
            n_mdn_components=cfg.n_mdn_components,
            use_attention=cfg.use_attention,
        )
        self.history = {'loss': [], 'val_metrics': []}
        self.best_model_state = None
        self.best_metric = float('inf')

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    def train(self):
        self.model.to(self.device)
        dataset = TensorDataset(self.X_tensor, self.cat_tensor, self.cont_tensor)
        train_size = int(0.9 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        if self.cfg.lr_schedule == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.cfg.lr / 10, max_lr=self.cfg.lr,
                step_size_up=train_size // self.cfg.batch_size, mode='triangular2'
            )
        elif self.cfg.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        else:
            scheduler = None

        for epoch in range(1, self.cfg.epochs + 1):
            # Calculate annealed beta value for KL divergence
            beta = self.get_beta_schedule(epoch)

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, beta, epoch)

            # Validate
            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs:
                val_metrics = self.validate(valid_loader, beta)

                if self.cfg.verbose:
                    print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_metrics['val_loss']:.4f} | "
                          f"Corr Diff: {val_metrics['corr_diff']:.4f} | "
                          f"Dist Diff: {val_metrics['dist_diff']:.4f}")

                # Save best model
                if self.cfg.save_best and val_metrics['combined_metric'] < self.best_metric:
                    self.best_metric = val_metrics['combined_metric']
                    self.best_model_state = self.model.state_dict()
                    if self.cfg.verbose:
                        print(f"New best model saved! Metric: {self.best_metric:.4f}")

                self.history['val_metrics'].append(val_metrics)

                # Handle reduce on plateau scheduler
                if self.cfg.lr_schedule == "reduce_on_plateau":
                    scheduler.step(val_metrics['val_loss'])
            else:
                if self.cfg.verbose:
                    print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f}")

            self.history['loss'].append(train_loss)

        # Restore best model
        if self.cfg.save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.cfg.verbose:
                print(f"Restored best model with metric: {self.best_metric:.4f}")

    def get_beta_schedule(self, epoch):
        cycle_length = self.cfg.beta_warmup_epochs
        return self.cfg.beta_end * (1 - np.cos(np.pi * (epoch % cycle_length) / cycle_length)) / 2

    def train_epoch(self, dataloader, optimizer, scheduler, beta, epoch):
        self.model.train()
        total_loss = 0.0
        for xb, catb, contb in dataloader:
            xb, catb, contb = xb.to(self.device), catb.to(self.device), contb.to(self.device)
            optimizer.zero_grad()

            # Forward pass
            tau = max(0.5, np.exp(-epoch / 30))
            outputs = self.model(xb, tau=tau)

            # Calculate loss
            loss = self.model.loss_function(
                outputs,
                contb,
                catb,
                beta=beta,
                lambda_mmd=self.cfg.lambda_mmd,
                lambda_corr=self.cfg.lambda_corr
            )

            # Backward pass
            loss['loss'].backward()
            optimizer.step()

            if scheduler is not None and self.cfg.lr_schedule == "cyclic":
                scheduler.step()

            total_loss += loss['loss'].item()

        return total_loss / len(dataloader)

    def validate(self, dataloader, beta):
        self.model.eval()
        val_loss = 0.0
        all_real = []
        all_fake = []

        with torch.no_grad():
            for xb, catb, contb in dataloader:
                xb, catb, contb = xb.to(self.device), catb.to(self.device), contb.to(self.device)

                # Forward pass
                outputs = self.model(xb)

                # Calculate loss
                loss = self.model.loss_function(
                    outputs,
                    contb,
                    catb,
                    beta=beta,
                    lambda_mmd=self.cfg.lambda_mmd,
                    lambda_corr=self.cfg.lambda_corr
                )

                val_loss += loss['loss'].item()

                # Collect real and generated data for metrics
                all_real.append(contb.cpu().numpy())
                all_fake.append(outputs['cont_out'].cpu().numpy())

        # Concatenate all batches
        all_real = np.concatenate(all_real, axis=0)
        all_fake = np.concatenate(all_fake, axis=0)

        # Calculate metrics
        metrics = self.calculate_metrics(all_real, all_fake)
        metrics['val_loss'] = val_loss / len(dataloader)
        metrics['combined_metric'] = metrics['val_loss'] + metrics['corr_diff'] + metrics['dist_diff']

        return metrics

    def calculate_metrics(self, real_data, fake_data):
        # Correlation difference
        real_corr = np.corrcoef(real_data.T) if real_data.shape[1] > 1 else np.array([[1.0]])
        fake_corr = np.corrcoef(fake_data.T) if fake_data.shape[1] > 1 else np.array([[1.0]])
        corr_diff = np.mean(np.abs(real_corr - fake_corr))

        # Distribution difference using KS test
        dist_diff = 0
        for i in range(real_data.shape[1]):
            ks_stat, _ = ks_2samp(real_data[:, i], fake_data[:, i])
            dist_diff += ks_stat
        dist_diff /= real_data.shape[1]

        return {
            'corr_diff': corr_diff,
            'dist_diff': dist_diff
        }

    def generate(self, n: int) -> pd.DataFrame:
        """Generate synthetic data using the trained model"""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.cfg.latent_dim).to(self.device)
            generated = self.model.generate(z)
            cont_out = generated['cont_out'].cpu().numpy()
            cat_indices = [x.cpu().numpy() for x in generated['cat_indices']]

        # Create dataframe with generated data
        data = {}

        # Process continuous columns
        for i, col in enumerate(self.num_cols):
            # Inverse transform the scaled values
            col_data = cont_out[:, i].reshape(-1, 1)
            data[col] = self.scalers[col].inverse_transform(col_data).flatten()

        # Process categorical columns
        for i, col in enumerate(self.cat_cols):
            if i < len(cat_indices):  # Check index is valid
                data[col] = [self.inverse_cat_maps[col].get(int(x), list(self.inverse_cat_maps[col].values())[0])
                             for x in cat_indices[i]]

        return pd.DataFrame(data)

    def evaluate_model(self, n_samples=1000):
        """Evaluate model quality with multiple metrics and plots"""
        # Generate synthetic data
        synthetic_df = self.generate(n_samples)

        # Original data (unscaled for visualization)
        original_data = {}
        for col in self.num_cols:
            original_data[col] = self.scalers[col].inverse_transform(
                self.cont_tensor[:, self.num_cols.index(col)].numpy().reshape(-1, 1)
            ).flatten()
        original_df = pd.DataFrame(original_data)

        # Calculate metrics
        metrics = self._calculate_evaluation_metrics(original_df, synthetic_df)

        # Create plots
        fig, axes = plt.subplots(len(self.num_cols), 2, figsize=(12, 4 * len(self.num_cols)))
        if len(self.num_cols) == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(self.num_cols):
            # Distribution plot
            axes[i, 0].hist(original_df[col], bins=30, alpha=0.5, label='Original')
            axes[i, 0].hist(synthetic_df[col], bins=30, alpha=0.5, label='Synthetic')
            axes[i, 0].set_title(f'Distribution: {col}')
            axes[i, 0].legend()

            # QQ plot
            if i == 0 and len(self.num_cols) > 1:
                # Correlation heatmap difference
                orig_corr = original_df.corr().values
                syn_corr = synthetic_df.corr().values
                diff = np.abs(orig_corr - syn_corr)
                im = axes[i, 1].imshow(diff, cmap='hot', vmin=0, vmax=1)
                axes[i, 1].set_title('Correlation Difference')
                plt.colorbar(im, ax=axes[i, 1])
                axes[i, 1].set_xticks(range(len(self.num_cols)))
                axes[i, 1].set_yticks(range(len(self.num_cols)))
                axes[i, 1].set_xticklabels(self.num_cols, rotation=45)
                axes[i, 1].set_yticklabels(self.num_cols)
            else:
                # Scatter plot for pairs or empty plot as fallback
                if len(self.num_cols) > 1 and i < len(self.num_cols) - 1:
                    next_col = self.num_cols[i + 1]
                    axes[i, 1].scatter(original_df[col], original_df[next_col],
                                       alpha=0.5, s=10, label='Original')
                    axes[i, 1].scatter(synthetic_df[col], synthetic_df[next_col],
                                       alpha=0.5, s=10, label='Synthetic')
                    axes[i, 1].set_title(f'Scatter: {col} vs {next_col}')
                    axes[i, 1].legend()
                else:
                    axes[i, 1].set_visible(False)

        plt.tight_layout()
        return metrics, fig

    def _calculate_evaluation_metrics(self, original_df, synthetic_df):
        """Calculate detailed evaluation metrics"""
        metrics = {
            'column_ks_tests': {},
            'correlation_difference': None,
        }

        # Calculate KS test for each column
        for col in self.num_cols:
            ks_stat, p_value = ks_2samp(original_df[col], synthetic_df[col])
            metrics['column_ks_tests'][col] = {
                'ks_stat': ks_stat,
                'p_value': p_value
            }

        # Calculate correlation difference if we have multiple columns
        if len(self.num_cols) > 1:
            orig_corr = original_df.corr().values
            syn_corr = synthetic_df.corr().values
            corr_diff = np.mean(np.abs(orig_corr - syn_corr))
            metrics['correlation_difference'] = corr_diff

        return metrics


class EnhancedVAE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            cont_dim: int,
            cat_sizes: list[int],
            latent_dim: int,
            hidden_dims: list[int],
            architecture: str = "mlp",
            use_latent_dropout: bool = True,
            dropout_rate: float = 0.2,
            use_flows: bool = True,
            flow_layers: int = 3,
            use_mdn: bool = True,
            n_mdn_components: int = 5,
            use_attention: bool = True,
    ):
        super().__init__()
        self.architecture = architecture
        self.use_latent_dropout = use_latent_dropout
        self.latent_dropout = nn.Dropout(dropout_rate)
        self.cont_dim = cont_dim
        self.cat_sizes = cat_sizes
        self.latent_dim = latent_dim
        self.use_flows = use_flows
        self.use_mdn = use_mdn
        self.n_mdn_components = n_mdn_components
        self.use_attention = use_attention

        # Encoder
        enc_layers, prev = [], input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(nn.LeakyReLU(0.2))

            if architecture == "residual":
                enc_layers.append(ResBlock(h, dropout_rate))

            if use_attention and h >= 64:  # Only use attention on larger hidden dims
                enc_layers.append(SelfAttention(h))

            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        # Normalizing Flow (if used)
        if use_flows:
            self.flow = NormalizingFlow(latent_dim, flow_layers)

        # Decoder
        dec_layers, prev = [], latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.BatchNorm1d(h))
            dec_layers.append(nn.LeakyReLU(0.2))

            if architecture == "residual":
                dec_layers.append(ResBlock(h, dropout_rate))

            if use_attention and h >= 64:  # Only use attention on larger hidden dims
                dec_layers.append(SelfAttention(h))

            prev = h
        self.decoder_trunk = nn.Sequential(*dec_layers)

        # Output heads
        if use_mdn:
            self.cont_mdn = MixtureDensityNetwork(prev, cont_dim, n_mdn_components)
        else:
            self.cont_head = nn.Linear(prev, cont_dim)

        self.cat_heads = nn.ModuleList([nn.Linear(prev, sz) for sz in cat_sizes]) if cat_sizes else None

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if self.use_latent_dropout:
            z = self.latent_dropout(z)
        return z

    def transform_latent(self, z):
        """Apply normalizing flows to transform latent space"""
        if self.use_flows:
            z_transformed, log_det = self.flow(z)
            return z_transformed, log_det
        return z, torch.zeros(z.size(0), device=z.device)

    def decode(self, z, use_gumbel=False, tau=1.0):
        h = self.decoder_trunk(z)

        # Generate continuous outputs
        if self.use_mdn:
            # For training we'll need the distribution params, just return means for inference
            weights, means, log_vars = self.cont_mdn.forward(h)
            if self.use_mdn:
                weights, means, log_vars = self.cont_mdn.forward(h)
                cont_out = torch.sum(weights.unsqueeze(-1) * means, dim=1)  # Use expectation
                cont_params = {'weights': weights, 'means': means, 'log_vars': log_vars}
        else:
            cont_out = self.cont_head(h)
            cont_params = None

        # Generate categorical outputs
        if self.cat_heads is not None:
            if use_gumbel:
                cat_logits = [self.gumbel_softmax(head(h), tau) for head in self.cat_heads]
            else:
                cat_logits = [head(h) for head in self.cat_heads]
        else:
            cat_logits = []

        return cont_out, cont_params, cat_logits

    def gumbel_softmax(self, logits, tau):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        return F.softmax((logits + gumbel_noise) / tau, dim=-1)

    def forward(self, x, tau=1.0):
        # Encode
        mu, logvar = self.encode(x)

        # Sample from latent distribution
        z = self.reparam(mu, logvar)

        # Transform latent space with normalizing flows
        z_transformed, log_det = self.transform_latent(z)

        # Decode
        cont_out, cont_params, cat_logits = self.decode(z_transformed, use_gumbel=True, tau=tau)

        return {
            'cont_out': cont_out,
            'cont_params': cont_params,
            'cat_logits': cat_logits,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'z_transformed': z_transformed,
            'log_det': log_det
        }

    def generate(self, z=None, n=None):
        """Generate samples from the model"""
        if z is None:
            assert n is not None, "Either provide latent vectors z or specify n"
            z = torch.randn(n, self.latent_dim).to(next(self.parameters()).device)

        cont_out, _, cat_logits = self.decode(z, use_gumbel=False)

        # For categorical variables, take argmax to select categories
        cat_indices = [torch.argmax(logit, dim=1) for logit in cat_logits] if cat_logits else []

        return {
            'cont_out': cont_out,
            'cat_indices': cat_indices
        }

    def loss_function(self, outputs, real_cont, real_cat, beta, lambda_mmd=1.0, lambda_corr=0.1):
        cont_out = outputs['cont_out']
        cont_params = outputs['cont_params']
        cat_logits = outputs['cat_logits']
        mu = outputs['mu']
        logvar = outputs['logvar']
        z = outputs['z']
        z_transformed = outputs['z_transformed']
        log_det = outputs['log_det']

        # Continuous reconstruction loss
        if self.use_mdn:
            cont_log_prob = self.cont_mdn.log_prob(self.decoder_trunk(z_transformed), real_cont)
            recon_loss_cont = -cont_log_prob.mean()
        else:
            recon_loss_cont = F.mse_loss(cont_out, real_cont, reduction='mean')

        # Categorical reconstruction loss
        recon_loss_cat = 0.0
        if cat_logits:
            for i, logits in enumerate(cat_logits):
                class_count = logits.size(1)
                weight = 1.0 / class_count
                recon_loss_cat += weight * F.cross_entropy(logits, real_cat[:, i], reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()

        # MMD penalty
        prior_z = torch.randn_like(z)
        mmd_loss = self.mmd(z, prior_z)

        # Correlation penalty
        corr_loss = self.correlation_penalty(real_cont, cont_out)

        loss = recon_loss_cont + recon_loss_cat + beta * kl_loss + lambda_mmd * mmd_loss + lambda_corr * corr_loss

        return {
            'loss': loss,
            'recon_loss_cont': recon_loss_cont,
            'recon_loss_cat': recon_loss_cat,
            'kl_loss': kl_loss,
            'mmd_loss': mmd_loss,
            'corr_loss': corr_loss,
        }

    def mmd(self, x, y):
        """Maximum Mean Discrepancy (MMD) using RBF kernel"""
        xx, yy, xy = self.rbf_kernel(x, x), self.rbf_kernel(y, y), self.rbf_kernel(x, y)
        return xx.mean() + yy.mean() - 2 * xy.mean()

    def rbf_kernel(self, x, y, sigma=1.0):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))

    def correlation_penalty(self, real, generated):
        """Penalize difference in correlation matrices"""
        real_corr = self.corrcoef(real)
        gen_corr = self.corrcoef(generated)
        return torch.mean(torch.abs(real_corr - gen_corr))

    def corrcoef(self, x):
        x = x - x.mean(dim=0)
        cov = x.T @ x / (x.size(0) - 1)
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        return cov / (std[:, None] * std[None, :] + 1e-8)
