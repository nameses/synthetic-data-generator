"""
Improved WGAN‑GP for tabular data
================================
• λ (GP‑weight) lowered to 2.5 because spectral‑norm is used
• ε‑drift penalty keeps critic outputs bounded without mean‑centering
• Numeric transformer clips 0.5 / 99.5 % tails before rank‑gauss
• Adaptive n_critic: 5 steps while |W| < 0.02, otherwise 1
• Separate Wasserstein estimate + losses in the log line
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from faker import Faker
from scipy.stats import norm, rankdata, wasserstein_distance
from sklearn.model_selection import train_test_split
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.enums import DataType
from models.field_metadata import FieldMetadata


# main.py  – after logging.basicConfig(...)
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fh = logging.FileHandler(f"train_{ts}.log", mode="w", encoding="utf‑8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(fh)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_autocast = lambda: torch.amp.autocast("cuda", enabled=True)

# ------------------------------------------------------------------#
# CONFIG
# ------------------------------------------------------------------#


@dataclass(slots=True)
class GanConfig:
    latent_dim: int = 128
    hidden_g: int = 512
    hidden_d: int = 384
    batch_size: int = 256
    n_critic_initial: int = 5
    gp_weight: float = 4.0
    drift_epsilon: float = 5e-4
    g_lr: float = 2e-4
    d_lr: float = 4e-4
    max_epochs: int = 300
    seed: int = 42
    # cosine schedule
    lr_min_ratio: float = 0.05
    # early stop
    patience: int = 200
    # gumbel
    tau_start: float = 2.5
    tau_end: float = 0.25
    val_interval: int = 5
    
    # Advanced transformation options
    use_conditional_gan: bool = True  # Enable conditional GAN for related columns
    advanced_transformers: bool = True  # Enable advanced transformers
    post_process_outliers: bool = True  # Clamp outliers in post-processing
    enhanced_network_capacity: bool = False  # Use larger networks for complex distributions
    generator_depth: int = 3  # Number of hidden layers in generator
    discriminator_depth: int = 3  # Number of hidden layers in discriminator
    additional_g_capacity: int = 256  # Additional capacity for enhanced networks


# ------------------------------------------------------------------#
# TRANSFORMERS  (numeric / datetime)
# ------------------------------------------------------------------#


class _BaseTf:
    def fit(self, s: pd.Series): ...
    def transform(self, s: pd.Series) -> np.ndarray: ...
    def inverse(self, v: np.ndarray) -> pd.Series: ...


class _StdTf(_BaseTf):
    def fit(self, s):
        self.mean, self.std = s.mean(), s.std()
        return self
    def transform(self, s):
        return (s - self.mean) / self.std
    def inverse(self, v):
        return pd.Series(v * self.std + self.mean)


class _MinMaxTf(_BaseTf):
    """Scale to [–1,1] by true min/max—preserves heavy tails."""
    def fit(self, s: pd.Series):
        self.min_, self.max_ = float(s.min()), float(s.max())
        return self
    def transform(self, s: pd.Series) -> np.ndarray:
        r = (s.astype(float) - self.min_) / (self.max_ - self.min_)
        return 2 * r - 1
    def inverse(self, v: np.ndarray) -> pd.Series:
        r = (v + 1) / 2
        raw = r * (self.max_ - self.min_) + self.min_
        return pd.Series(raw)
    
    
class LogTransformer:
    """Log transformer to handle skewed data: log(1+x) with safe bounds handling"""

    def __init__(self):
        self.zeros_mask = None
        # Store original data statistics for safe mapping back
        self.orig_min = None
        self.orig_max = None
        self.orig_median = None
        self.orig_mean = None
        self.orig_percentiles = None
        self.transformed_stats = None

    def fit(self, s: pd.Series) -> 'LogTransformer':
        # Store indices of zeros to preserve them if needed
        if hasattr(s, 'metadata') and getattr(s.metadata, 'preserve_zeros', False):
            self.zeros_mask = (s == 0)
        
        # Store original data statistics before transformation
        valid_data = s.dropna().values
        self.orig_min = float(valid_data.min())
        self.orig_max = float(valid_data.max())
        self.orig_median = float(np.median(valid_data))
        self.orig_mean = float(np.mean(valid_data))
        
        # Store percentiles for better distribution matching
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.orig_percentiles = {p: float(np.percentile(valid_data, p)) for p in percentiles}
        
        # Apply the transformation and store transformed statistics
        transformed = np.log1p(valid_data)
        self.transformed_stats = {
            'min': float(transformed.min()),
            'max': float(transformed.max()),
            'median': float(np.median(transformed)),
            'percentiles': {p: float(np.percentile(transformed, p)) for p in percentiles}
        }
        
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        # Apply log(1+x) transform
        transformed = np.log1p(s.values)
        return transformed

    def inverse(self, v: np.ndarray) -> pd.Series:
        """
        Safely inverse transform log-transformed values, ensuring:
        1. Proper scaling to the original range
        2. No overflow errors
        3. Distribution shape is preserved
        """
        # First clip the incoming values to the transformed range to avoid extreme outliers
        if self.transformed_stats:
            # Add a small buffer to the transformed range
            t_min = self.transformed_stats['min'] - 0.1
            t_max = self.transformed_stats['max'] + 0.1
            v = np.clip(v, t_min, t_max)
        
        # Apply the basic inverse transformation
        raw = np.expm1(v)
        
        # Clip to safe integer bounds if very large values might be created
        if self.orig_max > 1e6:
            # If the original data had large values, be more conservative
            int_max = float(2**31 - 1)  # int32 max
            raw = np.clip(raw, 0, int_max / 2)
        
        # Restore exact zeros if needed
        if self.zeros_mask is not None and len(self.zeros_mask) == len(raw):
            raw[self.zeros_mask] = 0
            
        # Enforce the same range as the original data
        if self.orig_min is not None and self.orig_max is not None:
            raw = np.clip(raw, self.orig_min, self.orig_max)

        return pd.Series(raw)


class YeoJohnsonTransformer:
    """Yeo-Johnson power transformer for handling skewed data without sign restrictions"""

    def __init__(self):
        self.lambda_ = None
        self.zeros_mask = None

    def fit(self, s: pd.Series) -> 'YeoJohnsonTransformer':
        from scipy import stats
        # If your SciPy ≥ 1.3 you can use yeojohnson_normmax directly:
        if hasattr(stats, "yeojohnson_normmax"):
            self.lambda_ = stats.yeojohnson_normmax(s.values)
        else:
            # Fallback: brute‑force search over a grid using yeojohnson_llf
            lmbdas = np.linspace(-2, 2, 401)
            llfs = [stats.yeojohnson_llf(l, s.values) for l in lmbdas]
            self.lambda_ = float(lmbdas[np.argmax(llfs)])
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        from scipy import stats
        transformed = stats.yeojohnson(s.values, lmbda=self.lambda_)
        return transformed

    def inverse(self, v: np.ndarray) -> pd.Series:
        """Inverse Yeo-Johnson transformation"""
        result = np.zeros_like(v, dtype=float)

        # Lambda cases for inverse transform
        if self.lambda_ == 0:
            result = np.expm1(v)
        elif self.lambda_ < 0:
            # x < 0 case
            result = 1 - np.power(-(self.lambda_*v + 1), 1/self.lambda_)
        elif self.lambda_ > 0 and self.lambda_ != 2:
            # x ≥ 0 case
            result = np.power((self.lambda_*v + 1), 1/self.lambda_) - 1
        elif self.lambda_ == 2:
            # Special case
            result = np.expm1(v)

        # Restore exact zeros if needed
        if self.zeros_mask is not None and len(self.zeros_mask) == len(result):
            result[self.zeros_mask] = 0

        return pd.Series(result)


class TransformerChain:
    """Chain of transformers to be applied sequentially"""

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, s: pd.Series) -> 'TransformerChain':
        # Fit each transformer in the chain
        current = s.copy()
        for transformer in self.transformers:
            transformer.fit(current)
            current = pd.Series(transformer.transform(current))
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        # Apply each transformer in sequence
        current = s
        for transformer in self.transformers:
            current = pd.Series(transformer.transform(current))
        return current.values

    def inverse(self, v: np.ndarray) -> pd.Series:
        # Apply inverse transformations in reverse order
        current = v
        for transformer in reversed(self.transformers):
            current = transformer.inverse(current)
        return current


class _ContTf(_BaseTf):
    """Rank‑gauss with light tail‑clipping for stability."""

    def fit(self, s):
        q_low, q_hi = s.quantile([0.0025, 0.9975])
        self.sorted_ = np.sort(s.clip(q_low, q_hi).to_numpy(copy=True))
        return self

    def transform(self, s):
        u = rankdata(s, method="average") / (len(s) + 1)
        return norm.ppf(u)

    def inverse(self, v):
        u = norm.cdf(v).clip(0, 1)
        idx = np.floor(u * (len(self.sorted_) - 1)).astype(int)
        return pd.Series(self.sorted_[idx])


class _DtTf(_ContTf):
    def __init__(self, fmt): self.fmt = fmt
    def fit(self, s):  return super().fit(self._to_sec(s))
    def transform(self, s): return super().transform(self._to_sec(s))
    def inverse(self, v):
        return pd.to_datetime(super().inverse(v), unit="s").dt.strftime(self.fmt)
    def _to_sec(self, s):
        return pd.to_datetime(s, format=self.fmt, errors="coerce").astype("int64") // 10**9


# ------------------------------------------------------------------#
# NETWORK BLOCKS
# ------------------------------------------------------------------#


def lin_sn(i, o): return spectral_norm(nn.Linear(i, o))


class MinibatchStd(nn.Module):
    def forward(self, x):
        std = x.std(0, keepdim=True).mean().expand(x.size(0), 1)
        return torch.cat([x, std], 1)


class _Generator(nn.Module):
    def __init__(self, cfg: GanConfig, num_out: int, cat_dims: List[int], total_steps: int):
        super().__init__()
        self.cfg, self.total_steps = cfg, total_steps
        self.register_buffer("step", torch.zeros(1))

        hid = cfg.hidden_g
        self.backbone = nn.Sequential(
            nn.Linear(cfg.latent_dim + sum(cat_dims), hid),
            nn.LeakyReLU(0.2),
            nn.Linear(hid, hid),
            nn.LeakyReLU(0.2),
            MinibatchStd(),
            nn.Linear(hid + 1, hid),
            nn.LeakyReLU(0.2),
        )
        self.num_head = nn.Linear(hid, num_out) if num_out else None
        self.cat_heads = nn.ModuleList(nn.Linear(hid, k) for k in cat_dims)

    def forward(self, z, cond_vec, hard=False):
        h = self.backbone(torch.cat([z, cond_vec], 1))
        out = [self.num_head(h)] if self.num_head else []
        # cosine‑annealed Gumbel τ
        tau = self.cfg.tau_end + 0.5 * (self.cfg.tau_start - self.cfg.tau_end) * (
            1 + math.cos(math.pi * self.step.item() / self.total_steps)
        )
        for head in self.cat_heads:
            out.append(F.gumbel_softmax(head(h), tau=tau, hard=hard))
        self.step += 1
        return torch.cat(out, 1)


class _Discriminator(nn.Module):
    def __init__(self, inp_dim, cond_dim, cfg: GanConfig):
        super().__init__()
        hid = cfg.hidden_d
        self.net = nn.Sequential(
            lin_sn(inp_dim + cond_dim, hid),
            nn.LeakyReLU(0.2),
            lin_sn(hid, hid),
            nn.LeakyReLU(0.2),
            lin_sn(hid, hid // 2),
            nn.LeakyReLU(0.2),
            lin_sn(hid // 2, 1),
        )

    def forward(self, x, c): return self.net(torch.cat([x, c], 1)).view(-1)


# ------------------------------------------------------------------#
# MAIN DRIVER
# ------------------------------------------------------------------#


class WGAN:
    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        self.cfg, self.meta, self.faker = cfg, meta, Faker()
        _set_seed(cfg.seed)

        self.real_df = real.reset_index(drop=True)

        self.W_smooth = None  # running abs Wasserstein
        self.n_critic = 5  # start conservatively

        # ------------ column groups & transformers --------------
        self.num_cols, self.dt_cols, self.cat_cols, self.str_cols = [], [], [], []
        for c, m in meta.items():
            if m.data_type in {DataType.INTEGER, DataType.DECIMAL}:
                self.num_cols.append(c)
            elif m.data_type is DataType.DATETIME:
                self.dt_cols.append(c)
            elif m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}:
                self.cat_cols.append(c)
            elif m.data_type is DataType.STRING:
                self.str_cols.append(c)

        self.tfs: Dict[str, _BaseTf] = {}; mats = []

        for c in self.num_cols:
            m = self.meta[c]
            if m.transformer:
                # Build a chain of actual transformer instances
                instances = []
                for name in m.transformer:
                    if name == "standard":
                        instances.append(_StdTf())
                    elif name == "minmax":
                        instances.append(_MinMaxTf())
                    elif name == "log":
                        instances.append(LogTransformer())
                    elif name == "yeo-johnson":
                        instances.append(YeoJohnsonTransformer())
                    else:
                        raise ValueError(f"Unknown transformer {name}")
                tf = TransformerChain(instances).fit(real[c])
            else:
                # Default to rank‑gauss
                tf = _ContTf().fit(real[c])
            self.tfs[c] = tf
            mats.append(tf.transform(real[c]))

        for c in self.dt_cols:
            tf = _DtTf(meta[c].datetime_format).fit(real[c]); self.tfs[c] = tf; mats.append(tf.transform(real[c]))

        self.cat_sizes, self.cat_maps = [], {}
        for c in self.cat_cols:
            uniq = sorted(real[c].astype(str).unique())
            self.cat_sizes.append(len(uniq))
            mp = {v: i for i, v in enumerate(uniq)}; self.cat_maps[c] = mp
            mats.append(F.one_hot(torch.tensor(real[c].astype(str).map(mp).values),
                                  num_classes=len(uniq)).T.numpy())

        # --- compute per-col sampling probs for any cat flagged 'empirical' ---
        self.cat_probs: dict[str, np.ndarray] = {}
        for c, size in zip(self.cat_cols, self.cat_sizes):
            m = meta[c]
            if m.sampling == "empirical":
                # map real values → integer indices
                inv = real[c].astype(str).map(self.cat_maps[c])
                counts = inv.value_counts().sort_index()
                self.cat_probs[c] = (counts / counts.sum()).values

        X = torch.tensor(np.vstack(mats).T, dtype=torch.float32)
        self.loader = DataLoader(TensorDataset(X), cfg.batch_size, shuffle=True, drop_last=True)
        LOGGER.info("Real tensor %s", X.shape)

        steps_per_epoch = math.ceil(len(self.loader))
        total_steps = cfg.max_epochs * steps_per_epoch
        cond_dim = sum(self.cat_sizes)

        self.G = _Generator(cfg, len(self.num_cols) + len(self.dt_cols), self.cat_sizes, total_steps).to(DEVICE)
        self.D = _Discriminator(X.shape[1], cond_dim, cfg).to(DEVICE)

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(0.0, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(0.0, 0.9))
        self.sch_g = CosineAnnealingLR(self.opt_g, cfg.max_epochs, eta_min=cfg.g_lr * cfg.lr_min_ratio)
        self.sch_d = CosineAnnealingLR(self.opt_d, cfg.max_epochs, eta_min=cfg.d_lr * cfg.lr_min_ratio)

        self.scaler_g = torch.amp.GradScaler("cuda", enabled=True)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=True)

        self.ema_G = copy.deepcopy(self.G).eval().requires_grad_(False)
        self.ema_beta, self.best_w, self.no_imp = 0.999, -float("inf"), 0
        self.n_critic = cfg.n_critic_initial

    # ------------------------------------------------------------------#
    def fit(self):
        # ── 0) split the full real→tensor dataset into train/val ───────────────
        #    we already built self.loader.dataset.tensors[0] as X_all
        X_all = self.loader.dataset.tensors[0]  # torch.Tensor [N, features]
        N = X_all.size(0)
        idx = np.arange(N)
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=self.cfg.seed)

        # rebuild a loader that only yields the TRAIN split
        X_train = X_all[train_idx]
        self.loader = DataLoader(
            TensorDataset(X_train),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True
        )

        # keep a pandas hold‑out for validation WD
        self.val_df = self.real_df.iloc[val_idx]

        # reset any early‑stop counters if you like
        self.best_w = -float("inf")
        self.no_imp = 0
        self.w_ema = None

        # ── 1) the usual WGAN training loop ────────────────────────────────────
        for epoch in range(1, self.cfg.max_epochs + 1):
            d_sum = g_sum = w_sum = 0.0

            for i, (real_batch,) in enumerate(self.loader, 1):
                real_batch = real_batch.to(DEVICE)
                cond = self._sample_cond(real_batch.size(0)).to(DEVICE)

                # train D
                w_est, d_loss = self._train_d(real_batch, cond)
                w_sum += w_est
                d_sum += d_loss

                # train G every n_critic steps
                if i % self.n_critic == 0:
                    g_loss = self._train_g(real_batch.size(0), cond)
                    g_sum += g_loss.item()

            # adapt n_critic, decay LRs, EMA‑stop as before…
            mean_W = w_sum / len(self.loader)
            self._adjust_n_critic(mean_W)
            self.sch_g.step()
            self.sch_d.step()

            d_mean = d_sum / len(self.loader)
            g_mean = g_sum / max(1, len(self.loader) / self.n_critic)
            lr_d = self.opt_d.param_groups[0]["lr"]
            lr_g = self.opt_g.param_groups[0]["lr"]

            LOGGER.info(
                "Ep %03d | W %.4f | D %.4f | G %.4f | n_c %d | lr_d %.6f | lr_g %.6f",
                epoch, mean_W, d_mean, g_mean, self.n_critic, lr_d, lr_g
            )

            if hasattr(self, "val_df") and epoch % self.cfg.val_interval == 0:
                syn = self.generate(len(self.val_df))
                # numeric features
                for c in self.num_cols:
                    # Use a safer approach for Wasserstein distance calculation
                    try:
                        # Special handling for log-transformed columns which can have extreme ranges
                        if hasattr(self.meta[c], 'transformer') and self.meta[c].transformer and 'log' in self.meta[c].transformer:
                            # For log-transformed columns, calculate WD in log space to avoid overflow
                            # This is more stable and still gives meaningful distance metrics
                            
                            # Convert both to float64 first (most important step)
                            real_vals = self.val_df[c].values.astype(np.float64)
                            syn_vals = syn[c].values.astype(np.float64)
                            
                            # Add a small epsilon to avoid log(0)
                            epsilon = 1e-10
                            real_vals = np.maximum(real_vals, epsilon)
                            syn_vals = np.maximum(syn_vals, epsilon)
                            
                            # Transform to log space (similar to the original transform)
                            real_log = np.log1p(real_vals)
                            syn_log = np.log1p(syn_vals)
                            
                            # Calculate WD in log space
                            wd = wasserstein_distance(real_log, syn_log)
                            LOGGER.info(f"VAL WD (log) {c:30s}: {wd:.4f}")
                            
                        # Regular handling for non-log-transformed columns
                        else:
                            # Convert values to float64 but handle potential overflow
                            # For integer columns, first convert to float64 which has a wider range
                            if self.meta[c].data_type == DataType.INTEGER:
                                # For real data
                                real_vals = self.val_df[c].values
                                if real_vals.dtype.kind in 'iu':  # If integer type
                                    real_vals = real_vals.astype(np.float64)
                                
                                # For synthetic data
                                syn_vals = syn[c].values
                                if syn_vals.dtype.kind in 'iu':  # If integer type
                                    syn_vals = syn_vals.astype(np.float64)
                            else:
                                # For DECIMAL type, already float
                                real_vals = self.val_df[c].values
                                syn_vals = syn[c].values
                                
                            # Replace any non-finite values with finite ones
                            real_vals = np.nan_to_num(real_vals, nan=0.0)
                            syn_vals = np.nan_to_num(syn_vals, nan=0.0)
                            
                            # Clip to reasonable range to avoid extreme values
                            # Use more conservative bounds for numeric calculations
                            max_bound = 1e6
                            real_vals = np.clip(real_vals, -max_bound, max_bound)
                            syn_vals = np.clip(syn_vals, -max_bound, max_bound)
                        
                            # Calculate WD with safe values
                            wd = wasserstein_distance(real_vals, syn_vals)
                            LOGGER.info(f"VAL WD (num) {c:30s}: {wd:.4f}")
                    except Exception as e:
                        LOGGER.warning(f"Could not calculate WD for {c}: {e}")
                        # Print more debug info
                        LOGGER.warning(f"Real dtype: {self.val_df[c].dtype}, Syn dtype: {syn[c].dtype}")
                        LOGGER.warning(f"Real range: [{self.val_df[c].min()} to {self.val_df[c].max()}], " 
                                      f"Syn range: [{syn[c].min()} to {syn[c].max()}]")
                
                # datetime features → epoch seconds
                for c in self.dt_cols:
                    try:
                        real_ts = (
                                pd.to_datetime(self.val_df[c], format=self.meta[c].datetime_format, errors='coerce')
                                .astype("int64") // 10 ** 9
                        ).values.astype(np.float64)
                        syn_ts = (
                                pd.to_datetime(syn[c], format=self.meta[c].datetime_format, errors='coerce')
                                .astype("int64") // 10 ** 9
                        ).values.astype(np.float64)
                        # Ensure finite values
                        real_ts = np.nan_to_num(real_ts, nan=0)
                        syn_ts = np.nan_to_num(syn_ts, nan=0)
                        wd = wasserstein_distance(real_ts, syn_ts)
                        LOGGER.info(f"VAL WD (dt)  {c:30s}: {wd:.4f}")
                    except Exception as e:
                        LOGGER.warning(f"Could not calculate WD for datetime {c}: {e}")

            # ── 3) early‐stop on W‐EMA if you like ────────────────────────────────
            w_est = -d_mean
            self.w_ema = w_est if self.w_ema is None else 0.9 * self.w_ema + 0.1 * w_est
            if self.w_ema > self.best_w + 1e-4:
                self.best_w, self.no_imp = self.w_ema, 0
                self.best_state = copy.deepcopy(self.G.state_dict())
            else:
                self.no_imp += 1

            if self.no_imp >= self.cfg.patience:
                LOGGER.info("Early stop after %d epochs without improvement", self.cfg.patience)
                break

    # ------------------------------------------------------------------#
    def generate(self, n: int) -> pd.DataFrame:
        self.ema_G.eval()
        rows, bs = [], self.cfg.batch_size
        with torch.no_grad():
            for _ in range(0, n, bs):
                cur = min(bs, n - len(rows))
                z = torch.randn(cur, self.cfg.latent_dim, device=DEVICE)
                cond = self._sample_cond(cur).to(DEVICE)
                rows.append(self.ema_G(z, cond, hard=True).cpu().numpy())
        mat = np.vstack(rows)[:n]

        ptr, data = 0, {}
        
        # Get original data ranges for all numeric columns for safer bounds enforcement
        orig_ranges = {}
        for c in self.num_cols:
            if c in self.real_df.columns:
                orig_ranges[c] = {
                    'min': float(self.real_df[c].min()),
                    'max': float(self.real_df[c].max()),
                    'median': float(self.real_df[c].median()),
                    'mean': float(self.real_df[c].mean())
                }
        
        for c in self.num_cols:
            inv = self.tfs[c].inverse(mat[:, ptr])
            ptr += 1
    
            # Apply any needed post-processing
            m = self.meta[c]
    
            # Handle NaN and inf values first - use more conservative bounds
            safe_max = min(1e9, np.finfo(np.float64).max / 1e6)
            safe_min = max(-1e9, np.finfo(np.float64).min / 1e6)
            inv = np.nan_to_num(inv, nan=0, posinf=safe_max, neginf=safe_min)

            # 1) Restore zeros if needed (important for delay columns)
            if hasattr(m, 'preserve_zeros') and m.preserve_zeros:
                # Generate random mask for zeros (probability based on real data)
                real_zeros_pct = (self.real_df[c] == 0).mean()
                if real_zeros_pct > 0.01:  # Only if zeros are meaningfully present
                    zero_mask = np.random.random(len(inv)) < real_zeros_pct
                    inv[zero_mask] = 0

            # 2) Handle log-transformed data with our improved transformer
            if hasattr(m, 'transformer') and m.transformer and 'log' in m.transformer:
                # The LogTransformer has already been applied by now, but we need 
                # to ensure the values are in the appropriate range and distribution
                
                # Get original data range
                orig_min = float(self.real_df[c].min())
                orig_max = float(self.real_df[c].max())
                
                # Get the current range of generated values
                curr_min, curr_max = float(inv.min()), float(inv.max())
                
                # First ensure we don't have infinite or NaN values
                inv = np.nan_to_num(inv, nan=0, posinf=orig_max, neginf=orig_min)
                
                # Find columns with extreme ranges that might cause overflow
                if orig_max > 1e6 or curr_max > 1e6:
                    # For columns with very large values, use distribution matching
                    # instead of exact value matching to avoid overflow
                    
                    # Get percentiles from real data
                    real_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    real_values = np.percentile(self.real_df[c].values, real_percentiles)
                    
                    # Get same percentiles from synthetic data
                    syn_values = np.percentile(inv, real_percentiles)
                    
                    # For each synthetic value, find where it falls in the synthetic percentiles
                    # and map to the corresponding real percentile
                    result = np.zeros_like(inv)
                    
                    # For each value, find the closest percentile bin and scale within that bin
                    for i, val in enumerate(inv):
                        # Find which percentile bin this value falls into
                        if val <= syn_values[0]:
                            # Below the 1st percentile
                            result[i] = real_values[0] * (val / max(syn_values[0], 1e-10))
                        elif val >= syn_values[-1]:
                            # Above the 99th percentile
                            ratio = min((val - syn_values[-1]) / max((curr_max - syn_values[-1]), 1e-10), 10.0)
                            result[i] = real_values[-1] + ratio * (orig_max - real_values[-1])
                        else:
                            # Find which bin the value is in
                            for j in range(len(real_percentiles)-1):
                                if syn_values[j] <= val <= syn_values[j+1]:
                                    # Linear interpolation within this bin
                                    bin_ratio = (val - syn_values[j]) / max((syn_values[j+1] - syn_values[j]), 1e-10)
                                    result[i] = real_values[j] + bin_ratio * (real_values[j+1] - real_values[j])
                                    break
                    
                    # Use the mapped values but ensure we respect bounds
                    inv = np.clip(result, orig_min, orig_max)
                else:
                    # For columns with moderate values, a simpler approach is sufficient
                    # Simple range scaling from current range to original range
                    scaling_factor = (orig_max - orig_min) / max((curr_max - curr_min), 1e-10)
                    inv = orig_min + (inv - curr_min) * scaling_factor
                    inv = np.clip(inv, orig_min, orig_max)

            # 3) Handle outliers with clamping if enabled
            if self.cfg.post_process_outliers:
                if hasattr(m, 'clamp_min') and m.clamp_min is not None:
                    if 0 < m.clamp_min < 1:  # Treat as percentile
                        min_val = np.percentile(self.real_df[c], m.clamp_min * 100)
                    else:  # Treat as absolute value
                        min_val = m.clamp_min
                    inv = np.maximum(inv, min_val)

                if hasattr(m, 'clamp_max') and m.clamp_max is not None:
                    if 0 < m.clamp_max < 1:  # Treat as percentile
                        max_val = np.percentile(self.real_df[c], m.clamp_max * 100)
                    else:  # Treat as absolute value
                        max_val = m.clamp_max
                    inv = np.minimum(inv, max_val)

            # 4) Make sure conditionally dependent columns respect their dependencies
            if hasattr(m, 'conditional_on') and m.conditional_on and self.cfg.use_conditional_gan:
                parent_col = m.conditional_on
                if parent_col in data:  # If parent already processed
                    # Apply relationship-specific adjustments
                    if c == 'Arrival Delay in Minutes' and parent_col == 'Departure Delay in Minutes':
                        # Arrival delays are typically slightly larger than departure delays
                        # Allow some flights to arrive early relative to their departure delay
                        early_mask = np.random.random(len(inv)) < 0.2  # 20% chance of early arrival
                        delay_factor = np.random.uniform(0.8, 1.2, len(inv))  # Realistic variation
                        inv = np.where(early_mask,
                                      np.maximum(0, data[parent_col] - np.random.uniform(5, 20, len(inv))),
                                      data[parent_col] * delay_factor)

            # 5) round to integer or fixed decimals with safe casting
            if m.data_type is DataType.INTEGER:
                # For columns that had log transform, be especially careful
                if hasattr(m, 'transformer') and m.transformer and 'log' in m.transformer:
                    # Get the range of the original integer column
                    col_min = max(0, self.real_df[c].min())  # Ensure non-negative for log-transformed columns
                    col_max = min(np.iinfo(np.int32).max / 2, self.real_df[c].max())  # Safer upper bound
                    
                    # Clip to the actual range seen in the data, not the maximum possible range
                    inv = np.clip(inv, col_min, col_max)
                    inv = np.round(inv)
                    
                    # Try to safely convert to int32
                    try:
                        inv = inv.astype(np.int32)
                    except (OverflowError, ValueError):
                        # If still having issues, use an even more conservative approach
                        LOGGER.warning(f"Issues with {c} - using conservative approach")
                        # First convert to float64 which has much larger range
                        inv_float = inv.astype(np.float64)
                        # Then round and clip to int32 range
                        inv_float = np.round(inv_float)
                        inv_float = np.clip(inv_float, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                        # Finally convert to int32
                        inv = inv_float.astype(np.int32)
                else:
                    # Standard integer handling for non-log-transformed columns
                    INT_MIN, INT_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max
                    inv = np.clip(inv, INT_MIN, INT_MAX)
                    inv = np.round(inv)
                    
                    try:
                        inv = inv.astype(np.int32)
                    except (OverflowError, ValueError):
                        LOGGER.warning(f"Integer overflow in column {c} - forcing clipping to int32 range")
                        inv = np.clip(inv, INT_MIN, INT_MAX).astype(np.int32)
            else:  # DECIMAL
                inv = np.round(inv, m.decimal_places)

            data[c] = inv
        for c in self.dt_cols:
            data[c] = self.tfs[c].inverse(mat[:, ptr])
            ptr += 1
        for c, k in zip(self.cat_cols, self.cat_sizes):
            idx = mat[:, ptr:ptr + k].argmax(1); ptr += k
            inv = {v: k for k, v in self.cat_maps[c].items()}
            data[c] = [inv[int(i)] for i in idx]
        for c in self.str_cols:
            meta = self.meta[c]; fn: Callable = meta.faker_method or self.faker.word
            data[c] = [fn(**(meta.faker_args or {})) for _ in range(n)]

        df = pd.DataFrame(data)
        return df[list(self.meta.keys())]

    # ------------------------------------------------------------------#
    # TRAIN HELPERS
    # ------------------------------------------------------------------#
    def _train_d(self, real, cond):
        self.opt_d.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.latent_dim, device=DEVICE)
        fake = self.G(z, cond).detach()
        with amp_autocast():
            d_real = self.D(real, cond)
            d_fake = self.D(fake, cond)
            gp = self._gp(real, fake, cond)

            # Wasserstein estimate & full loss
            w_est = d_real.mean() - d_fake.mean()
            loss = -w_est + self.cfg.gp_weight * gp
            loss += self.cfg.drift_epsilon * (d_real.pow(2).mean() + d_fake.pow(2).mean())

        self.scaler_d.scale(loss).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()
        return w_est.item(), loss.item()

    def _train_g(self, bsz, cond):
        self.opt_g.zero_grad(set_to_none=True)
        z = torch.randn(bsz, self.cfg.latent_dim, device=DEVICE)
        with amp_autocast():
            loss = -self.D(self.G(z, cond), cond).mean()
        self.scaler_g.scale(loss).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        # EMA
        with torch.no_grad():
            for p, p_ema in zip(self.G.parameters(), self.ema_G.parameters()):
                p_ema.mul_(self.ema_beta).add_(p.data, alpha=1 - self.ema_beta)
        return loss

    # ------------------------------------------------------------------#
    def _sample_cond(self, bsz):
        if not self.cat_cols:
            return torch.zeros(bsz, 0, device=DEVICE)
        cond = torch.zeros(bsz, sum(self.cat_sizes), device=DEVICE)
        off = 0
        for c, k in zip(self.cat_cols, self.cat_sizes):
            if c in self.cat_probs:
                # empirical sampling
                p = self.cat_probs[c]
                idx = np.random.choice(k, size=bsz, p=p)
            else:
                # uniform fallback
                idx = np.random.randint(k, size=bsz)
            cond[range(bsz), off + idx] = 1.0
            off += k
        return cond

    def _gp(self, real, fake, cond):
        alpha = torch.rand(real.size(0), 1, device=DEVICE)
        mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        with amp_autocast():
            score = self.D(mix, cond)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        return ((grad.view(grad.size(0), -1).norm(2, 1) - 1) ** 2).mean()

    def _adjust_n_critic(self, mean_W: float):
        """Update self.W_smooth and adapt self.n_critic via three‑level hysteresis."""
        alpha = 0.9
        # update smoothed W
        self.W_smooth = mean_W if self.W_smooth is None else alpha * self.W_smooth + (1 - alpha) * mean_W

        # hysteresis bounds: increase up to initial n_critic, decrease down to 1
        if self.W_smooth < 0.015 and self.n_critic < self.cfg.n_critic_initial:
            self.n_critic += 1
        elif self.W_smooth > 0.04 and self.n_critic > 1:
            self.n_critic -= 1



def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)