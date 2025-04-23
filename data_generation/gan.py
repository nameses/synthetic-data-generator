from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from data_generation.transformers import _BaseTf, _LogTf, _ZeroInflatedTf, _BoundedTf, _StdTf, _MinMaxTf, _ContTf, _DtTf
from models.enums import DataType
from models.field_metadata import FieldMetadata


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
    n_critic_initial: int = 4
    gp_weight: float = 2.5
    drift_epsilon: float = 5e-4
    g_lr: float = 1e-4
    d_lr: float = 2e-4
    epochs: int = 500
    seed: int = 42
    # cosine schedule
    lr_min_ratio: float = 0.05
    # early stop
    patience: int = 320
    # gumbel
    tau_start: float = 2.5
    tau_end: float = 0.25
    val_interval: int = 5
    # visualization
    plot_interval: int = 10
    plot_samples: int = 1000
    save_plots: bool = False


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


class GAN:
    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        self.cfg, self.meta, self.faker = cfg, meta, Faker()
        _set_seed(cfg.seed)

        self.real_df = real.dropna().reset_index(drop=True)

        self.W_smooth = None  # running abs Wasserstein
        self.n_critic = 5  # start conservatively
        
        # Initialize training metrics
        self.metrics = {
            'epochs': [],
            'w_distance': [],
            'd_loss': [],
            'g_loss': [],
            'n_critic': [],
            'lr_d': [],
            'lr_g': [],
            'val_wd': {},  # To store validation Wasserstein distances per feature
        }

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
            mode = getattr(self.meta[c], "transformer", None)
            if mode == "standard":
                tf = _StdTf().fit(real[c])
            elif mode == "minmax":
                tf = _MinMaxTf().fit(real[c])
            elif mode == "log":
                tf = _LogTf().fit(real[c])
            elif mode == "zero_inflated":
                tf = _ZeroInflatedTf().fit(real[c])
            elif mode == "bounded":
                tf = _BoundedTf().fit(real[c])
            else:  # default rank‑Gauss
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
        total_steps = cfg.epochs * steps_per_epoch
        cond_dim = sum(self.cat_sizes)

        self.G = _Generator(cfg, len(self.num_cols) + len(self.dt_cols), self.cat_sizes, total_steps).to(DEVICE)
        self.D = _Discriminator(X.shape[1], cond_dim, cfg).to(DEVICE)

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(0.0, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(0.0, 0.9))
        self.sch_g = CosineAnnealingLR(self.opt_g, cfg.epochs, eta_min=cfg.g_lr * cfg.lr_min_ratio)
        self.sch_d = CosineAnnealingLR(self.opt_d, cfg.epochs, eta_min=cfg.d_lr * cfg.lr_min_ratio)

        self.scaler_g = torch.amp.GradScaler("cuda", enabled=True)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=True)

        self.ema_G = copy.deepcopy(self.G).eval().requires_grad_(False)
        self.ema_beta, self.best_w, self.no_imp = 0.999, -float("inf"), 0
        self.n_critic = cfg.n_critic_initial

    # ------------------------------------------------------------------#
    def fit(self, verbose=False):
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
    
        # Reset metrics
        self.metrics = {
            'epochs': [],
            'w_distance': [],
            'd_loss': [],
            'g_loss': [],
            'n_critic': [],
            'lr_d': [],
            'lr_g': [],
            'val_wd': { c: [] for c in (self.num_cols + self.dt_cols) },
        }
    
        # ── 1) the usual GAN training loop ────────────────────────────────────
        for epoch in range(1, self.cfg.epochs + 1):
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
    
            # Store metrics
            self.metrics['epochs'].append(epoch)
            self.metrics['w_distance'].append(mean_W)
            self.metrics['d_loss'].append(d_mean)
            self.metrics['g_loss'].append(g_mean)
            self.metrics['n_critic'].append(self.n_critic)
            self.metrics['lr_d'].append(lr_d)
            self.metrics['lr_g'].append(lr_g)
    
            if verbose:
                LOGGER.info(
                    "Ep %03d | W %.4f | D %.4f | G %.4f | n_c %d | lr_d %.6f | lr_g %.6f",
                    epoch, mean_W, d_mean, g_mean, self.n_critic, lr_d, lr_g
                )
    
            if hasattr(self, "val_df") and epoch % self.cfg.val_interval == 0:
                syn = self.generate(len(self.val_df))
                val_wd_epoch = {}
                
                # numeric features
                for c in self.num_cols:
                    real_vals = self.val_df[c].astype(float).values
                    syn_vals = syn[c].astype(float).values
                    wd = wasserstein_distance(real_vals, syn_vals)
                    val_wd_epoch[c] = wd
                    self.metrics['val_wd'][c].append((epoch, wd))
                    if verbose:
                        LOGGER.info(f"VAL WD (num) {c:30s}: {wd:.4f}")
                
                # datetime features → epoch seconds
                for c in self.dt_cols:
                    real_ts = (
                            pd.to_datetime(self.val_df[c], format=self.meta[c].datetime_format)
                            .astype("int64") // 10 ** 9
                    ).values.astype(float)
                    syn_ts = (
                            pd.to_datetime(syn[c], format=self.meta[c].datetime_format)
                            .astype("int64") // 10 ** 9
                    ).values.astype(float)
                    wd = wasserstein_distance(real_ts, syn_ts)
                    val_wd_epoch[c] = wd
                    self.metrics['val_wd'][c].append((epoch, wd))
                    if verbose:
                        LOGGER.info(f"VAL WD (dt)  {c:30s}: {wd:.4f}")
                
                # Plot progress if configured
                # if epoch % self.cfg.plot_interval == 0:
                #    self.plot_training_metrics()
    
            # ── 3) early‐stop on W‐EMA if you like ────────────────────────────────
            w_est = -d_mean
            self.w_ema = w_est if self.w_ema is None else 0.9 * self.w_ema + 0.1 * w_est
            if self.w_ema > self.best_w + 1e-4:
                self.best_w, self.no_imp = self.w_ema, 0
                self.best_state = copy.deepcopy(self.G.state_dict())
            else:
                self.no_imp += 1
    
            if self.no_imp >= self.cfg.patience:
                if verbose:
                    LOGGER.info("Early stop after %d epochs without improvement", self.cfg.patience)
                break
                
        # Final visualization after training
        self.plot_training_metrics()
        
        return self.metrics

    # ------------------------------------------------------------------#
    def generate(self, n: int, use_best_model=False, temperature=1.0) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Parameters:
            n: Number of samples to generate
            use_best_model: Whether to use the best model saved during training
            temperature: Temperature for sampling (higher = more diversity)
        
        Returns:
            DataFrame with synthetic data
        """
        # Use best model if available and requested
        if use_best_model and hasattr(self, 'best_state'):
            orig_state = copy.deepcopy(self.G.state_dict())
            self.G.load_state_dict(self.best_state)
            self.ema_G = copy.deepcopy(self.G)
        
        self.ema_G.eval()
        rows, bs = [], self.cfg.batch_size
        with torch.no_grad():
            for _ in range(0, n, bs):
                cur = min(bs, n - len(rows))
                # Apply temperature to noise distribution
                z = torch.randn(cur, self.cfg.latent_dim, device=DEVICE) * temperature
                cond = self._sample_cond(cur).to(DEVICE)
                rows.append(self.ema_G(z, cond, hard=True).cpu().numpy())
        mat = np.vstack(rows)[:n]
        
        # Restore original model if we switched
        if use_best_model and hasattr(self, 'best_state') and 'orig_state' in locals():
            self.G.load_state_dict(orig_state)
            self.ema_G = copy.deepcopy(self.G)

        ptr, data = 0, {}
        for c in self.num_cols:
            inv = self.tfs[c].inverse(mat[:, ptr])
            ptr += 1

            # 1) invert any upstream log-transform
            if getattr(self.meta[c], 'transformer', None) == 'log':
                inv = np.expm1(inv)

            # 2) round to integer or fixed decimals
            m = self.meta[c]
            if m.data_type is DataType.INTEGER:
                inv = np.round(inv).astype(int)
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



    def plot_training_metrics(self, figsize=(15, 10)):
        """
        Plot training metrics including Wasserstein distance, losses, and other training parameters.
        """
        if not self.metrics['epochs']:
            LOGGER.warning("No training metrics available to plot")
            return
            
        plt.figure(figsize=figsize)
        
        # Create a 2x2 subplot layout
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['epochs'], self.metrics['w_distance'], 'b-', label='Wasserstein Distance')
        plt.title('Wasserstein Distance')
        plt.xlabel('Epoch')
        plt.ylabel('W-Distance')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['epochs'], self.metrics['d_loss'], 'r-', label='Discriminator Loss')
        plt.plot(self.metrics['epochs'], self.metrics['g_loss'], 'g-', label='Generator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['epochs'], self.metrics['n_critic'], 'k-')
        plt.title('n_critic Adaptation')
        plt.xlabel('Epoch')
        plt.ylabel('n_critic')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['epochs'], self.metrics['lr_d'], 'r-', label='D Learning Rate')
        plt.plot(self.metrics['epochs'], self.metrics['lr_g'], 'g-', label='G Learning Rate')
        plt.title('Learning Rates')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # Save if configured
        # if self.cfg.save_plots:
        #    plt.savefig(f'gan_training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
            
        plt.show()


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def visualize_gan_training(gan_model):
    """
    Visualize GAN training metrics and generated data.
    
    Parameters:
        gan_model: Trained GAN model
    """
    # Plot overall training metrics
    gan_model.plot_training_metrics()
    
    # Generate synthetic data and plot distributions
    synthetic_data = gan_model.generate(gan_model.cfg.plot_samples)
    gan_model.plot_distributions(synthetic_data)
    
    # Plot correlation matrices
    if len(gan_model.num_cols) >= 2:
        gan_model.plot_correlation_matrix(synthetic_data)