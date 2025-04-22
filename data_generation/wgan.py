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
            mode = getattr(self.meta[c], "transformer", None)
            if mode == "standard":
                tf = _StdTf().fit(real[c])
            elif mode == "minmax":
                tf = _MinMaxTf().fit(real[c])
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
                    real_vals = self.val_df[c].astype(float).values
                    syn_vals = syn[c].astype(float).values
                    wd = wasserstein_distance(real_vals, syn_vals)
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
                    LOGGER.info(f"VAL WD (dt)  {c:30s}: {wd:.4f}")

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



def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)