# data_generation/wgan.py
"""Gaussian‑copula + categorical‑aware WGAN‑GP with modern tricks."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from faker import Faker
from scipy.stats import norm, rankdata
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# CONFIG
# -----------------------------------------------------------------------------#


@dataclass(slots=True)
class GanConfig:
    latent_dim: int = 128
    hidden: int = 256
    batch_size: int = 256
    n_critic: int = 5
    gp_weight: float = 10.0
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    max_epochs: int = 300
    seed: int = 42
    amp: bool = True
    # learning‑rate schedule
    t0: int = 10
    t_mult: int = 2
    # early stopping
    patience: int = 200
    # gumbel‑softmax temperature
    tau_start: float = 1.5
    tau_end: float = 0.3


# -----------------------------------------------------------------------------#
# TRANSFORMERS –  numeric / datetime  (categoricals handled separately)
# -----------------------------------------------------------------------------#


class _BaseTransformer:
    def fit(self, series: pd.Series): ...
    def transform(self, series: pd.Series) -> np.ndarray: ...
    def inverse(self, values: np.ndarray) -> pd.Series: ...


class _ContinuousTransformer(_BaseTransformer):
    def fit(self, series: pd.Series):
        self.sorted_ = np.sort(series.to_numpy(copy=True))
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        u = rankdata(series, method="average") / (len(series) + 1.0)
        return norm.ppf(u)

    def inverse(self, values: np.ndarray) -> pd.Series:
        u = norm.cdf(values).clip(0, 1)
        idx = np.floor(u * (len(self.sorted_) - 1)).astype(int)
        return pd.Series(self.sorted_[idx], index=np.arange(len(values)))


class _DateTimeTransformer(_ContinuousTransformer):
    def __init__(self, fmt: str): self.fmt = fmt

    def fit(self, s): return super().fit(_to_epoch(s, self.fmt))
    def transform(self, s): return super().transform(_to_epoch(s, self.fmt))
    def inverse(self, v): return pd.to_datetime(super().inverse(v), unit="s").dt.strftime(self.fmt)


def _to_epoch(s: pd.Series, fmt: str):
    return pd.to_datetime(s, format=fmt, errors="coerce").astype("int64") // 10**9


# -----------------------------------------------------------------------------#
# NETWORK BUILDING BLOCKS
# -----------------------------------------------------------------------------#


def linear_sn(inp: int, out: int) -> nn.Linear:
    return spectral_norm(nn.Linear(inp, out))


class MinibatchStd(nn.Module):
    """Append minibatch‑std‑dev as an extra feature (helps mode coverage)."""

    def __init__(self): super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x, dim=0, keepdim=True)  # [1, F]
        mean_std = std.mean().expand(x.size(0), 1)  # [B,1]
        return torch.cat([x, mean_std], dim=1)      # [B, F+1]


# -----------------------------------------------------------------------------#
# GENERATOR  (numeric + categorical heads)
# -----------------------------------------------------------------------------#


class _Generator(nn.Module):
    def __init__(self, cfg: GanConfig, out_num: int, cat_dims: List[int]):
        super().__init__()
        self.cfg = cfg
        self.tau_start, self.tau_end = cfg.tau_start, cfg.tau_end
        self.register_buffer("step", torch.zeros(1))  # updated externally

        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.LeakyReLU(0.2),
            MinibatchStd(),
            nn.Linear(cfg.hidden + 1, cfg.hidden),
            nn.LeakyReLU(0.2),
        )

        self.num_head = nn.Linear(cfg.hidden, out_num) if out_num else None
        self.cat_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden, k) for k in cat_dims]
        )

    # ------------------------------------------------------------------#
    def forward(self, z: torch.Tensor, hard: bool = False) -> torch.Tensor:
        h = self.fc(z)
        out = []

        if self.num_head is not None:
            out.append(self.num_head(h))  # numeric block

        # adaptive temperature (cosine decay)
        total_steps = (self.cfg.max_epochs * math.ceil(real_len / self.cfg.batch_size))
        tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
            1 + math.cos(math.pi * self.step.item() / total_steps)
        )

        for head in self.cat_heads:
            logits = head(h)
            y = F.gumbel_softmax(logits, tau=tau, hard=hard)
            out.append(y)

        self.step += 1
        return torch.cat(out, dim=1)


# -----------------------------------------------------------------------------#
# DISCRIMINATOR
# -----------------------------------------------------------------------------#


class _Discriminator(nn.Module):
    def __init__(self, inp_dim: int, cfg: GanConfig):
        super().__init__()
        self.net = nn.Sequential(
            linear_sn(inp_dim, cfg.hidden),
            nn.LeakyReLU(0.2),
            linear_sn(cfg.hidden, cfg.hidden),
            nn.LeakyReLU(0.2),
            linear_sn(cfg.hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# -----------------------------------------------------------------------------#
# MAIN DRIVER
# -----------------------------------------------------------------------------#


class WGAN:
    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        global real_len; real_len = len(real)
        self.cfg, self.meta = cfg, meta
        self.faker = Faker()
        _set_seed(cfg.seed)

        # -------- column analysis ------------------------------------------------
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

        # ------------- transformers & one‑hots -----------------------------------
        self.transformers: Dict[str, _BaseTransformer] = {}
        mats = []

        # numeric + datetime
        for col in self.num_cols:
            tr = _ContinuousTransformer().fit(real[col])
            self.transformers[col] = tr
            mats.append(tr.transform(real[col]))
        for col in self.dt_cols:
            tr = _DateTimeTransformer(meta[col].datetime_format).fit(real[col])
            self.transformers[col] = tr
            mats.append(tr.transform(real[col]))

        # categoricals → one‑hot
        self.cat_sizes, self.cat_maps = [], {}
        for col in self.cat_cols:
            uniq = sorted(real[col].astype(str).unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(uniq)}
            self.cat_sizes.append(len(uniq))
            oh = F.one_hot(
                torch.tensor(real[col].astype(str).map(self.cat_maps[col]).values),
                num_classes=len(uniq),
            ).numpy()
            mats.append(oh.T)  # each class as its own row

        self.X_real = torch.tensor(np.vstack(mats).T, dtype=torch.float32)
        self.inp_dim = self.X_real.shape[1]

        # ------------- networks & opt -------------------------------------------
        self.G = _Generator(cfg, len(self.num_cols) + len(self.dt_cols), self.cat_sizes).to(DEVICE)
        self.D = _Discriminator(self.inp_dim, cfg).to(DEVICE)

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.9))

        self.sch_g = CosineAnnealingWarmRestarts(self.opt_g, T_0=cfg.t0, T_mult=cfg.t_mult)
        self.sch_d = CosineAnnealingWarmRestarts(self.opt_d, T_0=cfg.t0, T_mult=cfg.t_mult)

        self.scaler_g = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        self.scaler_d = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        self.loader = DataLoader(TensorDataset(self.X_real), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # ------------------------------------------------------------------#
    # TRAINING
    # ------------------------------------------------------------------#
    def fit(self):
        best_g_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            g_running, d_running = 0.0, 0.0

            for i, (real_batch,) in enumerate(self.loader, 1):
                real_batch = real_batch.to(DEVICE)
                g_loss = torch.tensor(0.0, device=DEVICE)

                d_loss = self._train_d(real_batch)
                d_running += d_loss.item()

                if i % self.cfg.n_critic == 0:
                    g_loss = self._train_g(real_batch.size(0))
                    g_running += g_loss.item()

                if (
                        torch.isnan(d_loss) or torch.isinf(d_loss) or
                        torch.isnan(g_loss) or torch.isinf(g_loss)
                ):
                    LOGGER.error("NaN/Inf detected – early stop.")
                    return

            # schedules
            self.sch_g.step(epoch - 1)
            self.sch_d.step(epoch - 1)

            g_mean = g_running / max(1, len(self.loader) / self.cfg.n_critic)
            LOGGER.info("Epoch %03d | D %.4f | G %.4f", epoch, d_running / len(self.loader), g_mean)

            # early stopping
            if g_mean + 1e-4 < best_g_loss:
                best_g_loss = g_mean
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve > self.cfg.patience:
                LOGGER.info("Early stopping: generator loss did not improve for %d epochs.", self.cfg.patience)
                break

    # ------------------------------------------------------------------#
    # SYNTHESIS
    # ------------------------------------------------------------------#
    def generate(self, n: int) -> pd.DataFrame:
        self.G.eval()
        out_rows = []
        bs = self.cfg.batch_size

        with torch.no_grad():
            for _ in range(0, n, bs):
                cur = min(bs, n - len(out_rows))
                z = torch.randn(cur, self.cfg.latent_dim, device=DEVICE)
                fake = self.G(z, hard=True).cpu().numpy()
                out_rows.append(fake)
        mat = np.vstack(out_rows)[:n]

        # -------------- inverse transforms -----------------------------
        ptr = 0
        data: Dict[str, Sequence] = {}

        # numeric
        for col in self.num_cols:
            tr = self.transformers[col]
            data[col] = tr.inverse(mat[:, ptr])
            ptr += 1
        # datetime
        for col in self.dt_cols:
            tr = self.transformers[col]
            data[col] = tr.inverse(mat[:, ptr])
            ptr += 1
        # categorical
        for col, size in zip(self.cat_cols, self.cat_sizes):
            one_hot = mat[:, ptr:ptr + size]
            idx = one_hot.argmax(1).astype(int)
            inv = {v: k for k, v in self.cat_maps[col].items()}
            data[col] = pd.Series([inv[i] for i in idx])
            ptr += size
        # strings via Faker
        for col in self.str_cols:
            meta = self.meta[col]
            fn: Callable = meta.faker_method or self.faker.word
            data[col] = [fn(**(meta.faker_args or {})) for _ in range(n)]

        return pd.DataFrame(data)[list(self.meta.keys())]

    # ------------------------------------------------------------------#
    # INTERNAL
    # ------------------------------------------------------------------#
    def _train_d(self, real: torch.Tensor) -> torch.Tensor:
        self.D.train()
        self.opt_d.zero_grad(set_to_none=True)

        z = torch.randn(real.size(0), self.cfg.latent_dim, device=DEVICE)
        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            fake = self.G(z).detach()
            gp = self._gradient_penalty(real, fake)
            loss = self.D(fake).mean() - self.D(real).mean() + self.cfg.gp_weight * gp

        self.scaler_d.scale(loss).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()
        return loss

    def _train_g(self, batch: int) -> torch.Tensor:
        self.G.train()
        self.opt_g.zero_grad(set_to_none=True)

        z = torch.randn(batch, self.cfg.latent_dim, device=DEVICE)
        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            fake = self.G(z)
            loss = -self.D(fake).mean()

        self.scaler_g.scale(loss).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        return loss

    # ------------------------------------------------------------------#
    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real.size(0), 1, device=DEVICE)
        mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            score = self.D(mix)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        return ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


# -----------------------------------------------------------------------------#
# UTILS
# -----------------------------------------------------------------------------#


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
