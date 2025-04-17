from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from faker import Faker
from scipy.stats import rankdata, norm
from torch.utils.data import DataLoader, TensorDataset

from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
@dataclass
class GanConfig:
    latent_dim: int = 128
    batch_size: int = 128
    n_critic: int = 5
    gp_weight: float = 10.0
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    max_epochs: int = 300
    use_amp: bool = True
    seed: int = 42

# ---------------------------------------------------------------------------
# SIMPLE MLP GENERATOR / DISCRIMINATOR
# ---------------------------------------------------------------------------
class _Gen(nn.Module):
    def __init__(self, cfg: GanConfig, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim)
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)


class _Disc(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).view(-1)

# ---------------------------------------------------------------------------
# WGAN‑GP DRIVER
# ---------------------------------------------------------------------------
class WGAN:
    """Minimal GP‑WGAN for fully‑numeric matrix.  Categorical/boolean handled
    via one‑hot; STRING columns filled afterwards with Faker; datetime numeric
    encoded ordinal/epoch.
    """

    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg, self.meta = cfg, meta
        self.faker = Faker()
        self.real_df = real.copy()

        # -------- preprocessing (drop NaNs) --------
        self.real_df.dropna(inplace=True)

        # Identify column groups
        self.num_cols = [c for c, m in meta.items() if m.data_type in (DataType.INTEGER, DataType.DECIMAL)]
        self.cat_cols = [c for c, m in meta.items() if m.data_type == DataType.CATEGORICAL]
        self.bool_cols = [c for c, m in meta.items() if m.data_type == DataType.BOOLEAN]
        self.dt_cols  = [c for c, m in meta.items() if m.data_type == DataType.DATETIME]
        self.str_cols = [c for c, m in meta.items() if m.data_type == DataType.STRING]

        # encode categoricals + booleans
        self.enc_maps: Dict[str, Dict] = {}
        for c in self.cat_cols + self.bool_cols:
            uniques = sorted(self.real_df[c].fillna("missing").unique())
            self.enc_maps[c] = {v: i for i, v in enumerate(uniques)}
            self.real_df[c] = self.real_df[c].map(self.enc_maps[c])

        # encode datetime → ordinal/epoch as float32
        for c in self.dt_cols:
            fmt = meta[c].datetime_format
            self.real_df[c] = pd.to_datetime(self.real_df[c], format=fmt, errors="coerce").astype(np.int64) / 1e9

        # final numeric matrix
        self.matrix = torch.tensor(self.real_df[self.num_cols + self.cat_cols + self.bool_cols + self.dt_cols].values,
                                   dtype=torch.float32)
        self.out_dim = self.matrix.shape[1]

        # ---------- models / opt / scaler ----------
        self.G = _Gen(cfg, self.out_dim).to(device)
        self.D = _Disc(self.out_dim).to(device)

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.9))

        # independent grad scalers
        self.scaler_g = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        self.scaler_d = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        self.loader = DataLoader(self.matrix, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

        LOGGER.info("Starting WGAN training on %s samples", len(self.matrix))

    # ---------------------------------------------------------------------
    # training helpers
    # ---------------------------------------------------------------------
    def _gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1, device=device)
        mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            score = self.D(mix)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()

    # ---------------------------------------------------------------------
    def fit(self):
        for epoch in range(self.cfg.max_epochs):
            for i, real in enumerate(self.loader, 1):
                real = real.to(device)

                # -------- discriminator --------
                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    z = torch.randn(real.size(0), self.cfg.latent_dim, device=device)
                    fake = self.G(z).detach()
                    d_real = self.D(real)
                    d_fake = self.D(fake)
                    gp = self._gradient_penalty(real, fake)
                    d_loss = d_fake.mean() - d_real.mean() + self.cfg.gp_weight * gp

                self.opt_d.zero_grad(set_to_none=True)
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.step(self.opt_d)
                self.scaler_d.update()

                # -------- generator (every n_critic) --------
                if i % self.cfg.n_critic == 0:
                    z = torch.randn(real.size(0), self.cfg.latent_dim, device=device)
                    with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                        g_fake = self.G(z)
                        g_loss = -self.D(g_fake).mean()
                    self.opt_g.zero_grad(set_to_none=True)
                    self.scaler_g.scale(g_loss).backward()
                    self.scaler_g.step(self.opt_g)
                    self.scaler_g.update()

            if epoch % 1 == 0 or epoch == self.cfg.max_epochs - 1:
                LOGGER.info("Epoch %d | D %.3f | G %.3f", epoch, d_loss.item(), g_loss.item())

    # ---------------------------------------------------------------------
    # synthetic generation
    # ---------------------------------------------------------------------
    def generate(self, n: int) -> pd.DataFrame:
        self.G.eval()
        out: List[pd.DataFrame] = []
        bs = 512
        with torch.no_grad():
            for i in range(0, n, bs):
                cur = min(bs, n - i)
                z = torch.randn(cur, self.cfg.latent_dim, device=device)
                sample = self.G(z).cpu().numpy()
                out.append(sample)
        mat = np.vstack(out)
        synth = pd.DataFrame(mat, columns=(self.num_cols + self.cat_cols + self.bool_cols + self.dt_cols))

        # inverse decode categoricals / booleans
        for c in self.cat_cols + self.bool_cols:
            inv = {v: k for k, v in self.enc_maps[c].items()}
            synth[c] = synth[c].round().clip(0, len(inv) - 1).astype(int).map(inv)

        # inverse datetime
        for c in self.dt_cols:
            fmt = self.meta[c].datetime_format
            synth[c] = pd.to_datetime(synth[c], unit="s", errors="coerce").dt.strftime(fmt)

        # clip / round numeric types
        for c in self.num_cols:
            m = self.meta[c]
            if m.data_type == DataType.INTEGER:
                synth[c] = synth[c].round().astype(int)
            elif m.data_type == DataType.DECIMAL and m.decimal_places is not None:
                synth[c] = synth[c].round(m.decimal_places)

        # fill STRING with Faker
        for c in self.str_cols:
            meta = self.meta[c]
            faker_fn: Callable = meta.faker_method or self.faker.word
            synth[c] = [faker_fn(**meta.faker_args) for _ in range(n)]

        return synth[self.meta.keys()]  # keep original column order