"""
Gaussian‑Copula ⇒ WGAN‑GP synthetic‑data engine.

The pipeline:

    1.  Fit per‑column transformers that map each feature to a
        *standard‑normal* space (copula space).
    2.  Train a Wasserstein GAN with gradient penalty on that
        dense numeric matrix.
    3.  Sample the generator and **inverse‑transform** every column
        back to its original domain.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from faker import Faker
from scipy.stats import norm, rankdata
from torch.utils.data import DataLoader, TensorDataset

from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------#
# CONFIG & MODELS
# ---------------------------------------------------------------------#


@dataclass(slots=True)
class GanConfig:
    latent_dim: int = 128
    batch_size: int = 256
    n_critic: int = 5
    gp_weight: float = 10.0
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    max_epochs: int = 300
    seed: int = 42
    amp: bool = True  # Automatic Mixed Precision


class _MLPGen(nn.Module):
    def __init__(self, cfg: GanConfig, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [B, out_dim]
        return self.net(z)


class _MLPDisc(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B]
        return self.net(x).view(-1)


# ---------------------------------------------------------------------#
# COLUMN‑LEVEL TRANSFORMERS  (copula helpers)
# ---------------------------------------------------------------------#


class _BaseTransformer:
    """Abstract base‑class for fit/transform/inverse routines."""

    def fit(self, series: pd.Series) -> "_BaseTransformer":  # noqa: D401
        raise NotImplementedError

    def transform(self, series: pd.Series) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, values: np.ndarray) -> pd.Series:
        raise NotImplementedError


class _ContinuousTransformer(_BaseTransformer):
    """Ranks ⇒ uniform ⇒ Φ⁻¹  (as in Gaussian copula)."""

    def __init__(self) -> None:
        self.sorted_: np.ndarray | None = None

    # --------------------------#
    def fit(self, series: pd.Series) -> "_ContinuousTransformer":
        self.sorted_ = np.sort(series.to_numpy(copy=True))
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        # rankdata returns 1…n
        u = rankdata(series, method="average") / (len(series) + 1.0)
        return norm.ppf(u)

    def inverse(self, values: np.ndarray) -> pd.Series:
        u = norm.cdf(values).clip(0, 1)
        # position in sorted empirical distribution
        idx = np.floor(u * (len(self.sorted_) - 1)).astype(int)
        return pd.Series(self.sorted_[idx], index=np.arange(len(values)))


class _CategoricalTransformer(_BaseTransformer):
    def __init__(self) -> None:
        self.cat2idx_: Dict = {}
        self.idx2cat_: Dict[int, str] = {}

    def fit(self, series: pd.Series) -> "_CategoricalTransformer":
        uniques = series.astype(str).unique().tolist()
        self.cat2idx_ = {c: i for i, c in enumerate(sorted(uniques))}
        self.idx2cat_ = {i: c for c, i in self.cat2idx_.items()}
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        idx = series.astype(str).map(self.cat2idx_)
        # map to Gaussian quantiles so we preserve correlations
        u = (idx + 0.5) / len(self.cat2idx_)
        return norm.ppf(u)

    def inverse(self, values: np.ndarray) -> pd.Series:
        u = norm.cdf(values).clip(0, 1 - 1e-12)
        idx = np.floor(u * len(self.idx2cat_)).astype(int)
        return pd.Series([self.idx2cat_[i] for i in idx], index=np.arange(len(values)))


_BoolTransformer = _CategoricalTransformer  # identical handling


class _DateTimeTransformer(_ContinuousTransformer):
    """Treat epoch seconds as a continuous quantity."""

    def __init__(self, dt_format: str):
        super().__init__()
        self.fmt = dt_format

    def fit(self, series: pd.Series) -> "_DateTimeTransformer":
        unix = pd.to_datetime(series, format=self.fmt, errors="coerce").astype("int64") // 10**9
        return super().fit(unix)

    def transform(self, series: pd.Series) -> np.ndarray:
        unix = pd.to_datetime(series, format=self.fmt, errors="coerce").astype("int64") // 10**9
        return super().transform(unix)

    def inverse(self, values: np.ndarray) -> pd.Series:
        unix = super().inverse(values)
        return pd.to_datetime(unix.astype(int), unit="s").dt.strftime(self.fmt)


# ---------------------------------------------------------------------#
# MAIN DRIVER
# ---------------------------------------------------------------------#


class WGAN:
    """
    Gaussian‑copula WGAN‑GP for tabular data.

    Use:
        gan = WGAN(real_df, meta, GanConfig())
        gan.fit()
        synthetic = gan.generate(n_samples)
    """

    # --------------------------#
    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        self.cfg, self.meta = cfg, meta
        self.faker = Faker()
        self._set_seed(cfg.seed)

        # build per‑column transformers
        self.transformers: Dict[str, _BaseTransformer] = {}
        transformed_cols: List[np.ndarray] = []

        for col, m in meta.items():
            tr = self._make_transformer(col, m)
            self.transformers[col] = tr.fit(real[col])
            transformed_cols.append(tr.transform(real[col]))

        # dense matrix in copula space
        self._matrix = torch.tensor(np.stack(transformed_cols, axis=1), dtype=torch.float32)
        self._output_dim: int = self._matrix.shape[1]

        # models / optimisers
        self.G = _MLPGen(cfg, self._output_dim).to(DEVICE)
        self.D = _MLPDisc(self._output_dim).to(DEVICE)
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.9))
        self.scaler_g = torch.amp.GradScaler(device="cuda", enabled=cfg.amp)
        self.scaler_d = torch.amp.GradScaler(device="cuda", enabled=cfg.amp)

        self.loader = DataLoader(
            TensorDataset(self._matrix), batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )

        LOGGER.info("Initialised WGAN‑GP with shape %s, device=%s", self._matrix.shape, DEVICE)

    # ------------------------------------------------------------------#
    # public api
    # ------------------------------------------------------------------#
    def fit(self) -> None:
        LOGGER.info("Training for %d epochs ...", self.cfg.max_epochs)
        for epoch in range(1, self.cfg.max_epochs + 1):
            d_running, g_running = 0.0, 0.0
            for i, (real,) in enumerate(self.loader, 1):
                real = real.to(DEVICE)
                d_loss = self._train_discriminator(real)
                d_running += d_loss.item()

                if i % self.cfg.n_critic == 0:
                    g_loss = self._train_generator(real.size(0))
                    g_running += g_loss.item()

            LOGGER.info(
                "Epoch %3d | D: %.4f | G: %.4f",
                epoch,
                d_running / len(self.loader),
                g_running / max(1, len(self.loader) // self.cfg.n_critic),
            )

    def generate(self, n: int) -> pd.DataFrame:
        self.G.eval()
        samples: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(0, n, self.cfg.batch_size):
                cur = min(self.cfg.batch_size, n - len(samples))
                z = torch.randn(cur, self.cfg.latent_dim, device=DEVICE)
                fake = self.G(z).cpu().numpy()
                samples.append(fake)
        mat = np.vstack(samples)[:n]

        # inverse‑transform every column
        columns: Dict[str, pd.Series] = {}
        for i, (col, tr) in enumerate(self.transformers.items()):
            series = tr.inverse(mat[:, i])
            columns[col] = self._post_process(col, series)

        df = pd.DataFrame(columns)
        # re‑order exactly as metadata
        df = df[list(self.meta.keys())]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------#
    # internal helpers
    # ------------------------------------------------------------------#
    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_transformer(self, col: str, m: FieldMetadata) -> _BaseTransformer:
        if m.data_type in {DataType.INTEGER, DataType.DECIMAL}:
            return _ContinuousTransformer()
        if m.data_type is DataType.DATETIME:
            if not m.datetime_format:
                raise ValueError(f"datetime_format missing for column {col}")
            return _DateTimeTransformer(m.datetime_format)
        if m.data_type in {DataType.CATEGORICAL}:
            return _CategoricalTransformer()
        if m.data_type is DataType.BOOLEAN:
            return _BoolTransformer()
        if m.data_type is DataType.STRING:
            # Not fed into the GAN at all; keep placeholder transformer that returns zeros
            class _Stub(_BaseTransformer):
                def fit(self, series): return self
                def transform(self, series): return np.zeros(len(series))
                def inverse(self, values): return pd.Series([None] * len(values))
            return _Stub()
        raise NotImplementedError(m)

    # ---------------- GAN steps ---------------- #
    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real.size(0), 1, device=DEVICE)
        mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        with torch.amp.autocast('cuda', enabled=self.cfg.amp):
            score = self.D(mix)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        norm_grad = grad.view(grad.size(0), -1).norm(2, dim=1)
        return ((norm_grad - 1) ** 2).mean()

    def _train_discriminator(self, real: torch.Tensor) -> torch.Tensor:
        self.opt_d.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.latent_dim, device=DEVICE)
        with torch.amp.autocast('cuda', enabled=self.cfg.amp):
            fake = self.G(z).detach()
            d_loss = (
                self.D(fake).mean()
                - self.D(real).mean()
                + self.cfg.gp_weight * self._gradient_penalty(real, fake)
            )
        self.scaler_d.scale(d_loss).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()
        return d_loss

    def _train_generator(self, batch: int) -> torch.Tensor:
        self.opt_g.zero_grad(set_to_none=True)
        z = torch.randn(batch, self.cfg.latent_dim, device=DEVICE)
        with torch.amp.autocast('cuda', enabled=self.cfg.amp):
            fake = self.G(z)
            g_loss = -self.D(fake).mean()
        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        return g_loss

    # ---------------- post‑processing helpers ---------------- #
    def _post_process(self, col: str, series: pd.Series) -> pd.Series:
        m = self.meta[col]
        if m.data_type is DataType.INTEGER:
            return series.round().astype(int)
        if m.data_type is DataType.DECIMAL:
            if m.decimal_places is not None:
                return series.round(m.decimal_places)
            return series
        if m.data_type is DataType.STRING:
            faker_fn: Callable = m.faker_method or self.faker.word
            return pd.Series([faker_fn(**(m.faker_args or {})) for _ in range(len(series))])
        return series
