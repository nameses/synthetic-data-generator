from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from faker import Faker
from scipy.stats import norm, rankdata
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
    latent_dim: int = 512
    hidden_g: int = 2048
    hidden_d: int = 1024
    batch_size: int = 256
    n_critic_initial: int = 5
    gp_weight: float = 2.5
    corr_weight: float = 0.25
    drift_epsilon: float = 5e-4
    g_lr: float = 2e-4
    d_lr: float = 4e-4
    max_epochs: int = 300
    seed: int = 42
    # cosine schedule
    lr_min_ratio: float = 0.05
    # early stop
    patience: int = 300
    # gumbel
    tau_start: float = 2.5
    tau_end: float = 0.25
    d_cls_weight: float = 0.1  # weight for D’s classification loss
    g_cls_weight: float = 0.1  # weight for G’s classification loss
    pretrain_epochs: int = 75
    mm_weight: float = 0.25


# ------------------------------------------------------------------#
# TRANSFORMERS  (numeric / datetime)
# ------------------------------------------------------------------#


class _BaseTf:
    def fit(self, s: pd.Series): ...

    def transform(self, s: pd.Series) -> np.ndarray: ...

    def inverse(self, v: np.ndarray) -> pd.Series: ...

class _LogTf(_BaseTf):
    """Applies log1p + z-score normalization and supports inverse."""
    def fit(self, s: pd.Series):
        log_vals = np.log1p(s)
        self.mean = log_vals.mean()
        self.std = log_vals.std()
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        return ((np.log1p(s) - self.mean) / self.std).to_numpy(copy=True)

    def inverse(self, v: np.ndarray) -> pd.Series:
        return np.expm1(v * self.std + self.mean)


class _StdTf(_BaseTf):
    """Simple z‑score scaler for roughly Gaussian columns."""

    def fit(self, s: pd.Series):
        self.mean = s.mean()
        self.std = s.std()
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        return ((s - self.mean) / self.std).to_numpy(copy=True)

    def inverse(self, v: np.ndarray) -> pd.Series:
        return pd.Series(v * self.std + self.mean)


class _ContTf(_BaseTf):
    """Rank‑gauss with light tail‑clipping for stability."""

    def fit(self, s):
        q_low, q_hi = s.quantile([0.001, 0.999])
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
        return pd.to_datetime(s, format=self.fmt, errors="coerce").astype("int64") // 10 ** 9


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
    def __init__(self, inp_dim, cond_dim, cfg: GanConfig, sat_dim: Optional[int]):
        super().__init__()
        hid = cfg.hidden_d
        # shared backbone
        self.shared = nn.Sequential(
            lin_sn(inp_dim + cond_dim, hid),
            nn.LeakyReLU(0.2),
            lin_sn(hid, hid),
            nn.LeakyReLU(0.2),
            lin_sn(hid, hid // 2),
            nn.LeakyReLU(0.2),
        )
        # adversarial head (Wasserstein)
        self.adv_head = lin_sn(hid // 2, 1)
        # optional classification head
        self.use_cls = sat_dim is not None
        if self.use_cls:
            self.cls_head = lin_sn(hid // 2, sat_dim)
        else:
            self.cls_head = None

    def forward(self, x, c):
        h = self.shared(torch.cat([x, c], 1))
        adv = self.adv_head(h).view(-1)
        cls = self.cls_head(h) if self.use_cls else None
        return adv, cls


# ------------------------------------------------------------------#
# MAIN DRIVER
# ------------------------------------------------------------------#


class WGAN:
    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        self.cfg, self.meta, self.faker = cfg, meta, Faker()
        _set_seed(cfg.seed)

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

        self.tfs: Dict[str, _BaseTf] = {}
        mats = []

        for c in self.num_cols:
            m = meta[c]
            series = real[c]

            if getattr(m, "transformer", None) == "log":
                tf = _LogTf().fit(series)
            elif m.transformer == "standard":
                tf = _StdTf().fit(series)
            else:
                tf = _ContTf().fit(series)

            self.tfs[c] = tf
            mats.append(tf.transform(series))
        for c in self.dt_cols:
            tf = _DtTf(meta[c].datetime_format).fit(real[c])
            self.tfs[c] = tf
            mats.append(tf.transform(real[c]))

        self.cat_sizes, self.cat_maps = [], {}
        for c in self.cat_cols:
            uniq = sorted(real[c].astype(str).unique())
            self.cat_sizes.append(len(uniq))
            mp = {v: i for i, v in enumerate(uniq)}
            self.cat_maps[c] = mp
            mats.append(F.one_hot(torch.tensor(real[c].astype(str).map(mp).values),
                                  num_classes=len(uniq)).T.numpy())

        # compute per-col sampling probs for any cat flagged 'empirical'
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

        # ── discover which column (if any) is flagged for prediction
        preds = [
            c for c, m in meta.items()
            if m.data_type is DataType.CATEGORICAL and m.is_prediction_used
        ]
        if len(preds) > 1:
            raise ValueError(f"Only one column can be is_prediction_used, got {preds}")
        if preds:
            self.pred_col = preds[0]
            self.use_cls = True
            self.sat_idx = self.cat_cols.index(self.pred_col)
            self.sat_dim = self.cat_sizes[self.sat_idx]
        else:
            self.pred_col = None
            self.use_cls = False
            self.sat_idx = None
            self.sat_dim = None

        self.G = _Generator(cfg, len(self.num_cols) + len(self.dt_cols), self.cat_sizes, total_steps).to(DEVICE)
        self.D = _Discriminator(
            X.shape[1], cond_dim, cfg,
            sat_dim=self.sat_dim if self.use_cls else None
        ).to(DEVICE)

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
    def fit(self, plot_training: bool = False):
        if plot_training:
            history = {"W": [], "D": [], "G": [], "n_c": [], "lr_d": [], "lr_g": []}

        for epoch in range(1, self.cfg.max_epochs + 1):
            ramp = min(1.0, epoch / self.cfg.pretrain_epochs)
            # compute current penalty weights
            dcls_w = self.cfg.d_cls_weight * ramp
            gcls_w = self.cfg.g_cls_weight * ramp
            corr_w = self.cfg.corr_weight * ramp

            d_sum = g_sum = w_sum = 0.0

            for i, (real,) in enumerate(self.loader, 1):
                real = real.to(DEVICE)
                cond = self._sample_cond(real.size(0)).to(DEVICE)
                # pass in dynamic D classification weight
                w_est = self._train_d(real, cond, dcls_w=dcls_w)
                d_sum += w_est[1]
                w_sum += w_est[0]

                g_loss = torch.tensor(0.0, device=DEVICE)
                if i % self.n_critic == 0:
                    # pass in dynamic G classification & correlation weights
                    g_loss = self._train_g(real.size(0), cond, cls_w=gcls_w, corr_w=corr_w)
                    g_sum += g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss

            # adaptive n_critic
            mean_W = w_sum / len(self.loader)
            alpha = 0.9
            self.W_smooth = (mean_W if self.W_smooth is None
                             else alpha * self.W_smooth + (1 - alpha) * mean_W)

            # three‑level hysteresis
            if self.W_smooth < 0.015 and self.n_critic < 5:
                self.n_critic += 1
            elif self.W_smooth > 0.04 and self.n_critic > 1:
                self.n_critic -= 1

            # LR decay
            self.sch_g.step()
            self.sch_d.step()

            g_mean = g_sum / max(1, len(self.loader) / self.n_critic)
            d_mean = d_sum / len(self.loader)

            lr_d = self.opt_d.param_groups[0]["lr"]
            lr_g = self.opt_g.param_groups[0]["lr"]
            LOGGER.info(
                "Ep %03d | W %.4f | D %.4f | G %.4f | n_c %d | lr_d %.6f | lr_g %.6f",
                epoch, mean_W, d_mean, g_mean, self.n_critic, lr_d, lr_g
            )

            # ── record metrics for plotting if requested
            if plot_training:
                history["W"].append(mean_W)
                history["D"].append(d_mean)
                history["G"].append(g_mean)
                history["n_c"].append(self.n_critic)
                history["lr_d"].append(lr_d)
                history["lr_g"].append(lr_g)

            # EMA‑Wasserstein early stop
            w_est = -d_mean
            self.w_ema = w_est if epoch == 1 else 0.9 * self.w_ema + 0.1 * w_est
            if self.w_ema > self.best_w + 1e-4:
                self.best_w, self.no_imp = self.w_ema, 0
            else:
                self.no_imp += 1
            if self.no_imp >= self.cfg.patience:
                LOGGER.info("Early stop after %d stagnant epochs", self.cfg.patience);
                break

        if plot_training:
            import matplotlib.pyplot as plt

            epochs = list(range(1, len(history["W"]) + 1))

            fig, axs = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle("WGAN Training Metrics", fontsize=16)

            axs[0, 0].plot(epochs, history["W"], label="Wasserstein", color="purple")
            axs[0, 0].set_title("Wasserstein Estimate")
            axs[0, 0].set_xlabel("Epoch")

            axs[0, 1].plot(epochs, history["D"], label="Discriminator loss", color="red")
            axs[0, 1].set_title("Discriminator Loss")
            axs[0, 1].set_xlabel("Epoch")

            axs[1, 0].plot(epochs, history["G"], label="Generator loss", color="blue")
            axs[1, 0].set_title("Generator Loss")
            axs[1, 0].set_xlabel("Epoch")

            axs[1, 1].plot(epochs, history["n_c"], label="n_critic", color="orange")
            axs[1, 1].set_title("Number of Generator Updates (n_critic)")
            axs[1, 1].set_xlabel("Epoch")

            axs[2, 0].plot(epochs, history["lr_d"], label="LR Discriminator", color="green")
            axs[2, 0].set_title("Discriminator Learning Rate")
            axs[2, 0].set_xlabel("Epoch")

            axs[2, 1].plot(epochs, history["lr_g"], label="LR Generator", color="teal")
            axs[2, 1].set_title("Generator Learning Rate")
            axs[2, 1].set_xlabel("Epoch")

            # Optional: tidy up layout
            for ax in axs.flat:
                ax.grid(True)
                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

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
            col = self.tfs[c].inverse(mat[:, ptr])
            # round per decimal_places / integer
            if self.meta[c].data_type is DataType.INTEGER:
                col = col.round().astype(int)
            elif self.meta[c].data_type is DataType.DECIMAL:
                dp = self.meta[c].decimal_places or 0
                col = col.round(dp)
            data[c] = col
            ptr += 1
        for c in self.dt_cols:
            data[c] = self.tfs[c].inverse(mat[:, ptr])
            ptr += 1
        for c, k in zip(self.cat_cols, self.cat_sizes):
            idx = mat[:, ptr:ptr + k].argmax(1)
            ptr += k
            inv = {v: k for k, v in self.cat_maps[c].items()}
            data[c] = [inv[int(i)] for i in idx]
        for c in self.str_cols:
            meta = self.meta[c]
            fn: Callable = meta.faker_method or self.faker.word
            data[c] = [fn(**(meta.faker_args or {})) for _ in range(n)]

        df = pd.DataFrame(data)
        return df[list(self.meta.keys())]

    # ------------------------------------------------------------------#
    # TRAIN HELPERS
    # ------------------------------------------------------------------#
    def _train_d(self, real, cond, dcls_w: float):
        self.opt_d.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.latent_dim, device=DEVICE)
        fake = self.G(z, cond).detach()

        with amp_autocast():
            # 1) forward real & fake through both heads
            d_real, cls_real = self.D(real, cond)
            d_fake, cls_fake = self.D(fake, cond)

            # 2) gradient penalty
            gp = self._gp(real, fake, cond)

            # 3) adversarial (WGAN‑GP) loss
            w_est = d_real.mean() - d_fake.mean()
            adv_loss = -w_est + self.cfg.gp_weight * gp
            adv_loss += self.cfg.drift_epsilon * (
                    d_real.pow(2).mean() + d_fake.pow(2).mean()
            )

            # 4) classification loss (only if flagged)
            if self.use_cls:
                off = sum(self.cat_sizes[: self.sat_idx])
                true_lbls = cond[:, off: off + self.sat_dim].argmax(1)
                cls_loss = (
                        F.cross_entropy(cls_real, true_lbls)
                        + F.cross_entropy(cls_fake, true_lbls)
                )
            else:
                cls_loss = torch.tensor(0.0, device=real.device)

            # 5) total D loss
            loss = adv_loss + dcls_w * cls_loss

        self.scaler_d.scale(loss).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()
        return w_est.item(), loss.item()

    def _train_g(self, bsz, cond, cls_w: float, corr_w: float):
        self.opt_g.zero_grad(set_to_none=True)
        z = torch.randn(bsz, self.cfg.latent_dim, device=DEVICE)

        # 1) adversarial + classification loss
        with amp_autocast():
            fake_logits, cls_fake = self.D(self.G(z, cond), cond)
            adv_loss = -fake_logits.mean()

            if self.use_cls:
                off = sum(self.cat_sizes[: self.sat_idx])
                true_lbls = cond[:, off: off + self.sat_dim].argmax(1)
                cls_loss = F.cross_entropy(cls_fake, true_lbls)
            else:
                cls_loss = torch.tensor(0.0, device=fake_logits.device)

        # 2) correlation penalty (already in your code)
        real_batch = next(iter(self.loader))[0].to(DEVICE)
        fake_batch = self.G(z, cond)
        N = len(self.num_cols)
        C_real = torch.corrcoef(real_batch[:, :N].T)
        C_fake = torch.corrcoef(fake_batch[:, :N].T)
        corr_loss = F.mse_loss(C_real, C_fake)

        # 3) total G loss
        loss = adv_loss + cls_w * cls_loss + corr_w * corr_loss
        # ── moment matching on flagged numeric columns
        # real_batch & fake_batch already from above
        for idx, col in enumerate(self.num_cols):
            if self.meta[col].match_moments:
                real_vals = real_batch[:, idx]
                fake_vals = fake_batch[:, idx]
                loss += self.cfg.mm_weight * (
                        (real_vals.mean() - fake_vals.mean()) ** 2
                        + (real_vals.std() - fake_vals.std()) ** 2
                )

        self.scaler_g.scale(loss).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        # EMA update
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
            score, _ = self.D(mix, cond)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        return ((grad.view(grad.size(0), -1).norm(2, 1) - 1) ** 2).mean()


def _set_seed(s):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
