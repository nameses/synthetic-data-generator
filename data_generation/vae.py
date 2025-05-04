from __future__ import annotations
import logging, math, random
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List
from scipy.stats import norm as scipy_stats_norm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_generation.transformers import (
    _StdTf, _MinMaxTf, _LogTf, _ZeroInflatedTf, _BoundedTf, _ContTf, _DtTf, _BaseTf)
from models.field_metadata import FieldMetadata
from models.enums import DataType
from faker import Faker
from torch.amp import GradScaler, autocast
import bitsandbytes as bnb
import math
from torch.nn.functional import huber_loss

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
@dataclass(slots=True)
class VAEConfig:
    latent_dim: int = 128
    hidden_dims: List[int] = (512, 256)
    dropout: float = 0.15
    batch_size: int = 256
    accum_steps: int = 2
    epochs: int = 250
    lr: float = 3e-4
    weight_decay: float = 1e-5
    lr_min_ratio: float = 0.1
    seed: int = 42
    # KL annealing
    beta_start: float = 0.0
    beta_end: float = 1.0
    kl_anneal: str = "cyclic"
    kl_cycles: int = 4
    kl_ratio: float = 0.25
    kl_warmup_epochs: int = 60
    # regularisation weights
    corr_weight: float = 3.0
    corr_warm: int = 40
    catnum_weight: float = 15.0
    moment_weight: float = 1.0
    # gumbel
    use_gumbel: bool = True
    tau_start: float = 2.5
    tau_end: float = 0.5

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --------------------------------------------------
# MODEL PARTS
# --------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, inp: int, hidden: List[int], z: int, drop: float):
        super().__init__(); layers=[]; d=inp
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.fc_mu, self.fc_lv = nn.Linear(d, z), nn.Linear(d, z)
    def forward(self, x):
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig, num_out: int, cat_dims: List[int], total_steps: int):
        super().__init__()
        self.cfg, self.total_steps = cfg, total_steps
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(k, embedding_dim=8) for k in cat_dims
        ])
        self.register_buffer("step", torch.zeros(1))
        d = cfg.latent_dim + sum(cat_dims)
        layers = []
        for h in cfg.hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.num_head = nn.Linear(d, num_out) if num_out else None
        self.cat_heads = nn.ModuleList(nn.Linear(d, k) for k in cat_dims)

    def forward(self, z, cond):
        if cond.dtype == torch.long:
            cond = self.embed_categories(cond)
        h = self.backbone(torch.cat([z, cond], 1))
        num = self.num_head(h) if self.num_head else None  # keep linear – better for quantile mapping
        logits = [head(h) for head in self.cat_heads]
        return num, logits

    def sample(self, z, cond, hard=True):
        if cond.dtype == torch.long:
            cond = self.embed_categories(cond)
        num, logits = self.forward(z, cond)
        angle = math.pi * self.step.item() / self.total_steps
        tau = self.cfg.tau_end + 0.5 * (self.cfg.tau_start - self.cfg.tau_end) * (1 + math.cos(angle))
        self.step += 1
        out = [num] if num is not None else []
        for lg in logits:
            out.append(F.gumbel_softmax(lg, tau=tau, hard=hard) if self.cfg.use_gumbel else F.softmax(lg, 1))
        return torch.cat(out, 1)

    def embed_categories(self, cat_idx: torch.Tensor) -> torch.Tensor:
        embeddings = [emb(cat_idx[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        return torch.cat(embeddings, dim=1)  # (B, total_embedding_dim)


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------
class VAEPipeline:
    """VAE with empirical quantile mapping for numeric & datetime outputs."""

    def __init__(self, df: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: VAEConfig):
        self.cfg, self.meta = cfg, meta
        _set_seed(cfg.seed)
        self.real_df = df.dropna().reset_index(drop=True)
        self.faker = Faker()
        self.num_cols = [c for c, m in meta.items() if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
        self.dt_cols  = [c for c, m in meta.items() if m.data_type is DataType.DATETIME]
        self.cat_cols = [c for c, m in meta.items() if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}]
        self.str_cols = [c for c in meta if c not in self.num_cols + self.dt_cols + self.cat_cols]

        # store sorted real arrays for quantile mapping
        self.real_sorted = {c: np.sort(self.real_df[c].to_numpy()) for c in self.num_cols}
        self.real_dt_sorted = {c: np.sort(pd.to_datetime(self.real_df[c], format=meta[c].datetime_format)
                                          .astype('int64').to_numpy()) for c in self.dt_cols}

        # transformers ------------------------------------------------------
        self.tfs: Dict[str, _BaseTf] = {}
        mats: List[np.ndarray] = []
        tf_map = {'standard': _StdTf,'minmax': _MinMaxTf,'log': _LogTf,
                  'zero_inflated': _ZeroInflatedTf,'bounded': _BoundedTf}
        for c in self.num_cols:
            tf = tf_map.get(getattr(meta[c],'transformer',None), _ContTf)().fit(self.real_df[c])
            self.tfs[c] = tf; mats.append(tf.transform(self.real_df[c]))
        for c in self.dt_cols:
            tf = _DtTf(meta[c].datetime_format).fit(self.real_df[c])
            self.tfs[c] = tf; mats.append(tf.transform(self.real_df[c]))

        # categorical one‑hots ---------------------------------------------
        self.cat_maps, self.cat_sizes, self.cat_probs = {}, [], {}
        cat_idx_mat: List[np.ndarray] = []
        for c in self.cat_cols:
            uniq = sorted(self.real_df[c].astype(str).unique())
            mp = {v:i for i,v in enumerate(uniq)}
            self.cat_maps[c] = mp
            self.cat_sizes.append(len(uniq))
            idx = self.real_df[c].astype(str).map(mp).values.astype(int)
            cat_idx_mat.append(idx)
            # mats.append(np.eye(len(uniq))[idx].T)
            cnt = pd.Series(idx).value_counts().sort_index()
            self.cat_probs[c] = (cnt/cnt.sum()).values

        X = torch.tensor(np.vstack(mats).T, dtype=torch.float32)
        self.cat_idx = torch.tensor(np.vstack(cat_idx_mat).T, dtype=torch.long)

        weights = torch.bincount(self.cat_idx[:, 0]).float()
        weights = 1.0 / weights[self.cat_idx[:, 0]]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

        self.loader = DataLoader(TensorDataset(X, self.cat_idx), cfg.batch_size, sampler=sampler, drop_last=True)
        self.num_out = len(self.num_cols) + len(self.dt_cols)
        self.cond_dim = sum(self.cat_sizes)
        self.cat_offset = self.num_out

        steps = cfg.epochs * math.ceil(len(self.loader))
        self.encoder = Encoder(X.shape[1], cfg.hidden_dims, cfg.latent_dim, cfg.dropout).to(DEVICE)
        self.decoder = Decoder(cfg, self.num_out, self.cat_sizes, steps).to(DEVICE)

        self.opt = bnb.optim.Adam8bit(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=cfg.lr,
            betas=(0.9, 0.95),  # bnb default
            weight_decay=cfg.weight_decay,
        )
        self.sch = CosineAnnealingLR(self.opt, cfg.epochs, eta_min=cfg.lr * cfg.lr_min_ratio)
        self.metrics = {k: [] for k in ('rec','cat','kl','corr','group','total')}
        # per-feature std (in transformed space) for scaled reconstruction loss
        feats = X[:, :self.num_out]
        stds = feats.std(0, unbiased=False) + 1e-6
        self.num_stds = stds.to(DEVICE).view(1, -1)

        self.prev_corr = None

    # ------------------------------------------------------------------
    def _reparam(self, mu, lv):
        std = torch.exp(0.5 * lv); return mu + std * torch.randn_like(std)
    def _beta(self, epoch):
        warm = int(self.cfg.epochs * 0.40)
        return 0.2 * min(1.0, epoch / warm)

    def _batch_group_means(self, one_hot: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        one_hot : (B, K)   – category indicators (hard or soft)
        values  : (B,)     – numeric column
        returns : (K,)     – mean of *values* for every category
        """
        weights = one_hot.sum(0).clamp_min(1.0)  # avoid /0
        return (one_hot.T @ values) / weights  # (K,)

    # ------------------------------------------------------------------
    def fit(self, verbose: bool=False):
        LOGGER.info("Training VAE for %d epochs", self.cfg.epochs)

        scaler = GradScaler(enabled=torch.cuda.is_available())

        for ep in range(1, self.cfg.epochs + 1):
            sums = {k: 0.0 for k in self.metrics}

            accum_n = 0
            for x, ci in self.loader:
                x, ci = x.to(DEVICE), ci.to(DEVICE)
                with autocast(device_type='cuda',dtype=torch.float16, enabled=torch.cuda.is_available()):
                    mu, lv = self.encoder(x)
                    lv = lv.clamp(min=-8.0, max=8.0)
                    z = self._reparam(mu, lv)

                    # ---------- STEP ① : unconditional category prediction ----------
                    zero_cond = torch.zeros(x.size(0), self.cond_dim, device=DEVICE)
                    _, logits_cat = self.decoder(z, zero_cond)  # (B, ΣK)
                    cat_soft_blocks = torch.cat(
                        [F.softmax(lg, 1) for lg in logits_cat], 1)  # (B, ΣK)

                    # categorical CE (before any mixing)
                    cat = sum(
                        F.cross_entropy(lg, ci[:, i].clamp(0, lg.size(1) - 1))
                        for i, lg in enumerate(logits_cat)
                    ) / len(logits_cat)

                    # ---------- teacher-forcing schedule ----------
                    teacher_ratio = max(0.0, 1.0 - ep / (self.cfg.epochs * 0.9))
                    mask = (torch.rand(x.size(0), 1, device=DEVICE) < teacher_ratio).float()
                    real_cond = torch.cat([
                        F.one_hot(ci[:, i], num_classes=k).float()
                        for i, k in enumerate(self.cat_sizes)
                    ], dim=1)
                    cond_mix = real_cond * mask + cat_soft_blocks * (1.0 - mask)
                    # ---------- STEP ② : numeric / datetime prediction ----------
                    u, _ = self.decoder(z, cond_mix)  # (B, N)

                    # ---------- CORRELATION REGULARISER (GPU, fast) ----------
                    eps = 1e-6
                    num_real = x[:, :self.num_out].float()  # already on DEVICE
                    num_pred = u.float()

                    num_real_z = (num_real - num_real.mean(0)) / (num_real.std(0) + 1e-6)
                    num_pred_z = (num_pred - num_pred.mean(0)) / (num_pred.std(0) + 1e-6)

                    mask = (num_real.std(0) > eps) & (num_pred.std(0) > eps)
                    if mask.sum() < 2:
                        real_corr_loss = torch.tensor(0.0, device=DEVICE)
                    else:
                        def _corr(m):  # m: (B,D)
                            m = m[:, mask]
                            xm = m - m.mean(0)
                            cov = (xm.T @ xm) / (xm.size(0) - 1)
                            std = xm.std(0).unsqueeze(1) + eps
                            return cov / (std @ std.T)

                        cur_loss = F.mse_loss(_corr(num_real_z), _corr(num_pred_z))
                        if self.prev_corr is None:
                            real_corr_loss = cur_loss
                        else:
                            real_corr_loss = 0.5 * self.prev_corr + 0.5 * cur_loss
                        self.prev_corr = real_corr_loss.detach()

                    rec = huber_loss(
                        (u - x[:, :self.num_out]) / self.num_stds,
                        torch.zeros_like(u),
                        delta=0.3
                    )

                    # KL (latent)
                    beta = self._beta(ep)
                    alpha = 1.0 if ep < self.cfg.corr_warm else min(3.0, 1.0 + 0.05 * (ep - self.cfg.corr_warm))
                    kl = -0.5 * torch.mean(
                        torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1)
                    ) * beta

                    # KL (categorical vs empirical prior)
                    kl_cat = 0.0
                    for i, k in enumerate(self.cat_sizes):
                        prior = torch.tensor(self.cat_probs[self.cat_cols[i]], device=DEVICE)
                        kl_cat += F.kl_div(
                            F.log_softmax(logits_cat[i], 1),
                            prior.repeat(self.cfg.batch_size, 1),
                            reduction="batchmean",
                        )

                    one_hot_all = cond_mix.split(self.cat_sizes, dim=1)
                    catnum, var_loss = 0.0, 0.0

                    for k, one_hot in zip(self.cat_sizes, one_hot_all):
                        w = one_hot.sum(0).clamp_min(1.0)  # (K,)
                        w = w.unsqueeze(1)  # (K,1)

                        # means
                        mean_real = (one_hot.T @ num_real) / w  # (K,N)
                        mean_pred = (one_hot.T @ u) / w
                        catnum += F.mse_loss(mean_pred, mean_real)

                        # variances
                        var_real = (one_hot.T @ (num_real ** 2)) / w - mean_real ** 2
                        var_pred = (one_hot.T @ (u ** 2)) / w - mean_pred ** 2
                        var_loss += F.mse_loss(var_pred, var_real)

                    catnum = catnum / len(self.cat_sizes)
                    var_loss = var_loss / len(self.cat_sizes)

                    corr = real_corr_loss

                    var_pen = ((u.std(0) - 1.0) ** 2).mean()

                    group_mse = 0.0
                    offset = 0
                    for k in self.cat_sizes:
                        mask = cond_mix[:, offset:offset + k]
                        offset += k
                        denom = mask.sum(0).clamp_min(1.0)  # (k,)
                        # broadcast mask → (B,k,1) and compute per-bucket mean
                        mean_real = (mask.T @ num_real) / denom.unsqueeze(1)
                        mean_pred = (mask.T @ num_pred) / denom.unsqueeze(1)
                        group_mse += F.mse_loss(mean_pred, mean_real)
                    group_mse = group_mse / len(self.cat_sizes)

                    w_corr   = (6.0 if ep < self.cfg.corr_warm else 12.0) * alpha
                    w_catnum = (0.2 if ep < self.cfg.corr_warm else 0.4)  * alpha
                    w_var    = (0.4 if ep < self.cfg.corr_warm else 0.8)  * alpha
                    w_group  = (0.0 if ep < self.cfg.corr_warm else 2.0)  * alpha

                    loss = (
                       rec
                       + cat
                       + kl
                       + w_corr * corr
                       + w_catnum * catnum
                       + w_var * var_loss
                       + w_group * group_mse
                       + 0.1 * kl_cat
                       + 0.2 * var_pen
                    ) / self.cfg.accum_steps

                    # self.opt.zero_grad()
                    # loss.backward()
                    # self.opt.step()

                    for k, v in zip(self.metrics, (rec, cat, kl, corr, group_mse, loss)):
                        sums[k] += v.item()

                scaler.scale(loss).backward()
                accum_n += 1

                if accum_n == self.cfg.accum_steps:
                    scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)

                    scaler.step(self.opt)
                    scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    accum_n = 0
            if verbose:
                for k in self.metrics:
                    self.metrics[k].append(sums[k] / len(self.loader))

            LOGGER.info(
                "Ep %03d | Rec %.4f | Cat %.4f | KL %.4f | Corr %.4f | Group %.4f | Tot %.4f",
                ep,
                *[self.metrics[k][-1] for k in ("rec", "cat", "kl", "corr", "group", "total")],
            )
            self.sch.step()

            if accum_n > 0:  # flush leftover grads
                scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)
                scaler.step(self.opt)
                scaler.update()
                self.opt.zero_grad(set_to_none=True)
                accum_n = 0

        # plot
        ep_range = range(1, self.cfg.epochs+1)
        if verbose:
            for k, v in self.metrics.items():
                plt.figure(figsize=(6,4)); plt.plot(ep_range, v); plt.title(k); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    def _quantile_map(self, u: np.ndarray, sorted_real: np.ndarray) -> np.ndarray:
        """Map uniform *u* in [0,1] onto empirical sorted_real via linear interpolation."""
        u = np.clip(u, 1e-6, 1-1e-6)
        pos = u * (len(sorted_real)-1)
        low = np.floor(pos).astype(int); high = np.clip(low+1,0,len(sorted_real)-1)
        frac = pos - low
        return sorted_real[low]*(1-frac) + sorted_real[high]*frac

    # ------------------------------------------------------------------
    def generate(self, n: int, temperature: float=1.0) -> pd.DataFrame:
        self.decoder.eval(); rows=[]; bs=self.cfg.batch_size
        with torch.no_grad():
            for start in range(0, n, bs):
                cur = min(bs, n - start)

                z = torch.randn(cur, self.cfg.latent_dim, device=DEVICE) * temperature

                out1 = self.decoder.sample(
                    z,
                    torch.zeros(cur, self.cond_dim, device=DEVICE),
                    hard=True)

                num1 = out1[:, :self.num_out]
                cond_pred = out1[:, self.num_out:]

                num2, _ = self.decoder.forward(z, cond_pred)

                dec_out = torch.cat([num2, cond_pred], dim=1).cpu().numpy()
                rows.append(dec_out)
        mat=np.vstack(rows)[:n]
        data:Dict[str,List]= {}
        ptr=0

        # numeric
        ptr = 0
        for c in self.num_cols + self.dt_cols:
            raw = mat[:, ptr]
            ptr += 1

            z = (raw - raw.mean()) / (raw.std() + 1e-6)
            u = scipy_stats_norm.cdf(z).clip(1e-6, 1 - 1e-6)

            tf = self.tfs[c]

            if isinstance(tf, _ZeroInflatedTf):
                # -------- zero-inflated (delays) ----------
                mask_zero = (u < tf.zero_rate)  # keep exact zero-rate
                positives = u[~mask_zero]
                pos_real = self.real_df[c][self.real_df[c] > 0].sort_values().to_numpy()

                # linear interpolation on the positive part
                idx = positives * (len(pos_real) - 1)
                lo = np.floor(idx).astype(int)
                hi = np.ceil(idx).astype(int)
                frac = idx - lo
                val = pos_real[lo] * (1 - frac) + pos_real[hi] * frac

                gen = np.zeros_like(u)
                gen[~mask_zero] = val
                data[c] = gen

            elif isinstance(tf, _BoundedTf):
                # -------- bounded (Age) ----------
                sorted_real = self.real_df[c].sort_values().to_numpy()
                idx = u * (len(sorted_real) - 1)
                lo = np.floor(idx).astype(int)
                hi = np.ceil(idx).astype(int)
                frac = idx - lo
                gen = sorted_real[lo] * (1 - frac) + sorted_real[hi] * frac
                data[c] = np.round(gen).astype(int)


            elif isinstance(tf, _DtTf):
                sorted_sec = (
                        pd.to_datetime(self.real_df[c], format=self.meta[c].datetime_format, errors="coerce")
                        .astype("int64")
                        .to_numpy() // 10 ** 9
                )
                idx = u * (len(sorted_sec) - 1)
                lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
                frac = idx - lo
                sec = sorted_sec[lo] * (1 - frac) + sorted_sec[hi] * frac
                dates = pd.Series(pd.to_datetime(sec, unit="s"), name=c)
                min_d, max_d = self.real_df[c].min(), self.real_df[c].max()
                dates = dates.clip(lower=min_d, upper=max_d)
                data[c] = dates.dt.strftime(self.meta[c].datetime_format)


            else:
                # -------- plain continuous / log-scaled ----------
                sorted_real = self.real_df[c].sort_values().to_numpy()
                idx = u * (len(sorted_real) - 1)
                lo = np.floor(idx).astype(int)
                hi = np.ceil(idx).astype(int)
                frac = idx - lo
                gen = sorted_real[lo] * (1 - frac) + sorted_real[hi] * frac

                # If original was DECIMAL – round to required precision
                m = self.meta[c]
                if m.data_type is DataType.INTEGER:
                    gen = np.round(gen).astype(int)
                elif m.data_type is DataType.DECIMAL:
                    gen = np.round(gen, m.decimal_places or 0)

                data[c] = gen

        # categoricals ----------------------------------------------------
        for c, k in zip(self.cat_cols, self.cat_sizes):
            idx = mat[:, ptr:ptr + k].argmax(1)
            inv = {v: k for k, v in self.cat_maps[c].items()}
            data[c] = [inv[int(j)] for j in idx]
            ptr += k

        # strings ---------------------------------------------------------
        for c in self.str_cols:
            meta = self.meta[c]
            fn: Callable = meta.faker_method or self.faker.word
            data[c] = [fn(**(meta.faker_args or {})) for _ in range(n)]

        return pd.DataFrame(data)[self.real_df.columns]