from __future__ import annotations
import logging, math, random
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
from scipy.stats import norm as _norm
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
    catnum_weight: float = 20.0
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
        h = self.backbone(torch.cat([z, cond], 1))
        num = self.num_head(h) if self.num_head else None  # keep linear – better for quantile mapping
        logits = [head(h) for head in self.cat_heads]
        return num, logits

    def sample(self, z, cond, hard=True):
        num, logits = self.forward(z, cond)
        angle = math.pi * self.step.item() / self.total_steps
        tau = self.cfg.tau_end + 0.5 * (self.cfg.tau_start - self.cfg.tau_end) * (1 + math.cos(angle))
        self.step += 1
        out = [num] if num is not None else []
        for lg in logits:
            out.append(F.gumbel_softmax(lg, tau=tau, hard=hard)
                       if self.cfg.use_gumbel else F.softmax(lg, 1))
        return torch.cat(out, 1)

# --------------------------------------------------
# PIPELINE
# --------------------------------------------------
class VAEPipeline:
    """VAE with empirical quantile mapping for numeric & datetime outputs."""

    def __init__(self, df: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: VAEConfig):
        self.cfg, self.meta = cfg, meta; _set_seed(cfg.seed)
        self.real_df = df.dropna().reset_index(drop=True); self.faker = Faker()
        self.num_cols = [c for c, m in meta.items() if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
        self.dt_cols  = [c for c, m in meta.items() if m.data_type is DataType.DATETIME]
        self.cat_cols = [c for c, m in meta.items() if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}]
        self.str_cols = [c for c in meta if c not in self.num_cols + self.dt_cols + self.cat_cols]

        # store sorted real arrays for quantile mapping
        self.real_sorted = {c: np.sort(self.real_df[c].to_numpy()) for c in self.num_cols}
        self.real_dt_sorted = {c: np.sort(pd.to_datetime(self.real_df[c], format=meta[c].datetime_format)
                                          .astype('int64').to_numpy()) for c in self.dt_cols}

        # transformers ------------------------------------------------------
        self.tfs: Dict[str, _BaseTf] = {}; mats: List[np.ndarray] = []
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
            uniq = sorted(self.real_df[c].astype(str).unique()); mp = {v:i for i,v in enumerate(uniq)}
            self.cat_maps[c] = mp; self.cat_sizes.append(len(uniq))
            idx = self.real_df[c].astype(str).map(mp).values.astype(int)
            cat_idx_mat.append(idx); mats.append(np.eye(len(uniq))[idx].T)
            cnt = pd.Series(idx).value_counts().sort_index(); self.cat_probs[c] = (cnt/cnt.sum()).values

        X = torch.tensor(np.vstack(mats).T, dtype=torch.float32)
        cat_idx = torch.tensor(np.vstack(cat_idx_mat).T, dtype=torch.long)
        self.loader = DataLoader(TensorDataset(X, cat_idx), cfg.batch_size, shuffle=True, drop_last=True)
        self.num_out = len(self.num_cols) + len(self.dt_cols)
        self.cond_dim = sum(self.cat_sizes)
        self.cat_offset = self.num_out

        steps = cfg.epochs * math.ceil(len(self.loader))
        self.encoder = Encoder(X.shape[1], cfg.hidden_dims, cfg.latent_dim, cfg.dropout).to(DEVICE)
        self.decoder = Decoder(cfg, self.num_out, self.cat_sizes, steps).to(DEVICE)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sch = CosineAnnealingLR(self.opt, cfg.epochs, eta_min=cfg.lr * cfg.lr_min_ratio)
        self.metrics = {k: [] for k in ('rec','cat','kl','corr','total')}
        # per-feature std (in transformed space) for scaled reconstruction loss
        feats = X[:, :self.num_out]
        stds = feats.std(0, unbiased=False) + 1e-6
        self.num_stds = stds.to(DEVICE).view(1, -1)

    # ------------------------------------------------------------------
    def _reparam(self, mu, lv):
        std = torch.exp(0.5 * lv); return mu + std * torch.randn_like(std)
    def _beta(self,e):
        cycle_len = self.cfg.epochs // self.cfg.kl_cycles
        phase = (e - 1) % cycle_len
        beta = 0.0 if phase < int(cycle_len * (1 - self.cfg.kl_ratio)) \
            else (phase - int(cycle_len * (1 - self.cfg.kl_ratio))) / (cycle_len * self.cfg.kl_ratio)
        beta *= self.cfg.beta_end
        return beta

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
        for ep in range(1, self.cfg.epochs+1):
            sums = {k:0.0 for k in self.metrics}
            for x, ci in self.loader:
                x, ci = x.to(DEVICE), ci.to(DEVICE)
                mu, lv = self.encoder(x)
                z = self._reparam(mu, lv)
                teacher_ratio = max(0.0, 1.0 - ep / (self.cfg.epochs * 0.6))
                use_teacher = (torch.rand(self.cfg.batch_size, 1, device=DEVICE) < teacher_ratio)  # still bool
                cond = x[:, self.num_out: self.num_out + self.cond_dim]
                # invert with ~, then cast to float
                cond = cond * use_teacher.float() + 0.0 * (~use_teacher).float()
                u, logits = self.decoder(z, cond)
                # scaled reconstruction
                rec = F.mse_loss((u - x[:, :self.num_out]) / self.num_stds, torch.zeros_like(u))
                # categorical CE
                cat = sum(F.cross_entropy(lg, ci[:, i].clamp(0, lg.size(1)-1)) for i, lg in enumerate(logits)) / len(logits)
                # KL
                kl = -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1)) * self._beta(ep)

                kl_cat = 0.0
                for i, k in enumerate(self.cat_sizes):
                    prior = torch.tensor(self.cat_probs[self.cat_cols[i]], device=DEVICE)
                    kl_cat += F.kl_div(F.log_softmax(logits[i], 1), prior.repeat(self.cfg.batch_size, 1), reduction='batchmean')

                catnum = 0.0
                col_off = 0
                x_num = x[:, :self.num_out]  # (B, N)
                u_num = u  # (B, N)
                for i, k in enumerate(self.cat_sizes):
                    real_1h = x[:, self.num_out + col_off: self.num_out + col_off + k]  # (B, K)
                    pred_1h = F.softmax(logits[i], 1)  # (B, K)
                    col_off += k

                    # (K, N) mean matrix for all numeric heads at once
                    w_real = real_1h.sum(0).clamp_min(1.0).unsqueeze(1)  # (K, 1)
                    w_pred = pred_1h.sum(0).clamp_min(1.0).unsqueeze(1)

                    m_real = real_1h.T @ x_num / w_real  # (K, N)
                    m_pred = pred_1h.T @ u_num / w_pred

                    catnum += F.mse_loss(m_pred, m_real, reduction='mean')

                catnum = catnum / len(self.cat_sizes)

                # group variances
                var_loss = torch.tensor(0.0, device=DEVICE)
                col_off = 0
                x_num = x[:, :self.num_out]  # (B, N)
                u_num = u  # (B, N)
                for i, k in enumerate(self.cat_sizes):
                    real_1h = x[:, self.num_out + col_off: self.num_out + col_off + k]  # (B, K)
                    pred_1h = F.softmax(logits[i], 1)  # (B, K)
                    col_off += k

                    # 1) expand to (B, N, K) so that we can do mask * value²
                    real_mask = real_1h.unsqueeze(1)  # (B, 1, K)
                    pred_mask = pred_1h.unsqueeze(1)  # (B, 1, K)
                    x_sq = x_num.unsqueeze(2) ** 2  # (B, N, 1)
                    u_sq = u_num.unsqueeze(2) ** 2  # (B, N, 1)

                    # 2) sum over batch → (N, K)
                    sq_real = (real_mask * x_sq).sum(0)
                    sq_pred = (pred_mask * u_sq).sum(0)

                    # 3) divide by counts (1, K) → (N, K)
                    w_real = real_1h.sum(0).clamp_min(1.0).unsqueeze(0)
                    w_pred = pred_1h.sum(0).clamp_min(1.0).unsqueeze(0)
                    var_real_nk = sq_real / w_real  # (N, K)
                    var_pred_nk = sq_pred / w_pred  # (N, K)

                    # 4) transpose → (K, N) to match your m_real shape
                    var_real = var_real_nk.T
                    var_pred = var_pred_nk.T

                    var_loss += F.mse_loss(var_pred, var_real, reduction='mean')

                var_loss = var_loss / len(self.cat_sizes)

                # Pearson-cov for numeric↔numeric (keeps shape)
                corr = self.cfg.corr_weight * F.mse_loss(
                    torch.corrcoef(x[:, :self.num_out].T),
                    torch.corrcoef(u.T))

                # pairwise category one-hot covariance
                catcat = 0.0
                block = torch.cat([F.softmax(lg, 1) for lg in logits], 1)  # (B, ΣK)
                target = x[:, self.num_out: self.num_out + self.cond_dim]  # real one-hots
                catcat = F.mse_loss(torch.corrcoef(block.T), torch.corrcoef(target.T))

                loss = rec + cat + kl + corr + self.cfg.catnum_weight * catnum + 0.5 * catcat + 0.1 * kl_cat + 0.5 * var_loss
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                for k,v in zip(self.metrics,(rec,cat,kl,corr,loss)): sums[k]+=v.item()
            for k in self.metrics: self.metrics[k].append(sums[k]/len(self.loader))
            LOGGER.info("Ep %03d | Rec %.4f | Cat %.4f | KL %.4f | Corr %.4f | Tot %.4f",
                        ep, *[self.metrics[k][-1] for k in ('rec','cat','kl','corr','total')])
            self.sch.step()
        # plot
        ep_range = range(1, self.cfg.epochs+1)
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

                # 1) Sample raw numeric+cat in one go (sample() uses its internal τ schedule)
                z = torch.randn(cur, self.cfg.latent_dim, device=DEVICE) * temperature
                out1 = self.decoder.sample(z,
                                           torch.zeros(cur, self.cond_dim, device=DEVICE),
                                           hard=True)  # shape (cur, num_out+cond_dim)
                num1 = out1[:, :self.num_out]  # (cur, num_out)
                cond_pred = out1[:, self.num_out:]  # (cur, cond_dim)

                # 2) Re-forward numeric head with the predicted one-hots
                num2, _ = self.decoder.forward(z, cond_pred)  # (cur, num_out)

                # 3) Build final output matrix just like training data
                dec_out = torch.cat([num2, cond_pred], dim=1).cpu().numpy()
                rows.append(dec_out)
        mat=np.vstack(rows)[:n]
        data:Dict[str,List]= {}
        ptr=0

        # ---- NUMERIC + DATETIME heads --------------------------------------
        ptr = 0
        for c in self.num_cols + self.dt_cols:
            raw = mat[:, ptr]  # decoder’s raw output for column c
            ptr += 1

            # 1) Standardise per batch → wide range
            z = (raw - raw.mean()) / (raw.std() + 1e-6)
            u = scipy_stats_norm.cdf(z).clip(1e-6, 1 - 1e-6)  # avoid 0/1 edges

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