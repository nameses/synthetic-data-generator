import copy
import logging
import math
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from faker import Faker
from scipy.stats import norm, rankdata
from sklearn.mixture import GaussianMixture
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.enums import DataType
from models.field_metadata import FieldMetadata


# -------------------- CONFIG --------------------#
@dataclass(slots=True)
class GanConfig:
    latent_dim: int = 512
    hidden_g: int = 1024
    hidden_d: int = 512
    batch_size: int = 64
    n_critic_initial: int = 5
    gp_weight: float = 10.0
    drift_epsilon: float = 1e-3
    # Loss weights
    cls_weight: float = 1.0
    mm_weight: float = 0.5
    cov_weight: float = 0.25
    fm_weight: float = 2.0
    mmd_weight: float = 3.0
    # training schedule
    pretrain_epochs: int = 10
    max_epochs: int = 100
    eval_every: int = 5
    patience: int = 12
    # LR
    g_lr: float = 2e-4
    d_lr: float = 1e-4
    lr_min_ratio: float = 0.01
    seed: int = 42
    # temperature for Gumbel
    tau_start: float = 1.0
    tau_end: float = 0.1


# ------------------ TRANSFORMERS -----------------#
class _BaseTf:
    def fit(self, s: pd.Series): ...
    def transform(self, s: pd.Series) -> np.ndarray: ...
    def inverse(self, v: np.ndarray) -> pd.Series: ...


class _LogTf(_BaseTf):
    def fit(self, s: pd.Series):
        self.mean, self.std = np.log1p(s).mean(), np.log1p(s).std()
        return self
    def transform(self, s): return ((np.log1p(s) - self.mean) / self.std)
    def inverse(self, v): return np.expm1(v * self.std + self.mean)


class _StdTf(_BaseTf):
    def fit(self, s): self.mean, self.std = s.mean(), s.std(); return self
    def transform(self, s): return (s - self.mean) / self.std
    def inverse(self, v): return v * self.std + self.mean


class _ContTf(_BaseTf):
    def fit(self, s):
        q_low, q_hi = s.quantile([0.001, 0.999])
        self.sorted_ = np.sort(s.clip(q_low, q_hi))
        return self
    def transform(self, s):
        u = rankdata(s) / (len(s) + 1)
        return norm.ppf(u)
    def inverse(self, v):
        u = norm.cdf(v).clip(0, 1)
        idx = np.floor(u * (len(self.sorted_) - 1)).astype(int)
        return pd.Series(self.sorted_[idx])


class _DtTf(_ContTf):
    def __init__(self, fmt): self.fmt = fmt
    def fit(self, s): return super().fit(self._to_sec(s))
    def transform(self, s): return super().transform(self._to_sec(s))
    def inverse(self, v): return pd.to_datetime(super().inverse(v), unit='s').strftime(self.fmt)
    def _to_sec(self, s): return pd.to_datetime(s, format=self.fmt).astype('int64') // 10 ** 9


class _GMMTf(_BaseTf):
    def __init__(self, n_components: int = 3): self.n_components = n_components
    def fit(self, s):
        arr = s.dropna().values.reshape(-1, 1)
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=0)
        self.gmm.fit(arr)
        samples, _ = self.gmm.sample(len(arr) * 2)
        self.sorted_ = np.sort(samples.flatten())
        return self
    def transform(self, s):
        arr = s.values
        u = np.searchsorted(self.sorted_, arr) / len(self.sorted_)
        return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    def inverse(self, v):
        u = norm.cdf(v)
        idx = np.floor(u * (len(self.sorted_) - 1)).astype(int)
        return pd.Series(self.sorted_[idx])


# ---------------- NETWORK BLOCKS ----------------#
def lin_sn(i, o): return spectral_norm(nn.Linear(i, o))


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1, self.act = nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
    def forward(self, x):
        h = self.act(self.fc1(x)); h = self.fc2(h)
        skip = self.skip(x) if self.skip else x
        return self.act(h + skip)


class MinibatchStd(nn.Module):
    def forward(self, x):
        std = x.std(0, keepdim=True).mean().expand(x.size(0), 1)
        return torch.cat([x, std], 1)


class _Generator(nn.Module):
    def __init__(self, cfg: GanConfig, num_out: int, cat_dims: List[int], total_steps: int):
        super().__init__()
        self.cfg = cfg
        self.total = total_steps
        self.register_buffer('step', torch.zeros(1))
        hid, c = cfg.hidden_g, cfg.latent_dim + sum(cat_dims)
        self.input_proj = nn.Linear(c, hid)
        self.res_blocks = nn.ModuleList([ResidualBlock(hid, hid) for _ in range(2)])
        self.final_act = nn.LeakyReLU(0.2)
        self.num_head = nn.Linear(hid, num_out) if num_out else None
        self.cat_heads = nn.ModuleList([nn.Linear(hid, k) for k in cat_dims])
    def forward(self, z, cond, hard=False):
        x = F.leaky_relu(self.input_proj(torch.cat([z, cond], 1)), 0.2)
        for b in self.res_blocks: x = b(x)
        h = self.final_act(x)
        out = [self.num_head(h)] if self.num_head else []
        tau = self.cfg.tau_end + 0.5 * (self.cfg.tau_start - self.cfg.tau_end) * (
            1 + math.cos(math.pi * self.step.item() / self.total))
        for head in self.cat_heads:
            out.append(F.gumbel_softmax(head(h), tau=tau, hard=hard))
        self.step += 1
        return torch.cat(out, 1)


class _Discriminator(nn.Module):
    def __init__(self, inp_dim, cond_dim, cfg: GanConfig, sat_dim: Optional[int]):
        super().__init__()
        hid = cfg.hidden_d
        self.shared = nn.Sequential(
            lin_sn(inp_dim + cond_dim, hid), nn.LeakyReLU(0.2),
            lin_sn(hid, hid), nn.LeakyReLU(0.2),
            lin_sn(hid, hid // 2), nn.LeakyReLU(0.2)
        )
        self.adv_head = lin_sn(hid // 2, 1)
        self.use_cls = sat_dim is not None
        self.cls_head = lin_sn(hid // 2, sat_dim) if self.use_cls else None
    def forward(self, x, cond):
        h = self.shared(torch.cat([x, cond], 1))
        adv = self.adv_head(h).view(-1)
        cls = self.cls_head(h) if self.use_cls else None
        return h, adv, cls


# ------------------ MMD UTILS ------------------#
def _rbf_kernel(x, y, sigma=1.0):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2 * x @ y.t()
    return torch.exp(-dist / (2 * sigma ** 2))

def _mmd(x, y):
    Kxx = _rbf_kernel(x, x)
    Kyy = _rbf_kernel(y, y)
    Kxy = _rbf_kernel(x, y)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


# ------------------- MAIN WGAN ------------------#
class WGAN:
    def __init__(self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig):
        self.cfg, self.meta = cfg, meta
        _set_seed(cfg.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # column groups
        self.num_cols = [c for c, m in meta.items() if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
        self.dt_cols = [c for c, m in meta.items() if m.data_type == DataType.DATETIME]
        self.cat_cols = [c for c, m in meta.items() if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}]
        self.str_cols = [c for c, m in meta.items() if m.data_type == DataType.STRING]
        # prepare transformers and data matrix
        mats = []
        self.tfs = {}
        for c in self.num_cols:
            m = meta[c]
            series = real[c]
            if m.transformer == 'log': tf = _LogTf().fit(series)
            elif m.transformer == 'standard': tf = _StdTf().fit(series)
            elif m.transformer == 'gmm': tf = _GMMTf().fit(series)
            else: tf = _ContTf().fit(series)
            self.tfs[c] = tf
            arr = tf.transform(series)
            mats.append(np.asarray(arr).reshape(-1, 1))
        for c in self.dt_cols:
            tf = _DtTf(meta[c].datetime_format).fit(real[c])
            self.tfs[c] = tf
            arr = tf.transform(real[c])
            mats.append(np.asarray(arr).reshape(-1, 1))
        self.cat_sizes, self.cat_maps, self.cat_probs = [], {}, {}
        for c in self.cat_cols:
            uniq = sorted(real[c].astype(str).unique())
            self.cat_sizes.append(len(uniq))
            mp = {v: i for i, v in enumerate(uniq)}
            self.cat_maps[c] = mp
            onehot = np.eye(len(uniq))[real[c].astype(str).map(mp)]
            mats.append(onehot)
            if meta[c].sampling == 'empirical':
                counts = real[c].astype(str).map(mp).value_counts().sort_index()
                self.cat_probs[c] = (counts / counts.sum()).values
        # data loader
        X = torch.tensor(np.hstack(mats), dtype=torch.float32)
        self.loader = DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        # model init
        steps_per = math.ceil(len(self.loader))
        total_steps = cfg.max_epochs * steps_per
        cond_dim = sum(self.cat_sizes)
        preds = [c for c, m in meta.items() if m.is_prediction_used]
        self.sat_dim = (len(self.cat_maps[preds[0]]) if preds else None)
        self.G = _Generator(cfg, len(self.num_cols) + len(self.dt_cols), self.cat_sizes, total_steps).to(self.device)
        self.D = _Discriminator(X.shape[1], cond_dim, cfg, self.sat_dim).to(self.device)
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(0.0, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(0.0, 0.9))
        self.sch_g = CosineAnnealingLR(self.opt_g, cfg.max_epochs, eta_min=cfg.g_lr * cfg.lr_min_ratio)
        self.sch_d = CosineAnnealingLR(self.opt_d, cfg.max_epochs, eta_min=cfg.d_lr * cfg.lr_min_ratio)
        self.ema_G = copy.deepcopy(self.G).eval().requires_grad_(False)
        self.ema_beta = 0.999
        # training state
        self.best_val = float('inf')
        self.no_imp = 0
        self.n_critic = cfg.n_critic_initial
        # logger setup
        self.logger = logging.getLogger('WGAN')
        fh = logging.FileHandler('wgan_train.log')
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        # history for plotting
        self.history = {
            'd_total': [], 'd_adv': [], 'd_gp': [], 'd_drift': [], 'd_cls': [],
            'g_total': [], 'g_adv': [], 'g_cls': [], 'g_mm': [], 'g_cov': [], 'g_fm': [], 'g_mmd': []
        }

    def fit(self, plot_training: bool = False, val_split: float = 0.1):
        # prepare train/val split
        data = self.loader.dataset.tensors[0]
        n = len(data)
        val_len = int(val_split * n)
        train_ds, val_ds = random_split(TensorDataset(data), [n - val_len, val_len])
        self.loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)
        for epoch in range(1, self.cfg.max_epochs + 1):
            # ramped weights
            r = min(1, epoch / self.cfg.pretrain_epochs)
            cls_w = self.cfg.cls_weight * r
            mm_w = self.cfg.mm_weight * (epoch > self.cfg.pretrain_epochs)
            cov_w = self.cfg.cov_weight * (epoch > self.cfg.pretrain_epochs)
            # accumulators
            epoch_d, epoch_g = {'adv':[], 'gp':[], 'drift':[], 'cls':[], 'total':[]}, {'adv':[], 'cls':[], 'mm':[], 'cov':[], 'fm':[], 'mmd':[], 'total':[]}
            # training
            for i, (real,) in enumerate(self.loader, 1):
                real = real.to(self.device)
                cond = self._sample_cond(real.size(0))
                cond = cond.to(self.device)
                d_metrics = self._train_d(real, cond, cls_w)
                for k, v in d_metrics.items(): epoch_d[k].append(v)
                if i % self.n_critic == 0:
                    g_metrics = self._train_g(real, cond, cls_w, mm_w, cov_w)
                    for k, v in g_metrics.items(): epoch_g[k].append(v)
            # scheduler step
            self.sch_g.step(); self.sch_d.step()
            # compute epoch means
            d_means = {k: (sum(v)/len(v) if v else 0.0) for k,v in epoch_d.items()}
            g_means = {k: (sum(v)/len(v) if v else 0.0) for k,v in epoch_g.items()}
            # record history
            self.history['d_total'].append(d_means['total']); self.history['d_adv'].append(d_means['adv'])
            self.history['d_gp'].append(d_means['gp']); self.history['d_drift'].append(d_means['drift'])
            self.history['d_cls'].append(d_means['cls'])
            self.history['g_total'].append(g_means['total']); self.history['g_adv'].append(g_means['adv'])
            self.history['g_cls'].append(g_means['cls']); self.history['g_mm'].append(g_means['mm'])
            self.history['g_cov'].append(g_means['cov']); self.history['g_fm'].append(g_means['fm'])
            self.history['g_mmd'].append(g_means['mmd'])
            # validation & early stopping
            if epoch % self.cfg.eval_every == 0:
                val_losses = []
                for real_batch in self.val_loader:
                    real_val = real_batch[0].to(self.device)
                    cond_val = self._sample_cond(real_val.size(0)).to(self.device)
                    val_losses.append(self._val_step(real_val, cond_val))
                val_loss = sum(val_losses)/len(val_losses) if val_losses else 0.0
                self.logger.info(f"VAL Ep {epoch} | G {val_loss:.4f}")
                if val_loss < self.best_val:
                    self.best_val = val_loss; self.no_imp = 0
                    self.best_weights = self.ema_G.state_dict();
                    torch.save(self.best_weights, 'best.pt')
                else:
                    self.no_imp += 1
                if self.no_imp >= self.cfg.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            # logging summary
            self.logger.info(
                f"Ep {epoch} | D_total {d_means['total']:.4f} | D_adv {d_means['adv']:.4f} | D_gp {d_means['gp']:.4f} | "
                f"D_drift {d_means['drift']:.6f} | D_cls {d_means['cls']:.4f} | G_total {g_means['total']:.4f} | "
                f"G_adv {g_means['adv']:.4f} | G_cls {g_means['cls']:.4f} | G_mm {g_means['mm']:.4f} | G_cov {g_means['cov']:.4f} | "
                f"G_fm {g_means['fm']:.4f} | G_mmd {g_means['mmd']:.4f}"
            )
        if hasattr(self, 'best_weights'):
            self.ema_G.load_state_dict(self.best_weights)
        # plot training curves if requested
        if plot_training:
            import matplotlib.pyplot as plt

            epochs = list(range(1, len(self.history['d_total']) + 1))

            # Define groups of curves to plot separately
            plot_groups = {
                'Discriminator Total': ['d_total'],
                'Discriminator Components': ['d_adv', 'd_gp', 'd_drift', 'd_cls'],
                'Generator Total': ['g_total'],
                'Generator Components': ['g_adv', 'g_cls', 'g_mm', 'g_cov', 'g_fm', 'g_mmd'],
            }

            for title, keys in plot_groups.items():
                plt.figure(figsize=(6, 4))
                for k in keys:
                    plt.plot(epochs, self.history[k], label=k.replace('_', ' ').title(),
                             linestyle='-' if k.endswith('total') else '--')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                fname = f"{title.lower().replace(' ', '_')}.png"
                plt.savefig(fname)
                self.logger.info(f"Saved plot {fname}")
                plt.show()

    def _train_d(self, real, cond, cls_w):
        self.opt_d.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.latent_dim, device=self.device)
        fake = self.G(z, cond).detach()
        _, d_real, cls_real = self.D(real, cond)
        _, d_fake, cls_fake = self.D(fake, cond)
        adv_loss = -(d_real.mean() - d_fake.mean())
        gp = self._gp(real, fake, cond)
        drift = self.cfg.drift_epsilon * (d_real.pow(2).mean() + d_fake.pow(2).mean())
        cls_loss = torch.tensor(0.0, device=self.device)
        if cls_real is not None:
            true = cond[:, sum(self.cat_sizes[:1]):sum(self.cat_sizes[:1])+self.sat_dim].argmax(1)
            cls_loss = cls_w * (F.cross_entropy(cls_real, true) + F.cross_entropy(cls_fake, true))
        total_loss = adv_loss + self.cfg.gp_weight * gp + drift + cls_loss
        total_loss.backward(); self.opt_d.step()
        return {'adv': adv_loss.item(), 'gp': gp.item(), 'drift': drift.item(), 'cls': cls_loss.item(), 'total': total_loss.item()}

    def _train_g(self, real, cond, cls_w, mm_w, cov_w):
        self.opt_g.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.latent_dim, device=self.device)
        fake = self.G(z, cond)
        _, d_fake, cls_fake = self.D(fake, cond)
        adv_loss = -d_fake.mean()
        cls_loss = torch.tensor(0.0, device=self.device)
        if cls_fake is not None:
            true = cond[:, sum(self.cat_sizes[:1]):sum(self.cat_sizes[:1])+self.sat_dim].argmax(1)
            cls_loss = cls_w * F.cross_entropy(cls_fake, true)
        mm_loss = torch.tensor(0.0, device=self.device)
        for idx, c in enumerate(self.num_cols):
            if self.meta[c].match_moments:
                mm_loss = mm_loss + mm_w * ((real[:, idx].mean() - fake[:, idx].mean())**2 + (real[:, idx].std() - fake[:, idx].std())**2)
        cov_loss = torch.tensor(0.0, device=self.device)
        if cov_w > 0:
            real_num = real[:, :len(self.num_cols)]
            fake_num = fake[:, :len(self.num_cols)]
            R_real = torch.corrcoef(real_num.T)
            R_fake = torch.corrcoef(fake_num.T)
            cov_loss = cov_w * F.mse_loss(R_fake, R_real)
            # real_c = real_centered.T @ real_centered
            # fake_c = fake_centered.T @ fake_centered
            # cov_loss = cov_w * F.mse_loss(fake_c, real_c)
        fm_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.fm_weight > 0:
            real_feats = self.D.shared(torch.cat([real, cond], 1))
            fake_feats = self.D.shared(torch.cat([fake, cond], 1))
            fm_loss = self.cfg.fm_weight * F.mse_loss(real_feats.mean(0), fake_feats.mean(0))
        mmd_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.mmd_weight > 0:
            x = real[:, :len(self.num_cols)+len(self.dt_cols)]; y = fake[:, :len(self.num_cols)+len(self.dt_cols)]
            mmd_loss = self.cfg.mmd_weight * _mmd(x, y)
        total_loss = adv_loss + cls_loss + mm_loss + cov_loss + fm_loss + mmd_loss
        total_loss.backward(); self.opt_g.step()
        # EMA update
        with torch.no_grad():
            for p, pe in zip(self.G.parameters(), self.ema_G.parameters()):
                pe.mul_(self.ema_beta).add_(p.data, alpha=(1-self.ema_beta))
        return {'adv': adv_loss.item(), 'cls': cls_loss.item(), 'mm': mm_loss.item(), 'cov': cov_loss.item(), 'fm': fm_loss.item(), 'mmd': mmd_loss.item(), 'total': total_loss.item()}

    def _val_step(self, real, cond):
        with torch.no_grad():
            z = torch.randn(real.size(0), self.cfg.latent_dim, device=self.device)
            fake = self.G(z, cond)
            _, d_fake, _ = self.D(fake, cond)
            return -d_fake.mean().item()

    def generate(self, n: int) -> pd.DataFrame:
        self.ema_G.eval()
        rows, batch_size = [], self.cfg.batch_size
        for _ in range((n - 1)//batch_size + 1):
            cur = min(batch_size, n - len(rows)*batch_size)
            z = torch.randn(cur, self.cfg.latent_dim, device=self.device)
            cond = self._sample_cond(cur).to(self.device)
            rows.append(self.ema_G(z, cond, hard=True).cpu().numpy())
        mat = np.vstack(rows)[:n]
        ptr, data = 0, {}
        for c in self.num_cols:
            col = self.tfs[c].inverse(mat[:, ptr])
            col = col.round().astype(int) if self.meta[c].data_type == DataType.INTEGER else col.round(self.meta[c].decimal_places or 0)
            data[c] = np.clip(col, self.meta[c].min_value, self.meta[c].max_value)
            ptr += 1
        for c in self.dt_cols:
            data[c] = self.tfs[c].inverse(mat[:, ptr]); ptr += 1
        for c, k in zip(self.cat_cols, self.cat_sizes):
            idx = mat[:, ptr:ptr+k].argmax(1); ptr += k
            inv = {v:k for k,v in self.cat_maps[c].items()}
            data[c] = [inv[int(i)] for i in idx]
        for c in self.str_cols:
            fn = self.meta[c].faker_method or Faker().word
            data[c] = [fn(**(self.meta[c].faker_args or {})) for _ in range(n)]
        return pd.DataFrame(data)

    def _sample_cond(self, bsz: int):
        if not self.cat_cols: return torch.zeros(bsz, 0, device=self.device)
        cond = torch.zeros(bsz, sum(self.cat_sizes), device=self.device)
        off = 0
        for c,k in zip(self.cat_cols, self.cat_sizes):
            idx = np.random.choice(k, bsz, p=self.cat_probs.get(c, None) if c in self.cat_probs else None)
            cond[range(bsz), off+idx] = 1; off += k
        return cond

    def _gp(self, real, fake, cond):
        alpha = torch.rand(real.size(0), 1, device=self.device)
        mix = alpha*real + (1-alpha)*fake; mix.requires_grad_(True)
        _, score, _ = self.D(mix, cond)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        return ((grad.view(grad.size(0), -1).norm(2,1)-1)**2).mean()


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
