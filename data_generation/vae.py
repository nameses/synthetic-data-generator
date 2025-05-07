# ─────────────  data_generation/vae.py  ─────────────
import logging
import math, random, itertools, torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

import numpy as np, pandas as pd
from faker import Faker

from models.enums import DataType
from models.field_metadata import FieldMetadata
from data_generation.transformers import _ContTf, _DtTf

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fh = logging.FileHandler(f"train_{ts}.log", mode="w", encoding="utf‑8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(fh)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(43);  np.random.seed(43);  random.seed(43)
fake = Faker()

# ---------- helper losses -------------------------------------------------
def sliced_wasserstein(x, y, n_proj=128):
    P = F.normalize(torch.randn(x.size(1), n_proj, device=x.device), dim=0)
    return ( (x @ P).sort(0).values - (y @ P).sort(0).values ).abs().mean()

def batch_corr(m):                           # Pearson on a batch tensor
    m = m - m.mean(0, keepdim=True)
    m = m / (m.std(0, keepdim=True) + 1e-6)
    return (m.t() @ m) / m.size(0)

# ---------- encoder / decoder ---------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU()
        )
        self.mu  = nn.Linear(256, z_dim)
        self.logv= nn.Linear(256, z_dim)

    def forward(self, x):
        h = self.net(x)
        mu, logv = self.mu(h), self.logv(h).clamp(-6, 6)
        std = torch.exp(0.5*logv)
        return mu + std*torch.randn_like(std), mu, logv


class Decoder(nn.Module):
    def __init__(self, num_dim, cat_dims, z_dim):
        super().__init__()
        self.cat_dims = cat_dims
        self.cond_dim = sum(cat_dims)

        half_z = z_dim // 2
        if self.cond_dim > 0:
            self.cat_head = nn.Sequential(
                nn.Linear(half_z, 256), nn.GELU(),
                nn.Linear(256, self.cond_dim)
            )
        else:
            self.cat_head = None
        self.num_head = nn.Sequential(
            nn.Linear(half_z + self.cond_dim, 256), nn.GELU(),
            nn.Linear(256, num_dim)
        )

    def forward(self, z_cat, z_num, tau, hard):
        # if we have any categories, do gumbel‐softmax; else no‐op
        if self.cond_dim > 0:
            logits = self.cat_head(z_cat).split(self.cat_dims, 1)
            cats = [F.gumbel_softmax(l, tau=tau, hard=hard) for l in logits]
            cond = torch.cat(cats, 1)
        else:
            logits = []
            # empty conditioning vector of shape [batch, 0]
            cond = torch.empty(z_cat.size(0), 0, device=z_cat.device)
        num = self.num_head(torch.cat([z_num, cond], 1))
        return num, logits, cond

# ---------- main wrapper ---------------------------------------------------
class VAE:
    def __init__(self, real_df: pd.DataFrame, meta: dict[str, FieldMetadata], latent_dim=128):
        self.meta = meta
        self.real = real_df.reset_index(drop=True)

        # column groups
        self.num_cols = [c for c,m in meta.items()
                         if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
        self.dt_cols  = [c for c,m in meta.items() if m.data_type == DataType.DATETIME]
        self.cat_cols = [c for c,m in meta.items()
                         if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}]
        self.str_cols = [c for c,m in meta.items() if m.data_type == DataType.STRING]

        # transformers  -----------------------------------------------------
        self.tf_num = {c: _ContTf().fit(self.real[c]) for c in self.num_cols}
        self.tf_dt  = {c: _DtTf(meta[c].datetime_format).fit(self.real[c])
                       for c in self.dt_cols}
        self.cat_maps = {c: pd.Categorical(self.real[c]).categories
                         for c in self.cat_cols}

        # model dims
        self.cont_cols = self.num_cols + self.dt_cols
        self.num_dim   = len(self.cont_cols)
        self.cat_dims  = [len(self.cat_maps[c]) for c in self.cat_cols]
        self.latent    = latent_dim

        in_dim = self.num_dim + sum(self.cat_dims)
        self.enc = Encoder(in_dim, latent_dim).to(DEVICE)
        self.dec = Decoder(self.num_dim, self.cat_dims, latent_dim).to(DEVICE)
        self.opt = torch.optim.AdamW(
            itertools.chain(self.enc.parameters(), self.dec.parameters()),
            lr=2e-4, weight_decay=1e-4
        )

        # real corr-matrix (continuous only)
        real_cont = np.stack(
            [self.tf_num[c].transform(self.real[c]) for c in self.num_cols] +
            [self.tf_dt[c].transform(self.real[c])  for c in self.dt_cols],
            1
        )
        self.real_corr = torch.tensor(np.corrcoef(real_cont, rowvar=False),
                                      dtype=torch.float32, device=DEVICE)

    # ----------- utilities -------------------------------------------------
    def _encode(self, df):
        num = torch.stack([torch.tensor(self.tf_num[c].transform(df[c]).copy(), dtype=torch.float32)
                           for c in self.num_cols] +
                          [torch.tensor(self.tf_dt[c].transform(df[c]).copy(), dtype=torch.float32)
                           for c in self.dt_cols], 1)
        cats = []
        for c,dim in zip(self.cat_cols, self.cat_dims):
            idx = pd.Categorical(df[c], categories=self.cat_maps[c]).codes
            cats.append(torch.eye(dim, device=DEVICE)[idx])
        return torch.cat([num.to(DEVICE)] + cats, 1)

    # ----------- training --------------------------------------------------
    def fit(self, epochs=250, bs=512, kl_max=1.0, n_cycles=4, valid=0.1, verbose=True):
        idx = np.arange(len(self.real)); np.random.shuffle(idx)
        split = int(len(idx)*(1-valid))
        tr, va = idx[:split], idx[split:]

        best, patience = 1e9, 0
        if verbose:
            history_ep, history_swd, history_beta = [], [], []
        for ep in range(1, epochs+1):
            np.random.shuffle(tr)
            beta = kl_max * max(0,(math.cos(n_cycles*math.pi*ep/epochs)+1)/2)

            for beg in range(0,len(tr),bs):
                batch = self.real.iloc[tr[beg:beg+bs]].reset_index(drop=True)
                x = self._encode(batch)
                z, mu, logv = self.enc(x)
                z_cat, z_num = torch.chunk(z, 2, 1)

                num_hat, logits, cond = self.dec(
                    z_cat, z_num, tau=max(0.5,1-ep/epochs), hard=False
                )

                # losses
                loss_num = F.mse_loss(num_hat, x[:, :self.num_dim])
                # categorical loss (only if we actually have categories)
                if self.cat_dims:
                    loss_cat = 0
                    start = self.num_dim
                    for l, dim in zip(logits, self.cat_dims):
                        tgt = x[:, start:start + dim].argmax(1)
                        loss_cat += F.cross_entropy(l, tgt)
                        start += dim
                    loss_cat = loss_cat / len(self.cat_dims)
                else:
                    loss_cat = torch.tensor(0.0, device=DEVICE)

                kl = -0.5*(1+logv - mu.pow(2) - logv.exp()).mean()

                with torch.no_grad():
                    fake = self.generate(len(batch), temp=0.5, _cpu=False)['cont']
                swd = sliced_wasserstein(fake, x[:,:self.num_dim])
                if fake.size(1) > 1:
                    corr_mat = batch_corr(fake)
                    corr_loss = F.l1_loss(corr_mat, self.real_corr)
                else:
                    corr_loss = torch.tensor(0.0, device=fake.device)

                loss = loss_num + loss_cat + beta*kl + 5*swd + 10*corr_loss
                self.opt.zero_grad(); loss.backward(); self.opt.step()

            # validation & early stop
            if ep%5==0 or ep==epochs:
                val_fake = self.generate(len(va), temp=0.5, _cpu=False)['cont']
                val_real = self._encode(self.real.iloc[va])[:,:self.num_dim]
                swd_val  = sliced_wasserstein(val_fake, val_real).item()
                if verbose:
                    LOGGER.info(f"Ep {ep:03d}  β={beta:.3f}  SWD_val={swd_val:.4f}")
                val_fake = self.generate(len(va), temp=0.5, _cpu=False)['cont']
                val_real = self._encode(self.real.iloc[va])[:, :self.num_dim]
                swd_val = sliced_wasserstein(val_fake, val_real).item()
                if verbose:
                    # record for final plot
                    history_ep.append(ep)
                    history_swd.append(swd_val)
                    history_beta.append(beta)
                if swd_val < best-1e-4:
                    best, patience = swd_val, 0
                    torch.save({'enc':self.enc.state_dict(),
                                'dec':self.dec.state_dict()}, 'vae_best.pt')
                else:
                    patience += 1
                    if patience > 20:
                        if verbose:
                            LOGGER.info("Early stop")
                        break
        self.enc.load_state_dict(torch.load('vae_best.pt')['enc'])
        self.dec.load_state_dict(torch.load('vae_best.pt')['dec'])

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(history_ep, history_swd, '-o', label='SWD_val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('SWD_val', color='C0')
            ax.tick_params(axis='y', labelcolor='C0')
            ax2 = ax.twinx()
            ax2.plot(history_ep, history_beta, '-s', color='C1', label='β')
            ax2.set_ylabel('β (KL weight)', color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
            fig.tight_layout()
            plt.title('VAE training: SWD vs β')
            plt.show()

    # ------------ generation ---------------------------------------------
    def generate(self, n, temp=0.8, _cpu=True):
        self.dec.eval()
        with torch.no_grad():
            out_num, out_cat = [], []
            bs = 1024
            # “start” is how many samples have already been handled
            for start in range(0, n, bs):
                cur = min(bs, n - start)
                z = torch.randn(cur, self.latent, device=DEVICE)
                z_cat, z_num = torch.chunk(z, 2, 1)
                num, logits, cond = self.dec(z_cat, z_num, tau=temp, hard=True)
                out_num.append(num)
                out_cat.append(cond)
            # now we have exactly n samples
            cont = torch.cat(out_num, dim=0).detach()
            cats = torch.cat(out_cat, dim=0).detach()

        # inverse-transform continuous
        data, k = {}, 0
        for c in self.num_cols:
            vals = self.tf_num[c].inverse(cont[:,k].cpu().numpy())
            data[c] = np.round(vals).astype(int) if self.meta[c].data_type==DataType.INTEGER \
                      else np.round(vals, self.meta[c].decimal_places or 2)
            k += 1
        for c in self.dt_cols:
            data[c] = self.tf_dt[c].inverse(cont[:,k].cpu().numpy())
            k += 1

        # inverse-transform categoricals / booleans
        start = 0
        for c,dim in zip(self.cat_cols, self.cat_dims):
            idx  = cats[:,start:start+dim].argmax(1).cpu().numpy()
            data[c] = self.cat_maps[c][idx]
            start += dim

        # generate strings with Faker or empirical sampling
        for c in self.str_cols:
            m = self.meta[c]
            if m.faker_method is not None:
                data[c] = [m.faker_method(**getattr(m,'faker_args',{})) for _ in range(n)]
            else:
                # fall back to empirical sampling
                data[c] = self.real[c].sample(n, replace=True).to_list()

        df = pd.DataFrame(data)
        return {'df':df, 'cont':cont} if not _cpu else df
# ───────────────────────────────────────────────────────
