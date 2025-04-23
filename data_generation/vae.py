# vae_pipeline.py
import copy
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from faker import Faker
from typing import Dict, List

from models.enums import DataType
from models.field_metadata import FieldMetadata
from data_generation.transformers import _LogTf, _ZeroInflatedTf, _BoundedTf, _MinMaxTf

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fh = logging.FileHandler(f"train_{ts}.log", mode="w", encoding="utf‑8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(fh)

_TF_REGISTRY = {
    None: _MinMaxTf(),
    "log": _LogTf(),
    "zero_inflated": _ZeroInflatedTf(),
    "bounded": _BoundedTf(),
    "minmax": _MinMaxTf()
}

class VAEConfig:
    """
    Configuration for the VAE pipeline.
    Recommended: latent_dim=512, hidden_dims=[256,512,1024,512,256,128]
    """
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dims: List[int] = None,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 300,
        beta_kl: float = 0.7,
        lambda_corr: float = 1.2,
        lambda_mmd: float = 5.0,
        scheduler_type: str = 'cosine',  # 'cosine' or 'plateau'
        device: str = None
    ):
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024, 1024, 512, 256]
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.beta_kl = beta_kl
        self.lambda_corr = lambda_corr
        self.lambda_mmd = lambda_mmd
        self.scheduler_type = scheduler_type
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faker = Faker()
        self.verbose = True

class VAEModel(nn.Module):
    def __init__(self, input_dim: int, cfg: VAEConfig):
        super().__init__()
        dims = [input_dim] + cfg.hidden_dims
        enc_layers = []
        for i in range(len(dims)-1):
            enc_layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.LeakyReLU(0.2)]
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(dims[-1], cfg.latent_dim)
        self.fc_logvar = nn.Linear(dims[-1], cfg.latent_dim)
        dec_dims = [cfg.latent_dim] + cfg.hidden_dims[::-1] + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims)-1):
            act = nn.LeakyReLU(0.2) if i < len(dec_dims)-2 else nn.Identity()
            dec_layers += [nn.Linear(dec_dims[i], dec_dims[i+1]), act]
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

class VAEPipeline:
    def __init__(self, df: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: VAEConfig):
        self.cfg = cfg
        self.meta = meta
        self.df = df.dropna().reset_index(drop=True)
        self.num_cols = [c for c,m in meta.items() if m.data_type in (DataType.INTEGER, DataType.DECIMAL)]
        self.cat_cols = [c for c,m in meta.items() if m.data_type in (DataType.CATEGORICAL, DataType.BOOLEAN)]
        self.date_cols = [c for c,m in meta.items() if m.data_type == DataType.DATETIME]
        self.str_cols = [c for c,m in meta.items() if m.data_type == DataType.STRING]
        self._preprocess()
        # Precompute original numeric correlation matrix
        corr_parts = []
        for c in self.num_cols:
            tf = self.transformers[c]
            arr = tf.transform(self.df[c])
            corr_parts.append(arr)

        for c in self.cat_cols:
            vals = self.df[c].astype("category").cat.codes.values
            corr_parts.append(2 * (vals / vals.max()) - 1)

        self.orig_corr = torch.tensor(
            np.corrcoef(np.column_stack(corr_parts).T),
            dtype=torch.float32, device=self.cfg.device)

        self.model = VAEModel(self.X.shape[1], cfg).to(cfg.device)

    def _preprocess(self):
        from sklearn.preprocessing import MinMaxScaler
        self.transformers = {}
        num_arr = []
        for c in self.num_cols:
            tf_name = getattr(self.meta[c], "transformer", None)
            tf = copy.deepcopy(_TF_REGISTRY[tf_name])

            # sklearn.MinMaxScaler wants a 2D array; our custom transformers want a Series
            if isinstance(tf, MinMaxScaler):
                # fit & transform with a one-column DataFrame
                tf.fit(self.df[[c]])
                arr = tf.transform(self.df[[c]]).flatten()
            else:
                # custom transformer: feed it a Series
                tf.fit(self.df[c])
                arr = tf.transform(self.df[c])

            self.transformers[c] = tf
            num_arr.append(arr)
        X_num = np.column_stack(num_arr) if num_arr else np.zeros((len(self.df), 0))

        self.cat_maps = {}
        cat_arrs = []
        for c in self.cat_cols:
            vals = self.df[c].astype(str)
            cats = sorted(vals.unique())
            self.cat_maps[c] = cats
            idxs = vals.map({v:i for i,v in enumerate(cats)}).values
            cat_arrs.append(np.eye(len(cats))[idxs])
        X_cat = np.hstack(cat_arrs) if cat_arrs else np.zeros((len(self.df),0))
        date_arrs = []
        for c in self.date_cols:
            secs = (self.df[c] - self.df[c].min()).dt.total_seconds().values.reshape(-1,1)
            date_arrs.append((secs/secs.max())*2 - 1)
        X_date = np.hstack(date_arrs) if date_arrs else np.zeros((len(self.df),0))
        X_all = np.hstack([X_num, X_cat, X_date])
        self.X = torch.tensor(X_all, dtype=torch.float32)

    def _corr(self, x: torch.Tensor):
        m = x.mean(dim=0, keepdim=True)
        xc = x - m
        cov = (xc.t()@xc)/(x.size(0)-1)
        std = torch.sqrt(torch.diag(cov)).unsqueeze(1)
        return cov/(std@std.t()+1e-6)

    def _mmd(self, x: torch.Tensor, y: torch.Tensor, sigma=1.0):
        xx = torch.exp(-((x.unsqueeze(1)-x.unsqueeze(0))**2).sum(2)/(2*sigma**2))
        yy = torch.exp(-((y.unsqueeze(1)-y.unsqueeze(0))**2).sum(2)/(2*sigma**2))
        xy = torch.exp(-((x.unsqueeze(1)-y.unsqueeze(0))**2).sum(2)/(2*sigma**2))
        return xx.mean() + yy.mean() - 2*xy.mean()
        
    def _is_in_notebook(self):
        """Check if we're running in a Jupyter notebook environment"""
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return True
            return False
        except:
            return False

    def fit(self, verbose=None):
        # Use class verbose setting if not specified
        if verbose is None:
            verbose = self.cfg.verbose
            
        # Initialize metrics tracking
        metrics = {
            'epoch': [],
            'train_total': [], 'train_recon': [], 'train_kl': [], 'train_corr': [], 'train_mmd': [],
            'val_loss': []
        }

        dataset = TensorDataset(self.X)
        train_ds, val_ds = random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if self.cfg.scheduler_type=='cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min')
        best_val = float('inf'); os.makedirs('checkpoints',exist_ok=True)
        for epoch in range(1,self.cfg.epochs+1):
            self.model.train()
            sum_losses = {'total':0,'recon':0,'kl':0,'corr':0,'mmd':0}
            for xb, in train_loader:
                xb = xb.to(self.cfg.device)
                x_hat, mu, logvar = self.model(xb)
                recon = F.mse_loss(x_hat, xb)
                kl = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())

                # 1) numeric part
                num_hat = x_hat[:, : len(self.num_cols)]

                # 2) categorical “soft codes”
                cat_hat_parts = []
                idx = len(self.num_cols)
                for c in self.cat_cols:
                    k = len(self.cat_maps[c])  # number of categories for c
                    logits = x_hat[:, idx: idx + k]  # decoder outputs for this field
                    probs = torch.softmax(logits, dim=1)
                    # compute a “soft index” in [0…k-1]
                    idx_vals = torch.arange(k, device=logits.device, dtype=probs.dtype)
                    code = (probs * idx_vals).sum(dim=1)
                    # scale to [-1,1], matching orig_corr’s scaling
                    code_s = code / (k - 1) * 2 - 1
                    cat_hat_parts.append(code_s.unsqueeze(1))
                    idx += k

                # 3) assemble full and compute correlation
                x_corr = torch.cat([num_hat] + cat_hat_parts, dim=1)  # shape [batch, 23]
                corr_hat = self._corr(x_corr)

                corr_loss = F.mse_loss(corr_hat, self.orig_corr)

                mmd_loss = self._mmd(xb[:,:len(self.num_cols)], x_hat[:,:len(self.num_cols)])
                loss = recon + self.cfg.beta_kl*kl + self.cfg.lambda_corr*corr_loss + self.cfg.lambda_mmd*mmd_loss
                opt.zero_grad(); loss.backward(); opt.step()
                sum_losses['total'] += loss.item(); sum_losses['recon'] += recon.item()
                sum_losses['kl'] += kl.item(); sum_losses['corr'] += corr_loss.item(); sum_losses['mmd'] += mmd_loss.item()
            # scheduler
            if self.cfg.scheduler_type=='cosine': scheduler.step()
            else: scheduler.step(sum_losses['total']/len(train_loader))
            # val
            self.model.eval(); val_loss=0
            with torch.no_grad():
                for xb, in val_loader:
                    xb=xb.to(self.cfg.device)
                    x_hat, mu, logvar = self.model(xb)
                    recon=F.mse_loss(x_hat, xb); kl=-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
                    val_loss += (recon+ self.cfg.beta_kl*kl).item()
            val_loss/=len(val_loader)
            lr=opt.param_groups[0]['lr']
            
            # Store metrics for plotting
            train_total = sum_losses['total']/len(train_loader)
            train_recon = sum_losses['recon']/len(train_loader)
            train_kl = sum_losses['kl']/len(train_loader)
            train_corr = sum_losses['corr']/len(train_loader)
            train_mmd = sum_losses['mmd']/len(train_loader)
            
            metrics['epoch'].append(epoch)
            metrics['train_total'].append(train_total)
            metrics['train_recon'].append(train_recon)
            metrics['train_kl'].append(train_kl)
            metrics['train_corr'].append(train_corr)
            metrics['train_mmd'].append(train_mmd)
            metrics['val_loss'].append(val_loss)
            
            if val_loss<best_val:
                best_val=val_loss
                torch.save(self.model.state_dict(),f"checkpoints/vae_best_epoch{epoch}.pt")
                if verbose:
                    LOGGER.info("Checkpoint saved: epoch %d, val_loss=%.4f", epoch, val_loss)
            if verbose:
                LOGGER.info("Epoch %03d | LR:%.6f | Train Loss:%.4f (Recon:%.4f, KL:%.4f, Corr:%.4f, MMD:%.4f) | Val Loss:%.4f",
                        epoch, lr, train_total, train_recon, train_kl, train_corr, train_mmd, val_loss)

        import matplotlib.pyplot as plt

        # 1) Total vs Validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['epoch'], metrics['train_total'], label='Train Total')
        plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation')
        plt.title('VAE Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2) Reconstruction & KL divergence
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['epoch'], metrics['train_recon'], label='Reconstruction')
        plt.plot(metrics['epoch'], metrics['train_kl'], label='KL Divergence')
        plt.title('VAE Reconstruction & KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3) Correlation & MMD regularization terms
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['epoch'], metrics['train_corr'], label='Correlation Loss')
        plt.plot(metrics['epoch'], metrics['train_mmd'], label='MMD Loss')
        plt.title('VAE Regularization Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return metrics

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Round every decimal column to its metadata-defined precision
        for c in self.num_cols:
            if self.meta[c].data_type == DataType.DECIMAL:
                # default to 2 if decimal_places is None or zero
                decimals = getattr(self.meta[c], 'decimal_places', None) or 2
                df[c] = df[c].round(decimals)
        return df[list(self.meta.keys())]


    def generate(self, n:int)->pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            z=torch.randn(n,self.cfg.latent_dim).to(self.cfg.device)
            x_fake=self.model.decoder(z).cpu().numpy()
        idx=0; data={}
        # numeric
        for c in self.num_cols:
            v=x_fake[:,idx]
            idx+=1
            v_clipped = np.clip(v, -1.0, 1.0)
            arr = self.transformers[c].inverse(v_clipped)
            if self.meta[c].data_type == DataType.INTEGER:
                arr = arr.round().astype(int)
            else:
                arr = arr.round(self.meta[c].decimal_places or 2)
            data[c]=arr
        # categorical
        for c in self.cat_cols:
            k=len(self.cat_maps[c]); logits=x_fake[:,idx:idx+k]; idx+=k
            choices=np.argmax(logits,axis=1)
            data[c]=[self.cat_maps[c][i] for i in choices]
        # date
        for c in self.date_cols:
            v=x_fake[:,idx]; idx+=1
            frac=(v+1)/2; base=self.df[c].min(); span=(self.df[c].max()-base).total_seconds()
            dates = [base+pd.to_timedelta(f*span,'s') for f in frac]
            # Format datetime using the format specified in metadata
            if self.meta[c].datetime_format:
                data[c] = [date.strftime(self.meta[c].datetime_format) for date in dates]
            else:
                data[c] = dates
        # string
        for c in self.str_cols:
            method=self.meta[c].faker_method or self.cfg.faker.word; args=self.meta[c].faker_args
            data[c]=[method(**args) if callable(method) else method for _ in range(n)]
        return self._postprocess(pd.DataFrame(data))
