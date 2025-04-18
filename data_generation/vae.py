# vae_pipeline.py
import os
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

class VAEConfig:
    """
    Configuration for the VAE pipeline.
    Recommended: latent_dim=512, hidden_dims=[256,512,1024,512,256,128]
    """
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dims: List[int] = [256, 512, 1024, 512, 256, 128],
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 80,
        beta_kl: float = 1.0,
        lambda_cov: float = 0.5,
        lambda_sign: float = 0.5,
        lambda_mmd: float = 5.0,
        scheduler_type: str = 'cosine',  # 'cosine' or 'plateau'
        device: str = None
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.beta_kl = beta_kl       # KL weight
        self.lambda_cov = lambda_cov # covariance penalty weight
        self.lambda_sign = lambda_sign # sign mismatch penalty weight
        self.lambda_mmd = lambda_mmd # MMD marginal matching weight
        self.scheduler_type = scheduler_type
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faker = Faker()
        self.verbose = True

class VAEModel(nn.Module):
    def __init__(self, input_dim: int, cfg: VAEConfig):
        super().__init__()
        dims = [input_dim] + cfg.hidden_dims
        # Encoder
        enc_layers = []
        for i in range(len(dims)-1):
            enc_layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.LeakyReLU(0.2)]
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(dims[-1], cfg.latent_dim)
        self.fc_logvar = nn.Linear(dims[-1], cfg.latent_dim)
        # Decoder
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
        # Identify columns by type
        self.num_cols = [c for c,m in meta.items() if m.data_type in (DataType.INTEGER, DataType.DECIMAL)]
        self.cat_cols = [c for c,m in meta.items() if m.data_type in (DataType.CATEGORICAL, DataType.BOOLEAN)]
        self.date_cols = [c for c,m in meta.items() if m.data_type == DataType.DATETIME]
        self.str_cols = [c for c,m in meta.items() if m.data_type == DataType.STRING]
        # Preprocess
        self._preprocess()
        # Model
        self.model = VAEModel(self.X.shape[1], cfg).to(cfg.device)

    def _preprocess(self):
        # Numeric scaling
        self.scalers = {c: MinMaxScaler((-1,1)).fit(self.df[[c]]) for c in self.num_cols}
        num_arr = [self.scalers[c].transform(self.df[[c]]) for c in self.num_cols]
        X_num = np.hstack(num_arr) if num_arr else np.zeros((len(self.df),0))
        # Categorical one-hot
        self.cat_maps = {}
        cat_arrs = []
        for c in self.cat_cols:
            vals = self.df[c].astype(str)
            cats = sorted(vals.unique())
            self.cat_maps[c] = cats
            idxs = vals.map({v:i for i,v in enumerate(cats)}).values
            cat_arrs.append(np.eye(len(cats))[idxs])
        X_cat = np.hstack(cat_arrs) if cat_arrs else np.zeros((len(self.df),0))
        # Date scaling
        date_arrs = []
        for c in self.date_cols:
            secs = (self.df[c] - self.df[c].min()).dt.total_seconds().values.reshape(-1,1)
            date_arrs.append((secs/secs.max())*2 - 1)
        X_date = np.hstack(date_arrs) if date_arrs else np.zeros((len(self.df),0))
        # Combine
        X_all = np.hstack([X_num, X_cat, X_date])
        self.X = torch.tensor(X_all, dtype=torch.float32)

    def _covariance(self, x: torch.Tensor):
        B = x.size(0)
        m = x.mean(dim=0, keepdim=True)
        xc = x - m
        return (xc.t() @ xc) / (B - 1)

    def _mmd(self, x: torch.Tensor, y: torch.Tensor, sigma=1.0):
        xx = torch.exp(-((x.unsqueeze(1)-x.unsqueeze(0))**2).sum(2)/(2*sigma**2))
        yy = torch.exp(-((y.unsqueeze(1)-y.unsqueeze(0))**2).sum(2)/(2*sigma**2))
        xy = torch.exp(-((x.unsqueeze(1)-y.unsqueeze(0))**2).sum(2)/(2*sigma**2))
        return xx.mean() + yy.mean() - 2*xy.mean()

    def train(self):
        # Train/val split
        dataset = TensorDataset(self.X)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size)
        # Optimizer and scheduler
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        if self.cfg.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min')
        best_val = float('inf')
        os.makedirs('checkpoints', exist_ok=True)
        # Training loop
        for epoch in range(1, self.cfg.epochs+1):
            self.model.train()
            train_loss = 0.0
            for xb, in train_loader:
                xb = xb.to(self.cfg.device)
                x_hat, mu, logvar = self.model(xb)
                # Loss terms
                recon = F.mse_loss(x_hat, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                cov_loss = F.mse_loss(self._covariance(xb), self._covariance(x_hat))
                sign_loss = torch.mean(F.relu(-self._covariance(xb) * self._covariance(x_hat)))
                num_slice = xb[:, :len(self.num_cols)]
                mmd_loss = self._mmd(num_slice, x_hat[:, :len(self.num_cols)])
                loss = recon + self.cfg.beta_kl * kl \
                       + self.cfg.lambda_cov * cov_loss \
                       + self.cfg.lambda_sign * sign_loss \
                       + self.cfg.lambda_mmd * mmd_loss
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, in val_loader:
                    xb = xb.to(self.cfg.device)
                    x_hat, mu, logvar = self.model(xb)
                    recon = F.mse_loss(x_hat, xb)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    val_loss += (recon + self.cfg.beta_kl * kl).item()
            val_loss /= len(val_loader)
            # Scheduler step
            if self.cfg.scheduler_type == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_loss)
            # Checkpoint
            if val_loss < best_val:
                best_val = val_loss
                path = f"checkpoints/vae_best.pt"
                torch.save(self.model.state_dict(), path)
            if self.cfg.verbose:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def generate(self, n: int) -> pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.cfg.latent_dim).to(self.cfg.device)
            x_fake = self.model.decoder(z).cpu().numpy()
        idx = 0; data = {}
        # Numeric
        for c in self.num_cols:
            v = x_fake[:, idx]; idx += 1
            orig = self.scalers[c].inverse_transform(v.reshape(-1,1)).flatten()
            if self.meta[c].data_type == DataType.INTEGER:
                data[c] = np.round(orig).astype(int)
            else:
                data[c] = np.round(orig, self.meta[c].decimal_places or 2)
        # Categorical
        for c in self.cat_cols:
            k = len(self.cat_maps[c])
            logits = x_fake[:, idx:idx+k]; idx += k
            choices = logits.argmax(axis=1)
            data[c] = [self.cat_maps[c][i] for i in choices]
        # Date
        for c in self.date_cols:
            v = x_fake[:, idx]; idx += 1
            frac = (v + 1) / 2
            base = self.df[c].min(); span = (self.df[c].max() - base).total_seconds()
            data[c] = [base + pd.to_timedelta(f*span, 's') for f in frac]
        # String
        for c in self.str_cols:
            method = self.meta[c].faker_method or self.cfg.faker.word; args = self.meta[c].faker_args
            data[c] = [method(**args) if callable(method) else method for _ in range(n)]
        return pd.DataFrame(data)
