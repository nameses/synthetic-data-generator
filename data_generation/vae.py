# vae_pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from faker import Faker
from typing import Dict, List

from models.enums import DataType
from models.field_metadata import FieldMetadata


class VAEConfig:
    def __init__(
            self,
            latent_dim: int = 512,
            hidden_dims: List[int] = [256, 512, 1024, 512, 256, 128],
            batch_size: int = 512,
            lr: float = 1e-3,
            epochs: int = 80,
            beta_kl: float = 1.0,
            lambda_cov: float = 0.5,
            lambda_sign: float = 0.5,
            lambda_mmd: float = 5.0,
            device: str = None
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.beta_kl = beta_kl  # KL weight
        self.lambda_cov = lambda_cov  # covariance magnitude penalty
        self.lambda_sign = lambda_sign  # penalty for sign mismatch
        self.lambda_mmd = lambda_mmd  # MMD marginal matching
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faker = Faker()
        self.verbose = True


class VAEModel(nn.Module):
    def __init__(self, input_dim: int, config: VAEConfig):
        super().__init__()
        dims = [input_dim] + config.hidden_dims
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.LeakyReLU(0.2)]
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(dims[-1], config.latent_dim)
        self.fc_logvar = nn.Linear(dims[-1], config.latent_dim)

        dec_dims = [config.latent_dim] + config.hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(dec_dims) - 1):
            act = nn.LeakyReLU(0.2) if i < len(dec_dims) - 2 else nn.Identity()
            decoder_layers += [nn.Linear(dec_dims[i], dec_dims[i + 1]), act]
        self.decoder = nn.Sequential(*decoder_layers)

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
        # columns
        self.num_cols = [c for c, m in meta.items() if m.data_type in (DataType.INTEGER, DataType.DECIMAL)]
        self.cat_cols = [c for c, m in meta.items() if m.data_type in (DataType.CATEGORICAL, DataType.BOOLEAN)]
        self.date_cols = [c for c, m in meta.items() if m.data_type == DataType.DATETIME]
        self.str_cols = [c for c, m in meta.items() if m.data_type == DataType.STRING]
        # preprocess
        self._preprocess()
        # model
        self.model = VAEModel(self.X.shape[1], cfg).to(cfg.device)

    def _preprocess(self):
        # numeric
        self.scalers = {c: MinMaxScaler(feature_range=(-1, 1)).fit(self.df[[c]]) for c in self.num_cols}
        num = [self.scalers[c].transform(self.df[[c]]) for c in self.num_cols]
        X_num = np.hstack(num) if num else np.zeros((len(self.df), 0))
        # categorical
        self.cat_maps = {}
        X_cat_list = []
        for c in self.cat_cols:
            vals = self.df[c].astype(str)
            categories = sorted(vals.unique())
            self.cat_maps[c] = categories
            idx = vals.map({v: i for i, v in enumerate(categories)}).values
            onehot = np.eye(len(categories))[idx]
            X_cat_list.append(onehot)
        X_cat = np.hstack(X_cat_list) if X_cat_list else np.zeros((len(self.df), 0))
        # date
        X_date_list = []
        for c in self.date_cols:
            seconds = (self.df[c] - self.df[c].min()).dt.total_seconds().values.reshape(-1, 1)
            scaled = (seconds / seconds.max()) * 2 - 1
            X_date_list.append(scaled)
        X_date = np.hstack(X_date_list) if X_date_list else np.zeros((len(self.df), 0))
        # combine
        X_all = np.hstack([X_num, X_cat, X_date])
        self.X = torch.tensor(X_all, dtype=torch.float32)

    def _covariance(self, x: torch.Tensor):
        B = x.size(0)
        m = x.mean(dim=0, keepdim=True)
        xc = x - m
        cov = (xc.t() @ xc) / (B - 1)
        return cov

    def _mmd(self, x: torch.Tensor, y: torch.Tensor):
        sigma = 1.0
        xx = torch.exp(-((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(2) / (2 * sigma ** 2))
        yy = torch.exp(-((y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2) / (2 * sigma ** 2))
        xy = torch.exp(-((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2) / (2 * sigma ** 2))
        return xx.mean() + yy.mean() - 2 * xy.mean()

    def train(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        ds = TensorDataset(self.X)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        for epoch in range(1, self.cfg.epochs + 1):
            loss_e = 0
            for (xb,) in loader:
                xb = xb.to(self.cfg.device)
                x_hat, mu, logvar = self.model(xb)
                # losses
                recon = F.mse_loss(x_hat, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                # covariance
                cov_x = self._covariance(xb)
                cov_h = self._covariance(x_hat)
                cov_loss = F.mse_loss(cov_x, cov_h)
                # sign mismatch penalty
                sign_loss = torch.mean(F.relu(-cov_x * cov_h))
                # mmd on numeric marginals
                mmd_loss = self._mmd(xb[:, :len(self.num_cols)], x_hat[:, :len(self.num_cols)])
                loss = recon + self.cfg.beta_kl * kl \
                       + self.cfg.lambda_cov * cov_loss \
                       + self.cfg.lambda_sign * sign_loss \
                       + self.cfg.lambda_mmd * mmd_loss
                opt.zero_grad();
                loss.backward();
                opt.step()
                loss_e += loss.item()
            if self.cfg.verbose:
                print(f"Epoch {epoch:03d} | Loss: {loss_e / len(loader):.4f}")

    def generate(self, n: int) -> pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.cfg.latent_dim).to(self.cfg.device)
            x_fake = self.model.decoder(z).cpu().numpy()
        idx = 0;
        data = {}
        # numeric
        for c in self.num_cols:
            v = x_fake[:, idx]
            orig = self.scalers[c].inverse_transform(v.reshape(-1, 1)).flatten()
            if self.meta[c].data_type == DataType.INTEGER:
                data[c] = np.round(orig).astype(int)
            else:
                data[c] = np.round(orig, self.meta[c].decimal_places or 2)
            idx += 1
        # categorical
        for c in self.cat_cols:
            k = len(self.cat_maps[c]);
            logits = x_fake[:, idx:idx + k];
            idx += k
            choices = np.argmax(logits, axis=1)
            data[c] = [self.cat_maps[c][i] for i in choices]
        # date
        for c in self.date_cols:
            v = x_fake[:, idx];
            idx += 1
            frac = (v + 1) / 2;
            base = self.df[c].min();
            span = (self.df[c].max() - base).total_seconds()
            data[c] = [base + pd.to_timedelta(f * span, 's') for f in frac]
        # string
        for c in self.str_cols:
            method = self.meta[c].faker_method or self.cfg.faker.word;
            args = self.meta[c].faker_args
            data[c] = [method(**args) if callable(method) else method for _ in range(n)]
        return pd.DataFrame(data)
