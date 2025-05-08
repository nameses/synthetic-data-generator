"""VAE models for the data generation task."""

import torch
from torch import nn
from torch.nn import functional


class Encoder(nn.Module):
    """Encoder network for the VAE. Maps input data to a latent space."""

    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logv = nn.Linear(256, z_dim)

    def forward(self, x):
        """Forward pass through the encoder."""
        h = self.net(x)
        mu, logv = self.mu(h), self.logv(h).clamp(-6, 6)
        std = torch.exp(0.5 * logv)
        return mu + std * torch.randn_like(std), mu, logv


class Decoder(nn.Module):
    """Decoder network for the VAE. Maps latent space back to input space."""

    def __init__(self, num_dim, cat_dims, z_dim):
        super().__init__()
        self.cat_dims = cat_dims
        self.cond_dim = sum(cat_dims)

        half_z = z_dim // 2
        if self.cond_dim > 0:
            self.cat_head = nn.Sequential(
                nn.Linear(half_z, 256), nn.GELU(), nn.Linear(256, self.cond_dim)
            )
        else:
            self.cat_head = None
        self.num_head = nn.Sequential(
            nn.Linear(half_z + self.cond_dim, 256), nn.GELU(), nn.Linear(256, num_dim)
        )

    def forward(self, z_cat, z_num, tau, hard):
        """Forward pass through the decoder."""
        # if we have any categories, do gumbel‐softmax else no‐op
        if self.cond_dim > 0:
            logits = self.cat_head(z_cat).split(self.cat_dims, 1)
            cats = [functional.gumbel_softmax(l, tau=tau, hard=hard) for l in logits]
            cond = torch.cat(cats, 1)
        else:
            logits = []
            # empty conditioning vector of shape [batch, 0]
            cond = torch.empty(z_cat.size(0), 0, device=z_cat.device)
        num = self.num_head(torch.cat([z_num, cond], 1))
        return num, logits, cond
