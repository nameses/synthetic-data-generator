import pandas as pd
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = enc_out.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def generate_synthetic_data_vae(df: pd.DataFrame, input_dim: int, synthetic_size: int):
    vae = VAE(input_dim)
    vae.eval()
    z = torch.randn(synthetic_size, 2)
    generated_data = vae.decoder(z).detach().numpy()
    return pd.DataFrame(generated_data, columns=df.select_dtypes(include=[np.number]).columns)
