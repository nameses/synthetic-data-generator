from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from models.enums import DataType
from models.field_metadata import FieldMetadata
from data_generation.tools import rescale_numeric_values, convert_categorical_values

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2, cat_dims=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log variance
        )

        self.decoder_num = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Output scaled between 0 and 1
        )

        # Separate categorical outputs
        self.cat_outputs = nn.ModuleList([nn.Linear(latent_dim, dim) for dim in cat_dims]) if cat_dims else None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = enc_out.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        num_out = self.decoder_num(z)  # Numeric output

        if self.cat_outputs:
            cat_outs = [torch.softmax(layer(z), dim=1) for layer in self.cat_outputs]  # Categorical output
        else:
            cat_outs = None

        return num_out, cat_outs, mu, logvar


def generate_synthetic_data_vae(df: pd.DataFrame, metadata: Dict[str, FieldMetadata], synthetic_size: int):
    numerical_cols = [col for col, meta in metadata.items() if meta.data_type in {DataType.INTEGER, DataType.DECIMAL}]
    categorical_cols = [col for col, meta in metadata.items() if meta.data_type == DataType.CATEGORICAL]

    input_dim = len(numerical_cols)  # VAE takes numerical input
    cat_dims = [df[col].nunique() for col in categorical_cols]  # Number of unique categories per column

    vae = VAE(input_dim, latent_dim=2, cat_dims=cat_dims)
    vae.eval()

    # Generate synthetic latent variables
    z = torch.randn(synthetic_size, vae.latent_dim)
    num_data, cat_data, _, _ = vae.forward(z)

    # Rescale numeric values
    num_df = rescale_numeric_values(df, num_data, numerical_cols, metadata)

    # Convert categorical values
    cat_df = convert_categorical_values(df, cat_data, categorical_cols)

    return pd.concat([num_df, cat_df], axis=1)