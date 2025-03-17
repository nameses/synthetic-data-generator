from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.enums import DataType
from models.field_metadata import FieldMetadata
from data_generation.tools import rescale_numeric_values, convert_categorical_values

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cat_dims):
        super(Generator, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.num_output = nn.Linear(128, output_dim)  # For numeric data
        self.cat_outputs = nn.ModuleList([nn.Linear(128, dim) for dim in cat_dims])  # For categorical data

    def forward(self, x):
        shared = self.shared_layers(x)
        num_out = torch.sigmoid(self.num_output(shared))  # Numeric values

        cat_outs = [torch.softmax(layer(shared), dim=1) for layer in self.cat_outputs]  # Categorical values
        return num_out, cat_outs


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def generate_synthetic_data_gan(df: pd.DataFrame, metadata: Dict[str, FieldMetadata], synthetic_size: int):
    numerical_cols = [col for col, meta in metadata.items() if meta.data_type in {DataType.INTEGER, DataType.DECIMAL}]
    categorical_cols = [col for col, meta in metadata.items() if meta.data_type == DataType.CATEGORICAL]

    input_dim = len(numerical_cols)  # GAN only takes numerical inputs
    cat_dims = [df[col].nunique() for col in categorical_cols]  # Category sizes

    generator = Generator(input_dim, input_dim, cat_dims)
    generator.eval()

    # Generate synthetic numeric + categorical data
    z = torch.randn(synthetic_size, input_dim)
    num_data, cat_data = generator(z)
    # Convert from tensor to NumPy
    #num_data = num_data.detach().numpy()
    # Convert each tensor in the list
    #cat_data = [t.detach().numpy() for t in cat_data]

    # Rescale numeric values
    num_df = rescale_numeric_values(df, num_data, numerical_cols, metadata)

    # Convert categorical values
    cat_df = convert_categorical_values(df, cat_data, categorical_cols)

    return pd.concat([num_df, cat_df], axis=1)


