import pandas as pd
import torch
import numpy as np
from models.enums import DataType


def rescale_numeric_values(df: pd.DataFrame, num_data: torch.Tensor, numerical_cols: list, metadata: dict):
    """
    Rescales numeric values from [0,1] back to the original range of the dataset.
    """
    num_df = pd.DataFrame(num_data.detach().numpy(), columns=numerical_cols)

    for col in numerical_cols:
        min_val, max_val = df[col].min(), df[col].max()
        num_df[col] = num_df[col] * (max_val - min_val) + min_val
        if metadata[col].data_type == DataType.INTEGER:
            num_df[col] = num_df[col].round().astype(int)

    return num_df


def convert_categorical_values(df: pd.DataFrame, cat_data: list, categorical_cols: list):
    """
    Converts categorical tensor output to actual category values.
    """
    cat_df = pd.DataFrame({col: torch.argmax(cat_data[i], dim=1).numpy() for i, col in enumerate(categorical_cols)})

    for col in categorical_cols:
        cat_df[col] = df[col].unique()[cat_df[col]]  # Convert indices back to category names

    return cat_df
