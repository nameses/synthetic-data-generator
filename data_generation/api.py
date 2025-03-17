import torch
from models.enums import MethodType
from models.field_metadata import FieldMetadata
from typing import Dict, Any
import pandas as pd
from data_generation.gan import generate_synthetic_data_gan
from data_generation.vae import generate_synthetic_data_vae
import numpy as np


def generate_synthetic_data(df: pd.DataFrame,
                            metadata: Dict[str, FieldMetadata],
                            method_type: MethodType,
                            synthetic_size: int) -> pd.DataFrame:
    numerical_data = df.select_dtypes(include=[np.number]).values
    input_dim = numerical_data.shape[1]


    if method_type == MethodType.GAN:
        return generate_synthetic_data_gan(df, metadata, synthetic_size)
    elif method_type == MethodType.VAE:
        return generate_synthetic_data_vae(df, input_dim, synthetic_size)
    else:
        raise ValueError("Unsupported method type")

    #if not isinstance(synthetic_data, pd.DataFrame):
    #    synthetic_data = pd.DataFrame(synthetic_data, columns=df.select_dtypes(include=[np.number]).columns)

    #return synthetic_data




