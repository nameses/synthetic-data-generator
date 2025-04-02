# data_generation/api.py
import torch
from models.enums import MethodType
from models.field_metadata import FieldMetadata
from typing import Dict, Any
import pandas as pd
import numpy as np

# Import the WGAN implementation
from data_generation.wgan import generate_synthetic_data_wgan


def generate_synthetic_data(df: pd.DataFrame,
                            metadata: Dict[str, FieldMetadata],
                            method_type: MethodType,
                            synthetic_size: int) -> pd.DataFrame:
    print(f"Using WGAN method for synthetic data generation")
    return generate_synthetic_data_wgan(
        df,
        metadata,
        synthetic_size
    )