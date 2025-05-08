"""Module for dataclasses that are common for neural networks"""

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from models.enums import DataType
from models.field_metadata import FieldMetadata


@dataclass
class DataSchema:
    """
    Cleans raw DataFrame and categorizes columns by type.
    """

    metadata: Dict[str, FieldMetadata]
    real_df: pd.DataFrame
    num_cols: list[str]
    dt_cols: list[str]
    cat_cols: list[str]
    str_cols: list[str]

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, meta: Dict[str, FieldMetadata]
    ) -> "DataSchema":
        """Converts a DataFrame into a DataSchema."""
        metadata = meta
        # Clean and reset index
        real_df = df.dropna().reset_index(drop=True)
        # Derive column lists based on metadata
        num_cols = [
            c
            for c, m in metadata.items()
            if m.data_type in {DataType.INTEGER, DataType.DECIMAL}
        ]
        dt_cols = [c for c, m in metadata.items() if m.data_type == DataType.DATETIME]
        cat_cols = [
            c
            for c, m in metadata.items()
            if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}
        ]
        str_cols = [c for c, m in metadata.items() if m.data_type == DataType.STRING]
        return cls(
            metadata=metadata,
            real_df=real_df,
            num_cols=num_cols,
            dt_cols=dt_cols,
            cat_cols=cat_cols,
            str_cols=str_cols,
        )
