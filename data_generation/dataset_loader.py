from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from models.enums import DataType
from models.field_metadata import FieldMetadata

logger = logging.getLogger(__name__)


class DatasetLoader:
    def load_real_data(
        self,
        data_path: str,
        metadata: Dict[str, FieldMetadata],
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        # Identify datetime columns for parsing
        dt_cols = [col for col, meta in metadata.items() if meta.data_type == DataType.DATETIME]
        df = pd.read_csv(data_path, parse_dates=dt_cols, infer_datetime_format=True)

        # Drop NaNs everywhere (requirement)
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.warning("Dropped %d rows containing NaNs (%.1f%%)", before - len(df), 100 * (before - len(df)) / before)

        if sample_size is not None and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        return df