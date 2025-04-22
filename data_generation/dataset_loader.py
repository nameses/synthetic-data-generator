"""Robust CSV loader for GAN training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)


class DatasetLoader:
    """Load and pre‑clean user data for the synthetic‑data pipeline."""

    @staticmethod
    def _datetime_columns(meta: Dict[str, FieldMetadata]) -> List[str]:
        return [c for c, m in meta.items() if m.data_type is DataType.DATETIME]

    # ---------------------------------------------------------------------#
    # PUBLIC API
    # ---------------------------------------------------------------------#
    def load_real_data(
        self,
        data_path: str | Path,
        metadata: Dict[str, FieldMetadata],
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Read a CSV file, parse datetime columns (if any),
        **drop every row with a NaN**, optionally subsample.

        Parameters
        ----------
        data_path:
            Path to the user dataset.
        metadata:
            Column‑level metadata (types / formats).
        sample_size:
            Optional down‑sample size (useful for quick experiments).

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe ready for modelling.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(data_path)

        dt_cols = self._datetime_columns(metadata)
        df = pd.read_csv(data_path, parse_dates=dt_cols)
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        LOGGER.info(
            "Dropped %d rows with NaNs (%.1f%%). Final shape: %s",
            before - len(df),
            100 * (before - len(df)) / max(before, 1),
            df.shape,
        )

        # record min/max for numeric columns
        for col, m in metadata.items():
            if m.data_type in {DataType.INTEGER, DataType.DECIMAL}:
                m.min_value = float(df[col].min())
                m.max_value = float(df[col].max())
                # optionally apply user-specified log_transform
                if m.transformer and 'log' in m.transformer:
                    df[col] = np.log1p(df[col])

        if sample_size is not None and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
            LOGGER.info("Sub‑sampled to %d rows", sample_size)

        return df
