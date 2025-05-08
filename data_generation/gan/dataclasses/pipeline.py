"""Dataclasses for the training and generation pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from data_generation.gan.dataclasses.training import GanConfig
from data_generation.transformers import _ContTf, _DtTf
from models.enums import DataType
from models.field_metadata import FieldMetadata
if TYPE_CHECKING:
    from data_generation.gan.models import _Generator, _Discriminator


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
            if m.data_type in {DataType.Integer, DataType.Decimal}
        ]
        dt_cols = [c for c, m in metadata.items() if m.data_type == DataType.Datetime]
        cat_cols = [
            c
            for c, m in metadata.items()
            if m.data_type in {DataType.Categorical, DataType.Boolean}
        ]
        str_cols = [c for c, m in metadata.items() if m.data_type == DataType.String]
        return cls(
            metadata=metadata,
            real_df=real_df,
            num_cols=num_cols,
            dt_cols=dt_cols,
            cat_cols=cat_cols,
            str_cols=str_cols,
        )


@dataclass
class DataPipeline:
    """All things related to preparing and loading the real dataset."""

    schema: DataSchema
    loader: DataLoader
    tfs: dict
    cat_maps: dict
    cat_sizes: list[int]
    cat_probs: dict[str, np.ndarray]
    delta_real: torch.Tensor | None

    @classmethod
    def from_schema(cls, schema: DataSchema, cfg: GanConfig) -> "DataPipeline":
        """Converts a DataSchema into a DataPipeline."""
        mats = []
        tfs = {}

        # Continuous transformers
        for c in schema.num_cols:
            tf = _ContTf().fit(schema.real_df[c])
            tfs[c] = tf
            mats.append(tf.transform(schema.real_df[c]))

        # Datetime transformers
        for c in schema.dt_cols:
            tf = _DtTf(schema.metadata[c].datetime_format).fit(schema.real_df[c])
            tfs[c] = tf
            mats.append(tf.transform(schema.real_df[c]))

        # Categorical one-hot encoding
        cat_maps, cat_sizes = {}, []
        for c in schema.cat_cols:
            uniq = sorted(schema.real_df[c].astype(str).unique())
            cat_maps[c] = {v: i for i, v in enumerate(uniq)}
            cat_sizes.append(len(uniq))
            mats.append(
                one_hot(
                    torch.tensor(schema.real_df[c].astype(str).map(cat_maps[c]).values),
                    num_classes=len(uniq),
                ).T.numpy()
            )

        # Build tensor and loader
        real_tensor = torch.tensor(np.vstack(mats).T, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(real_tensor),
            cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

        cat_probs: dict[str, np.ndarray] = {}
        for c, _ in zip(schema.cat_cols, cat_sizes):
            # empirical sampling
            counts = (
                schema.real_df[c]
                .astype(str)
                .map(cat_maps[c])
                .value_counts()
                .sort_index()
            )
            cat_probs[c] = (counts / counts.sum()).values

        if cfg.loss.delta.use_delta_loss and (schema.num_cols or schema.dt_cols):
            delta_real = torch.tensor(
                np.diff(
                    real_tensor[:, : (len(schema.num_cols) + len(schema.dt_cols))]
                    .cpu()
                    .numpy(),
                    axis=0,
                ),
                dtype=torch.float32,
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            delta_real = None

        return cls(
            schema=schema,
            tfs=tfs,
            cat_maps=cat_maps,
            cat_sizes=cat_sizes,
            loader=loader,
            cat_probs=cat_probs,
            delta_real=delta_real,
        )


@dataclass
class ModelContainer:
    """Generator, Discriminator, and EMA snapshot."""

    generator: _Generator
    discriminator: _Discriminator
    ema_g: _Generator


@dataclass
class Optimizers:
    """Optimizers, schedulers, and AMP scalers."""

    opt_g: torch.optim.Optimizer
    opt_d: torch.optim.Optimizer
    sch_g: torch.optim.lr_scheduler.LRScheduler
    sch_d: torch.optim.lr_scheduler.LRScheduler
    scaler_g: torch.amp.GradScaler
    scaler_d: torch.amp.GradScaler


@dataclass
class TrainingState:
    """All of your metrics, early-stop flags, and best-model snapshots."""

    metrics: dict = field(default_factory=dict)
    best_w: float = -float("inf")
    no_imp: int = 0
    best_state: dict = None
    w_ema: float = None
    n_critic: int = 0
    val_df: pd.DataFrame = None
