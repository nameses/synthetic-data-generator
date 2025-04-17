from __future__ import annotations
import os
from typing import Dict, List
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error

from models.enums import DataType
from models.field_metadata import FieldMetadata

logger = logging.getLogger(__name__)


sns.set(style="whitegrid", context="talk")


def generate_report(real: pd.DataFrame, synth: pd.DataFrame, meta: Dict[str, FieldMetadata], out_dir: str = "reports") -> str:  # noqa: D401
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"report_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    txt: List[str] = ["SYNTHETIC DATA QUALITY REPORT", f"Created: {datetime.now()}"]

    # Numerical columns – distribution & Wasserstein
    num_cols = [c for c, m in meta.items() if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
    for col in num_cols:
        if col not in synth.columns:
            continue
        w = wasserstein_distance(real[col], synth[col])
        txt.append(f"{col}: Wasserstein = {w:.3f}")
        plt.figure(figsize=(6, 4))
        sns.kdeplot(real[col], label="Real")
        sns.kdeplot(synth[col], label="Synthetic")
        plt.title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{col}_dist.png"))
        plt.close()

    # Correlation matrices
    real_corr = real[num_cols].corr()
    synth_corr = synth[num_cols].corr()
    diff = real_corr - synth_corr
    mae = mean_absolute_error(real_corr.values, synth_corr.values)
    txt.append(f"\nMean abs error of correlation matrices: {mae:.4f}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(real_corr, ax=axes[0], vmin=-1, vmax=1, cmap="vlag", square=True)
    axes[0].set_title("Real")
    sns.heatmap(synth_corr, ax=axes[1], vmin=-1, vmax=1, cmap="vlag", square=True)
    axes[1].set_title("Synthetic")
    sns.heatmap(diff, ax=axes[2], vmin=-1, vmax=1, cmap="vlag", square=True)
    axes[2].set_title("Diff")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "correlation.png"))
    plt.close(fig)

    # Save plain text summary
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf‑8") as f:
        f.write("\n".join(txt))
    logger.info("Report written to %s", run_dir)
    return run_dir