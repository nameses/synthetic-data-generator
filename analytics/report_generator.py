import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_absolute_error

from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)
sns.set(style="whitegrid")


def generate_report(real: pd.DataFrame, synth: pd.DataFrame, meta: Dict[str, FieldMetadata], out_dir: str = "reports") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"report_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    synth.to_csv(run_dir / "synthetic.csv", index=False)

    summary_lines = [
        "SYNTHETIC DATA QUALITY REPORT",
        f"Created: {datetime.now()}",
        f"Real shape: {real.shape}",
        f"Synthetic shape: {synth.shape}",
        f"NaNs in real: {real.isna().sum().sum()}",
        f"NaNs in synth: {synth.isna().sum().sum()}",
        "\n=== Column Metrics ==="
    ]

    num_cols = [c for c, m in meta.items() if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
    cat_cols = [c for c, m in meta.items() if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}]
    dt_cols = [c for c, m in meta.items() if m.data_type == DataType.DATETIME]

    def safe_plot(fig, filename):
        try:
            fig.tight_layout()
            fig.savefig(run_dir / "images" / filename)
            plt.close(fig)
        except Exception as e:
            LOGGER.warning("Failed to save %s: %s", filename, e)

    def correlation_matrix_plot(df1, df2, filename_prefix):
        corr1 = df1.corr()
        corr2 = df2.corr()
        diff = (corr1 - corr2).abs()
        mae = mean_absolute_error(corr1.values.flatten(), corr2.values.flatten())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (mat, title) in enumerate(zip([corr1, corr2, diff], ["Real", "Synthetic", "Difference"])):
            sns.heatmap(mat, vmin=-1, vmax=1, cmap="vlag", ax=axes[i])
            axes[i].set_title(title)
        safe_plot(fig, f"{filename_prefix}_correlation_matrix.png")
        return mae

    def eta_squared_matrix(df, meta):
        eta = np.zeros((len(cat_cols), len(num_cols)))
        for i, cat_col in enumerate(cat_cols):
            for j, num_col in enumerate(num_cols):
                try:
                    groups = df.groupby(cat_col)[num_col]
                    eta[i, j] = np.nan_to_num(
                        sum(len(g) * (g.mean() - df[num_col].mean()) ** 2 for _, g in groups) /
                        ((df[num_col] - df[num_col].mean()) ** 2).sum()
                    )
                except:
                    eta[i, j] = np.nan
        return eta

    for col in real.columns:
        if col not in synth.columns:
            summary_lines.append(f"!! {col} missing in synthetic")
            continue

        r, s = real[col], synth[col]

        if col in num_cols:
            w_dist = wasserstein_distance(r, s)
            ks_stat, _ = ks_2samp(r, s)
            summary_lines.append(f"{col:30}| W-dist {w_dist:.3f} | KS {ks_stat:.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            bins = min(100, int(np.sqrt(len(r))))
            axes[0].hist(r, bins=bins, density=True, alpha=0.8, label="Real", color="tab:blue")
            axes[0].hist(s, bins=bins, density=True, alpha=0.8, label="Synthetic", color="tab:orange")
            axes[0].set_title("Distribution Comparison", fontsize=12)
            axes[0].legend()
            axes[0].set_ylabel("Density")

            # sns.kdeplot(r, ax=axes[0], label="Real", lw=2)
            # sns.kdeplot(s, ax=axes[0], label="Synthetic", lw=2)
            # axes[0].legend()
            # axes[0].set_title(f"KDE: {col}")

            probs = np.linspace(0.01, 0.99, 100)
            qr = np.quantile(r, probs)
            qs = np.quantile(s, probs)
            axes[1].scatter(qr, qs, s=8)
            axes[1].plot([qr.min(), qr.max()], [qr.min(), qr.max()], 'k--')
            axes[1].set_title(f"QQ Plot: {col}")

            safe_plot(fig, f"{col}_numeric.png")

        elif col in cat_cols:
            real_freq = r.value_counts(normalize=True)
            synth_freq = s.value_counts(normalize=True)
            idx = sorted(set(real_freq.index).union(synth_freq.index))
            real_vals = real_freq.reindex(idx, fill_value=0)
            synth_vals = synth_freq.reindex(idx, fill_value=0)
            mae = np.abs(real_vals - synth_vals).mean()
            summary_lines.append(f"{col:30}| Cat-MAE {mae:.3f}")

            fig, ax = plt.subplots(figsize=(10, 4))
            width = 0.4
            x = np.arange(len(idx))
            ax.bar(x - width / 2, real_vals.values, width, label="Real")
            ax.bar(x + width / 2, synth_vals.values, width, label="Synthetic")
            ax.set_xticks(x)
            ax.set_xticklabels(idx, rotation=30)
            ax.set_title(f"Frequencies: {col}")
            ax.legend()

            safe_plot(fig, f"{col}_categorical.png")

        elif col in dt_cols:
            fmt = meta[col].datetime_format or "%Y-%m-%d"
            r_dt = pd.to_datetime(r, errors='coerce', format=fmt)
            s_dt = pd.to_datetime(s, errors='coerce', format=fmt)
            summary_lines.append(f"{col:30}| Range real [{r_dt.min()} – {r_dt.max()}] | synth [{s_dt.min()} – {s_dt.max()}]")

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(r_dt, stat="density", bins=50, label="Real", kde=True, ax=ax)
            sns.histplot(s_dt, stat="density", bins=50, label="Synthetic", kde=True, ax=ax)
            ax.legend()
            ax.set_title(f"Datetime: {col}")
            safe_plot(fig, f"{col}_datetime.png")

    if len(num_cols) >= 2:
        mae_corr = correlation_matrix_plot(real[num_cols], synth[num_cols], "num")
        summary_lines.append(f"MAE of numeric correlations: {mae_corr:.4f}")

    if cat_cols and num_cols:
        eta_real = eta_squared_matrix(real, meta)
        eta_synth = eta_squared_matrix(synth, meta)
        eta_diff = np.abs(eta_real - eta_synth)

        for mat, title, fname in zip(
            [eta_real, eta_synth, eta_diff],
            ["η² Real", "η² Synthetic", "η² Abs Diff"],
            ["eta_real.png", "eta_synth.png", "eta_diff.png"]
        ):
            fig, ax = plt.subplots(figsize=(1.5 * len(num_cols), 0.5 * len(cat_cols) + 3))
            sns.heatmap(mat, annot=True, xticklabels=num_cols, yticklabels=cat_cols, cmap="rocket_r", vmin=0, vmax=1, ax=ax)
            ax.set_title(title)
            safe_plot(fig, fname)

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    LOGGER.info("Report written to: %s", run_dir)
    return str(run_dir)
