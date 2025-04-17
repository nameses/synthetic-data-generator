"""
analytics/report_generator.py

Creates a timestamped folder inside `reports/` with:
  • synthetic.csv                     (full generated data)
  • summary.txt                       (plain‑text metrics)
  • <col>_numeric.png                 KDE + QQ plot   (numeric)
  • <col>_categorical.png             grouped bars + |diff| bars
  • <col>_datetime.png                histogram + KDE
  • corr_numeric.png                  Pearson ρ (num+dt ↔ num+dt)
  • eta_real/synth/diff.png           η² (category → numeric) matrices
  • dt_corr_real/synth/diff.png       ρ (datetime ↔ numeric/datetime)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error

from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------#
# GLOBAL STYLE
# ------------------------------------------------------------------#
sns.set_theme(
    style="white",
    context="talk",
    palette=sns.color_palette(["#2563eb", "#f97316"]),  # blue | orange
)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 14,
    }
)

# ------------------------------------------------------------------#
# HELPERS
# ------------------------------------------------------------------#


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """η² — fraction of variance in *num* explained by *cat* (ANOVA)."""
    cat = cat.astype("category")
    if cat.nunique() < 2:
        return np.nan
    grand_mean = num.mean()
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for _, g in num.groupby(cat, observed=False)
    )
    ss_total = ((num - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total else np.nan


def _qq_plot(ax, real: pd.Series, synth: pd.Series, title: str, n: int = 1000) -> None:
    """Empirical QQ plot (synthetic vs real)."""
    probs = np.linspace(0, 1, n + 2)[1:-1]
    q_real = np.quantile(real, probs)
    q_synth = np.quantile(synth, probs)

    ax.scatter(q_real, q_synth, alpha=0.6, s=20)
    lims = [np.nanmin([q_real, q_synth]), np.nanmax([q_real, q_synth])]
    ax.plot(lims, lims, ls="--", lw=1, c="black")
    ax.set_title(f"{title} - QQ plot")
    ax.set_xlabel("Real quantiles")
    ax.set_ylabel("Synthetic quantiles")


def _to_epoch(series: pd.Series, fmt: str) -> pd.Series:
    """Datetime string → epoch seconds (int)."""
    return (
        pd.to_datetime(series, format=fmt, errors="coerce")
        .astype("int64")
        .floordiv(10**9)
    )


# ------------------------------------------------------------------#
# MAIN REPORT FUNCTION
# ------------------------------------------------------------------#
def generate_report(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    meta: Dict[str, FieldMetadata],
    out_dir: str | os.PathLike = "reports",
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"report_{ts}"
    _ensure_dir(run_dir)

    # ------------------------------------------------------------------#
    # GLOBAL SUMMARY & SAVE SYNTHETIC CSV
    # ------------------------------------------------------------------#
    synth.to_csv(run_dir / "synthetic.csv", index=False)

    summary: List[str] = [
        "SYNTHETIC DATA QUALITY REPORT",
        f"Created: {datetime.now()}",
        f"Real shape:      {real.shape}",
        f"Synthetic shape: {synth.shape}",
        f"NaNs in real:  {real.isna().sum().sum()}",
        f"NaNs in synth: {synth.isna().sum().sum()}",
        "",
        "=== Column metrics ===",
    ]

    # column groups
    num_cols = [c for c, m in meta.items() if m.data_type in {DataType.INTEGER, DataType.DECIMAL}]
    dt_cols = [c for c, m in meta.items() if m.data_type is DataType.DATETIME]
    num_cols_with_dt = num_cols + dt_cols
    cat_cols = [c for c, m in meta.items() if m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}]

    # ------------------------------------------------------------------#
    # PER‑COLUMN PLOTS & METRICS
    # ------------------------------------------------------------------#
    for col, m in meta.items():
        if col not in synth.columns:
            summary.append(f"!! {col} missing in synthetic")
            continue

        # ---------- numeric ----------
        if m.data_type in {DataType.INTEGER, DataType.DECIMAL}:
            r, s = real[col], synth[col]
            w = wasserstein_distance(r, s)
            summary.append(f"{col:30s}| W-dist {w:7.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.kdeplot(r, ax=axes[0], label="Real", linewidth=2)
            sns.kdeplot(s, ax=axes[0], label="Synthetic", linewidth=2)
            axes[0].set_title(col + " - KDE")
            axes[0].legend()

            _qq_plot(axes[1], r, s, col)
            fig.tight_layout()
            fig.savefig(run_dir / f"{col}_numeric.png")
            plt.close(fig)

        # ---------- categorical / boolean ----------
        elif m.data_type in {DataType.CATEGORICAL, DataType.BOOLEAN}:
            r_counts = real[col].value_counts(normalize=True)
            s_counts = synth[col].value_counts(normalize=True)
            idx = sorted(set(r_counts.index).union(s_counts.index))
            r_vals = r_counts.reindex(idx, fill_value=0)
            s_vals = s_counts.reindex(idx, fill_value=0)
            diff = (r_vals - s_vals).abs()
            mae = diff.mean()
            summary.append(f"{col:30s}| Cat-MAE {mae:7.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            width = 0.4
            axes[0].bar(np.arange(len(idx)) - width / 2, r_vals.values, width, label="Real")
            axes[0].bar(np.arange(len(idx)) + width / 2, s_vals.values, width, label="Synthetic")
            axes[0].set_xticks(np.arange(len(idx)))
            axes[0].set_xticklabels(idx, rotation=40, ha="right")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title(col + " - frequencies")
            axes[0].legend()

            axes[1].bar(np.arange(len(idx)), diff.values, color="gray", alpha=0.7)
            axes[1].set_xticks(np.arange(len(idx)))
            axes[1].set_xticklabels(idx, rotation=40, ha="right")
            axes[1].set_ylabel("|Real - Synth|")
            axes[1].set_title("Absolute difference")
            fig.tight_layout()
            fig.savefig(run_dir / f"{col}_categorical.png")
            plt.close(fig)

        # ---------- datetime ----------
        elif m.data_type is DataType.DATETIME:
            r_dt = pd.to_datetime(real[col], format=m.datetime_format, errors="coerce")
            s_dt = pd.to_datetime(synth[col], format=m.datetime_format, errors="coerce")

            summary.append(
                f"{col:30s}| Range real [{r_dt.min()} – {r_dt.max()}] "
                f"| synth [{s_dt.min()} – {s_dt.max()}]"
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(r_dt, bins=50, stat="density", alpha=0.6, label="Real", ax=ax)
            sns.kdeplot(s_dt, lw=2, label="Synthetic", ax=ax)
            ax.set_title(col + " - datetime distribution")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / f"{col}_datetime.png")
            plt.close(fig)

    # ------------------------------------------------------------------#
    # NUMERIC (+DATETIME) ↔ NUMERIC Pearson CORRELATION
    # ------------------------------------------------------------------#
    def _numeric_like(df: pd.DataFrame) -> pd.DataFrame:
        """Return numeric columns plus datetime converted to epoch seconds."""
        out = df[num_cols].copy()
        for c in dt_cols:
            out[c] = _to_epoch(df[c], meta[c].datetime_format)
        return out.astype(float)

    if len(num_cols_with_dt) >= 2:
        real_num = _numeric_like(real)
        synth_num = _numeric_like(synth)

        real_corr = real_num.corr()
        synth_corr = synth_num.corr()
        diff_corr = real_corr - synth_corr
        mae_corr = mean_absolute_error(real_corr.values.flatten(), synth_corr.values.flatten())
        summary.extend(["", f"MAE of Pearson correlations (num+dt): {mae_corr:.4f}"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for mat, ax, title in [
            (real_corr, axes[0], "Real"),
            (synth_corr, axes[1], "Synthetic"),
            (diff_corr, axes[2], "Difference"),
        ]:
            sns.heatmap(mat, vmin=-1, vmax=1, cmap="vlag", square=True, ax=ax, cbar=False)
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(run_dir / "corr_numeric.png")
        plt.close(fig)

    # ------------------------------------------------------------------#
    # CATEGORY → NUMERIC/DATETIME η² MATRICES
    # ------------------------------------------------------------------#
    if cat_cols and num_cols_with_dt:
        def _eta_matrix(df: pd.DataFrame) -> np.ndarray:
            eta = np.zeros((len(cat_cols), len(num_cols_with_dt)))
            num_df = _numeric_like(df)
            for i, c_cat in enumerate(cat_cols):
                for j, c_num in enumerate(num_cols_with_dt):
                    eta[i, j] = _correlation_ratio(df[c_cat], num_df[c_num])
            return eta

        eta_real = _eta_matrix(real)
        eta_synth = _eta_matrix(synth)
        eta_diff = eta_real - eta_synth
        mae_eta = np.nanmean(np.abs(eta_diff))
        summary.append(f"MAE of η² (cat→num/dt): {mae_eta:.4f}")

        for mat, fname, title in [
            (eta_real, "eta_real.png", "η² - real"),
            (eta_synth, "eta_synth.png", "η² - synthetic"),
            (eta_diff, "eta_diff.png", "η² - difference"),
        ]:
            fig, ax = plt.subplots(figsize=(1.6 * len(num_cols_with_dt), 0.6 * len(cat_cols) + 3))
            sns.heatmap(
                mat,
                vmin=0,
                vmax=1,
                cmap="rocket_r",
                square=False,
                ax=ax,
                xticklabels=num_cols_with_dt,
                yticklabels=cat_cols,
                cbar_kws={"label": "η²"},
            )
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(run_dir / fname)
            plt.close(fig)

    # ------------------------------------------------------------------#
    # DATETIME ↔ NUMERIC/DATETIME Pearson CORRELATIONS
    # ------------------------------------------------------------------#
    if dt_cols:
        def _dt_corr(df: pd.DataFrame) -> pd.DataFrame:
            num_df = _numeric_like(df)
            out = pd.DataFrame(index=dt_cols, columns=num_cols_with_dt, dtype=float)
            for d in dt_cols:
                d_epoch = _to_epoch(df[d], meta[d].datetime_format).astype(float)
                for n in num_cols_with_dt:
                    out.loc[d, n] = np.corrcoef(d_epoch, num_df[n])[0, 1]
            return out

        real_dt_corr = _dt_corr(real)
        synth_dt_corr = _dt_corr(synth)
        diff_dt_corr = real_dt_corr - synth_dt_corr

        for mat, fname, title in [
            (real_dt_corr, "dt_corr_real.png", "Datetime corr - real"),
            (synth_dt_corr, "dt_corr_synth.png", "Datetime corr - synthetic"),
            (diff_dt_corr, "dt_corr_diff.png", "Datetime corr - diff (real-synth)"),
        ]:
            fig, ax = plt.subplots(figsize=(1.6 * len(num_cols_with_dt), 0.6 * len(dt_cols) + 3))
            sns.heatmap(
                mat,
                vmin=-1,
                vmax=1,
                cmap="vlag",
                square=False,
                ax=ax,
                cbar=False,
                xticklabels=num_cols_with_dt,
                yticklabels=dt_cols,
            )
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(run_dir / fname)
            plt.close(fig)

    # ------------------------------------------------------------------#
    # SAVE SUMMARY
    # ------------------------------------------------------------------#
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    LOGGER.info("Report written to %s", run_dir)
    return str(run_dir)
