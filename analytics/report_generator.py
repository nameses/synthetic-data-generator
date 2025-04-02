import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple
from pathlib import Path

from models.enums import DataType
from models.field_metadata import FieldMetadata


def generate_comparison_report(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: Dict[str, FieldMetadata],
        report_dir: str = "reports"
) -> None:
    """
    Generate comprehensive comparison report with statistics and visualizations

    Parameters:
        real_data: Original DataFrame
        synthetic_data: Generated DataFrame
        metadata: Dictionary of FieldMetadata objects
        report_dir: Base directory for reports
    """
    # Create report directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(report_dir) / f"report-{timestamp}"
    report_path.mkdir(parents=True, exist_ok=True)

    # Save original and synthetic data as CSV
    real_data.to_csv(report_path / "original_data.csv", index=False)
    synthetic_data.to_csv(report_path / "synthetic_data.csv", index=False)

    # Initialize report file
    report_file = report_path / "comparison_report.txt"

    # Generate statistics and visualizations
    with open(report_file, 'w') as f:
        f.write("SYNTHETIC DATA QUALITY REPORT\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        f.write(f"Real data shape: {real_data.shape}\n")
        f.write(f"Synthetic data shape: {synthetic_data.shape}\n\n")

        # Compare each column
        for col in real_data.columns:
            if col not in synthetic_data.columns:
                f.write(f"Column {col} missing in synthetic data!\n")
                continue

            f.write(f"\n=== Column: {col} ===\n")
            col_type = metadata[col].data_type if col in metadata else 'unknown'
            f.write(f"Data type: {col_type}\n")

            try:
                # Basic statistics
                real_stats = get_column_stats(real_data[col])
                synth_stats = get_column_stats(synthetic_data[col])

                f.write("\nReal Data Statistics:\n")
                for stat, value in real_stats.items():
                    f.write(f"{stat}: {value}\n")

                f.write("\nSynthetic Data Statistics:\n")
                for stat, value in synth_stats.items():
                    f.write(f"{stat}: {value}\n")

                # Type-specific comparisons
                if col_type == DataType.INTEGER or col_type == DataType.DECIMAL:
                    # Numerical comparison
                    ks_stat, p_value = stats.ks_2samp(
                        real_data[col].dropna(),
                        synthetic_data[col].dropna()
                    )
                    f.write(f"\nKolmogorov-Smirnov Test:\n")
                    f.write(f"Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}\n")

                    # Plot distributions
                    plot_distributions(
                        real_data[col],
                        synthetic_data[col],
                        col,
                        report_path
                    )

                elif col_type == DataType.CATEGORICAL:
                    # Categorical comparison
                    real_counts = real_data[col].value_counts(normalize=True)
                    synth_counts = synthetic_data[col].value_counts(normalize=True)

                    f.write("\nReal Value Proportions:\n")
                    for val, prop in real_counts.items():
                        f.write(f"{val}: {prop:.4f}\n")

                    f.write("\nSynthetic Value Proportions:\n")
                    for val, prop in synth_counts.items():
                        f.write(f"{val}: {prop:.4f}\n")

                    # Plot categorical distributions
                    plot_categorical_distributions(
                        real_data[col],
                        synthetic_data[col],
                        col,
                        report_path
                    )


                elif col_type == DataType.BOOLEAN:
                    # Boolean comparison
                    try:
                        real_prop = real_data[col].astype(int).mean()
                        synth_prop = synthetic_data[col].astype(int).mean()
                        f.write(f"\nReal True Proportion: {real_prop:.4f}\n")
                        f.write(f"Synthetic True Proportion: {synth_prop:.4f}\n")
                        # Plot boolean distributions
                        plot_boolean_distributions(
                            real_data[col].astype(int),
                            synthetic_data[col].astype(int),
                            col,
                            report_path
                        )
                    except Exception as e:
                        f.write(f"\nError processing boolean column {col}: {str(e)}\n")

                elif col_type == DataType.DATE_TIME:
                    # Date comparison
                    real_dates = pd.to_datetime(real_data[col])
                    synth_dates = pd.to_datetime(synthetic_data[col])

                    f.write("\nReal Date Range:\n")
                    f.write(f"Min: {real_dates.min()}, Max: {real_dates.max()}\n")
                    f.write("\nSynthetic Date Range:\n")
                    f.write(f"Min: {synth_dates.min()}, Max: {synth_dates.max()}\n")

                    # Plot date distributions
                    plot_date_distributions(
                        real_data[col],
                        synthetic_data[col],
                        col,
                        report_path
                    )

            except Exception as e:
                f.write(f"\nError analyzing column {col}: {str(e)}\n")
                continue

    print(f"Report generated at: {report_path}")


# Update the get_column_stats function in report_generator.py
def get_column_stats(series: pd.Series) -> Dict[str, float]:
    """Calculate basic statistics for a column with proper type handling"""
    stats = {
        'Count': len(series),
        'Missing': series.isna().sum(),
        'Unique': series.nunique()
    }

    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            'Mean': series.mean(),
            'Std': series.std(),
            'Min': series.min(),
            '25%': series.quantile(0.25),
            '50%': series.quantile(0.5),
            '75%': series.quantile(0.75),
            'Max': series.max()
        })
    elif pd.api.types.is_bool_dtype(series):
        # Convert boolean to int for calculations
        int_series = series.astype(int)
        stats.update({
            'True Proportion': int_series.mean(),
            'False Proportion': 1 - int_series.mean()
        })
    elif pd.api.types.is_datetime64_any_dtype(series):
        stats.update({
            'Min': series.min(),
            'Max': series.max()
        })
    elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        value_counts = series.value_counts(normalize=True)
        stats.update({
            'Most Common': value_counts.idxmax() if len(value_counts) > 0 else None,
            'Most Common Freq': value_counts.max() if len(value_counts) > 0 else 0,
            'Mean Encoding': value_counts.mean()  # Add mean for categorical types
        })

    return {k: str(v) if not isinstance(v, (int, float)) else v for k, v in stats.items()}


def plot_distributions(
        real_series: pd.Series,
        synth_series: pd.Series,
        col_name: str,
        report_path: Path
) -> None:
    """Plot numerical distributions"""
    plt.figure(figsize=(12, 6))

    # Histograms
    sns.histplot(real_series, color='blue', label='Real', kde=True, alpha=0.5)
    sns.histplot(synth_series, color='orange', label='Synthetic', kde=True, alpha=0.5)

    plt.title(f'Distribution Comparison - {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Density')
    plt.legend()

    plot_path = report_path / f"{col_name}_distribution.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def plot_categorical_distributions(
        real_series: pd.Series,
        synth_series: pd.Series,
        col_name: str,
        report_path: Path
) -> None:
    """Plot categorical distributions"""
    plt.figure(figsize=(12, 6))

    # Get top 10 categories
    real_counts = real_series.value_counts(normalize=True).head(10)
    synth_counts = synth_series.value_counts(normalize=True).head(10)

    # Combine for plotting
    df = pd.DataFrame({
        'Real': real_counts,
        'Synthetic': synth_counts
    }).fillna(0)

    df.plot(kind='bar', color=['blue', 'orange'])
    plt.title(f'Category Proportions - {col_name}')
    plt.xlabel('Category')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)

    plot_path = report_path / f"{col_name}_categories.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def plot_boolean_distributions(
        real_series: pd.Series,
        synth_series: pd.Series,
        col_name: str,
        report_path: Path
) -> None:
    """Plot boolean distributions"""
    plt.figure(figsize=(8, 6))

    real_props = real_series.value_counts(normalize=True)
    synth_props = synth_series.value_counts(normalize=True)

    df = pd.DataFrame({
        'Real': real_props,
        'Synthetic': synth_props
    }).T

    df.plot(kind='bar', stacked=True, color=['green', 'red'])
    plt.title(f'Boolean Distribution - {col_name}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)

    plot_path = report_path / f"{col_name}_boolean.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def plot_date_distributions(
        real_series: pd.Series,
        synth_series: pd.Series,
        col_name: str,
        report_path: Path
) -> None:
    """Plot date distributions"""
    plt.figure(figsize=(12, 6))

    real_dates = pd.to_datetime(real_series)
    synth_dates = pd.to_datetime(synth_series)

    # Plot histograms by year-month
    real_counts = real_dates.dt.to_period('M').value_counts().sort_index()
    synth_counts = synth_dates.dt.to_period('M').value_counts().sort_index()

    plt.plot(real_counts.index.astype(str), real_counts,
             label='Real', color='blue')
    plt.plot(synth_counts.index.astype(str), synth_counts,
             label='Synthetic', color='orange')

    plt.title(f'Date Distribution - {col_name}')
    plt.xlabel('Year-Month')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()

    plot_path = report_path / f"{col_name}_dates.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()