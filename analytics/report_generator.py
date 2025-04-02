import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import ks_2samp, wasserstein_distance

from models.enums import DataType
from models.field_metadata import FieldMetadata


def generate_comparison_report(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: Dict[str, FieldMetadata],
        report_dir: str = "reports"
) -> str:
    """
    Generate comprehensive comparison report with visualizations and metrics.

    Args:
        real_data: Original dataset
        synthetic_data: Generated synthetic dataset
        metadata: Dictionary of FieldMetadata objects
        report_dir: Base directory for reports

    Returns:
        Path to the generated report directory
    """
    # Create report directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"report_{timestamp}")
    os.makedirs(report_path, exist_ok=True)

    # Save datasets
    real_data.to_csv(os.path.join(report_path, "original_data.csv"), index=False)
    synthetic_data.to_csv(os.path.join(report_path, "synthetic_data.csv"), index=False)

    # Initialize report content
    report_content = []
    report_content.append("SYNTHETIC DATA QUALITY REPORT")
    report_content.append(f"Generated at: {datetime.now()}")
    report_content.append(f"\nOriginal data shape: {real_data.shape}")
    report_content.append(f"Synthetic data shape: {synthetic_data.shape}\n")

    # Set up visualization style
    #plt.style.use('seaborn')
    sns.set_theme()
    sns.set_palette("husl")

    # Analyze each column
    for col in real_data.columns:
        report_content.append(f"\n=== Column: {col} ===")
        meta = metadata.get(col)
        col_type = meta.data_type if meta else "unknown"
        report_content.append(f"Data type: {col_type}")

        if col not in synthetic_data.columns:
            report_content.append("Warning: Column missing in synthetic data\n")
            continue

        # Handle different data types
        if meta and meta.data_type in [DataType.DECIMAL, DataType.INTEGER]:
            report_content.extend(_analyze_numerical(col, real_data, synthetic_data, report_path))
        elif meta and meta.data_type == DataType.CATEGORICAL:
            report_content.extend(_analyze_categorical(col, real_data, synthetic_data, report_path))
        elif meta and meta.data_type == DataType.BOOLEAN:
            report_content.extend(_analyze_boolean(col, real_data, synthetic_data, report_path))
        elif meta and meta.data_type == DataType.DATE_TIME:
            report_content.extend(_analyze_datetime(col, real_data, synthetic_data, report_path))
        else:
            report_content.extend(_analyze_generic(col, real_data, synthetic_data, report_path))

    # Add correlation comparison
    _compare_correlations(real_data, synthetic_data, metadata, report_path)
    report_content.append("\n=== Correlation Matrices ===\n")
    report_content.append("Visualizations saved as: correlation_comparison.png")

    # Save text report
    report_txt_path = os.path.join(report_path, "report.txt")
    with open(report_txt_path, 'w') as f:
        f.write("\n".join(report_content))

    return report_path


def _analyze_numerical(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Analyze numerical column and generate visualizations"""
    content = []

    # Calculate statistics
    real_stats = real_data[col].describe()
    synth_stats = synthetic_data[col].describe()

    # KS test
    ks_stat, ks_p = ks_2samp(real_data[col], synthetic_data[col])
    wass_dist = wasserstein_distance(real_data[col], synthetic_data[col])

    content.append("\nReal Data Statistics:")
    content.append(f"Count: {real_stats['count']:.0f}")
    content.append(f"Mean: {real_stats['mean']:.4f}")
    content.append(f"Std: {real_stats['std']:.4f}")
    content.append(f"Min: {real_stats['min']:.4f}")
    content.append(f"25%: {real_stats['25%']:.4f}")
    content.append(f"50%: {real_stats['50%']:.4f}")
    content.append(f"75%: {real_stats['75%']:.4f}")
    content.append(f"Max: {real_stats['max']:.4f}")

    content.append("\nSynthetic Data Statistics:")
    content.append(f"Count: {synth_stats['count']:.0f}")
    content.append(f"Mean: {synth_stats['mean']:.4f}")
    content.append(f"Std: {synth_stats['std']:.4f}")
    content.append(f"Min: {synth_stats['min']:.4f}")
    content.append(f"25%: {synth_stats['25%']:.4f}")
    content.append(f"50%: {synth_stats['50%']:.4f}")
    content.append(f"75%: {synth_stats['75%']:.4f}")
    content.append(f"Max: {synth_stats['max']:.4f}")

    content.append("\nStatistical Tests:")
    content.append(f"Kolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    content.append(f"Wasserstein Distance: {wass_dist:.4f}")

    # Generate visualizations
    plt.figure(figsize=(12, 6))

    # Histogram comparison
    plt.subplot(1, 2, 1)
    sns.histplot(real_data[col], color='blue', label='Real', kde=True, alpha=0.5)
    sns.histplot(synthetic_data[col], color='orange', label='Synthetic', kde=True, alpha=0.5)
    plt.title(f'Distribution Comparison: {col}')
    plt.legend()

    # QQ plot
    plt.subplot(1, 2, 2)
    real_sorted = np.sort(real_data[col].dropna())
    synth_sorted = np.sort(synthetic_data[col].dropna())

    # Fix: Sample the larger dataset to match the size of the smaller one
    if len(real_sorted) > len(synth_sorted):
        # Sample real data to match synthetic data size
        indices = np.linspace(0, len(real_sorted) - 1, len(synth_sorted)).astype(int)
        real_sorted_sampled = real_sorted[indices]
        plt.plot(real_sorted_sampled, synth_sorted, 'o', alpha=0.5)
    else:
        # Sample synthetic data to match real data size
        indices = np.linspace(0, len(synth_sorted) - 1, len(real_sorted)).astype(int)
        synth_sorted_sampled = synth_sorted[indices]
        plt.plot(real_sorted, synth_sorted_sampled, 'o', alpha=0.5)

    # Rest of QQ plot code
    min_val = min(real_sorted.min(), synth_sorted.min())
    max_val = max(real_sorted.max(), synth_sorted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Real Data Quantiles')
    plt.ylabel('Synthetic Data Quantiles')
    plt.title(f'QQ Plot: {col}')

    plt.tight_layout()
    plt.savefig(os.path.join(report_path, f"{col}_comparison.png"))
    plt.close()

    content.append(f"\nVisualizations saved as: {col}_comparison.png")
    return content


def _analyze_categorical(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[
    str]:
    """Analyze categorical column and generate visualizations"""
    content = []

    # Calculate value counts
    real_counts = real_data[col].value_counts(normalize=True)
    synth_counts = synthetic_data[col].value_counts(normalize=True)

    # Fix: Sample a subset of real data to match synthetic data size for mutual info calculation
    if len(real_data) > len(synthetic_data):
        # Sample real data to match synthetic data size
        real_sample = real_data.sample(n=len(synthetic_data), random_state=42)
        mi = mutual_info_score(
            real_sample[col].astype('category').cat.codes,
            synthetic_data[col].astype('category').cat.codes
        )
    else:
        # Sample synthetic data to match real data size
        synth_sample = synthetic_data.sample(n=len(real_data), random_state=42)
        mi = mutual_info_score(
            real_data[col].astype('category').cat.codes,
            synth_sample[col].astype('category').cat.codes
        )

    content.append("\nReal Value Proportions:")
    for val, prop in real_counts.items():
        content.append(f"{val}: {prop:.4f}")

    content.append("\nSynthetic Value Proportions:")
    for val, prop in synth_counts.items():
        content.append(f"{val}: {prop:.4f}")

    content.append(f"\nMutual Information Score: {mi:.4f}")

    # Generate visualization
    plt.figure(figsize=(10, 6))

    df = pd.DataFrame({
        'Real': real_counts,
        'Synthetic': synth_counts
    }).fillna(0)

    df.plot(kind='bar', ax=plt.gca())
    plt.title(f'Category Proportions: {col}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, f"{col}_comparison.png"))
    plt.close()

    content.append(f"\nVisualization saved as: {col}_comparison.png")
    return content


def _analyze_boolean(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Analyze boolean column"""
    content = []

    real_prop = real_data[col].mean()
    synth_prop = synthetic_data[col].mean()

    content.append("\nReal Data:")
    content.append(f"True proportion: {real_prop:.4f}")
    content.append(f"False proportion: {1 - real_prop:.4f}")

    content.append("\nSynthetic Data:")
    content.append(f"True proportion: {synth_prop:.4f}")
    content.append(f"False proportion: {1 - synth_prop:.4f}")

    # Generate visualization
    plt.figure(figsize=(8, 5))

    pd.DataFrame({
        'Real': [real_prop, 1 - real_prop],
        'Synthetic': [synth_prop, 1 - synth_prop]
    }, index=['True', 'False']).plot(kind='bar', ax=plt.gca())

    plt.title(f'Boolean Proportions: {col}')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, f"{col}_comparison.png"))
    plt.close()

    content.append(f"\nVisualization saved as: {col}_comparison.png")
    return content


def _analyze_datetime(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Analyze datetime column"""
    content = []

    real_dates = pd.to_datetime(real_data[col])
    synth_dates = pd.to_datetime(synthetic_data[col])

    content.append("\nReal Data Date Range:")
    content.append(f"Start: {real_dates.min()}")
    content.append(f"End: {real_dates.max()}")

    content.append("\nSynthetic Data Date Range:")
    content.append(f"Start: {synth_dates.min()}")
    content.append(f"End: {synth_dates.max()}")

    # Generate visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    real_dates.hist(ax=plt.gca(), bins=20, color='blue', alpha=0.7)
    plt.title('Real Date Distribution')

    plt.subplot(1, 2, 2)
    synth_dates.hist(ax=plt.gca(), bins=20, color='orange', alpha=0.7)
    plt.title('Synthetic Date Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(report_path, f"{col}_comparison.png"))
    plt.close()

    content.append(f"\nVisualization saved as: {col}_comparison.png")
    return content


def _analyze_generic(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Analyze generic/unknown column type"""
    content = []

    content.append("\nWarning: Unknown column type - basic analysis only")

    # For strings, show most common values
    if real_data[col].dtype == 'object':
        content.append("\nReal Data Top Values:")
        for val, count in real_data[col].value_counts().head(5).items():
            content.append(f"{val}: {count}")

        content.append("\nSynthetic Data Top Values:")
        for val, count in synthetic_data[col].value_counts().head(5).items():
            content.append(f"{val}: {count}")

    return content


def _compare_correlations(real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                          metadata: Dict[str, FieldMetadata], report_path: str):
    """Compare correlation matrices between real and synthetic data"""
    numerical_cols = [col for col, meta in metadata.items()
                      if meta and meta.data_type in [DataType.DECIMAL, DataType.INTEGER]]

    if len(numerical_cols) < 2:
        return

    plt.figure(figsize=(16, 8))

    # Real data correlations
    plt.subplot(1, 2, 1)
    real_corr = real_data[numerical_cols].corr()
    sns.heatmap(real_corr, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, square=True)
    plt.title('Real Data Correlations')

    # Synthetic data correlations
    plt.subplot(1, 2, 2)
    synth_corr = synthetic_data[numerical_cols].corr()
    sns.heatmap(synth_corr, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, square=True)
    plt.title('Synthetic Data Correlations')

    plt.tight_layout()
    plt.savefig(os.path.join(report_path, "correlation_comparison.png"))
    plt.close()