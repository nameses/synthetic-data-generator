import os
import gc
import logging
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
from sklearn.metrics import mutual_info_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from warnings import filterwarnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter specific warnings
filterwarnings('ignore', category=RuntimeWarning)
filterwarnings('ignore', category=UserWarning)

# Set style - using modern seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100  # Reduced for better performance
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.max_open_warning'] = 50

# Custom color palette
QUALITATIVE_PALETTE = sns.color_palette("husl", 10)
DIVERGING_PALETTE = sns.color_palette("vlag", as_cmap=True)
SEQUENTIAL_PALETTE = sns.color_palette("flare", as_cmap=True)

from models.enums import DataType
from models.field_metadata import FieldMetadata


def generate_comparison_report(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: Dict[str, FieldMetadata],
        report_dir: str = "reports",
        sample_size: Optional[int] = 2000,
        skip_memory_intensive: bool = True
) -> str:
    """
    Generate comprehensive comparison report between real and synthetic data.

    Args:
        real_data: Original dataset
        synthetic_data: Generated synthetic dataset
        metadata: Dictionary of field metadata
        report_dir: Output directory for reports
        sample_size: Number of samples to use for analysis (None for all)
        skip_memory_intensive: Skip analyses that require significant memory

    Returns:
        Path to the generated report directory
    """
    # Create report directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"report_{timestamp}")
    os.makedirs(report_path, exist_ok=True)

    # Initialize memory monitoring
    process = psutil.Process()
    mem_threshold = 0.8 * psutil.virtual_memory().available

    try:
        # Sample data if requested
        if sample_size is not None:
            real_data = real_data.sample(min(sample_size, len(real_data)), random_state=42)
            synthetic_data = synthetic_data.sample(min(sample_size, len(synthetic_data)), random_state=42)

        # Save datasets
        real_data.to_csv(os.path.join(report_path, "original_data.csv"), index=False)
        synthetic_data.to_csv(os.path.join(report_path, "synthetic_data.csv"), index=False)

        # Initialize report content
        report_content = []
        report_content.append("SYNTHETIC DATA QUALITY REPORT")
        report_content.append(f"Generated at: {datetime.now()}")
        report_content.append(f"\nOriginal data shape: {real_data.shape}")
        report_content.append(f"Synthetic data shape: {synthetic_data.shape}")
        report_content.append(f"\nColumns analyzed: {len(real_data.columns)}")
        report_content.append(f"Memory usage at start: {process.memory_info().rss / (1024 ** 2):.2f} MB\n")

        # 1. Dataset-level analyses
        report_content.append("\n=== DATASET-LEVEL ANALYSIS ===")
        report_content.extend(
            _analyze_dataset_level(real_data, synthetic_data, metadata, report_path, skip_memory_intensive))

        # 2. Column-level analyses
        report_content.append("\n=== COLUMN-LEVEL ANALYSIS ===")
        all_columns = set(real_data.columns).union(set(synthetic_data.columns)).union(set(metadata.keys()))
        for col_idx, col in enumerate(all_columns):
            try:
                meta = metadata.get(col)
                col_type = meta.data_type if meta else "unknown"

                if col not in synthetic_data.columns:
                    report_content.append(f"\n=== Column: {col} ===")
                    report_content.append(f"Warning: Column missing in synthetic data (Type: {col_type})\n")
                    continue

                if col not in real_data.columns:
                    report_content.append(f"\n=== Column: {col} ===")
                    report_content.append(f"Warning: Column missing in real data (Type: {col_type})\n")
                    continue

                # Memory check
                if process.memory_info().rss > mem_threshold:
                    logger.warning("Memory threshold reached, cleaning up")
                    plt.close('all')
                    gc.collect()
                    mem_threshold = 0.8 * psutil.virtual_memory().available



                logger.info(f"Processing column {col_idx + 1}/{len(real_data.columns)}: {col} ({col_type})")

                if col not in synthetic_data.columns:
                    report_content.append(f"\n=== Column: {col} ===")
                    report_content.append("Warning: Column missing in synthetic data\n")
                    continue

                report_content.append(f"\n=== Column: {col} ===")
                report_content.append(f"Data type: {col_type}")

                if meta and meta.data_type in [DataType.DECIMAL, DataType.INTEGER]:
                    report_content.extend(_analyze_numerical(col, real_data, synthetic_data, report_path))
                elif meta and meta.data_type == DataType.CATEGORICAL:
                    report_content.extend(_analyze_categorical(col, real_data, synthetic_data, report_path))
                elif meta and meta.data_type == DataType.BOOLEAN:
                    report_content.extend(_analyze_boolean(col, real_data, synthetic_data, report_path))
                elif meta and meta.data_type == DataType.DATETIME:
                    report_content.extend(_analyze_datetime(col, real_data, synthetic_data, report_path, datetime_format = meta.datetime_format))
                else:
                    report_content.extend(_analyze_generic(col, real_data, synthetic_data, report_path))

            except Exception as e:
                logger.error(f"Error processing column {col}: {str(e)}", exc_info=True)
                report_content.append(f"\nError processing column {col}: {str(e)}")
                plt.close('all')
                continue

        # 3. Relationship analyses (only if not skipped and we have enough data)
        if not skip_memory_intensive and len(real_data) > 0:
            try:
                report_content.append("\n=== RELATIONSHIP ANALYSIS ===")
                report_content.extend(_analyze_relationships(real_data, synthetic_data, metadata, report_path))
            except Exception as e:
                logger.error(f"Error in relationship analysis: {str(e)}")
                report_content.append("\nError in relationship analysis")

        # Final memory report
        report_content.append(f"\nFinal memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close('all')
        gc.collect()

    # Save text report
    report_txt_path = os.path.join(report_path, "report.txt")
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    return report_path


def _analyze_dataset_level(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: Dict[str, FieldMetadata],
        report_path: str,
        skip_memory_intensive: bool
) -> List[str]:
    """Perform dataset-level analyses"""
    content = []

    # 1. Basic statistics
    content.append("\nDataset Statistics:")
    content.append(f"Real data missing values: {real_data.isna().sum().sum()} ({real_data.isna().mean().mean():.2%})")
    content.append(
        f"Synthetic data missing values: {synthetic_data.isna().sum().sum()} ({synthetic_data.isna().mean().mean():.2%})")

    # 2. Data type consistency
    type_mismatches = [
        col for col in real_data.columns
        if col in synthetic_data.columns
           and real_data[col].dtype != synthetic_data[col].dtype
    ]
    content.append(f"\nData type mismatches: {len(type_mismatches)}")
    for col in type_mismatches[:5]:  # Show first 5
        content.append(f"- {col}: Real={real_data[col].dtype}, Synthetic={synthetic_data[col].dtype}")

    # 3. Duplicates analysis
    content.append(f"\nDuplicate rows:")
    content.append(f"Real data: {real_data.duplicated().sum()} ({real_data.duplicated().mean():.2%})")
    content.append(f"Synthetic data: {synthetic_data.duplicated().sum()} ({synthetic_data.duplicated().mean():.2%})")

    # 4. Visualization: Data type distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        type_counts = pd.DataFrame({
            'Real': real_data.dtypes.value_counts(),
            'Synthetic': synthetic_data.dtypes.value_counts()
        }).fillna(0)
        type_counts.plot(kind='bar', ax=ax)
        ax.set_title('Data Type Distribution Comparison')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(report_path, "data_type_distribution.png"))
        plt.close(fig)
        content.append("\nVisualization saved: data_type_distribution.png")
    except Exception as e:
        logger.warning(f"Could not create data type distribution plot: {str(e)}")

    # 5. Dimensionality reduction visualization (if not too large and not skipped)
    if not skip_memory_intensive and len(real_data) > 0 and len(real_data) < 5000:
        try:
            numerical_cols = real_data.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:  # Need at least 2 columns for DR
                content.extend(_visualize_dimensionality_reduction(real_data, synthetic_data, report_path))
        except Exception as e:
            logger.warning(f"Dimensionality reduction failed: {str(e)}")
            content.append("\nSkipped dimensionality reduction due to error")

    return content


def _analyze_numerical(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Optimized numerical column analysis for faster integer column processing"""
    content = []
    fig = None

    try:
        # Clean data - optimized for speed
        real_clean = real_data[col].replace([np.inf, -np.inf], np.nan).dropna().values
        synth_clean = synthetic_data[col].replace([np.inf, -np.inf], np.nan).dropna().values

        if len(real_clean) == 0 or len(synth_clean) == 0:
            content.append("\nWarning: No valid data after cleaning")
            return content

        # Calculate statistics using numpy for speed
        content.append("\nReal Data Statistics:")
        content.append(f"Count: {len(real_clean):.0f}")
        content.append(f"Mean: {np.mean(real_clean):.4f} ± {np.std(real_clean):.4f}")
        content.append(f"Range: [{np.min(real_clean):.4f}, {np.max(real_clean):.4f}]")

        content.append("\nSynthetic Data Statistics:")
        content.append(f"Count: {len(synth_clean):.0f}")
        content.append(f"Mean: {np.mean(synth_clean):.4f} ± {np.std(synth_clean):.4f}")
        content.append(f"Range: [{np.min(synth_clean):.4f}, {np.max(synth_clean):.4f}]")

        # Only calculate statistical tests if we have enough data
        if len(real_clean) > 10 and len(synth_clean) > 10:
            try:
                ks_stat, ks_p = ks_2samp(real_clean, synth_clean)
                wass_dist = wasserstein_distance(real_clean, synth_clean)
                content.append("\nStatistical Tests:")
                content.append(f"Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_p:.4f}")
                content.append(f"Wasserstein Distance: {wass_dist:.4f}")
            except Exception as e:
                logger.warning(f"Statistical tests failed for {col}: {str(e)}")
        else:
            content.append("\nWarning: Not enough data for statistical tests")

        # Create optimized visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Analysis of {col}', y=1.05)

        # Optimized distribution plot
        if len(real_clean) > 1000 or len(synth_clean) > 1000:
            # For large datasets, use histograms without KDE
            ax1.hist(real_clean, bins=50, color='blue', alpha=0.5, label='Real', density=True)
            ax1.hist(synth_clean, bins=50, color='orange', alpha=0.5, label='Synthetic', density=True)
        else:
            # For smaller datasets, use KDE
            sns.kdeplot(real_clean, color='blue', label='Real', ax=ax1)
            sns.kdeplot(synth_clean, color='orange', label='Synthetic', ax=ax1)
        ax1.set_title('Distribution Comparison')
        ax1.legend()

        # Optimized QQ plot
        sample_size = min(len(real_clean), len(synth_clean), 500)  # Max 500 points for performance
        real_sample = np.random.choice(real_clean, size=sample_size, replace=False)
        synth_sample = np.random.choice(synth_clean, size=sample_size, replace=False)
        real_q = np.percentile(real_sample, np.linspace(0, 100, sample_size))
        synth_q = np.percentile(synth_sample, np.linspace(0, 100, sample_size))

        ax2.scatter(real_q, synth_q, alpha=0.5)
        min_val = min(real_q.min(), synth_q.min())
        max_val = max(real_q.max(), synth_q.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax2.set_title('QQ Plot')
        ax2.set_xlabel('Real Quantiles')
        ax2.set_ylabel('Synthetic Quantiles')

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, f"{col}_comparison.png"),
                    dpi=100,  # Lower DPI for faster saving
                    bbox_inches='tight')
        content.append(f"\nVisualizations saved as: {col}_comparison.png")

    except Exception as e:
        logger.error(f"Error in numerical analysis for {col}: {str(e)}")
        content.append(f"\nError generating visualizations for {col}")
    finally:
        if fig:
            plt.close(fig)

    return content


def _analyze_categorical(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    content = []
    fig = None

    try:
        # Handle missing values and ensure string type
        real_clean = real_data[col].fillna('missing').astype(str)
        synth_clean = synthetic_data[col].fillna('missing').astype(str)

        # Get all unique categories from both datasets
        all_categories = sorted(list(set(real_clean.unique()).union(set(synth_clean.unique()))))

        # Create normalized value counts
        real_counts = real_clean.value_counts(normalize=True).reindex(all_categories, fill_value=0)
        synth_counts = synth_clean.value_counts(normalize=True).reindex(all_categories, fill_value=0)

        # Add basic statistics
        content.append("\nReal Data Value Proportions (Top 10):")
        for val, prop in real_counts.nlargest(10).items():
            content.append(f"{val}: {prop:.4f}")

        content.append("\nSynthetic Data Value Proportions (Top 10):")
        for val, prop in synth_counts.nlargest(10).items():
            content.append(f"{val}: {prop:.4f}")

        # Only perform statistical tests if we have sufficient data
        if len(all_categories) > 1 and len(real_clean) > 10 and len(synth_clean) > 10:
            try:
                # Create contingency table with small pseudo-count to avoid zeros
                contingency = pd.DataFrame({
                    'Real': real_clean.value_counts().reindex(all_categories, fill_value=0.5),
                    'Synthetic': synth_clean.value_counts().reindex(all_categories, fill_value=0.5)
                }).T.fillna(0.5)

                # Calculate statistical tests
                chi2, chi2_p, _, _ = chi2_contingency(contingency)
                js_div = jensenshannon(real_counts, synth_counts)

                content.append("\nStatistical Tests:")
                content.append(f"Chi-square test: χ²={chi2:.2f}, p={chi2_p:.4f}")
                content.append(f"Jensen-Shannon Divergence: {js_div:.4f}")
            except Exception as e:
                content.append("\nWarning: Statistical tests failed - " + str(e))

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(10, len(all_categories))
        plot_data = pd.DataFrame({
            'Real': real_counts.head(top_n),
            'Synthetic': synth_counts.reindex(real_counts.head(top_n).index, fill_value=0)
        })
        plot_data.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {top_n} Category Proportions: {col}')
        ax.set_ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(os.path.join(report_path, f"{col}_categorical_comparison.png"))
        content.append(f"\nVisualization saved as: {col}_categorical_comparison.png")

    except Exception as e:
        logger.error(f"Error in categorical analysis for {col}: {str(e)}")
        content.append(f"\nError generating visualizations for {col}")
    finally:
        if fig:
            plt.close(fig)

    return content


def _analyze_boolean(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Enhanced boolean column analysis"""
    content = []
    fig = None

    try:
        real_prop = real_data[col].mean()
        synth_prop = synthetic_data[col].mean()
        prop_diff = abs(real_prop - synth_prop)

        # Statistical test
        contingency = pd.DataFrame({
            'Real': [real_data[col].sum(), len(real_data) - real_data[col].sum()],
            'Synthetic': [synthetic_data[col].sum(), len(synthetic_data) - synthetic_data[col].sum()]
        })
        chi2, chi2_p, _, _ = chi2_contingency(contingency)

        content.append("\nReal Data:")
        content.append(f"True proportion: {real_prop:.4f}")
        content.append(f"False proportion: {1 - real_prop:.4f}")

        content.append("\nSynthetic Data:")
        content.append(f"True proportion: {synth_prop:.4f}")
        content.append(f"False proportion: {1 - synth_prop:.4f}")

        content.append("\nStatistical Test:")
        content.append(f"Proportion difference: {prop_diff:.4f}")
        content.append(f"Chi-square test: χ²={chi2:.2f}, p={chi2_p:.4f}")

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 5))

        plot_data = pd.DataFrame({
            'Real': [real_prop, 1 - real_prop],
            'Synthetic': [synth_prop, 1 - synth_prop]
        }, index=['True', 'False'])

        plot_data.plot(kind='bar', ax=ax)
        ax.set_title(f'Boolean Proportions: {col}')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, f"{col}_boolean_comparison.png"))
        content.append(f"\nVisualization saved as: {col}_boolean_comparison.png")

    except Exception as e:
        logger.error(f"Error in boolean analysis for {col}: {str(e)}")
        content.append(f"\nError generating visualization for {col}")
    finally:
        if fig:
            plt.close(fig)

    return content


def _analyze_datetime(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str, datetime_format: str) -> List[str]:
    """Enhanced datetime analysis with temporal patterns"""
    content = []
    fig = None

    try:
        if datetime_format is None and hasattr(real_data[col], 'dt'):
            datetime_format = real_data[col].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[0]

        # Convert to datetime
        real_dates = pd.to_datetime(real_data[col], format=datetime_format, errors='coerce')
        synth_dates = pd.to_datetime(synthetic_data[col], format=datetime_format, errors='coerce')

        # Basic statistics
        real_min, real_max = real_dates.min(), real_dates.max()
        synth_min, synth_max = synth_dates.min(), synth_dates.max()
        real_duration = real_max - real_min
        synth_duration = synth_max - synth_min

        # Wasserstein distance on ordinal dates
        real_ordinal = real_dates.apply(lambda x: x.toordinal())
        synth_ordinal = synth_dates.apply(lambda x: x.toordinal())
        wass_dist = wasserstein_distance(real_ordinal, synth_ordinal)

        # Basic statistics
        content.append("\nReal Data Date Range:")
        if len(real_dates.dropna()) > 0:
            content.append(f"Start: {real_min}")
            content.append(f"End: {real_max}")
            content.append(f"Duration: {real_duration}")
        else:
            content.append("No valid dates found")

        content.append("\nSynthetic Data Date Range:")
        if len(synth_dates.dropna()) > 0:
            content.append(f"Start: {synth_min}")
            content.append(f"End: {synth_max}")
            content.append(f"Duration: {synth_duration}")
        else:
            content.append("No valid dates found")

        content.append("\nStatistical Test:")
        content.append(f"Wasserstein Distance: {wass_dist:.2f}")

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Datetime Analysis: {col}', y=1.02)

        # Temporal distribution
        sns.histplot(real_dates, color='blue', label='Real', kde=True, ax=axes[0, 0])
        sns.histplot(synth_dates, color='orange', label='Synthetic', kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Date Distribution')
        axes[0, 0].legend()

        # Day of week comparison
        real_dow = real_dates.dt.dayofweek.value_counts(normalize=True).sort_index()
        synth_dow = synth_dates.dt.dayofweek.value_counts(normalize=True).sort_index()
        dow_df = pd.DataFrame({'Real': real_dow, 'Synthetic': synth_dow})
        dow_df.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_df.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Day of Week Distribution')

        # Month comparison
        real_month = real_dates.dt.month.value_counts(normalize=True).sort_index()
        synth_month = synth_dates.dt.month.value_counts(normalize=True).sort_index()
        month_df = pd.DataFrame({'Real': real_month, 'Synthetic': synth_month})
        month_df.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Month Distribution')

        # Hour comparison (if datetime has time component)
        if any(real_dates.dt.hour != 0) or any(real_dates.dt.minute != 0):
            real_hour = real_dates.dt.hour.value_counts(normalize=True).sort_index()
            synth_hour = synth_dates.dt.hour.value_counts(normalize=True).sort_index()
            hour_df = pd.DataFrame({'Real': real_hour, 'Synthetic': synth_hour})
            hour_df.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Hour of Day Distribution')
        else:
            # Year comparison if no time component
            real_year = real_dates.dt.year.value_counts(normalize=True).sort_index()
            synth_year = synth_dates.dt.year.value_counts(normalize=True).sort_index()
            year_df = pd.DataFrame({'Real': real_year, 'Synthetic': synth_year})
            year_df.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Year Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, f"{col}_datetime_comparison.png"))
        content.append(f"\nVisualizations saved as: {col}_datetime_comparison.png")

    except Exception as e:
        logger.error(f"Error in datetime analysis for {col}: {str(e)}")
        content.append(f"\nError generating visualizations for {col}")
    finally:
        if fig:
            plt.close(fig)

    return content


def _analyze_generic(col: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, report_path: str) -> List[str]:
    """Analyze generic/unknown column type"""
    content = []

    content.append("\nWarning: Unknown column type - basic analysis only")

    # For strings, show most common values
    if real_data[col].dtype == 'object':
        content.append("\nReal Data Top Values:")
        for val, count in real_data[col].value_counts().head(10).items():
            content.append(f"{val}: {count}")

        content.append("\nSynthetic Data Top Values:")
        for val, count in synthetic_data[col].value_counts().head(10).items():
            content.append(f"{val}: {count}")

    return content


def _analyze_relationships(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: Dict[str, FieldMetadata],
        report_path: str
) -> List[str]:
    """Analyze relationships between variables"""
    content = []

    try:
        # 1. Correlation comparison
        numerical_cols = [col for col, meta in metadata.items()
                          if meta and meta.data_type in [DataType.DECIMAL, DataType.INTEGER]]

        if len(numerical_cols) >= 2:
            content.extend(_compare_correlations(real_data, synthetic_data, numerical_cols, report_path))

        # 2. Pairwise relationships
        if len(numerical_cols) >= 2 and len(real_data) <= 5000:
            content.extend(_compare_pairwise_relationships(real_data, synthetic_data, numerical_cols, report_path))

        # 3. Cross-type relationships (numerical vs categorical)
        categorical_cols = [col for col, meta in metadata.items()
                            if meta and meta.data_type == DataType.CATEGORICAL]

        if numerical_cols and categorical_cols and len(real_data) <= 5000:
            content.extend(_compare_cross_type_relationships(
                real_data, synthetic_data, numerical_cols, categorical_cols, report_path
            ))

    except Exception as e:
        logger.error(f"Error in relationship analysis: {str(e)}")
        content.append("\nError in relationship analysis")

    return content


def _compare_correlations(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        numerical_cols: List[str],
        report_path: str
) -> List[str]:
    """Compare correlation matrices"""
    content = []
    fig = None

    try:
        # Calculate correlations
        real_corr = real_data[numerical_cols].corr()
        synth_corr = synthetic_data[numerical_cols].corr()

        # Correlation difference
        corr_diff = real_corr - synth_corr
        mae = mean_absolute_error(real_corr.values, synth_corr.values)

        content.append("\nCorrelation Analysis:")
        content.append(f"Mean Absolute Error between correlation matrices: {mae:.4f}")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle('Correlation Matrix Comparison', y=1.02)

        # Real data correlations
        sns.heatmap(real_corr, annot=True, fmt=".2f", cmap=DIVERGING_PALETTE,
                    vmin=-1, vmax=1, square=True, ax=axes[0], cbar=False)
        axes[0].set_title('Real Data Correlations')

        # Synthetic data correlations
        sns.heatmap(synth_corr, annot=True, fmt=".2f", cmap=DIVERGING_PALETTE,
                    vmin=-1, vmax=1, square=True, ax=axes[1], cbar=False)
        axes[1].set_title('Synthetic Data Correlations')

        # Difference
        sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap=DIVERGING_PALETTE,
                    center=0, vmin=-1, vmax=1, square=True, ax=axes[2])
        axes[2].set_title('Difference (Real - Synthetic)')

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, "correlation_comparison.png"))
        content.append("\nVisualization saved: correlation_comparison.png")

    except Exception as e:
        logger.error(f"Error in correlation comparison: {str(e)}")
        content.append("\nError generating correlation comparison")
    finally:
        if fig:
            plt.close(fig)

    return content


def _compare_pairwise_relationships(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        numerical_cols: List[str],
        report_path: str
) -> List[str]:
    """Compare pairwise relationships between numerical variables"""
    content = []
    fig = None

    try:
        # Select top 5 numerical columns by variance
        top_cols = real_data[numerical_cols].var().sort_values(ascending=False).head(5).index.tolist()

        if len(top_cols) < 2:
            return content

        # Create pairplot comparison
        fig, axes = plt.subplots(len(top_cols), len(top_cols), figsize=(20, 20))
        fig.suptitle('Pairwise Relationships Comparison', y=1.02)

        for i, col1 in enumerate(top_cols):
            for j, col2 in enumerate(top_cols):
                if i == j:
                    # Diagonal - show histograms
                    sns.histplot(real_data[col1], color='blue', label='Real', ax=axes[i, j], kde=True)
                    sns.histplot(synthetic_data[col1], color='orange', label='Synthetic', ax=axes[i, j], kde=True)
                    axes[i, j].legend()
                else:
                    # Off-diagonal - show scatter plots
                    axes[i, j].scatter(real_data[col1], real_data[col2], color='blue', alpha=0.3, label='Real')
                    axes[i, j].scatter(synthetic_data[col1], synthetic_data[col2], color='orange', alpha=0.3,
                                       label='Synthetic')
                    axes[i, j].legend()

                if i == len(top_cols) - 1:
                    axes[i, j].set_xlabel(col2)
                if j == 0:
                    axes[i, j].set_ylabel(col1)

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, "pairwise_relationships.png"))
        content.append("\nVisualization saved: pairwise_relationships.png")

    except Exception as e:
        logger.error(f"Error in pairwise relationship comparison: {str(e)}")
        content.append("\nError generating pairwise relationships visualization")
    finally:
        if fig:
            plt.close(fig)

    return content


def _compare_cross_type_relationships(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        report_path: str
) -> List[str]:
    """Compare relationships between numerical and categorical variables"""
    content = []
    fig = None

    try:
        # Select top numerical and categorical columns
        top_num = real_data[numerical_cols].var().sort_values(ascending=False).head(3).index.tolist()
        top_cat = [col for col in categorical_cols if len(real_data[col].unique()) <= 10][:3]

        if not top_num or not top_cat:
            return content

        # Create visualization
        fig, axes = plt.subplots(len(top_num), len(top_cat), figsize=(18, 6 * len(top_num)))
        fig.suptitle('Numerical-Categorical Relationships', y=1.02)

        if len(top_num) == 1 or len(top_cat) == 1:
            axes = np.array(axes).reshape(len(top_num), len(top_cat))

        for i, num_col in enumerate(top_num):
            for j, cat_col in enumerate(top_cat):
                # Boxplot comparison
                real_df = real_data[[num_col, cat_col]].copy()
                real_df['source'] = 'Real'
                synth_df = synthetic_data[[num_col, cat_col]].copy()
                synth_df['source'] = 'Synthetic'
                combined = pd.concat([real_df, synth_df])

                sns.boxplot(x=cat_col, y=num_col, hue='source', data=combined,
                            ax=axes[i, j], palette={'Real': 'blue', 'Synthetic': 'orange'})
                axes[i, j].set_title(f'{num_col} by {cat_col}')
                axes[i, j].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, "cross_type_relationships.png"))
        content.append("\nVisualization saved: cross_type_relationships.png")

    except Exception as e:
        logger.error(f"Error in cross-type relationship comparison: {str(e)}")
        content.append("\nError generating cross-type relationships visualization")
    finally:
        if fig:
            plt.close(fig)

    return content


def _visualize_dimensionality_reduction(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        report_path: str
) -> List[str]:
    """Visualize datasets using dimensionality reduction"""
    content = []
    fig = None

    try:
        # Prepare data - only use numerical columns
        numerical_cols = real_data.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) < 2:
            return content

        # Combine and label data - drop NA values first
        real_data_clean = real_data[numerical_cols].dropna()
        synth_data_clean = synthetic_data[numerical_cols].dropna()

        # Check we still have data after dropping NA
        if len(real_data_clean) == 0 or len(synth_data_clean) == 0:
            logger.warning("No valid numerical data after dropping NA values")
            return content

        # Sample if too large
        max_samples = 2000  # Reduced for performance
        if len(real_data_clean) > max_samples:
            real_data_clean = real_data_clean.sample(max_samples, random_state=42)
        if len(synth_data_clean) > max_samples:
            synth_data_clean = synth_data_clean.sample(max_samples, random_state=42)

        # Standardize
        scaler = StandardScaler()
        try:
            real_scaled = scaler.fit_transform(real_data_clean)
            synth_scaled = scaler.transform(synth_data_clean)
        except ValueError as e:
            logger.warning(f"Scaling failed: {str(e)}")
            return content

        # Create combined dataset with labels
        combined = np.vstack([real_scaled, synth_scaled])
        labels = ['Real'] * len(real_scaled) + ['Synthetic'] * len(synth_scaled)

        # 1. PCA Visualization
        pca = PCA(n_components=2)
        try:
            pca_result = pca.fit_transform(combined)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels,
                            palette={'Real': 'blue', 'Synthetic': 'orange'},
                            alpha=0.6, ax=ax)
            ax.set_title('PCA: Real vs Synthetic Data')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            plt.tight_layout()
            plt.savefig(os.path.join(report_path, "pca_comparison.png"))
            plt.close(fig)

            content.append("\nVisualization saved: pca_comparison.png")
        except Exception as e:
            logger.warning(f"PCA failed: {str(e)}")
            if fig:
                plt.close(fig)

    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {str(e)}")
        content.append("\nError generating dimensionality reduction visualizations")
    finally:
        if fig:
            plt.close(fig)

    return content
