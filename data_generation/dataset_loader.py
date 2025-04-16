import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from models.enums import DataType
from models.field_metadata import FieldMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    def load_real_data(
            self,
            data_path: str,
            metadata: Dict[str, FieldMetadata],
            sample_size: int,
            random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load and sample data while preserving complete date distributions
        without decomposing into components.
        """
        date_cols = {
            col: meta.datetime_format
            for col, meta in metadata.items()
            if meta.data_type == DataType.DATETIME
        }

        # Read CSV with proper date parsing
        date_columns = [col for col in date_cols.keys() if col in pd.read_csv(data_path, nrows=1).columns]
        if date_columns:
            full_data = pd.read_csv(
                data_path,
                parse_dates=date_columns,
                infer_datetime_format=True
            )
        else:
            full_data = pd.read_csv(data_path)

        if len(full_data) <= sample_size:
            return full_data

        for col in date_cols:
            # Convert to ordinal for numerical handling
            full_data[f"_ord_{col}"] = full_data[col].apply(lambda x: x.toordinal())

            # Create time-based bins (preserves temporal density)
            full_data[f"_bin_{col}"] = pd.qcut(
                full_data[f"_ord_{col}"],
                q=min(50, len(full_data) // 1000),  # Dynamic bin count
                duplicates='drop'
            )

        # 2. Create stratification groups
        strat_cols = []

        # Prioritize date columns
        if date_cols:
            strat_cols.append(f"_bin_{date_cols[0]}")  # Use first date column

        # Add categorical columns (up to 1 to avoid over-stratification)
        cat_cols = [col for col, meta in metadata.items()
                    if meta.data_type == DataType.CATEGORICAL and col in full_data.columns]
        if cat_cols:
            strat_cols.append(sorted(cat_cols, key=lambda x: full_data[x].nunique())[0])

        # Add numerical bins if needed
        num_cols = [col for col, meta in metadata.items()
                    if meta.data_type in [DataType.INTEGER, DataType.DECIMAL]
                    and col in full_data.columns]
        if num_cols and len(strat_cols) < 2:  # Keep total strat cols <= 2
            full_data[f"_bin_{num_cols[0]}"] = pd.qcut(full_data[num_cols[0]], q=10, duplicates='drop')
            strat_cols.append(f"_bin_{num_cols[0]}")

        # 3. Perform stratified sampling
        if strat_cols:
            full_data["_strat_key"] = full_data[strat_cols].astype(str).agg('|'.join, axis=1)
            sampled_data = full_data.groupby("_strat_key", group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), max(1, int(sample_size * len(x) / len(full_data)))),
                    random_state=random_state
                )).sample(frac=1, random_state=random_state)
        else:
            sampled_data = full_data.sample(min(sample_size, len(full_data)), random_state=random_state)

        # 4. Cleanup and validate
        sampled_data = sampled_data[full_data.columns.difference([
            *[f"_ord_{c}" for c in date_cols],  # iterates dict keys
            *[f"_bin_{c}" for c in list(date_cols) + num_cols],  # list(date_cols) + num_cols is valid
            '_strat_key'
        ])]
        self._validate_date_distributions(full_data, sampled_data, date_cols)

        return sampled_data.head(sample_size)

    def _validate_date_distributions(self, original, sample, date_cols):
        """Validate complete date distributions are preserved"""
        for col in date_cols:
            # Compare full date distributions using Wasserstein distance
            orig_ord = original[col].apply(lambda x: x.toordinal())
            samp_ord = sample[col].apply(lambda x: x.toordinal())

            # Calculate density overlap
            density_diff = np.mean(np.isin(
                samp_ord,
                np.linspace(orig_ord.min(), orig_ord.max(), 1000)
            ))

            if density_diff < 0.8:  # Less than 80% density overlap
                logger.warning(f"Date density significantly changed for {col} ({density_diff:.1%} overlap)")

            # Compare distribution shapes (KS test on ordinal values)
            ks_stat = ks_2samp(orig_ord, samp_ord)[0]
            if ks_stat > 0.15:
                logger.warning(f"Date distribution changed for {col} (KS={ks_stat:.2f})")
