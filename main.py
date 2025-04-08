import pandas as pd
import numpy as np
from datetime import datetime
from faker import Faker
from typing import Dict, List, Optional
import torch
import logging
from enum import Enum

from analytics.report_generator import generate_comparison_report
from data_generation.wgan import GANConfig
from models.enums import DataType, MethodType
from models.field_metadata import FieldMetadata

# 5 int, 5 cat - https://www.kaggle.com/datasets/uciml/german-credit/data
# 7 int, 12 cat, 2 bool - https://www.kaggle.com/datasets/elsnkazm/german-credit-scoring-data
# 1 date, 1 time, !!id!! - https://www.kaggle.com/datasets/usgs/earthquake-database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self):
        self.faker = Faker()

    def load_real_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load real dataset from path or use demo data"""
        if data_path:
            logger.info(f"Loading data from {data_path}")
            return pd.read_csv(data_path)

        logger.info("Using demo diamond dataset")
        try:
            import seaborn as sns
            data = sns.load_dataset('diamonds').head(30000)

            # Add synthetic fields for demonstration
            data['purchase_date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(
                np.random.randint(0, 365 * 3, len(data)), unit='d'
            )
            data['customer_email'] = [self.faker.email() for _ in range(len(data))]
            data['is_premium'] = data['price'] > data['price'].quantile(0.8)
            return data

        except ImportError:
            logger.warning("Seaborn not available, using smaller demo dataset")
            titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
            return pd.read_csv(titanic_url)

    def define_metadata(self, data: pd.DataFrame) -> Dict[str, FieldMetadata]:
        """Define metadata with physics constraints"""
        metadata = {
            'person_age': FieldMetadata(
                DataType.INTEGER,
                min_value=18,
                max_value=100
            ),
            'person_income': FieldMetadata(
                DataType.INTEGER,
                min_value=0,
                max_value=1000000
            ),
            'person_home_ownership': FieldMetadata(
                DataType.CATEGORICAL
            ),
            'person_emp_length': FieldMetadata(
                DataType.DECIMAL,
                min_value=0,
                max_value=50  # More realistic max
            ),
            'loan_intent': FieldMetadata(DataType.CATEGORICAL),
            'loan_grade': FieldMetadata(
                DataType.CATEGORICAL
            ),
            'loan_amnt': FieldMetadata(
                DataType.INTEGER,
                min_value=500,
                max_value=35000
            ),
            'loan_int_rate': FieldMetadata(
                DataType.DECIMAL,
                min_value=5.0,
                max_value=30.0
            ),
            'loan_status': FieldMetadata(
                DataType.BOOLEAN
            ),
            'loan_percent_income': FieldMetadata(
                DataType.DECIMAL,
                min_value=0.0,
                max_value=1.0
            ),
            'cb_person_default_on_file': FieldMetadata(
                DataType.CATEGORICAL
            ),
            'cb_person_cred_hist_length': FieldMetadata(
                DataType.INTEGER,
                min_value=0,
                max_value=30
            )
        }

        # Only keep columns that exist in the data
        return {col: meta for col, meta in metadata.items() if col in data.columns}

    def generate_synthetic_data(
            self,
            real_data: pd.DataFrame,
            metadata: Dict[str, FieldMetadata],
            method: MethodType = MethodType.WGAN,
            synthetic_size: Optional[int] = None,
            epochs: int = 10
    ) -> pd.DataFrame:
        """Generate synthetic data using specified method"""
        synthetic_size = synthetic_size or len(real_data) // 5

        logger.info(f"Generating {synthetic_size} synthetic samples using {method.value}")

        if method == MethodType.WGAN:
            from data_generation.wgan import WGAN

            gan_config = GANConfig()

            generator = WGAN(
                real_data=real_data,
                metadata=metadata,
                config=gan_config
            )
            generator.train(epochs=epochs)
            synthetic_data = generator.generate(synthetic_size)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return synthetic_data

    def validate_constraints(self, synthetic_data: pd.DataFrame, metadata: Dict[str, FieldMetadata]):
        """Validate that physics constraints were respected"""
        violations = []

        for col, meta in metadata.items():
            if col not in synthetic_data.columns:
                continue

            # Check range constraints
            if meta.data_type in [DataType.INTEGER, DataType.DECIMAL]:
                if meta.min_value is not None and (synthetic_data[col] < meta.min_value).any():
                    violations.append(f"{col} violates min_value {meta.min_value}")
                if meta.max_value is not None and (synthetic_data[col] > meta.max_value).any():
                    violations.append(f"{col} violates max_value {meta.max_value}")

            # Check relational constraints
            for constraint in getattr(meta, 'constraints', []):
                other_col = constraint.get('other_column')
                if other_col not in synthetic_data.columns:
                    continue

                if constraint['type'] == 'greater_than':
                    margin = constraint.get('margin', 0)
                    if (synthetic_data[col] - synthetic_data[other_col] + margin < 0).any():
                        violations.append(
                            f"{col} not always > {other_col} (margin={margin})"
                        )

                elif constraint['type'] == 'less_than':
                    margin = constraint.get('margin', 0)
                    if (synthetic_data[other_col] - synthetic_data[col] + margin < 0).any():
                        violations.append(
                            f"{col} not always < {other_col} (margin={margin})"
                        )

                elif constraint['type'] == 'positive_correlation':
                    min_ratio = constraint.get('min_ratio', 0)
                    if (synthetic_data[col] * min_ratio > synthetic_data[other_col]).any():
                        violations.append(
                            f"{col} not maintaining min ratio {min_ratio} with {other_col}"
                        )

        if violations:
            logger.warning(f"Constraint violations detected:\n  - " + "\n  - ".join(violations))
        else:
            logger.info("All physics constraints validated successfully")

        return not bool(violations)


def main():
    """Main execution pipeline"""
    try:
        generator = DataGenerator()

        # 1. Load real data
        real_data = generator.load_real_data('datasets/credit_risk_dataset.csv')
        real_data = real_data.dropna()

        logger.info(f"Loaded real data with shape: {real_data.shape}")

        # 2. Define metadata with constraints
        metadata = generator.define_metadata(real_data)
        logger.info(f"Defined metadata for {len(metadata)} columns")

        # 3. Generate synthetic data
        synthetic_data = generator.generate_synthetic_data(
            real_data=real_data,
            metadata=metadata,
            method=MethodType.WGAN
        )
        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        # 4. Validate constraints
        generator.validate_constraints(synthetic_data, metadata)

        # 5. Generate comparison report
        report_path = generate_comparison_report(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata
        )
        logger.info(f"Generated comparison report at: {report_path}")

    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
