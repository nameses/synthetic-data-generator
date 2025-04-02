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
            data = sns.load_dataset('diamonds').head(40000)

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
            'carat': FieldMetadata(
                DataType.DECIMAL,
                min_value=0.2,
                max_value=5.01,
                constraints=[
                    {'type': 'positive_correlation', 'other_column': 'price', 'min_ratio': 1000}
                ]
            ),
            'cut': FieldMetadata(DataType.CATEGORICAL),
            'color': FieldMetadata(DataType.CATEGORICAL),
            'clarity': FieldMetadata(DataType.CATEGORICAL),
            'depth': FieldMetadata(
                DataType.DECIMAL,
                min_value=43,
                max_value=79
            ),
            'table': FieldMetadata(
                DataType.DECIMAL,
                min_value=43,
                max_value=95
            ),
            'price': FieldMetadata(DataType.INTEGER),
            'x': FieldMetadata(
                DataType.DECIMAL,
                constraints=[{'type': 'greater_than', 'other_column': 'y'}]
            ),
            'y': FieldMetadata(
                DataType.DECIMAL,
                constraints=[{'type': 'greater_than', 'other_column': 'z'}]
            ),
            'z': FieldMetadata(DataType.DECIMAL),
            'purchase_date': FieldMetadata(DataType.DATE_TIME),
            'customer_email': FieldMetadata(
                DataType.STRING,
                fake_strategy='email'
            ),
            'is_premium': FieldMetadata(DataType.BOOLEAN)
        }

        # Only keep columns that exist in the data
        return {col: meta for col, meta in metadata.items() if col in data.columns}

    def generate_synthetic_data(
            self,
            real_data: pd.DataFrame,
            metadata: Dict[str, FieldMetadata],
            method: MethodType = MethodType.WGAN,
            synthetic_size: Optional[int] = None,
            epochs: int = 100
    ) -> pd.DataFrame:
        """Generate synthetic data using specified method"""
        synthetic_size = synthetic_size or len(real_data) // 5  # Default 20% size

        logger.info(f"Generating {synthetic_size} synthetic samples using {method.value}")

        if method == MethodType.WGAN:
            from data_generation.wgan import PhysicsInformedWGAN

            gan_config = GANConfig(
                latent_dim=512,
                batch_size=256,
                n_critic=5,
                gp_weight=10.0,
                phys_weight=1.0,
                g_lr=1e-4,
                d_lr=1e-4,
                g_betas=(0.5, 0.9),
                d_betas=(0.5, 0.9),
                patience=25,
                clip_value=0.5,
                use_mixed_precision=True,
                spectral_norm=True,
                residual_blocks=True
            )

            generator = PhysicsInformedWGAN(
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
        real_data = generator.load_real_data()  # Or pass path to your data
        logger.info(f"Loaded real data with shape: {real_data.shape}")

        # 2. Define metadata with constraints
        metadata = generator.define_metadata(real_data)
        logger.info(f"Defined metadata for {len(metadata)} columns")

        # 3. Generate synthetic data
        synthetic_data = generator.generate_synthetic_data(
            real_data=real_data,
            metadata=metadata,
            method=MethodType.WGAN,
            synthetic_size=10000,
            epochs=100
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

        # 6. Save synthetic data
        output_path = "synthetic_data.csv"
        synthetic_data.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to: {output_path}")

    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
