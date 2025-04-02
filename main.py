import pandas as pd
import numpy as np
from datetime import datetime
from faker import Faker
from typing import Dict

from analytics.report_generator import generate_comparison_report
from data_generation.api import generate_synthetic_data
from models.enums import DataType, MethodType
from models.field_metadata import FieldMetadata

fake = Faker()


def load_real_datasets():
    """Load larger real datasets from public sources"""
    try:
        # Example with larger dataset (adjust as needed)
        import seaborn as sns
        data = sns.load_dataset('diamonds').head(40000)

        # Add some synthetic fields for demonstration
        fake = Faker()
        data['purchase_date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(
            np.random.randint(0, 365 * 3, len(data)), unit='d'
        )
        data['customer_email'] = [fake.email() for _ in range(len(data))]
        data['is_premium'] = data['price'] > data['price'].quantile(0.8)

        return data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to titanic dataset
        titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        return pd.read_csv(titanic_url)

# Define metadata with configurable string generation
def get_metadata() -> Dict[str, FieldMetadata]:
    """Example for diamonds dataset - adjust according to your actual dataset"""
    return {
        'carat': FieldMetadata(DataType.DECIMAL, min_value=0.2, max_value=5.01),
        'cut': FieldMetadata(DataType.CATEGORICAL),
        'color': FieldMetadata(DataType.CATEGORICAL),
        'clarity': FieldMetadata(DataType.CATEGORICAL),
        'depth': FieldMetadata(DataType.DECIMAL, min_value=43, max_value=79),
        'table': FieldMetadata(DataType.DECIMAL, min_value=43, max_value=95),
        'price': FieldMetadata(DataType.INTEGER),
        'x': FieldMetadata(DataType.DECIMAL),
        'y': FieldMetadata(DataType.DECIMAL),
        'z': FieldMetadata(DataType.DECIMAL),
        'purchase_date': FieldMetadata(DataType.DATE_TIME),
        'customer_email': FieldMetadata(DataType.STRING, fake_strategy='email'),
        'is_premium': FieldMetadata(DataType.BOOLEAN)
    }


# Update the main execution part
if __name__ == "__main__":
    # Load real data
    real_data = load_real_datasets()
    print("Loaded real data with shape:", real_data.shape)

    metadata = get_metadata()

    # Generate synthetic data at 50% size of original
    synthetic_size = int(len(real_data) * 0.2)

    print(f"Generating synthetic data at 5% size: {synthetic_size} samples")

    synthetic_data = generate_synthetic_data(
        real_data,
        metadata,
        MethodType.WGAN,
        synthetic_size=synthetic_size
    )

    # Generate comparison report
    generate_comparison_report(
        real_data,
        synthetic_data,
        metadata
    )