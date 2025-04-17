import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, Optional
import logging

from data_generation.dataset_loader import DatasetLoader
from data_generation.wgan import WGAN

from analytics.report_generator import generate_report
from data_generation.wgan import GanConfig
from models.enums import DataType, MethodType
from models.field_metadata import FieldMetadata

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # from pathlib import Path
    # import pandas as pd
    #
    # # Tiny dummy dataset for demonstration only.
    # data = pd.DataFrame({
    #     "num": np.random.randn(2000) * 10 + 50,
    #     "cat": np.random.choice(["A", "B", "C"], 2000),
    #     "flag": np.random.choice([True, False], 2000),
    #     "when": pd.date_range("2020-01-01", periods=2000, freq="H"),
    #     "name": ["stub"] * 2000,
    # })
    #
    # meta = {
    #     "num": FieldMetadata(DataType.DECIMAL, decimal_places=2),
    #     "cat": FieldMetadata(DataType.CATEGORICAL),
    #     "flag": FieldMetadata(DataType.BOOLEAN),
    #     "when": FieldMetadata(DataType.DATETIME, datetime_format="%Y-%m-%d %H:%M:%S"),
    #     "name": FieldMetadata(DataType.STRING, faker_method=Faker().name),
    # }
    #
    # gan = WGAN(data, meta, GanConfig())
    # gan.fit()
    # synth = gan.generate(10)
    # print(synth.head())
    try:
        synthetic_size = 10_000

        # define metadata
        metadata = {
            'Gender': FieldMetadata(DataType.CATEGORICAL),
            'Customer Type': FieldMetadata(DataType.CATEGORICAL),
            'Age': FieldMetadata(DataType.INTEGER),
            'Type of Travel': FieldMetadata(DataType.CATEGORICAL),
            'Class': FieldMetadata(DataType.CATEGORICAL),
            'Flight Distance': FieldMetadata(DataType.INTEGER),
            'Inflight wifi service': FieldMetadata(DataType.INTEGER),
            'Departure or Arrival time convenient': FieldMetadata(DataType.INTEGER),
            'Ease of Online booking': FieldMetadata(DataType.INTEGER),
            'Gate location': FieldMetadata(DataType.INTEGER),
            'Food and drink': FieldMetadata(DataType.INTEGER),
            'Online boarding': FieldMetadata(DataType.INTEGER),
            'Seat comfort': FieldMetadata(DataType.INTEGER),
            'Inflight entertainment': FieldMetadata(DataType.INTEGER),
            'On-board service': FieldMetadata(DataType.INTEGER),
            'Leg room service': FieldMetadata(DataType.INTEGER),
            'Baggage handling': FieldMetadata(DataType.INTEGER),
            'Checkin service': FieldMetadata(DataType.INTEGER),
            'Inflight service': FieldMetadata(DataType.INTEGER),
            'Cleanliness': FieldMetadata(DataType.INTEGER),
            'Departure Delay in Minutes': FieldMetadata(DataType.INTEGER),
            'Arrival Delay in Minutes': FieldMetadata(DataType.DECIMAL, decimal_places=1),
            'satisfaction': FieldMetadata(DataType.CATEGORICAL)
        }
        logger.info(f"Defined metadata for {len(metadata)} columns")

        datasetLoader = DatasetLoader()

        # load real data
        real_data = datasetLoader.load_real_data(
            data_path='datasets/airline-passenger-satisfaction.csv',
            metadata=metadata
        )

        logger.info(f"Loaded real data with shape: {real_data.shape}")

        logger.info(f"Generating {synthetic_size} synthetic samples")

        gan_config = GanConfig()
        generator = WGAN(
            real=real_data,
            meta=metadata,
            cfg=gan_config
        )

        generator.fit()
        synthetic_data = generator.generate(synthetic_size)
        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        # generate comparison
        report_path = generate_report(real_data, synthetic_data, metadata)
        logger.info(f"Generated comparison report at: {report_path}")

    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
