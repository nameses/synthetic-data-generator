import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, Optional
import logging

from data_generation.dataset_loader import DatasetLoader
from data_generation.wgan import WGAN

from analytics.report_generator import generate_comparison_report
from data_generation.wgan import GANConfig
from models.enums import DataType, MethodType
from models.field_metadata import FieldMetadata

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self):
        self.faker = Faker()
        self.datasetLoader = DatasetLoader()

    def define_metadata(self) -> Dict[str, FieldMetadata]:
        metadata = {
            'Gender': FieldMetadata(DataType.CATEGORICAL),
            'Customer Type': FieldMetadata(DataType.CATEGORICAL),
            'Age': FieldMetadata(DataType.INTEGER, min_value=7, max_value=85),
            'Type of Travel': FieldMetadata(DataType.CATEGORICAL),
            'Class': FieldMetadata(DataType.CATEGORICAL),
            'Flight Distance': FieldMetadata(DataType.INTEGER, min_value=31, max_value=4983),
            'Inflight wifi service': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Departure or Arrival time convenient': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Ease of Online booking': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Gate location': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Food and drink': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Online boarding': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Seat comfort': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Inflight entertainment': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'On-board service': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Leg room service': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Baggage handling': FieldMetadata(DataType.INTEGER, min_value=1, max_value=5),
            'Checkin service': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Inflight service': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Cleanliness': FieldMetadata(DataType.INTEGER, min_value=0, max_value=5),
            'Departure Delay in Minutes': FieldMetadata(DataType.INTEGER, min_value=0, max_value=1592),
            'Arrival Delay in Minutes': FieldMetadata(DataType.DECIMAL, min_value=0.0, max_value=1584.0, decimal_places=1),
            'satisfaction': FieldMetadata(DataType.CATEGORICAL)
        }

        return {col: meta for col, meta in metadata.items()}

    def load_real_data(self, data_path: str, metadata: Dict[str, FieldMetadata], sample_size: int = 100_000) -> pd.DataFrame:
        return self.datasetLoader.load_real_data(data_path, metadata, sample_size)

    def generate_synthetic_data(
            self,
            real_data: pd.DataFrame,
            metadata: Dict[str, FieldMetadata],
            method: MethodType,
            synthetic_size: Optional[int],
            epochs: int
    ) -> pd.DataFrame:
        synthetic_size = synthetic_size or len(real_data)

        logger.info(f"Generating {synthetic_size} synthetic samples using {method.value}")

        gan_config = GANConfig()
        generator = WGAN(
            real_data=real_data,
            metadata=metadata,
            config=gan_config
        )
        generator.train_loop(epochs=epochs)
        synthetic_data = generator.generate(synthetic_size)

        return synthetic_data


def main():
    try:
        EPOCHS = 100
        REAL_SIZE = 50_000
        SYNTHETIC_SIZE = 10_000

        generator = DataGenerator()

        # define metadata
        metadata = generator.define_metadata()
        logger.info(f"Defined metadata for {len(metadata)} columns")

        # load real data
        real_data = generator.load_real_data(
            data_path='datasets/airline-passenger-satisfaction.csv',
            metadata=metadata,
            sample_size=REAL_SIZE
        )

        logger.info(f"Loaded real data with shape: {real_data.shape}")

        # generate synthetic data
        synthetic_data = generator.generate_synthetic_data(
            real_data=real_data,
            metadata=metadata,
            method=MethodType.WGAN,
            synthetic_size=SYNTHETIC_SIZE,
            epochs=EPOCHS
        )
        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        # generate comparison
        report_path = generate_comparison_report(real_data, synthetic_data, metadata, sample_size=SYNTHETIC_SIZE)
        logger.info(f"Generated comparison report at: {report_path}")

    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
