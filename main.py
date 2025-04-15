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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self):
        self.faker = Faker()
        self.datasetLoader = DatasetLoader()

    def define_metadata(self) -> Dict[str, FieldMetadata]:
        # title,magnitude,date_time,cdi,mmi,alert,tsunami,sig,net,nst,dmin,gap,magType,depth,latitude,longitude,location,continent,country
        metadata = {
            'app_name': FieldMetadata(DataType.STRING, faker_method=self.faker.company),
            'magnitude': FieldMetadata(DataType.DECIMAL, min_value=6.5, max_value=9.1, decimal_places=1),
            'date_time': FieldMetadata(DataType.DATETIME, datetime_format='%d-%m-%Y %H:%M',
                                       datetime_min='01-01-1995 00:00',
                                       datetime_max='31-12-2023 23:59'),
            'cdi': FieldMetadata(DataType.INTEGER, min_value=0, max_value=9),
            'mmi': FieldMetadata(DataType.INTEGER, min_value=1, max_value=10),
            'alert': FieldMetadata(DataType.CATEGORICAL, allow_nans=True, nan_probability=0.2),
            'tsunami': FieldMetadata(DataType.BOOLEAN),
            'sig': FieldMetadata(DataType.INTEGER, min_value=650, max_value=2910),
            'net': FieldMetadata(DataType.CATEGORICAL),
            'nst': FieldMetadata(DataType.INTEGER, min_value=0, max_value=934),
            'dmin': FieldMetadata(DataType.DECIMAL, min_value=0.0, max_value=17.7, decimal_places=3),
            'gap': FieldMetadata(DataType.INTEGER, min_value=0, max_value=239),
            'magType': FieldMetadata(DataType.CATEGORICAL),
            'depth': FieldMetadata(DataType.DECIMAL, min_value=2.7, max_value=671.0, decimal_places=3),
            'latitude': FieldMetadata(DataType.DECIMAL, min_value=-61.8, max_value=71.6, decimal_places=4),
            'longitude': FieldMetadata(DataType.DECIMAL, min_value=-180.0, max_value=180.0, decimal_places=3),
            'location': FieldMetadata(DataType.STRING, faker_method=self.faker.city),
            'continent': FieldMetadata(DataType.CATEGORICAL, allow_nans=True, nan_probability=0.2),
            'country': FieldMetadata(DataType.STRING, faker_method=self.faker.country, allow_nans=True, nan_probability=0.2)
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
        REAL_SIZE = 1_000
        SYNTHETIC_SIZE = 1_000

        generator = DataGenerator()

        # define metadata
        metadata = generator.define_metadata()
        logger.info(f"Defined metadata for {len(metadata)} columns")

        # load real data
        real_data = generator.load_real_data(
            data_path='datasets/earthquake_1995-2023.csv',
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
