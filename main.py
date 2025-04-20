import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, Optional
import logging

from data_generation.dataset_loader import DatasetLoader
from data_generation.vae import VAEPipeline, VAEConfig
from data_generation.wgan import WGAN

from analytics.report_generator import generate_report
from data_generation.wgan import GanConfig
from models.enums import DataType, MethodType
from models.field_metadata import FieldMetadata
from models.metadata import metadata_airline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    try:
        synthetic_size = 50_000

        # define metadata
        metadata = metadata_airline

        datasetLoader = DatasetLoader()

        # load real data
        real_data = datasetLoader.load_real_data(
            data_path='datasets/airline-passenger-satisfaction.csv',
            # data_path='datasets/healthcare_dataset.csv',
            # data_path='datasets/adult.csv',
            metadata=metadata
        )  # .head(40_000)

        generator = WGAN(real=real_data, meta=metadata, cfg=GanConfig())
        generator.fit(True)
        synthetic_data = generator.generate(synthetic_size)

        # vae = VAEPipeline(df=real_data, meta=metadata, cfg=VAEConfig())
        # vae.train()
        # synthetic_data = vae.generate(synthetic_size)

        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        # generate comparison
        report_path = generate_report(real_data, synthetic_data, metadata)
        logger.info(f"Generated comparison report at: {report_path}")

        # synthetic_data = pd.read_csv('reports/reports_20250418/report_20250418_040050/synthetic.csv')

        # # generate comparison
        # report_path = generate_report(real_data, synthetic_data, metadata)
        # logger.info(f"Generated comparison report at: {report_path}")

    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
