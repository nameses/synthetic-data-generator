import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, Optional
import logging
from data_generation.dataset_loader import DatasetLoader
from data_generation.vae import VAEPipeline, VAEConfig
from data_generation.gan import GAN,GanConfig

from analytics.report_generator import generate_report
from models.enums import DataType
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

        # datasetLoader = DatasetLoader()

        real_data = pd.read_csv('datasets/airline-passenger-satisfaction.csv', parse_dates=['Date']).dropna()

        # load real data
        # real_data = datasetLoader.load_real_data(
        #     data_path='datasets/airline-passenger-satisfaction.csv',
        #     metadata=metadata
        # )

        # Configure GAN with optimal parameters for airline dataset
        # gan_config = GanConfig(
        #     epochs=10,
        #     patience=320,
        #     n_critic_initial=3,
        #     gp_weight=2.5,
        #     g_lr=1e-4,
        #     d_lr=2e-4,
        # )
        #
        # # Initialize and train GAN
        # generator = GAN(real=real_data, meta=metadata, cfg=gan_config)
        # # Train the model and get training metrics
        # generator.fit(True)
        # # Generate synthetic data using the best model
        # synthetic_data = generator.generate(synthetic_size)
        # logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        vae_cfg = VAEConfig(epochs=100)
        vae_pipe = VAEPipeline(df=real_data, meta=metadata_airline, cfg=vae_cfg)
        vae_pipe.fit(True)
        synthetic_data = vae_pipe.generate(30000)
        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        generate_report(real_data, synthetic_data, metadata)


    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
