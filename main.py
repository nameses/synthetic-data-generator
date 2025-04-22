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

        datasetLoader = DatasetLoader()

        # load real data
        real_data = datasetLoader.load_real_data(
            data_path='datasets/airline-passenger-satisfaction.csv',
            metadata=metadata
        )

        # Configure WGAN with optimal parameters for airline dataset
        gan_config = GanConfig(
            max_epochs=500,
            patience=14,
            n_critic_initial=4,
            gp_weight=2.5,
            drift_epsilon=1e-3,
            g_lr=1e-4,
            d_lr=2e-4,
            lr_min_ratio=0.05,
            # Enable advanced features for better numeric column generation
            use_conditional_gan=True,
            advanced_transformers=True,
            post_process_outliers=True,
            enhanced_network_capacity=True,
            generator_depth=4,
            discriminator_depth=3,
            additional_g_capacity=384
        )

        # Initialize and train WGAN
        generator = WGAN(real=real_data, meta=metadata, cfg=gan_config)
        # Train the model and get training metrics
        generator.fit()
        # Generate synthetic data using the best model
        synthetic_data = generator.generate(synthetic_size)
        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        generate_report(real_data, synthetic_data, metadata)


    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
