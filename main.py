import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, Optional
import logging
from data_generation.vae import VAE
from data_generation.gan import GAN,GanConfig

from analytics.report_generator import generate_report
from models.enums import DataType
from models.field_metadata import FieldMetadata
from models.metadata import metadata_airline, metadata_power_consumption

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    try:
        # synthetic_size = 50_000

        # define metadata
        metadata = metadata_power_consumption

        # real_data = pd.read_csv('datasets/airline-passenger-satisfaction.csv', parse_dates=['Date']).dropna()

        # Loading dataset
        raw = pd.read_csv('datasets/household_power_consumption.csv',sep=';',dayfirst=True,na_values='?')
        # Parse datetime explicitly to avoid FutureWarning
        raw['dt'] = pd.to_datetime(
            raw['Date'].str.strip() + ' ' + raw['Time'].str.strip(),
            format='%d/%m/%Y %H:%M:%S',
            dayfirst=True,
            errors='coerce'
        )
        # Drop rows where parsing failed
        df_power = raw.dropna(subset=['dt']).drop(columns=['Date', 'Time'])

        # Aggregate to hourly sums and drop any remaining NaNs
        df_hourly = df_power.dropna().set_index('dt').resample('h').sum()
        # Extract one full year (2008)
        real_data = df_hourly['2006-12-16':'2009-12-31']
        real_data.to_csv('hourly_household_power_consumption.csv',index=False)
        print("Loaded hourly data shape:", real_data.shape)
        synthetic_size = len(real_data) // 2
        # Configure GAN with optimal parameters for airline dataset
        # gan_config = GanConfig(
        #     epochs=320,
        #     patience=250
        # )
        #
        # # Initialize and train GAN
        # generator = GAN(real=real_data, meta=metadata, cfg=gan_config)
        # # Train the model and get training metrics
        # generator.fit(True)
        # # Generate synthetic data using the best model
        # synthetic_data = generator.generate(synthetic_size)
        # logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        vae = VAE(real_data, metadata)
        vae.fit(epochs=320, verbose=True)

        synthetic_data = vae.generate(synthetic_size)
        logger.info(f"Generated synthetic data with shape: {synthetic_data.shape}")

        generate_report(real_data, synthetic_data, metadata)


    except Exception as e:
        logger.error(f"Error in data generation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
