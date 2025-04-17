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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

metadata_airline = {
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

metadata_helthcare = {
    'Name': FieldMetadata(DataType.STRING, faker_method=Faker().name),
    'Age': FieldMetadata(DataType.INTEGER),
    'Gender': FieldMetadata(DataType.CATEGORICAL),
    'Blood Type': FieldMetadata(DataType.CATEGORICAL),
    'Medical Condition': FieldMetadata(DataType.CATEGORICAL),
    'Date of Admission': FieldMetadata(DataType.DATETIME, datetime_format='%Y-%m-%d'),
    'Doctor': FieldMetadata(DataType.STRING, faker_method=Faker().name),
    'Hospital': FieldMetadata(DataType.STRING, faker_method=Faker().company),
    'Insurance Provider': FieldMetadata(DataType.CATEGORICAL),
    'Billing Amount': FieldMetadata(DataType.DECIMAL, decimal_places=12),
    'Room Number': FieldMetadata(DataType.INTEGER),
    'Admission Type': FieldMetadata(DataType.CATEGORICAL),
    'Discharge Date': FieldMetadata(DataType.DATETIME, datetime_format='%Y-%m-%d'),
    'Medication': FieldMetadata(DataType.CATEGORICAL),
    'Test Results': FieldMetadata(DataType.CATEGORICAL),
}

metadata_adult = {
    'age': FieldMetadata(DataType.INTEGER),
    'workclass': FieldMetadata(DataType.CATEGORICAL),
    'fnlwgt': FieldMetadata(DataType.INTEGER),
    'education': FieldMetadata(DataType.CATEGORICAL),
    'education.num': FieldMetadata(DataType.INTEGER),
    'marital.status': FieldMetadata(DataType.CATEGORICAL),
    'occupation': FieldMetadata(DataType.CATEGORICAL),
    'relationship': FieldMetadata(DataType.CATEGORICAL),
    'race': FieldMetadata(DataType.CATEGORICAL),
    'sex': FieldMetadata(DataType.CATEGORICAL),
    'capital.gain': FieldMetadata(DataType.INTEGER),
    'capital.loss': FieldMetadata(DataType.INTEGER),
    'hours.per.week': FieldMetadata(DataType.INTEGER),
    'native.country': FieldMetadata(DataType.CATEGORICAL),
    'income': FieldMetadata(DataType.CATEGORICAL)
}


def main():
    try:
        synthetic_size = 40_000

        # define metadata
        metadata = metadata_airline

        datasetLoader = DatasetLoader()

        # load real data
        real_data = datasetLoader.load_real_data(
            data_path='datasets/airline-passenger-satisfaction.csv',
            # data_path='datasets/healthcare_dataset.csv',
            # data_path='datasets/adult.csv',
            metadata=metadata
        )

        generator = WGAN(real=real_data, meta=metadata, cfg=GanConfig())

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
