import logging

import pandas as pd

from analytics.report_generator import generate_report
from models.metadata import metadata_airline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

df_original = pd.read_csv('../datasets/airline-passenger-satisfaction.csv').dropna()
df_synthethic = pd.read_csv('synthetic.csv')

print(generate_report(df_original, df_synthethic, metadata_airline))