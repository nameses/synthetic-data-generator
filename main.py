import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp, entropy
from enum import Enum
from typing import Dict, Any
from faker import Faker

from data_generation.api import generate_synthetic_data
from models.enums import DataType, MethodType
from models.field_metadata import FieldMetadata

fake = Faker()

def mix_real_synthetic(real: pd.DataFrame, synthetic: pd.DataFrame, mix_ratio=0.5) -> pd.DataFrame:
    mix_size = int(len(real) * mix_ratio)
    mixed_data = pd.concat([real.sample(mix_size), synthetic.sample(mix_size)], ignore_index=True)
    return mixed_data


def compare_datasets(real: pd.DataFrame, synthetic: pd.DataFrame):
    stats = {}
    for column in real.columns:
        if real[column].dtype in [np.int64, np.float64]:
            stats[column] = {
                "KS_test": ks_2samp(real[column], synthetic[column]).statistic,
                "Mean_diff": abs(real[column].mean() - synthetic[column].mean()),
                "Std_diff": abs(real[column].std() - synthetic[column].std()),
            }
        elif real[column].dtype == object:
            stats[column] = {
                "JSD": entropy(pd.value_counts(real[column], normalize=True),
                               pd.value_counts(synthetic[column], normalize=True))
            }
    return stats

 
def visualize_data(real: pd.DataFrame, synthetic: pd.DataFrame):
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real.select_dtypes(include=[np.number]))
    synth_pca = pca.transform(synthetic.select_dtypes(include=[np.number]))

    plt.scatter(real_pca[:, 0], real_pca[:, 1], label='Real', alpha=0.5)
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], label='Synthetic', alpha=0.5)
    plt.legend()
    plt.title("PCA Projection of Real vs Synthetic Data")
    plt.show()


metadataExample = {
    "age": FieldMetadata(DataType.INTEGER),
    "salary": FieldMetadata(DataType.DECIMAL),
    "gender": FieldMetadata(DataType.CATEGORICAL),
    "name": FieldMetadata(DataType.STRING, faker_func=fake.name)
}
size = 10000
real_data = pd.DataFrame({
    "age": np.random.randint(18, 60, size),
    "salary": np.random.uniform(30000, 100000, size),
    "gender": np.random.choice(["Male", "Female"], size),
    "name": [fake.name() for _ in range(10000)]
})
synthetic_data = generate_synthetic_data(real_data, metadataExample, MethodType.GAN, size)
print(real_data)
print(synthetic_data)
#mixed_data = mix_real_synthetic(real_data, synthetic_data)
#stats = compare_datasets(real_data, synthetic_data)
visualize_data(real_data, synthetic_data)
