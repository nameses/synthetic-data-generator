# Synthetic Data Generation Toolkit

This project provides end-to-end pipelines for synthesizing tabular data using two deep learning approaches:

- **GAN-based Generator** (Wasserstein GAN with Gradient Penalty, feature-matching, covariance loss, adaptive critic schedule)
- **VAE-based Generator** (β-VAE with reconstruction, KL, SWD, correlation, and categorical-frequency losses)

Designed for practitioners in strategic decision-making, risk analysis, and data augmentation, this toolkit supports mixed data types (numeric, categorical, datetime) and produces high-fidelity synthetic samples.

---

## Features

- **Modular Pipelines**: `gan.pipeline.GAN` and `vae.pipeline.VAE` classes with configurable architectures and losses  
- **Data Schema & Transformers**: Automatic column classification and preprocessing via `models.dataclasses.DataSchema` and custom transformers (`_ContTf`, `DtTf`)  
- **Advanced Losses**:  
  - WGAN-GP with spectral normalization and minibatch standard deviation  
  - β-VAE with sliced-Wasserstein, correlation, and category-frequency regularizers  
- **Evaluator Utilities**: Logistic discriminator, statistical metrics (Wasserstein, KS, KL, JSD, Cramér’s V)  
- **Reproducibility**: Global RNG seeding, mixed-precision support, deterministic tests  

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-org/synthetic-data-generation.git
   cd synthetic-data-generation
   ```

2. Create a virtual environment and install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
   ```

# Quickstart

### GAN Pipeline usage example

```python
from gan.pipeline import GAN
from gan.dataclasses.training import GanConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType
import pandas as pd

# Load data and metadata
df = pd.read_csv("data/real_data.csv")
meta = {col: FieldMetadata(data_type=DataType.DECIMAL, decimal_places=2)
        for col in df.columns}

# Configure and train
gan = GAN(df, meta, GanConfig(epochs=100, batch_size=64))
gan.fit(verbose=True)

# Generate synthetic samples
syn_df = gan.generate(n_samples=1000)
```

### VAE Pipeline usage example

```python
from vae.pipeline import VAE
from vae.dataclasses.training import VaeConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType
import pandas as pd

# Load data and metadata
df = pd.read_csv("data/real_data.csv")
meta = {col: FieldMetadata(data_type=DataType.DECIMAL, decimal_places=2)
        for col in df.columns}

# Configure and train
vae = VAE(df, meta, VaeConfig(epochs=100, batch_size=64))
vae.fit(verbose=True)

# Generate synthetic samples
syn_df = vae.generate(n_samples=1000)
```

### Project Structure

```plaintext
├── common/
│ ├── dataclasses.py # DataSchema & metadata parser
│ ├── transformers.py # Continuous & datetime transformers
│ └── utilities.py # Global RNG seeding
├── gan/
│ ├── pipeline.py # GAN training pipeline
│ ├── utilities.py # lin_sn, CriticScheduler, AMP helpers
│ └── dataclasses/ # GAN config classes
├── vae/
│ ├── pipeline.py # VAE training pipeline
│ └── dataclasses/ # VAE config classes
├── models/
│ ├── enums.py # DataType definitions
│ └── field_metadata.py # FieldMetadata for each column
├── tests/
│ ├── test_common.py
│ ├── test_gan.py
│ └── test_vae.py
├── notebooks/ # Analysis & visualization examples
├── requirements.txt
└── README.md
```
