import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_generation.wgan import WGAN, GanConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType

# ——— 1. Build hourly series & sliding windows
df = (
    pd.read_csv("household_power_consumption.txt", sep=';',
                parse_dates={'dt': ['Date', 'Time']},
                infer_datetime_format=True, na_values='?')
    .dropna(subset=['Global_active_power'])
    .set_index('dt')
    .resample('H').sum()
    .reset_index(drop=True)
)
vals = df['Global_active_power'].values
W_size = 24
windows = np.stack([vals[i:i + W_size] for i in range(len(vals) - W_size + 1)])

# ——— 2. Label “peak” vs “normal”
thr = np.percentile(windows.max(axis=1), 90)
y_real = (windows.max(axis=1) > thr).astype(int)

# ——— 3. Make a DataFrame of 24 features t0…t23
cols = [f"t{i}" for i in range(W_size)]
W_df = pd.DataFrame(windows, columns=cols)

# ——— 4. Metadata for each lag
meta = {
    c: FieldMetadata(data_type=DataType.DECIMAL, decimal_places=3)
    for c in cols
}

# ——— 5. Train GAN on ALL windows
gan = WGAN(real=W_df, meta=meta, cfg=GanConfig(max_epochs=150))
gan.fit()

# ——— 6. Generate synthetic windows to balance classes
n0, n1 = (y_real == 0).sum(), (y_real == 1).sum()
k = abs(n0 - n1)
S = gan.generate(k)  # returns DataFrame with same 24 columns

# assign synthetic labels:
y_syn = np.zeros(k, int) if n0 < n1 else np.ones(k, int)

# ——— 7. Combine & train a RandomForest
X = np.vstack([W_df.values, S.values])
y = np.hstack([y_real, y_syn])
from sklearn.model_selection import train_test_split

Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(Xtr, ytr)
print("RF acc:", clf.score(Xte, yte))
