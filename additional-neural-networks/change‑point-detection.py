import pandas as pd
import ruptures as rpt
from data_generation.wgan import WGAN, GanConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType

# ——— 1. Hourly series over one calendar year
df = (
    pd.read_csv("household_power_consumption.txt", sep=';',
                parse_dates={'dt': ['Date', 'Time']},
                infer_datetime_format=True, na_values='?')
    .dropna(subset=['Global_active_power'])
    .set_index('dt')
    .resample('H').sum()
)
year = df["2008-01-01":"2008-12-31"].reset_index(drop=True)

# ——— 2. Detect real change‑points
algo = rpt.Pelt(model="rbf").fit(year.Global_active_power.values)
bkps_real = algo.predict(pen=10)
print("Real breakpoints:", bkps_real)

# ——— 3. Train & generate synthetic copy of the same length
meta = {
    'Global_active_power': FieldMetadata(data_type=DataType.DECIMAL, decimal_places=3)
}
gan = WGAN(real=year[['Global_active_power']], meta=meta,
           cfg=GanConfig(max_epochs=150))
gan.fit()
S = gan.generate(len(year))  # DataFrame with one column

# ——— 4. Slightly perturb each copy and re‑run ruptures
for delta in [0.05, -0.05, 0.10, -0.10]:
    pert = S.Global_active_power * (1 + delta)
    algo = rpt.Pelt(model="rbf").fit(pert.values)
    print(f"Δ={delta:+.0%} →", algo.predict(pen=10))
