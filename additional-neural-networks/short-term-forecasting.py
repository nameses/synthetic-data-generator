import pandas as pd
from datetime import timedelta
from data_generation.wgan import WGAN, GanConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType
from statsmodels.tsa.arima.model import ARIMA

from models.metadata import metadata_power_consumption

# ——— 1. Load & aggregate to daily
df = (
    pd.read_csv("household_power_consumption.txt", sep=';',
                parse_dates={'dt': ['Date', 'Time']},
                infer_datetime_format=True, na_values='?')
    .dropna(subset=['Global_active_power'])
    .set_index('dt')
    .resample('D').sum()
    .reset_index()
)
daily = df.rename(columns={'dt': 'Date', 'Global_active_power': 'Value'})

# ——— 2. Build metadata for the single series
meta = metadata_power_consumption

# ——— 3. Train your GAN
gan = WGAN(real=daily[['Value']], meta=meta,
           cfg=GanConfig(max_epochs=200))
gan.fit()

# ——— 4. Generate 60 days of fake history
S = gan.generate(60)
# assign real dates +1…+60
last_date = daily['Date'].iloc[-1]
S['Date'] = [last_date + timedelta(days=i + 1) for i in range(len(S))]

# ——— 5. Fit ARIMA on (a) real-only and (b) real+synthetic
train_real = daily.set_index('Date')['Value'][:-180]
train_aug = pd.concat([
    train_real,
    S.set_index('Date')['Value']
])
test = daily.set_index('Date')['Value'][-180:]


def one_step_mae(series, order=(5, 1, 0)):
    model = ARIMA(series, order=order).fit()
    preds = model.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    return (preds - test).abs().mean()


print("MAE real-only:", one_step_mae(train_real))
print("MAE real+syn :", one_step_mae(train_aug))
