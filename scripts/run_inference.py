from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import (
    load_trained_model,
    predict_anomalies,
)

# Doğrulama dönemi: 1990–2000
df = pd.read_csv(
    ROOT / "data/processed/validation.csv",
    parse_dates=["dt"]
)

# İlk şehir kaydını seç
city_keys = ["City", "Country", "Latitude", "Longitude"]

city = (
    df.groupby(city_keys, sort=False)
    .get_group(tuple(df.iloc[0][city_keys]))
    .sort_values("dt")
)

# İlk 60 ayı kullan: Ocak 1990 – Aralık 1994
history = city["temperature_anomaly"].iloc[:60].to_numpy()

model = load_trained_model(
    ROOT / "models/M02_pooled_lstm.keras"
)

predictions = predict_anomalies(
    model,
    history
)

# Gerçek gelecek 12 ay: Ocak – Aralık 1995
actual = city["temperature_anomaly"].iloc[60:72].to_numpy()

results = pd.DataFrame({
    "Month": city["dt"].iloc[60:72].dt.strftime("%Y-%m").to_numpy(),
    "Actual": actual,
    "Predicted": predictions
})

print("\n12-Month Temperature Anomaly Forecast")
print(results.round(3).to_string(index=False))