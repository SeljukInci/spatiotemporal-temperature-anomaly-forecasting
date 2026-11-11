
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import create_period_sequences
from src.evaluate import (
    calculate_metrics,
    zero_anomaly_baseline,
    persistence_baseline,
    seasonal_persistence_baseline,
)

CITY_KEYS = [
    "City",
    "Country",
    "Latitude",
    "Longitude",
]


def main():

    # 1. Verileri yükle
    processed_dir = ROOT / "data" / "processed"

    train_df = pd.read_csv(
        processed_dir / "train.csv",
        parse_dates=["dt"]
    )

    val_df = pd.read_csv(
        processed_dir / "validation.csv",
        parse_dates=["dt"]
    )

    context_df = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    # 2. Doğrulama pencereleri
    X_val, y_val = create_period_sequences(
        context_df,
        CITY_KEYS,
        target_start="1990-01-01",
        target_end="2000-12-01",
        input_length=60,
        horizon=12,
    )

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    assert X_val.shape == (24200, 60, 1)
    assert y_val.shape == (24200, 12)

    # 3. Baseline tahminleri
    b00_pred = zero_anomaly_baseline(y_val)

    b01_pred = persistence_baseline(
        X_val,
        horizon=12
    )

    b02_pred = seasonal_persistence_baseline(
        X_val,
        horizon=12,
        season_length=12
    )

    # 4. M02 modelini yükle
    model_path = ROOT / "models" / "M02_pooled_lstm.keras"

    model = tf.keras.models.load_model(model_path)

    m02_pred = model.predict(
        X_val,
        verbose=0
    )

    predictions = {
        "B00 Zero-Anomaly": b00_pred,
        "B01 Persistence": b01_pred,
        "B02 Seasonal Persistence": b02_pred,
        "M02 Pooled LSTM": m02_pred,
    }

    # 5. Genel performans
    results = pd.DataFrame({
        name: calculate_metrics(y_val, pred)
        for name, pred in predictions.items()
    }).T

    results.index.name = "Model"

    print("\nGENEL DOĞRULAMA SONUÇLARI")
    print(results.round(4))

    # 6. Büyük anomali analizi
    threshold = 2.0
    extreme_mask = np.abs(y_val) >= threshold

    extreme_results = {}

    for name, pred in predictions.items():

        errors = np.abs(y_val - pred)

        extreme_results[name] = {
            "Extreme MAE": float(
                errors[extreme_mask].mean()
            ),
            "Normal MAE": float(
                errors[~extreme_mask].mean()
            ),
        }

    extreme_df = pd.DataFrame.from_dict(
        extreme_results,
        orient="index"
    )

    extreme_df.index.name = "Model"

    print("\nBÜYÜK ANOMALİ ANALİZİ (|anomali| >= 2 °C)")
    print(
        "Büyük anomali oranı:",
        f"{extreme_mask.mean() * 100:.2f}%"
    )
    print(extreme_df.round(4))

    # 7. Tahmin değişkenliği
    variability = pd.DataFrame({
        "Model": ["Actual"] + list(predictions.keys()),
        "Std": [
            float(y_val.std()),
            *[
                float(pred.std())
                for pred in predictions.values()
            ]
        ]
    })

    print("\nTAHMİN DEĞİŞKENLİĞİ")
    print(variability.round(4).to_string(index=False))

    # 8. Sonuçları kaydet
    output_dir = ROOT / "reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "validation_comparison.csv"
    )

    extreme_df.to_csv(
        output_dir / "extreme_anomaly_comparison.csv"
    )

    variability.to_csv(
        output_dir / "prediction_variability.csv",
        index=False
    )

    print("\nDeğerlendirme tamamlandı.")
    print("Sonuçlar kaydedildi:", output_dir)


if __name__ == "__main__":
    main()
