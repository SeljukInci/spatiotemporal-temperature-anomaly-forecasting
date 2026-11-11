import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Boyut uyuşmazlığı: {y_true.shape} != {y_pred.shape}"
        )

    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    rmse = np.sqrt(
        mean_squared_error(y_true.ravel(), y_pred.ravel())
    )

    return {"MAE": float(mae), "RMSE": float(rmse)}


def zero_anomaly_baseline(y_true):
    return np.zeros_like(y_true)


def persistence_baseline(X, horizon=12):
    X = np.asarray(X)

    if X.ndim != 3 or X.shape[-1] != 1:
        raise ValueError("X boyutu (N, 60, 1) olmalı.")

    last_anomaly = X[:, -1, 0]

    return np.repeat(
        last_anomaly[:, np.newaxis],
        horizon,
        axis=1
    )

def evaluate_model(model, X, y):
    predictions = model.predict(X, verbose=0)
    return calculate_metrics(y, predictions)

def seasonal_persistence_baseline(X, horizon=12, season_length=12):
    X = np.asarray(X)

    if X.ndim != 3 or X.shape[-1] != 1:
        raise ValueError("X boyutu (N, 60, 1) olmalı.")

    if X.shape[1] < season_length:
        raise ValueError("Yeterli geçmiş veri bulunmuyor.")

    if horizon > season_length:
        raise ValueError(
            "Bu baseline için horizon <= season_length olmalı."
        )

    # Son 12 ayın anomalilerini sonraki 12 aya kopyala
    return X[:, -season_length:, 0][:, :horizon].copy()