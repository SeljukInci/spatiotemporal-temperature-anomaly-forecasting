from pathlib import Path

import numpy as np
import tensorflow as tf


def load_trained_model(model_path):
    """Kaydedilmiş Keras modelini yükler."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}"
        )

    return tf.keras.models.load_model(model_path)


def predict_anomalies(model, history, input_length=60):
    """
    Son 60 aylık anomaliden gelecek 12 ayı tahmin eder.

    Parameters
    ----------
    model : tf.keras.Model
        Eğitilmiş tahmin modeli.

    history : array-like
        Kronolojik sıralanmış aylık sıcaklık anomalileri.

    input_length : int
        Modelin beklediği geçmiş ay sayısı.

    Returns
    -------
    np.ndarray
        12 aylık tahmin vektörü.
    """

    history = np.asarray(history, dtype=np.float32)

    if history.ndim != 1:
        raise ValueError(
            "Geçmiş veriler tek boyutlu olmalı."
        )

    if len(history) < input_length:
        raise ValueError(
            f"En az {input_length} aylık veri gerekli."
        )

    if not np.isfinite(history).all():
        raise ValueError(
            "Geçmiş veriler NaN veya sonsuz değer içeriyor."
        )

    # Son 60 ayı al
    recent_history = history[-input_length:]

    # (60,) -> (1, 60, 1)
    X = recent_history.reshape(
        1, input_length, 1
    )

    # Tahmin
    predictions = model.predict(
        X,
        verbose=0
    )

    return predictions[0]