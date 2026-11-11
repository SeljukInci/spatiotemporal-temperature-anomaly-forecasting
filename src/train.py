import tensorflow as tf
from pathlib import Path


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name,
    epochs=30,
    batch_size=256,
    patience=5,
):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.keras"
    model.save(model_path)

    print(f"Model kaydedildi: {model_path}")

    return model, history