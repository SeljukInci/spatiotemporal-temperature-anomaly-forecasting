import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense


def build_m02_pooled_lstm(
    input_length=60,
    hidden_units=64,
    horizon=12,
    dropout_rate=0.2
):
    inputs = Input(shape=(input_length, 1), name="anomaly_input")

    x = LSTM(hidden_units, name="temporal_lstm")(inputs)
    x = Dropout(dropout_rate, name="dropout")(x)

    outputs = Dense(horizon, name="forecast")(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="M02_Pooled_LSTM"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model