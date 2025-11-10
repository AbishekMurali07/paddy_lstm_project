

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm_model(data_path="../data/final_tn_dataset.csv"):
    """Train the LSTM model and return the model, scaler, and test data."""
    df = pd.read_csv(data_path)

    # Select features for training
    features = ["Area", "Production", "Annual_Rainfall_mm", "Avg_Temperature_C",
                "Nitrogen", "Phosphorus", "Potassium"]
    target = "Yield"

    # Handle missing or invalid data
    df = df.dropna(subset=features + [target])

    X = df[features].values
    y = df[target].values.reshape(-1, 1)

    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split into training/testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Reshape for LSTM [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build & train
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    tf.compat.v1.reset_default_graph()  # fixed TensorFlow warning
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)

    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Model trained successfully. Test MAE: {mae:.4f}")

    return model, scaler_X, scaler_y, X_test, y_test, df
