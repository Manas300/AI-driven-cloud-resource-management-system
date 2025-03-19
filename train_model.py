import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("cpu_usage.csv", parse_dates=["timestamp"])

# Normalize CPU usage
scaler = MinMaxScaler()
df["cpu_usage"] = scaler.fit_transform(df[["cpu_usage"]])

# Convert to sequences (10 time steps each)
seq_length = 10
X_train = np.array([df["cpu_usage"].values[i:i+seq_length] for i in range(len(df) - seq_length)])

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build LSTM Autoencoder model
model = keras.Sequential([
    keras.layers.LSTM(16, activation="relu", return_sequences=True, input_shape=(seq_length, 1)),
    keras.layers.LSTM(4, activation="relu", return_sequences=False),
    keras.layers.RepeatVector(seq_length),
    keras.layers.LSTM(4, activation="relu", return_sequences=True),
    keras.layers.LSTM(16, activation="relu", return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.1)

# Save trained model
model.save("cpu_anomaly_model.h5")
print("✅ Model trained and saved as cpu_anomaly_model.h5!")
