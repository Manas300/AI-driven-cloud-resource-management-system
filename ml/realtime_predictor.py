import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Configurations
PROMETHEUS_URL = "http://localhost:9091/api/v1/query"
QUERY = 'rate(container_cpu_usage_seconds_total[1m])'
INTERVAL = 15
SEQUENCE_LENGTH = 10
EPOCHS = 10

# Fetch CPU data
def fetch_cpu_metrics():
    try:
        res = requests.get(PROMETHEUS_URL, params={'query': QUERY})
        results = res.json()["data"]["result"]
        metrics = []
        for item in results:
            container = item["metric"].get("container", "unknown")
            value = float(item["value"][1])
            metrics.append({
                "timestamp": datetime.utcnow(),
                "container": container,
                "value": value
            })
        return pd.DataFrame(metrics)
    except Exception as e:
        print("[ERROR] Fetching metrics:", e)
        return pd.DataFrame(columns=["timestamp", "container", "value"])

# Collect for N seconds
def collect_data(duration=120):
    data = []
    print("[INFO] Starting data collection...")
    start = time.time()
    while time.time() - start < duration:
        df = fetch_cpu_metrics()
        if not df.empty:
            print(f"[{datetime.utcnow()}] Polled {len(df)} datapoints.")
            data.append(df)
        time.sleep(INTERVAL)
    return pd.concat(data)

# Format for LSTM
def prepare_sequences(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)

# Train LSTM
def train_model(values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    X, y = prepare_sequences(scaled, SEQUENCE_LENGTH)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=EPOCHS, verbose=1)
    return model, scaler, X, y

# Main pipeline
def main():
    df = collect_data()
    container = df['container'].value_counts().idxmax()
    print(f"[INFO] Focusing on container: {container}")
    series = df[df["container"] == container].sort_values("timestamp")["value"].values

    model, scaler, X, y = train_model(series)
    pred = model.predict(X)
    actual = scaler.inverse_transform(y.reshape(-1, 1))
    predicted = scaler.inverse_transform(pred)

    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.legend()
    plt.title(f"CPU Usage Forecast for {container}")
    plt.xlabel("Time Step")
    plt.ylabel("CPU Usage")
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
