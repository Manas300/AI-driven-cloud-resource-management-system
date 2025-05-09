from flask import Flask, Response
from prometheus_client import Gauge, generate_latest
import pandas as pd
import numpy as np
import requests
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Create Flask app
app = Flask(__name__)

# Metrics
cpu_actual = Gauge('cpu_actual_usage', 'Actual CPU Usage')
cpu_predicted = Gauge('cpu_predicted_usage', 'Predicted CPU Usage')

# Other constants
PROMETHEUS_URL = "http://localhost:9091/api/v1/query"
SEQ_LENGTH = 10
BATCH_SIZE = 1
EPOCHS = 10
POLL_INTERVAL = 10  # seconds

data_points = []
scaler = MinMaxScaler()

# CPU usage fetcher
def fetch_cpu_usage():
    try:
        query = 'rate(container_cpu_usage_seconds_total[1m])'
        response = requests.get(PROMETHEUS_URL, params={'query': query})
        results = response.json()["data"]["result"]
        values = [float(item["value"][1]) for item in results if "value" in item]
        return np.mean(values) if values else None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

# Prepare LSTM sequences
def prepare_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

# LSTM model builder
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# Metrics endpoint
@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# Background job that keeps training and predicting
def background_job():
    global data_points, scaler
    while True:
        usage = fetch_cpu_usage()
        if usage is not None:
            data_points.append([usage])
            cpu_actual.set(usage)
            print(f"[INFO] Fetched CPU Usage: {usage:.6f}")

        if len(data_points) > SEQ_LENGTH + 1:
            df = pd.DataFrame(data_points, columns=["usage"])
            scaled_data = scaler.fit_transform(df)

            X, y = prepare_sequences(scaled_data, SEQ_LENGTH)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = build_model((SEQ_LENGTH, 1))
            model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            input_seq = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
            pred = model.predict(input_seq)[0][0]
            pred_val = scaler.inverse_transform([[pred]])[0][0]
            cpu_predicted.set(pred_val)
            print(f"[PREDICTION] CPU Predicted: {pred_val:.6f}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    from threading import Thread
    # Start background polling thread
    t = Thread(target=background_job)
    t.start()

    # Run Flask server
    app.run(host="0.0.0.0", port=9101)
