from flask import Flask
from prometheus_client import Gauge, start_http_server
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import logging
from tensorflow.keras.metrics import MeanSquaredError  # Explicit import for mse

# Initialize Flask
app = Flask(__name__)

# Load trained model with explicit 'mse' metric
model = tf.keras.models.load_model("cpu_anomaly_model.h5", custom_objects={'mse': MeanSquaredError()})

# Set up logging
logging.basicConfig(filename="cpu_anomaly.log", level=logging.INFO)

# Define Prometheus metrics
cpu_anomaly_score = Gauge("cpu_anomaly_score", "Anomaly Score of Synthetic CPU Usage")
cpu_warning = Gauge("cpu_warning", "1 if anomaly detected, 0 otherwise")

# Start Prometheus HTTP server
start_http_server(8000)

@app.route("/")
def home():
    return "CPU Anomaly Detection Running with Prometheus & Grafana!"

# Load dataset
df = pd.read_csv("cpu_usage.csv")

# Monitor in real time
THRESHOLD = 0.05  # Set anomaly threshold

for i in range(len(df) - 10):
    # Prepare input data
    input_data = np.array(df["cpu_usage"].values[i:i+10]).reshape(1, 10, 1)
    
    # Predict
    pred = model.predict(input_data)[0, 0]
    
    # Compute anomaly score
    score = np.abs(pred - df["cpu_usage"].values[i+10])
    
    # Push to Prometheus
    cpu_anomaly_score.set(score)
    
    # Check for anomaly
    if score > THRESHOLD:
        logging.warning(f"⚠️ Anomaly Detected at Time {i+10}: Score {score}")
        cpu_warning.set(1)  # Set warning flag
    else:
        cpu_warning.set(0)  # Reset flag
    
    time.sleep(1)  # Simulate real-time monitoring

