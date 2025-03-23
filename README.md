# Cloud Resource Optimization with Machine Learning and Real-Time Monitoring

## Overview  
This project focuses on optimizing cloud-based resource utilization by applying machine learning techniques to analyze and detect anomalies in CPU usage patterns. The system generates synthetic CPU usage data, applies both traditional and deep learning models for anomaly detection, and integrates Prometheus and Grafana for real-time monitoring and alerting.

## Features  
- **Synthetic Data Generation:** Simulates real-world CPU usage fluctuations.  
- **Multi-Model Approach:** Uses both traditional (ARIMA, Prophet) and deep learning (LSTM, Autoencoder, Transformer) models.  
- **Prometheus Integration:** Collects and exposes real-time CPU metrics.  
- **Grafana Dashboards:** Visualizes CPU utilization trends and anomalies.  
- **Anomaly Detection & Alerts:** Flags unusual CPU behavior and breaches predefined thresholds.  

### 1️⃣ Prerequisites  
Ensure you have the following installed:  
- Python 3.8+  
- Docker & Docker-Compose  
- Prometheus & Grafana  

### 2️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/cloud-optimization.git
cd cloud-optimization
