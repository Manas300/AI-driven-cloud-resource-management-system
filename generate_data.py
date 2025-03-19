import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate normal CPU usage (50% ± 10%)
normal_cpu = np.random.normal(loc=50, scale=10, size=1000)

# Introduce anomalies (CPU spikes above 90% or below 20%)
anomalies = np.random.choice([95, 98, 15, 10], size=20)
cpu_usage = np.concatenate([normal_cpu, anomalies])

# Create timestamps
timestamps = pd.date_range(start="2025-03-13 00:00:00", periods=len(cpu_usage), freq="S")

# Save to CSV
df = pd.DataFrame({"timestamp": timestamps, "cpu_usage": cpu_usage})
df.to_csv("cpu_usage.csv", index=False)

print("✅ Synthetic CPU data generated and saved as cpu_usage.csv!")
