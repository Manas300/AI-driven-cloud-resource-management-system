from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    metrics_data = """
# HELP dummy_metric A dummy metric for testing
# TYPE dummy_metric gauge
dummy_metric 42
"""
    return Response(metrics_data, mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040)
