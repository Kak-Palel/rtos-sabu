from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app, Summary, Gauge
from flask import Flask, jsonify, request

app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': make_wsgi_app()})

voltage = Gauge('voltage', 'Voltage gauge example')
current = Gauge('current', 'Current gauge example')
power = Gauge('power', 'Power gauge example')
energy = Gauge('energy', 'Energy gauge example')
frequency = Gauge('frequency', 'Frequency gauge example')
power_factor = Gauge('power_factor', 'Power Factor gauge example')

@app.route('/')
def home():
    return "" \
    "<h1>Welcome to the Home Page</h1>" \
    "<p>This is a simple Flask application.</p>"

@app.route('/inference', methods=['POST'])
def inference():
    print("Inference endpoint called")
    request_data = request.get_json()
    print(f"Received data for inference: {request_data}")

    voltage.set(request_data.get("Voltage", 0))
    current.set(request_data.get("Current", 0))
    power.set(request_data.get("Power", 0))
    energy.set(request_data.get("Energy", 0))
    frequency.set(request_data.get("Frequency", 0))
    power_factor.set(request_data.get("PF", 0))

    # response = {"prediction_conf": {"anomaly": 0.95, "normal": 0.05}}
    response = request_data
    response["prediction_conf"] = {"anomaly": 0.1, "normal": 0.9}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)