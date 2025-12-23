from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app, Enum, Gauge
from anomaly_detector_lightning import RealtimeAnomalyDetector
from flask import Flask, jsonify, request
import torch
import time
import numpy as np

app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': make_wsgi_app()})

DATA_RATE = 1 #hz
SAMPLE_EVERY = 1 #seconds
ENABLE_DATA_SAVING = False
data_received = 0

MODEL_PATH = 'weights/bilstm-autoencoder-epoch=199-val_loss=0.11.ckpt'
SCALER_PATH = 'scaler.pkl'

FEATURE_NAMES = ['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']
LOOKBACK = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anomaly_detector = RealtimeAnomalyDetector(
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    feature_names=FEATURE_NAMES,
    lookback=LOOKBACK,
    device=device
)

if ENABLE_DATA_SAVING:
    from data_saver import DataSaver
    data_saver = DataSaver("laptop_rian_palel.csv")
else:
    data_saver = None


voltage = Gauge('voltage', 'Voltage gauge')
current = Gauge('current', 'Current gauge')
power = Gauge('power', 'Power gauge')
energy = Gauge('energy', 'Energy gauge')
frequency = Gauge('frequency', 'Frequency gauge')
power_factor = Gauge('power_factor', 'Power Factor gauge')
reconstruction_error = Gauge('reconstruction_error', 'Reconstruction Error gauge from the BiLSTM Autoencoder')

@app.route('/')
def home():
    return "" \
    "<h1>Welcome to the Home Page</h1>" \
    "<p>This is a simple Flask application.</p>"

@app.route('/inference', methods=['POST'])
def inference():
    global data_received
    
    print("Inference endpoint called")
    request_data = request.get_json()
    print(f"Received data for inference: {request_data}")

    new_voltage = request_data.get("voltage", 0)
    new_current = request_data.get("current", 0)
    new_power = request_data.get("power", 0)
    new_energy = request_data.get("energy", 0)
    new_frequency = request_data.get("frequency", 0)
    new_pf = request_data.get("pf", 0)

    # print(new_voltage); print(new_current); print(new_power); print(new_energy); print(new_frequency);print(new_pf)

    voltage.set(new_voltage)
    current.set(new_current)
    power.set(new_power)
    energy.set(new_energy)
    frequency.set(new_frequency)
    power_factor.set(new_pf)

    # if data_saver and (data_received % SAMPLE_EVERY == 0):
    #     data_saver.save(
    #         new_voltage,
    #         new_current,
    #         new_power,
    #         new_energy,
    #         new_frequency,
    #         new_pf
    #     )
    # data_received += 1
    if ENABLE_DATA_SAVING:
        data_saver.save(
            new_voltage,
            new_current,
            new_power,
            new_energy,
            new_frequency,
            new_pf
        )

    data_point = np.array([new_voltage, new_current, new_power, new_energy, new_frequency, new_pf])
    error, original, reconstructed = anomaly_detector.add_data_point(data_point)
        
    timestamp = time.time()
    if error is not None:  # Only update after we have enough points
        anomaly_detector.update_history(timestamp, error, original, reconstructed)
        
        print(f"[{timestamp:4f}] Error: {error:.6f}")
        response = request_data
        response["reconstruction_error"] = error
        reconstruction_error.set(error)
        return jsonify(response)
    else:
        print(f"[{timestamp:4f}] Buffering... ({len(anomaly_detector.data_buffer)}/{anomaly_detector.lookback})")
        response = request_data
        response["reconstruction_error"] = 0
        reconstruction_error.set(0)
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')