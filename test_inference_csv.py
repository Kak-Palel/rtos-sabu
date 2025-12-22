import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from anomaly_detector import RealtimeAnomalyDetector
from realtime_plotter import RealtimePlotter
    
CSV_PATH = 'sensor_log.csv'
MODEL_PATH = 'bilstm_autoencoder.pth'
THRESHOLD_PATH = 'threshold.npy'
SCALER_PATH = 'scaler.pkl'

FEATURE_NAMES = ['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']
LOOKBACK = 10

UPDATE_INTERVAL = 1000  # Update every 1 second (1000ms)

def simulate_realtime_detection(csv_path, model_path, threshold_path, scaler_path,
                                feature_names, lookback=10, update_interval=1000,
                                device='cpu'):
    """
    Simulate real-time anomaly detection from CSV data.
    
    Args:
        csv_path: path to CSV file with new data
        model_path: path to saved model (.pth)
        threshold_path: path to saved threshold (.npy)
        scaler_path: path to saved scaler (.pkl)
        feature_names: list of feature names
        lookback: sequence length
        update_interval: update interval in milliseconds (1000 = 1 second)
        device: 'cuda' or 'cpu'
    """
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    data = df[feature_names].values
    
    print(f"Loaded {len(data)} data points")
    print(f"Features: {feature_names}\n")
    
    # Initialize detector
    detector = RealtimeAnomalyDetector(
        model_path=model_path,
        threshold_path=threshold_path,
        scaler_path=scaler_path,
        feature_names=feature_names,
        lookback=lookback,
        device=device,
        error_threshold=0.7
    )
    
    # Initialize plotter
    plotter = RealtimePlotter(detector, max_points=100)
    
    # Data point iterator
    data_idx = [0]  # Use list to modify in nested function
    
    def update_plot(frame):
        """Update function for animation."""
        if data_idx[0] >= len(data):
            print("\n✓ All data processed!")
            return
        
        # Get next data point
        data_point = data[data_idx[0]]
        print("data_point: ", data_point)
        timestamp = data_idx[0]
        
        # Process data point
        is_anomaly, error, original, reconstructed = detector.add_data_point(data_point)
        
        if is_anomaly is not None:  # Only update after we have enough points
            detector.update_history(timestamp, error, is_anomaly, original, reconstructed)
            
            # Print status
            status = "🔴 ANOMALY" if is_anomaly else "🟢 Normal "
            print(f"[{timestamp:4d}] {status} | Error: {error:.6f} | Threshold: {detector.threshold:.6f}")
        else:
            print(f"[{timestamp:4d}] Buffering... ({len(detector.data_buffer)}/{detector.lookback})")
        
        # Update plots
        plotter.update()
        
        data_idx[0] += 1
    
    # Create animation
    print(f"Starting real-time simulation (update every {update_interval}ms)...")
    print("Close the plot window to stop.\n")
    print("="*70)
    
    anim = FuncAnimation(plotter.fig, update_plot, interval=update_interval, 
                        cache_frame_data=False, repeat=False)
    
    plt.show()
    
    # Print final statistics
    if len(detector.prediction_history) > 0:
        total_anomalies = sum(detector.prediction_history)
        total_points = len(detector.prediction_history)
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS")
        print(f"{'='*70}")
        print(f"Total points processed: {total_points}")
        print(f"Total anomalies detected: {total_anomalies}")
        print(f"Anomaly rate: {(total_anomalies/total_points)*100:.2f}%")
        print(f"Average reconstruction error: {np.mean(detector.error_history):.6f}")
        print(f"Max reconstruction error: {np.max(detector.error_history):.6f}")
        print(f"Min reconstruction error: {np.min(detector.error_history):.6f}")
        print(f"{'='*70}")

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}\n')
    
    # Run simulation
    simulate_realtime_detection(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH,
        scaler_path=SCALER_PATH,
        feature_names=FEATURE_NAMES,
        lookback=LOOKBACK,
        update_interval=UPDATE_INTERVAL,
        device=device
    )