import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import deque

class BiLSTMAutoencoder(nn.Module):
    """
    Bi-LSTM Autoencoder for anomaly detection in time series data.
    
    Architecture:
    - Encoder: 2 Bi-LSTM layers (64 units, 32 units)
    - Decoder: 2 LSTM layers (32 units, 64 units)
    """
    def __init__(self, num_features = 6, hidden_dim1=64, hidden_dim2=32):
        super(BiLSTMAutoencoder, self).__init__()
        
        self.num_features = num_features
        
        # ENCODER
        self.encoder_lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
            bidirectional=True 
        )
        
        self.encoder_lstm2 = nn.LSTM(
            input_size=hidden_dim1 * 2,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # DECODER
        self.decoder_lstm1 = nn.LSTM(
            input_size=hidden_dim2 * 2,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder_lstm2 = nn.LSTM(
            input_size=hidden_dim2,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim1, num_features)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        
        # ENCODING
        # Pass through first Bi-LSTM layer
        encoded, _ = self.encoder_lstm1(x)
        # encoded shape: (batch_size, sequence_length, 128)
        
        # Pass through second Bi-LSTM layer
        encoded, _ = self.encoder_lstm2(encoded)
        # encoded shape: (batch_size, sequence_length, 64)
        
        # DECODING
        # Pass through first LSTM layer
        decoded, _ = self.decoder_lstm1(encoded)
        # decoded shape: (batch_size, sequence_length, 32)
        
        # Pass through second LSTM layer
        decoded, _ = self.decoder_lstm2(decoded)
        # decoded shape: (batch_size, sequence_length, 64)
        
        # Output layer to reconstruct original features
        output = self.output_layer(decoded)
        # output shape: (batch_size, sequence_length, num_features)
        
        return output


class RealtimeAnomalyDetector:
    """
    Real-time anomaly detection with live plotting.
    """
    def __init__(self, model_path, threshold_path, scaler_path, feature_names, 
                 lookback=10, device='cpu', error_threshold = None):
        """
        Args:
            model_path: path to saved model weights (.pth)
            threshold_path: path to saved threshold (.npy)
            scaler_path: path to saved scaler (.pkl)
            feature_names: list of feature names
            lookback: sequence length for LSTM
            device: 'cuda' or 'cpu'
            error_threshold: override the trained threshold, leave to None if using the trained threshold
        """
        self.device = device
        self.lookback = lookback
        self.feature_names = feature_names
        self.num_features = len(feature_names)

        self.voltage_offset = None
        self.current_offset = None
        self.power_offset = None
        self.energy_offset = None
        self.frequency_offset = None
        self.power_factor_offset = None

        self.data_count_without_offset_update = 0
        
        # Load scaler
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = BiLSTMAutoencoder(num_features=self.num_features)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Load threshold
        print(f"Loading threshold from {threshold_path}...")
        self.threshold = np.load(threshold_path) if error_threshold is None else error_threshold
        print(f"Threshold: {self.threshold:.6f}\n")
        
        # Buffer to store recent data points
        self.data_buffer = deque(maxlen=lookback)
        
        # History for plotting
        self.time_history = []
        self.error_history = []
        self.prediction_history = []
        self.original_history = {name: [] for name in feature_names}
        self.reconstructed_history = {name: [] for name in feature_names}
        
    def add_data_point(self, data_point):
        """
        Add a new data point and make prediction.
        
        Args:
            data_point: numpy array of shape (num_features,)
        
        Returns:
            is_anomaly: boolean (or None if not enough data yet)
            error: reconstruction error (or None)
            original_denorm: original data point (denormalized)
            reconstructed_denorm: reconstructed data point (denormalized)
        """
        # Normalize the data point
        data_point_normalized = self.scaler.transform(data_point.reshape(1, -1))[0]
        
        # Add to buffer
        self.data_buffer.append(data_point_normalized)
        
        # Need at least 'lookback' points to make prediction
        if len(self.data_buffer) < self.lookback:
            return None, None, None, None
        
        # Create sequence
        sequence = np.array(list(self.data_buffer))  # Shape: (lookback, num_features)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed_tensor = self.model(sequence_tensor)
        
        # Get the last timestep reconstruction (most recent)
        reconstructed = reconstructed_tensor[0, -1, :].cpu().numpy()
        original = sequence[-1]  # Last point in sequence

        print("original", original)
        print("reconstructed", reconstructed)

        if not self.voltage_offset or self.data_count_without_offset_update >= 10:
            self.voltage_offset = original[0] - reconstructed[0]
            self.current_offset = original[1] - reconstructed[1]
            self.power_offset = original[2] - reconstructed[2]
            self.energy_offset = original[3] - reconstructed[3]
            self.frequency_offset = original[4] - reconstructed[4]
            self.power_factor_offset = original[5] - reconstructed[5]
            self.data_count_without_offset_update = 0
        else:
            self.data_count_without_offset_update += 1
        
        # Apply offsets to reconstructed values
        reconstructed[0] += self.voltage_offset
        reconstructed[1] += self.current_offset
        reconstructed[2] += self.power_offset
        reconstructed[3] += self.energy_offset
        reconstructed[4] += self.frequency_offset
        reconstructed[5] += self.power_factor_offset
        
        # Calculate reconstruction error
        error = np.mean((original - reconstructed) ** 2)
        
        # Determine if anomaly
        is_anomaly = error > self.threshold
        
        # Inverse transform for plotting
        original_denorm = self.scaler.inverse_transform(original.reshape(1, -1))[0]
        reconstructed_denorm = self.scaler.inverse_transform(reconstructed.reshape(1, -1))[0]
        
        return is_anomaly, error, original_denorm, reconstructed_denorm
    
    def update_history(self, timestamp, error, is_anomaly, original, reconstructed):
        """Update history for plotting."""
        self.time_history.append(timestamp)
        self.error_history.append(error)
        self.prediction_history.append(1 if is_anomaly else 0)
        
        for i, name in enumerate(self.feature_names):
            self.original_history[name].append(original[i])
            self.reconstructed_history[name].append(reconstructed[i])