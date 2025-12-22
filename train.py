import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    """
    Creates sequences with a lookback window for LSTM input.
    
    Args:
        data: numpy array of shape (num_samples, num_features)
        lookback: number of past timesteps to use for prediction
    """
    def __init__(self, data, lookback=10):
        self.data = data
        self.lookback = lookback
        
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        # Get sequence of length 'lookback'
        sequence = self.data[idx:idx + self.lookback]
        return torch.FloatTensor(sequence)

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

def train_model(model, train_loader, val_loader, epochs=50, lr=0.0001, device='cpu'):
    """
    Train the Bi-LSTM Autoencoder model.
    
    Args:
        model: BiLSTMAutoencoder instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of training epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def calculate_reconstruction_errors(model, data_loader, device='cpu'):
    """
    Calculate reconstruction errors for each sample.
    
    Returns:
        errors: numpy array of reconstruction errors (one per sample)
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            
            # Calculate MSE for each sample in the batch
            batch_errors = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
            errors.extend(batch_errors.cpu().numpy())
    
    return np.array(errors)


def set_threshold(errors, percentile=99.5):
    """
    Set threshold based on reconstruction error distribution.
    
    Args:
        errors: numpy array of reconstruction errors from training data
        percentile: percentile value for threshold (e.g., 95, 99)
    
    Returns:
        threshold: value above which samples are considered anomalies
    """
    threshold = np.percentile(errors, percentile)
    return threshold


def detect_anomalies(model, data_loader, threshold, device='cpu'):
    """
    Detect anomalies in test data.
    
    Returns:
        predictions: binary array (1 = anomaly, 0 = normal)
        errors: reconstruction errors for each sample
    """
    errors = calculate_reconstruction_errors(model, data_loader, device)
    predictions = (errors > threshold).astype(int)
    return predictions, errors


def get_reconstructions(model, data_loader, device='cpu'):
    """
    Get original and reconstructed sequences for visualization.
    
    Returns:
        originals: list of original sequences
        reconstructed: list of reconstructed sequences
    """
    model.eval()
    originals = []
    reconstructed = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            recon = model(batch)
            
            originals.append(batch.cpu().numpy())
            reconstructed.append(recon.cpu().numpy())
    
    # Concatenate all batches
    originals = np.concatenate(originals, axis=0)
    reconstructed = np.concatenate(reconstructed, axis=0)
    
    return originals, reconstructed


def plot_reconstruction_comparison(originals, reconstructed, feature_names, 
                                   num_samples=5, save_path='reconstruction_comparison.png'):
    """
    Plot original vs reconstructed sequences for multiple samples and features.
    
    Args:
        originals: array of shape (num_samples, sequence_length, num_features)
        reconstructed: array of shape (num_samples, sequence_length, num_features)
        feature_names: list of feature names
        num_samples: number of samples to plot
        save_path: path to save the figure
    """
    num_features = originals.shape[2]
    sequence_length = originals.shape[1]
    
    # Select random samples to plot
    sample_indices = np.random.choice(len(originals), min(num_samples, len(originals)), replace=False)
    
    fig, axes = plt.subplots(num_samples, num_features, figsize=(20, 3*num_samples))
    
    # Handle case where num_samples = 1
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_idx in enumerate(sample_indices):
        for j, feature_name in enumerate(feature_names):
            ax = axes[i, j]
            
            # Plot original and reconstructed
            ax.plot(originals[sample_idx, :, j], 'b-', label='Original', linewidth=2, alpha=0.7)
            ax.plot(reconstructed[sample_idx, :, j], 'r--', label='Reconstructed', linewidth=2, alpha=0.7)
            
            # Calculate reconstruction error for this feature
            mse = np.mean((originals[sample_idx, :, j] - reconstructed[sample_idx, :, j])**2)
            
            # Add labels and title
            if i == 0:
                ax.set_title(f'{feature_name}\n(MSE: {mse:.6f})', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'MSE: {mse:.6f}', fontsize=9)
            
            if i == num_samples - 1:
                ax.set_xlabel('Timestep', fontsize=9)
            
            if j == 0:
                ax.set_ylabel(f'Sample {sample_idx}', fontsize=9)
            
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nReconstruction comparison plot saved as '{save_path}'")
    plt.show()


def plot_feature_reconstruction_errors(originals, reconstructed, feature_names, 
                                      save_path='feature_errors.png'):
    """
    Plot distribution of reconstruction errors for each feature.
    
    Args:
        originals: array of shape (num_samples, sequence_length, num_features)
        reconstructed: array of shape (num_samples, sequence_length, num_features)
        feature_names: list of feature names
        save_path: path to save the figure
    """
    num_features = originals.shape[2]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        # Calculate per-sample errors for this feature
        errors = np.mean((originals[:, :, i] - reconstructed[:, :, i])**2, axis=1)
        
        ax = axes[i]
        ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feature_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.6f}')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature reconstruction errors plot saved as '{save_path}'")
    plt.show()

def plot_anomaly_visualization(originals, reconstructed, predictions, errors, 
                               feature_names, threshold, num_normal=3, num_anomaly=3,
                               save_path='anomaly_visualization.png'):
    """
    Plot examples of normal and anomalous samples with their reconstructions.
    
    Args:
        originals: array of shape (num_samples, sequence_length, num_features)
        reconstructed: array of shape (num_samples, sequence_length, num_features)
        predictions: binary array (1 = anomaly, 0 = normal)
        errors: reconstruction errors for each sample
        feature_names: list of feature names
        threshold: anomaly detection threshold
        num_normal: number of normal samples to plot
        num_anomaly: number of anomalous samples to plot
        save_path: path to save the figure
    """
    # Get indices of normal and anomalous samples
    normal_indices = np.where(predictions == 0)[0]
    anomaly_indices = np.where(predictions == 1)[0]
    
    # Select samples with highest/lowest errors
    if len(normal_indices) > 0:
        # Select normal samples with highest errors (close to threshold)
        normal_errors = errors[normal_indices]
        top_normal = normal_indices[np.argsort(normal_errors)[-num_normal:]]
    else:
        top_normal = []
    
    if len(anomaly_indices) > 0:
        # Select anomalous samples with highest errors
        anomaly_errors = errors[anomaly_indices]
        top_anomaly = anomaly_indices[np.argsort(anomaly_errors)[-num_anomaly:]]
    else:
        top_anomaly = []
    
    num_features = len(feature_names)
    total_samples = len(top_normal) + len(top_anomaly)
    
    if total_samples == 0:
        print("No samples to plot!")
        return
    
    fig, axes = plt.subplots(total_samples, num_features, figsize=(20, 3*total_samples))
    
    # Handle single sample case
    if total_samples == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    # Plot normal samples
    for sample_idx in top_normal:
        for j, feature_name in enumerate(feature_names):
            ax = axes[plot_idx, j]
            
            ax.plot(originals[sample_idx, :, j], 'b-', label='Original', linewidth=2, alpha=0.7)
            ax.plot(reconstructed[sample_idx, :, j], 'r--', label='Reconstructed', linewidth=2, alpha=0.7)
            
            if plot_idx == 0:
                ax.set_title(feature_name, fontsize=11, fontweight='bold')
            
            if j == 0:
                ax.set_ylabel(f'NORMAL\nError: {errors[sample_idx]:.6f}', 
                            fontsize=9, color='green', fontweight='bold')
            
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot anomalous samples
    for sample_idx in top_anomaly:
        for j, feature_name in enumerate(feature_names):
            ax = axes[plot_idx, j]
            
            ax.plot(originals[sample_idx, :, j], 'b-', label='Original', linewidth=2, alpha=0.7)
            ax.plot(reconstructed[sample_idx, :, j], 'r--', label='Reconstructed', linewidth=2, alpha=0.7)
            
            if j == 0:
                ax.set_ylabel(f'ANOMALY\nError: {errors[sample_idx]:.6f}', 
                            fontsize=9, color='red', fontweight='bold')
            
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Add overall title
    fig.suptitle(f'Anomaly Detection Examples (Threshold: {threshold:.6f})', 
                 fontsize=14, fontweight='bold', y=1.001)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Anomaly visualization plot saved as '{save_path}'")
    plt.show()

def main():
    df = pd.read_csv('laptop_rian_palel.csv')
    
    features = ['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']
    data = df[features].values
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Split data: 70% train, 15% validation, 15% test
    train_size = int(0.8 * len(data_normalized))
    val_size = int(0.15 * len(data_normalized))
    
    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:train_size + val_size]
    test_data = data_normalized[train_size + val_size:]
    
    # Create datasets and dataloaders
    lookback = 10
    batch_size = 128
    
    train_dataset = TimeSeriesDataset(train_data, lookback=lookback)
    val_dataset = TimeSeriesDataset(val_data, lookback=lookback)
    test_dataset = TimeSeriesDataset(test_data, lookback=lookback)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_features = len(features)
    model = BiLSTMAutoencoder(num_features=num_features)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Train the model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=300, lr=0.0001, device=device
    )

    # exit()
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Calculate reconstruction errors on training data
    print("\nCalculating reconstruction errors on training data...")
    train_errors = calculate_reconstruction_errors(model, train_loader, device)
    
    # Set threshold (using 95th percentile)
    threshold = set_threshold(train_errors, percentile=95)
    print(f'Threshold set at: {threshold:.6f}')
    
    # Plot distribution of training errors
    plt.subplot(1, 2, 2)
    plt.hist(train_errors, bins=50, alpha=0.7, label='Training Errors')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Reconstruction Errors')
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Get reconstructions for visualization
    print("\nGetting reconstructions for visualization...")
    train_originals, train_reconstructed = get_reconstructions(model, train_loader, device)
    test_originals, test_reconstructed = get_reconstructions(model, test_loader, device)
    
    # Plot reconstruction comparison for training data (normal data)
    print("\n=== Plotting Training Data Reconstructions (Normal Data) ===")
    plot_reconstruction_comparison(
        train_originals, train_reconstructed, features,
        num_samples=5, save_path='train_reconstruction_comparison.png'
    )
    
    # Plot feature-wise reconstruction errors for training data
    plot_feature_reconstruction_errors(
        train_originals, train_reconstructed, features,
        save_path='train_feature_errors.png'
    )
    
    # Detect anomalies on test data
    print("\n=== Detecting Anomalies on Test Data ===")
    predictions, test_errors = detect_anomalies(model, test_loader, threshold, device)
    
    num_anomalies = np.sum(predictions)
    print(f'Number of anomalies detected: {num_anomalies} out of {len(predictions)} samples')
    print(f'Anomaly rate: {num_anomalies/len(predictions)*100:.2f}%')
    
    # Plot test errors
    plt.figure(figsize=(14, 4))
    plt.plot(test_errors, label='Reconstruction Error', alpha=0.7)
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    plt.scatter(np.where(predictions == 1)[0], test_errors[predictions == 1], 
                color='red', label='Anomalies', s=20, zorder=5)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.title('Anomaly Detection Results on Test Data')
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot reconstruction comparison for test data
    print("\n=== Plotting Test Data Reconstructions ===")
    plot_reconstruction_comparison(
        test_originals, test_reconstructed, features,
        num_samples=5, save_path='test_reconstruction_comparison.png'
    )
    
    # Plot feature-wise reconstruction errors for test data
    plot_feature_reconstruction_errors(
        test_originals, test_reconstructed, features,
        save_path='test_feature_errors.png'
    )
    
    # Plot examples of normal vs anomalous samples
    print("\n=== Plotting Normal vs Anomalous Examples ===")
    plot_anomaly_visualization(
        test_originals, test_reconstructed, predictions, test_errors,
        features, threshold, num_normal=3, num_anomaly=3,
        save_path='anomaly_examples.png'
    )
    
    # Save the model
    torch.save(model.state_dict(), 'bilstm_autoencoder_last.pth')
    print("\nModel saved as 'bilstm_autoencoder_last.pth'")
    
    # Save threshold for future use
    np.save('threshold_last.npy', threshold)
    print("Threshold saved as 'threshold_last.npy'")


if __name__ == '__main__':
    main()