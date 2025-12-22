import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torchmetrics
import matplotlib.pyplot as plt
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

class BiLSTMAutoencoder(L.LightningModule):
    """
    Bi-LSTM Autoencoder for anomaly detection in time series data.
    
    Architecture:
    - Encoder: 2 Bi-LSTM layers (64 units, 32 units)
    - Decoder: 2 LSTM layers (32 units, 64 units)
    """
    def __init__(self, num_features = 6, hidden_dim1=64, hidden_dim2=32):
        super(BiLSTMAutoencoder, self).__init__()

        self.num_features = num_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        
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

        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        reconstructed = self.forward(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        self.log('train_loss', loss)
        return loss

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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        reconstructed = self.forward(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        accuracy = torchmetrics.functional.accuracy(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        precision = torchmetrics.functional.precision(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        recall = torchmetrics.functional.recall(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        f1 = torchmetrics.functional.f1_score(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
        return loss
    
    def test_step(self, batch, batch_idx):
        reconstructed = self.forward(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        accuracy = torchmetrics.functional.accuracy(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        precision = torchmetrics.functional.precision(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        recall = torchmetrics.functional.recall(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        f1 = torchmetrics.functional.f1_score(reconstructed.round(), batch.round(), task='multiclass', num_classes=2)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        return loss

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
    epochs = 200
    
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
    
    wandb_logger = WandbLogger(project='bilstm_autoencoder_anomaly_detection')
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["lookback"] = lookback
    wandb_logger.experiment.config["epochs"] = epochs

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='weights/',
        filename='bilstm-autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        enable_version_counter=True
    )

    # Train the model
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Finish wandb run
    wandb.save('weights/*.ckpt')
    wandb.finish()
    
if __name__ == '__main__':
    main()