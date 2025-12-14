"""
Quick script to recreate and save the scaler if you already trained your model
but forgot to save the scaler.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load your ORIGINAL training data (the same CSV you used for training)
df = pd.read_csv('komputer_tf.csv')

# Select features (same as training)
features = ['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']
data = df[features].values

# Split data: 70% train (same as training script)
train_size = int(0.7 * len(data))
train_data = data[:train_size]

# Create and fit scaler on training data
scaler = StandardScaler()
scaler.fit(train_data)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ Scaler recreated and saved as 'scaler.pkl'")
print(f"  - Fitted on {len(train_data)} training samples")
print(f"  - Features: {features}")
print(f"\nScaler statistics:")
print(f"  Mean: {scaler.mean_}")
print(f"  Std:  {scaler.scale_}")