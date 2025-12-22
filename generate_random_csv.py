"""
Generate realistic PZEM sensor data with anomalies for testing.
Creates a CSV file with voltage, current, power, energy, frequency, and power_factor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_pzem_data(num_samples=1000, anomaly_ratio=0.05, seed=42):
    """
    Generate realistic PZEM sensor data with anomalies.
    
    Args:
        num_samples: number of data points to generate
        anomaly_ratio: percentage of anomalies (0.05 = 5%)
        seed: random seed for reproducibility
    
    Returns:
        DataFrame with generated data
    """
    np.random.seed(seed)
    
    # Number of anomalies to inject
    num_anomalies = int(num_samples * anomaly_ratio)
    
    # Initialize data dictionary
    data = {
        'voltage': [],
        'current': [],
        'power': [],
        'energy': [],
        'frequency': [],
        'power_factor': []
    }
    
    # Generate normal data first
    for i in range(num_samples):
        # Normal operating ranges for AC power (adjust based on your system)
        voltage = np.random.normal(220, 5)  # 220V ± 5V (AC voltage)
        current = np.random.normal(5, 0.5)  # 5A ± 0.5A
        frequency = np.random.normal(50, 0.1)  # 50Hz ± 0.1Hz
        power_factor = np.random.uniform(0.85, 0.99)  # Typical PF range
        
        # Power calculation: P = V * I * PF
        power = voltage * current * power_factor
        
        # Energy accumulation (cumulative kWh)
        if i == 0:
            energy = 0
        else:
            # Add power consumed in this timestep (assuming 10-minute intervals like the paper)
            energy = data['energy'][-1] + (power / 1000) * (10/60)  # Convert to kWh
        
        data['voltage'].append(voltage)
        data['current'].append(current)
        data['power'].append(power)
        data['energy'].append(energy)
        data['frequency'].append(frequency)
        data['power_factor'].append(power_factor)
    
    # Inject anomalies at random positions
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['voltage_spike', 'current_surge', 
                                         'frequency_drift', 'power_drop', 
                                         'low_pf', 'combined'])
        
        if anomaly_type == 'voltage_spike':
            # Voltage spike or drop
            data['voltage'][idx] *= np.random.choice([0.7, 1.5])  # ±30% or +50%
            
        elif anomaly_type == 'current_surge':
            # Current surge
            data['current'][idx] *= np.random.uniform(2, 3)  # 2-3x normal
            
        elif anomaly_type == 'frequency_drift':
            # Frequency drift (grid instability)
            data['frequency'][idx] += np.random.choice([-2, 2])  # ±2Hz
            
        elif anomaly_type == 'power_drop':
            # Sudden power drop
            data['power'][idx] *= np.random.uniform(0.3, 0.6)  # 30-60% drop
            
        elif anomaly_type == 'low_pf':
            # Low power factor (inefficient)
            data['power_factor'][idx] = np.random.uniform(0.5, 0.7)
            
        elif anomaly_type == 'combined':
            # Multiple issues at once
            data['voltage'][idx] *= np.random.uniform(0.8, 1.3)
            data['current'][idx] *= np.random.uniform(1.5, 2.5)
            data['frequency'][idx] += np.random.uniform(-1, 1)
            data['power_factor'][idx] = np.random.uniform(0.6, 0.8)
        
        # Recalculate power after anomaly injection
        data['power'][idx] = (data['voltage'][idx] * 
                             data['current'][idx] * 
                             data['power_factor'][idx])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add timestamp column (optional)
    df.insert(0, 'timestamp', pd.date_range(start='2024-01-01', periods=num_samples, freq='10min'))
    
    # Add label column (0 = normal, 1 = anomaly)
    df['is_anomaly'] = 0
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    
    return df, anomaly_indices


def plot_generated_data(df, anomaly_indices, save_path='generated_data_plot.png'):
    """
    Visualize the generated data with anomalies highlighted.
    """
    features = ['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Plot normal data
        ax.plot(df.index, df[feature], 'b-', alpha=0.6, linewidth=1, label='Normal')
        
        # Highlight anomalies
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, df.loc[anomaly_indices, feature], 
                      color='red', s=50, zorder=5, label='Anomaly', marker='x')
        
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{save_path}'")
    plt.show()


def main():
    """
    Main function to generate and save PZEM sensor data.
    """
    print("="*70)
    print("PZEM SENSOR DATA GENERATOR")
    print("="*70)
    
    # Configuration
    NUM_SAMPLES = 1000       # Total number of data points
    ANOMALY_RATIO = 0.1     # 10% anomalies
    OUTPUT_FILE = 'anomaly_data.csv'
    
    print(f"\nConfiguration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Anomaly ratio: {ANOMALY_RATIO*100:.1f}%")
    print(f"  - Output file: {OUTPUT_FILE}")
    
    # Generate data
    print(f"\nGenerating data...")
    df, anomaly_indices = generate_pzem_data(
        num_samples=NUM_SAMPLES,
        anomaly_ratio=ANOMALY_RATIO,
        seed=42
    )
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Data saved to '{OUTPUT_FILE}'")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("DATA STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"Normal samples: {len(df[df['is_anomaly'] == 0])}")
    print(f"Anomalous samples: {len(df[df['is_anomaly'] == 1])}")
    print(f"Anomaly percentage: {(len(anomaly_indices)/len(df))*100:.2f}%")
    
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS")
    print(f"{'='*70}")
    print(df[['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']].describe())
    
    # Plot the data
    print(f"\nGenerating visualization...")
    plot_generated_data(df, anomaly_indices)
    
    print(f"\n{'='*70}")
    print("✓ DONE! You can now use this CSV for training or testing.")
    print(f"{'='*70}")
    
    # Show first few rows
    print(f"\nFirst 10 rows of generated data:")
    print(df.head(10))
    
    # Show some anomalies
    if len(anomaly_indices) > 0:
        print(f"\nSample anomalies (first 5):")
        print(df.loc[anomaly_indices[:5]])


if __name__ == '__main__':
    main()