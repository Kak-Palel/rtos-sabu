import matplotlib.pyplot as plt
import numpy as np

class RealtimePlotter:
    """
    Real-time plotting with matplotlib animation.
    Only for debugging purposes, for production, use grafana. 
    """
    def __init__(self, detector, max_points=100):
        """
        Args:
            detector: RealtimeAnomalyDetector instance
            max_points: maximum number of points to display
        """
        self.detector = detector
        self.max_points = max_points
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # Top: Reconstruction error plot
        self.ax_error = self.fig.add_subplot(gs[0, :])
        
        # Middle: 6 feature plots (2 rows x 3 columns)
        self.ax_features = []
        for i in range(2):
            for j in range(3):
                ax = self.fig.add_subplot(gs[i+1, j])
                self.ax_features.append(ax)
        
        # Last row: Latest values display
        self.ax_status = self.fig.add_subplot(gs[3, :])
        self.ax_status.axis('off')
        
        # Initialize plots
        self.init_plots()
        
    def init_plots(self):
        """Initialize all plot elements."""
        # Error plot
        self.line_error, = self.ax_error.plot([], [], 'b-', linewidth=2, label='Reconstruction Error')
        self.threshold_line = self.ax_error.axhline(
            y=self.detector.threshold, color='r', linestyle='--', 
            linewidth=2, label=f'Threshold ({self.detector.threshold:.4f})'
        )
        self.scatter_anomaly = self.ax_error.scatter([], [], color='red', s=100, 
                                                      zorder=5, label='Anomaly', marker='X')
        self.ax_error.set_xlabel('Time', fontsize=11, fontweight='bold')
        self.ax_error.set_ylabel('Reconstruction Error', fontsize=11, fontweight='bold')
        self.ax_error.set_title('Real-time Anomaly Detection', fontsize=13, fontweight='bold')
        self.ax_error.legend(loc='upper left', fontsize=10)
        self.ax_error.grid(True, alpha=0.3)
        
        # Feature plots
        self.lines_original = []
        self.lines_reconstructed = []
        for i, (ax, name) in enumerate(zip(self.ax_features, self.detector.feature_names)):
            line_orig, = ax.plot([], [], 'b-', linewidth=2, label='Original', alpha=0.7)
            line_recon, = ax.plot([], [], 'r--', linewidth=2, label='Reconstructed', alpha=0.7)
            self.lines_original.append(line_orig)
            self.lines_reconstructed.append(line_recon)
            
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
    def update(self):
        """Update all plots with current data."""
        if len(self.detector.time_history) == 0:
            return
        
        # Get data (show only last max_points)
        start_idx = max(0, len(self.detector.time_history) - self.max_points)
        times = self.detector.time_history[start_idx:]
        errors = self.detector.error_history[start_idx:]
        predictions = self.detector.prediction_history[start_idx:]
        
        # Update error plot
        self.line_error.set_data(times, errors)
        
        # Update anomaly markers
        anomaly_indices = [i for i, p in enumerate(predictions) if p == 1]
        if anomaly_indices:
            anomaly_times = [times[i] for i in anomaly_indices]
            anomaly_errors = [errors[i] for i in anomaly_indices]
            self.scatter_anomaly.set_offsets(np.c_[anomaly_times, anomaly_errors])
        else:
            self.scatter_anomaly.set_offsets(np.empty((0, 2)))
        
        self.ax_error.relim()
        self.ax_error.autoscale_view()
        
        # Update feature plots
        for i, name in enumerate(self.detector.feature_names):
            original_vals = self.detector.original_history[name][start_idx:]
            reconstructed_vals = self.detector.reconstructed_history[name][start_idx:]
            
            self.lines_original[i].set_data(times, original_vals)
            self.lines_reconstructed[i].set_data(times, reconstructed_vals)
            
            self.ax_features[i].relim()
            self.ax_features[i].autoscale_view()
        
        # Update status text
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        if len(self.detector.time_history) > 0:
            latest_error = self.detector.error_history[-1]
            latest_prediction = self.detector.prediction_history[-1]
            status_color = 'red' if latest_prediction == 1 else 'green'
            status_text = 'ANOMALY DETECTED!' if latest_prediction == 1 else 'Normal'
            
            # Count total anomalies
            total_anomalies = sum(self.detector.prediction_history)
            total_points = len(self.detector.prediction_history)
            anomaly_rate = (total_anomalies / total_points) * 100 if total_points > 0 else 0
            
            status_str = f"""
            Latest Status: {status_text} | Error: {latest_error:.6f} | Threshold: {self.detector.threshold:.6f}
            Total Anomalies: {total_anomalies}/{total_points} ({anomaly_rate:.1f}%) | Points Processed: {total_points}
            """
            
            self.ax_status.text(0.5, 0.5, status_str, 
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=12, fontweight='bold',
                              color=status_color,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.canvas.draw_idle()