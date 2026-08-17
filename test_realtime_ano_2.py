"""
Professional-Grade Gait Anomaly Detection System
Advanced GUI | Multi-Algorithm Detection | Medical-Level Analysis
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠ PyTorch not installed. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠ Scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import zscore
import time

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """Manage system configuration"""
    
    def __init__(self, config_file='gait_config.json'):
        self.config_file = config_file
        self.default_config = {
            'video': {
                'resolution': [1280, 720],
                'fps': 30,
                'detection_confidence': 0.5,
                'tracking_confidence': 0.5
            },
            'detection': {
                'sequence_length': 30,
                'anomaly_threshold': 0.05,
                'sensitivity': 'medium',  # low, medium, high
                'min_confidence': 60.0,
                'streak_threshold': 5
            },
            'algorithms': {
                'autoencoder': True,
                'isolation_forest': True,
                'one_class_svm': False,
                'ensemble_voting': 'majority'  # majority, weighted
            },
            'visualization': {
                'show_skeleton': True,
                'show_angles': True,
                'show_elevation': True,
                'show_velocity': True,
                'graph_width': 1400,
                'graph_height': 900
            },
            'alerts': {
                'enable_sound': False,
                'enable_popup': True,
                'streak_alert': 5,
                'severity_alert': 'high'
            },
            'export': {
                'auto_save': True,
                'save_interval': 100,
                'export_format': 'csv',  # csv, json, excel
                'include_screenshots': False
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults
                    config = self.default_config.copy()
                    config.update(loaded)
                    return config
            except:
                pass
        return self.default_config.copy()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except:
            return False
    
    def get(self, category, key, default=None):
        """Get config value"""
        return self.config.get(category, {}).get(key, default)
    
    def set(self, category, key, value):
        """Set config value"""
        if category not in self.config:
            self.config[category] = {}
        self.config[category][key] = value
        self.save_config()


# ============================================================================
# ADVANCED VISUALIZATION SYSTEM
# ============================================================================

class AdvancedGraphVisualizer:
    """Professional multi-panel graph system"""
    
    def __init__(self, config_manager, width=1400, height=900):
        self.config = config_manager
        self.window_name = "Professional Gait Analysis Dashboard"
        self.width = width
        self.height = height
        
        # Data storage
        self.max_points = 300
        self.reconstruction_errors = deque(maxlen=self.max_points)
        self.anomaly_scores = deque(maxlen=self.max_points)
        self.confidence_scores = deque(maxlen=self.max_points)
        self.severity_levels = deque(maxlen=self.max_points)
        self.frame_numbers = deque(maxlen=self.max_points)
        
        # Biomechanical metrics
        self.stride_lengths = deque(maxlen=self.max_points)
        self.step_widths = deque(maxlen=self.max_points)
        self.knee_angles_left = deque(maxlen=self.max_points)
        self.knee_angles_right = deque(maxlen=self.max_points)
        self.cadence = deque(maxlen=self.max_points)
        
        # Threshold tracking
        self.threshold_values = deque(maxlen=self.max_points)
        
        # Statistics
        self.total_anomalies = 0
        self.total_frames = 0
        self.current_streak = 0
        self.max_streak = 0
        self.severity_counts = {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 0}
        
        # Algorithm results
        self.algo_results = {
            'autoencoder': deque(maxlen=self.max_points),
            'isolation_forest': deque(maxlen=self.max_points),
            'svm': deque(maxlen=self.max_points),
            'ensemble': deque(maxlen=self.max_points)
        }
        
        # Colors (BGR)
        self.colors = {
            'bg': (25, 25, 35),
            'grid': (60, 60, 70),
            'error': (255, 100, 100),
            'threshold': (0, 255, 255),
            'anomaly': (0, 0, 255),
            'normal': (0, 255, 0),
            'confidence': (255, 150, 0),
            'mild': (0, 255, 255),
            'moderate': (0, 165, 255),
            'severe': (0, 0, 255),
            'text': (255, 255, 255),
            'panel': (45, 45, 55)
        }
    
    def add_data_point(self, frame_num, error, threshold, is_anomaly, confidence, 
                       severity='normal', biomechanics=None, algo_votes=None):
        """Add comprehensive data point"""
        self.frame_numbers.append(frame_num)
        self.reconstruction_errors.append(error)
        self.threshold_values.append(threshold)
        self.confidence_scores.append(confidence)
        self.severity_levels.append(severity)
        
        # Calculate anomaly score (0-100)
        anomaly_score = (error / max(threshold, 0.001)) * 100
        self.anomaly_scores.append(min(anomaly_score, 200))
        
        # Biomechanics
        if biomechanics:
            self.stride_lengths.append(biomechanics.get('stride_length', 0))
            self.step_widths.append(biomechanics.get('step_width', 0))
            self.knee_angles_left.append(biomechanics.get('left_knee_angle', 0))
            self.knee_angles_right.append(biomechanics.get('right_knee_angle', 0))
            self.cadence.append(biomechanics.get('cadence', 0))
        
        # Algorithm votes
        if algo_votes:
            for algo, vote in algo_votes.items():
                if algo in self.algo_results:
                    self.algo_results[algo].append(1 if vote else 0)
        
        self.total_frames += 1
        if is_anomaly:
            self.total_anomalies += 1
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
            self.severity_counts[severity] = self.severity_counts.get(severity, 0) + 1
        else:
            self.current_streak = 0
    
    def draw_graph_panel(self, canvas, data, color, x, y, w, h, title, y_label, 
                         y_max=None, show_points=True, overlay_data=None):
        """Draw single graph panel"""
        if len(data) < 2:
            return
        
        # Panel background
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.colors['grid'], 2)
        
        # Title
        cv2.putText(canvas, title, (x+10, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Graph area
        margin = 40
        graph_x = x + margin
        graph_y = y + 40
        graph_w = w - 2*margin
        graph_h = h - 60
        
        # Grid
        for i in range(6):
            gy = graph_y + int((i/5) * graph_h)
            cv2.line(canvas, (graph_x, gy), (graph_x+graph_w, gy),
                    self.colors['grid'], 1)
        
        # Data
        if y_max is None:
            y_max = max(data) * 1.2 if max(data) > 0 else 1.0
        
        points = []
        for i, value in enumerate(data):
            px = graph_x + int((i / max(len(data)-1, 1)) * graph_w)
            py = graph_y + graph_h - int((value / max(y_max, 0.001)) * graph_h)
            py = max(graph_y, min(graph_y + graph_h, py))
            points.append((px, py))
        
        # Draw line
        for i in range(len(points)-1):
            cv2.line(canvas, points[i], points[i+1], color, 2, cv2.LINE_AA)
        
        # Draw points
        if show_points:
            for point in points[-20:]:
                cv2.circle(canvas, point, 3, color, -1)
        
        # Overlay data (e.g., threshold)
        if overlay_data and len(overlay_data) == len(data):
            overlay_points = []
            for i, value in enumerate(overlay_data):
                px = graph_x + int((i / max(len(overlay_data)-1, 1)) * graph_w)
                py = graph_y + graph_h - int((value / max(y_max, 0.001)) * graph_h)
                py = max(graph_y, min(graph_y + graph_h, py))
                overlay_points.append((px, py))
            
            for i in range(len(overlay_points)-1):
                cv2.line(canvas, overlay_points[i], overlay_points[i+1],
                        self.colors['threshold'], 2, cv2.LINE_AA)
        
        # Y-axis labels
        for i in range(6):
            gy = graph_y + int((i/5) * graph_h)
            value = y_max * (1 - i/5)
            text = f"{value:.2f}"
            cv2.putText(canvas, text, (x+5, gy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['text'], 1)
        
        # Y-axis title
        cv2.putText(canvas, y_label, (x+5, y+h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
    
    def draw_statistics_panel(self, canvas, x, y, w, h):
        """Draw comprehensive statistics panel"""
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.colors['grid'], 2)
        
        py = y + 25
        px = x + 15
        
        # Title
        cv2.putText(canvas, "DETECTION STATISTICS", (px, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['threshold'], 2)
        py += 35
        
        stats = [
            ("Frames Analyzed", str(self.total_frames)),
            ("Total Anomalies", str(self.total_anomalies)),
            ("Anomaly Rate", f"{(self.total_anomalies/max(self.total_frames,1)*100):.2f}%"),
            ("", ""),
            ("Current Streak", str(self.current_streak)),
            ("Max Streak", str(self.max_streak)),
            ("", ""),
            ("Severity Breakdown:", ""),
            ("  Mild", str(self.severity_counts.get('mild', 0))),
            ("  Moderate", str(self.severity_counts.get('moderate', 0))),
            ("  Severe", str(self.severity_counts.get('severe', 0))),
        ]
        
        for label, value in stats:
            if label:
                color = self.colors['text'] if value else self.colors['threshold']
                cv2.putText(canvas, f"{label}: {value}", (px, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            py += 22
    
    def draw_current_status(self, canvas, x, y, w, h):
        """Draw current status panel"""
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.colors['grid'], 2)
        
        py = y + 25
        px = x + 15
        
        cv2.putText(canvas, "CURRENT STATUS", (px, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['threshold'], 2)
        py += 35
        
        if len(self.reconstruction_errors) > 0:
            error = self.reconstruction_errors[-1]
            threshold = self.threshold_values[-1] if len(self.threshold_values) > 0 else 0
            confidence = self.confidence_scores[-1] if len(self.confidence_scores) > 0 else 0
            severity = self.severity_levels[-1] if len(self.severity_levels) > 0 else 'normal'
            
            # Status
            is_anomaly = error > threshold
            status = "ANOMALY DETECTED" if is_anomaly else "NORMAL GAIT"
            status_color = self.colors['anomaly'] if is_anomaly else self.colors['normal']
            
            cv2.putText(canvas, status, (px, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            py += 35
            
            # Metrics
            metrics = [
                (f"Error: {error:.6f}", self.colors['text']),
                (f"Threshold: {threshold:.6f}", self.colors['text']),
                (f"Confidence: {confidence:.1f}%", self.colors['confidence']),
                (f"Severity: {severity.upper()}", self.get_severity_color(severity))
            ]
            
            for text, color in metrics:
                cv2.putText(canvas, text, (px, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                py += 25
            
            # Confidence bar
            bar_w = w - 40
            bar_h = 20
            bar_x = px
            bar_y = py + 10
            
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                         (80, 80, 80), -1)
            
            fill = int((confidence / 100) * bar_w)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h),
                         self.colors['confidence'], -1)
            
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                         self.colors['text'], 2)
    
    def get_severity_color(self, severity):
        """Get color for severity level"""
        severity_colors = {
            'normal': self.colors['normal'],
            'mild': self.colors['mild'],
            'moderate': self.colors['moderate'],
            'severe': self.colors['severe']
        }
        return severity_colors.get(severity, self.colors['text'])
    
    def render(self):
        """Render complete dashboard"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.colors['bg']
        
        # Title
        cv2.putText(canvas, "PROFESSIONAL GAIT ANALYSIS DASHBOARD", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['threshold'], 2)
        
        if len(self.reconstruction_errors) < 2:
            cv2.putText(canvas, "Initializing analysis...", 
                       (self.width//2 - 150, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
            return canvas
        
        # Layout: 3x2 grid for graphs + 1 column for stats
        graph_w = 420
        graph_h = 250
        margin = 20
        
        # Row 1: Anomaly Score & Confidence
        self.draw_graph_panel(canvas, self.anomaly_scores, self.colors['error'],
                             margin, 70, graph_w, graph_h,
                             "Anomaly Score", "Score", 200, True)
        
        self.draw_graph_panel(canvas, self.confidence_scores, self.colors['confidence'],
                             margin*2 + graph_w, 70, graph_w, graph_h,
                             "Detection Confidence", "%", 100, True)
        
        # Row 2: Stride & Step Width
        self.draw_graph_panel(canvas, self.stride_lengths, (255, 200, 100),
                             margin, 70 + graph_h + margin, graph_w, graph_h,
                             "Stride Length", "Units", None, True)
        
        self.draw_graph_panel(canvas, self.step_widths, (200, 150, 255),
                             margin*2 + graph_w, 70 + graph_h + margin, graph_w, graph_h,
                             "Step Width", "Units", None, True)
        
        # Row 3: Knee Angles
        self.draw_graph_panel(canvas, self.knee_angles_left, (0, 255, 255),
                             margin, 70 + 2*(graph_h + margin), graph_w, graph_h,
                             "Left Knee Angle", "Degrees", 180, True)
        
        self.draw_graph_panel(canvas, self.knee_angles_right, (0, 255, 0),
                             margin*2 + graph_w, 70 + 2*(graph_h + margin), graph_w, graph_h,
                             "Right Knee Angle", "Degrees", 180, True)
        
        # Right column: Statistics & Status
        stats_x = margin*3 + graph_w*2
        stats_w = self.width - stats_x - margin
        
        self.draw_statistics_panel(canvas, stats_x, 70, stats_w, 400)
        self.draw_current_status(canvas, stats_x, 490, stats_w, 380)
        
        return canvas
    
    def show(self):
        """Display dashboard"""
        canvas = self.render()
        cv2.imshow(self.window_name, canvas)


# ============================================================================
# AUTOENCODER ENGINE
# ============================================================================

class AutoencoderEngine:
    """Autoencoder training and inference engine"""
    
    def __init__(self, input_dim=24, sequence_length=30, config=None):
        if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
            print("⚠ ML features disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.config = config
        
        # Model
        self.model = LegFocusedAutoencoder(input_dim, sequence_length).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.criterion = nn.MSELoss()
        
        # Scaler
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_fitted = False
        
        # Anomaly detection
        self.anomaly_threshold = 0.05
        self.baseline_errors = []
        self.reconstruction_errors = deque(maxlen=1000)
        
        # Training stats
        self.model_version = 0
        self.total_samples_trained = 0
        
        print(f"✓ AutoencoderEngine initialized on {self.device}")
    
    def train_on_data(self, data, feature_columns, epochs=15, batch_size=32):
        """Train autoencoder on normal gait data"""
        if not self.enabled or len(data) < 50:
            print("⚠ Insufficient data for training")
            return False
        
        try:
            self.feature_columns = feature_columns
            
            # Fit scaler
            if not self.is_fitted:
                self.scaler.fit(data)
                self.is_fitted = True
            
            # Normalize
            data_normalized = self.scaler.transform(data)
            
            # Create sequences
            sequences = []
            stride = max(1, self.sequence_length // 3)
            for i in range(0, len(data_normalized) - self.sequence_length + 1, stride):
                sequences.append(data_normalized[i:i+self.sequence_length])
            
            if len(sequences) < 20:
                print("⚠ Not enough sequences created")
                return False
            
            sequences = np.array(sequences)
            
            # Split train/val
            X_train, X_val = train_test_split(sequences, test_size=0.2, random_state=42)
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(X_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(X_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(X_train)),
                                     shuffle=True, drop_last=False, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(X_val)),
                                   shuffle=False, num_workers=0)
            
            print(f"  Training on {len(X_train)} sequences, validating on {len(X_val)}")
            
            # Training loop
            self.model.train()
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            for epoch in range(epochs):
                train_loss = 0
                train_batches = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    reconstructed, latent, attn = self.model(batch_x)
                    loss = self.criterion(reconstructed, batch_y)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                # Validation
                self.model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        reconstructed, _, _ = self.model(batch_x)
                        loss = self.criterion(reconstructed, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                
                self.model.train()
                
                if train_batches > 0 and val_batches > 0:
                    avg_train = train_loss / train_batches
                    avg_val = val_loss / val_batches
                    
                    self.scheduler.step(avg_val)
                    
                    if (epoch + 1) % 3 == 0:
                        print(f"    Epoch {epoch+1}/{epochs} - Train: {avg_train:.6f}, Val: {avg_val:.6f}")
                    
                    if avg_val < best_val_loss:
                        best_val_loss = avg_val
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            
            # Calculate baseline errors for threshold
            self.model.eval()
            baseline_errors = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    reconstructed, _, _ = self.model(batch_x)
                    
                    for i in range(batch_x.size(0)):
                        error = self.criterion(reconstructed[i:i+1], batch_y[i:i+1]).item()
                        baseline_errors.append(error)
            
            self.baseline_errors = baseline_errors
            
            # Set adaptive threshold
            mean_error = np.mean(baseline_errors)
            std_error = np.std(baseline_errors)
            self.anomaly_threshold = mean_error + (2.5 * std_error)
            
            if self.config:
                self.config.set('detection', 'anomaly_threshold', self.anomaly_threshold)
            
            self.total_samples_trained += len(X_train)
            self.model_version += 1
            
            print(f"  ✓ Training complete - Threshold: {self.anomaly_threshold:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_anomaly(self, sequence):
        """Detect anomaly in sequence"""
        if not self.enabled or not self.is_fitted:
            return False, 0.0, 0.0
        
        try:
            self.model.eval()
            
            # Normalize
            sequence_normalized = self.scaler.transform(sequence)
            x = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                reconstructed, latent, attn_weights = self.model(x)
                error = self.criterion(reconstructed, x).item()
            
            self.reconstruction_errors.append(error)
            
            # Adaptive threshold
            if len(self.reconstruction_errors) > 100:
                recent_errors = list(self.reconstruction_errors)[-200:]
                percentile_95 = np.percentile(recent_errors, 95)
                effective_threshold = max(self.anomaly_threshold, percentile_95)
            else:
                effective_threshold = self.anomaly_threshold
            
            is_anomaly = error > effective_threshold
            
            # Calculate confidence
            if is_anomaly:
                confidence = min((error / effective_threshold - 1.0) * 100, 100)
            else:
                confidence = max(100 - (error / effective_threshold) * 100, 0)
            
            return is_anomaly, error, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False, 0.0, 0.0
    
    def save_model(self, path):
        """Save model"""
        if not self.enabled:
            return False
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'anomaly_threshold': self.anomaly_threshold,
                'baseline_errors': self.baseline_errors,
                'model_version': self.model_version
            }, path)
            print(f"✓ Model saved: {path}")
            return True
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def load_model(self, path):
        """Load model"""
        if not self.enabled:
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.feature_columns = checkpoint['feature_columns']
            self.anomaly_threshold = checkpoint['anomaly_threshold']
            self.baseline_errors = checkpoint['baseline_errors']
            self.model_version = checkpoint.get('model_version', 0)
            self.is_fitted = True
            print(f"✓ Model loaded - Threshold: {self.anomaly_threshold:.6f}")
            return True
        except Exception as e:
            print(f"❌ Load error: {e}")
            return False


# ============================================================================
# ENSEMBLE ANOMALY DETECTOR
# ============================================================================

class EnsembleAnomalyDetector:
    """Multi-algorithm ensemble detector"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.enabled_algos = []
        
        # Initialize algorithms
        if SKLEARN_AVAILABLE:
            if config_manager.get('algorithms', 'isolation_forest', True):
                self.isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                self.enabled_algos.append('isolation_forest')
            
            if config_manager.get('algorithms', 'one_class_svm', False):
                self.one_class_svm = OneClassSVM(
                    kernel='rbf',
                    gamma='auto',
                    nu=0.1
                )
                self.enabled_algos.append('one_class_svm')
        
        self.is_fitted = {'isolation_forest': False, 'one_class_svm': False}
        print(f"✓ Ensemble detector initialized with: {self.enabled_algos}")
    
    def fit(self, data):
        """Train ensemble models"""
        if not SKLEARN_AVAILABLE or len(data) < 50:
            return
        
        try:
            if 'isolation_forest' in self.enabled_algos:
                self.isolation_forest.fit(data)
                self.is_fitted['isolation_forest'] = True
                print("✓ Isolation Forest trained")
            
            if 'one_class_svm' in self.enabled_algos:
                self.one_class_svm.fit(data)
                self.is_fitted['one_class_svm'] = True
                print("✓ One-Class SVM trained")
        
        except Exception as e:
            print(f"❌ Ensemble training error: {e}")
    
    def predict(self, sample):
        """Get ensemble prediction"""
        votes = {}
        
        try:
            if 'isolation_forest' in self.enabled_algos and self.is_fitted['isolation_forest']:
                pred = self.isolation_forest.predict([sample])[0]
                votes['isolation_forest'] = (pred == -1)
            
            if 'one_class_svm' in self.enabled_algos and self.is_fitted['one_class_svm']:
                pred = self.one_class_svm.predict([sample])[0]
                votes['svm'] = (pred == -1)
        
        except:
            pass
        
        return votes


# ============================================================================
# ENHANCED AUTOENCODER (Keeping previous implementation)
# ============================================================================

if TORCH_AVAILABLE:
    class LegFocusedAutoencoder(nn.Module):
        """Advanced autoencoder architecture"""
        def __init__(self, input_dim=24, sequence_length=30, latent_dim=16):
            super(LegFocusedAutoencoder, self).__init__()
            
            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.latent_dim = latent_dim
            
            # Encoder
            self.cnn_encoder = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Conv1d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
            
            self.lstm_encoder = nn.LSTM(
                input_size=64,
                hidden_size=latent_dim,
                num_layers=3,
                batch_first=True,
                dropout=0.3,
                bidirectional=True
            )
            
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim * 2,
                num_heads=4,
                dropout=0.2,
                batch_first=True
            )
            
            self.compress = nn.Linear(latent_dim * 2, latent_dim)
            
            # Decoder
            self.lstm_decoder = nn.LSTM(
                input_size=latent_dim,
                hidden_size=64,
                num_layers=3,
                batch_first=True,
                dropout=0.3
            )
            
            self.cnn_decoder = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Conv1d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Conv1d(64, input_dim, kernel_size=5, padding=2)
            )
        
        def forward(self, x):
            # CNN encoding
            x_cnn = x.permute(0, 2, 1)
            cnn_features = self.cnn_encoder(x_cnn)
            cnn_features = cnn_features.permute(0, 2, 1)
            
            # LSTM encoding
            lstm_out, _ = self.lstm_encoder(cnn_features)
            
            # Attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Compress
            latent = self.compress(attn_out[:, -1, :])
            
            # Decode
            latent_repeated = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
            lstm_decoded, _ = self.lstm_decoder(latent_repeated)
            
            lstm_decoded = lstm_decoded.permute(0, 2, 1)
            reconstructed = self.cnn_decoder(lstm_decoded)
            reconstructed = reconstructed.permute(0, 2, 1)
            
            return reconstructed, latent, attn_weights


# ============================================================================
# ADVANCED BIOMECHANICAL ANALYZER
# ============================================================================

class AdvancedBiomechanicalAnalyzer:
    """Professional-grade gait biomechanics analysis"""
    
    def __init__(self):
        self.previous_landmarks = None
        self.fps = 30
        self.feature_dim = 24  # Enhanced feature set
        
        # Temporal tracking
        self.left_heel_heights = deque(maxlen=120)
        self.right_heel_heights = deque(maxlen=120)
        self.velocity_history = deque(maxlen=60)
        
        # Gait cycle detection
        self.last_left_strike = None
        self.last_right_strike = None
        self.stride_times = deque(maxlen=20)
        
        # Filters
        self.b, self.a = butter(2, 0.1, btype='low')
    
    def calculate_distance(self, p1, p2):
        """3D Euclidean distance"""
        return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
    
    def calculate_2d_distance(self, p1, p2):
        """2D distance"""
        return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)
    
    def calculate_angle(self, p1, p2, p3):
        """3-point angle"""
        v1 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        v2 = np.array([p3.x-p2.x, p3.y-p2.y, p3.z-p2.z])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def calculate_velocity(self, current, previous):
        """Velocity calculation"""
        if previous is None:
            return 0.0
        displacement = self.calculate_distance(current, previous)
        return displacement * self.fps
    
    def detect_heel_strike(self, heel_height, heel_history):
        """Detect heel strike event"""
        if len(heel_history) < 5:
            return False
        
        recent = list(heel_history)[-5:]
        if heel_height == max(recent) and heel_height > np.mean(recent) + 0.02:
            return True
        return False
    
    def calculate_cadence(self):
        """Steps per minute"""
        if len(self.stride_times) < 2:
            return 0
        avg_stride_time = np.mean(self.stride_times)
        return (60.0 / avg_stride_time) if avg_stride_time > 0 else 0
    
    def analyze_comprehensive_gait(self, landmarks):
        """Extract 24 advanced biomechanical features"""
        try:
            # Landmark indices
            L_HIP, R_HIP = 23, 24
            L_KNEE, R_KNEE = 25, 26
            L_ANKLE, R_ANKLE = 27, 28
            L_HEEL, R_HEEL = 29, 30
            L_FOOT, R_FOOT = 31, 32
            
            features = []
            
            # 1-2: Stride metrics
            stride_length = self.calculate_2d_distance(landmarks[L_HEEL], landmarks[R_HEEL])
            step_width = abs(landmarks[L_ANKLE].x - landmarks[R_ANKLE].x)
            
            # 3-6: Hip angles
            left_hip_flex = self.calculate_angle(landmarks[11], landmarks[L_HIP], landmarks[L_KNEE])
            right_hip_flex = self.calculate_angle(landmarks[12], landmarks[R_HIP], landmarks[R_KNEE])
            left_hip_abd = abs(landmarks[L_HIP].x - landmarks[L_KNEE].x)
            right_hip_abd = abs(landmarks[R_HIP].x - landmarks[R_KNEE].x)
            
            # 7-8: Knee angles
            left_knee = 180 - self.calculate_angle(landmarks[L_HIP], landmarks[L_KNEE], landmarks[L_ANKLE])
            right_knee = 180 - self.calculate_angle(landmarks[R_HIP], landmarks[R_KNEE], landmarks[R_ANKLE])
            
            # 9-10: Ankle angles
            left_ankle = self.calculate_angle(landmarks[L_KNEE], landmarks[L_ANKLE], landmarks[L_HEEL])
            right_ankle = self.calculate_angle(landmarks[R_KNEE], landmarks[R_ANKLE], landmarks[R_HEEL])
            
            # 11-14: Elevation metrics
            left_heel_elev = landmarks[L_HEEL].y
            right_heel_elev = landmarks[R_HEEL].y
            left_toe_elev = landmarks[L_FOOT].y
            right_toe_elev = landmarks[R_FOOT].y
            
            self.left_heel_heights.append(left_heel_elev)
            self.right_heel_heights.append(right_heel_elev)
            
            # 15-18: Velocities
            left_knee_vel = 0.0
            right_knee_vel = 0.0
            left_ankle_vel = 0.0
            right_ankle_vel = 0.0
            
            if self.previous_landmarks:
                left_knee_vel = self.calculate_velocity(
                    landmarks[L_KNEE], self.previous_landmarks[L_KNEE]
                )
                right_knee_vel = self.calculate_velocity(
                    landmarks[R_KNEE], self.previous_landmarks[R_KNEE]
                )
                left_ankle_vel = self.calculate_velocity(
                    landmarks[L_ANKLE], self.previous_landmarks[L_ANKLE]
                )
                right_ankle_vel = self.calculate_velocity(
                    landmarks[R_ANKLE], self.previous_landmarks[R_ANKLE]
                )
            
            # 19: Symmetry index
            knee_symmetry = min(left_knee, right_knee) / (max(left_knee, right_knee) + 1e-8)
            
            # 20: Cadence
            cadence = self.calculate_cadence()
            
            # 21-22: Center of mass tracking
            com_x = (landmarks[L_HIP].x + landmarks[R_HIP].x) / 2
            com_y = (landmarks[L_HIP].y + landmarks[R_HIP].y) / 2
            
            # 23-24: Foot clearance
            left_clearance = abs(left_heel_elev - min(self.left_heel_heights)) if len(self.left_heel_heights) > 0 else 0
            right_clearance = abs(right_heel_elev - min(self.right_heel_heights)) if len(self.right_heel_heights) > 0 else 0
            
            features = [
                stride_length, step_width,
                left_hip_flex, right_hip_flex,
                left_hip_abd, right_hip_abd,
                left_knee, right_knee,
                left_ankle, right_ankle,
                left_heel_elev, right_heel_elev,
                left_toe_elev, right_toe_elev,
                left_knee_vel, right_knee_vel,
                left_ankle_vel, right_ankle_vel,
                knee_symmetry, cadence,
                com_x, com_y,
                left_clearance, right_clearance
            ]
            
            self.previous_landmarks = landmarks
            
            # Biomechanics dict for visualization
            biomechanics = {
                'stride_length': stride_length,
                'step_width': step_width,
                'left_knee_angle': left_knee,
                'right_knee_angle': right_knee,
                'cadence': cadence
            }
            
            return features, biomechanics
            
        except Exception as e:
            return None, None


# ============================================================================
# GUI CONTROL PANEL
# ============================================================================

class GUIControlPanel:
    """Tkinter-based control panel"""
    
    def __init__(self, config_manager, detector_callback):
        self.config = config_manager
        self.detector_callback = detector_callback
        
        self.root = tk.Tk()
        self.root.title("Gait Analysis Control Panel")
        self.root.geometry("400x700")
        self.root.resizable(False, False)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title = tk.Label(self.root, text="PROFESSIONAL GAIT ANALYZER",
                        font=("Arial", 14, "bold"), fg="blue")
        title.pack(pady=10)
        
        # Detection Settings
        settings_frame = ttk.LabelFrame(self.root, text="Detection Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Sensitivity
        ttk.Label(settings_frame, text="Sensitivity:").grid(row=0, column=0, sticky="w")
        self.sensitivity_var = tk.StringVar(value=self.config.get('detection', 'sensitivity', 'medium'))
        sensitivity_combo = ttk.Combobox(settings_frame, textvariable=self.sensitivity_var,
                                        values=['low', 'medium', 'high'], state='readonly')
        sensitivity_combo.grid(row=0, column=1, padx=5)
        sensitivity_combo.bind('<<ComboboxSelected>>', self.on_sensitivity_change)
        
        # Threshold
        ttk.Label(settings_frame, text="Threshold:").grid(row=1, column=0, sticky="w", pady=5)
        self.threshold_var = tk.DoubleVar(value=self.config.get('detection', 'anomaly_threshold', 0.05))
        threshold_scale = ttk.Scale(settings_frame, from_=0.01, to=0.2, 
                                   variable=self.threshold_var, orient="horizontal")
        threshold_scale.grid(row=1, column=1, padx=5)
        
        self.threshold_label = ttk.Label(settings_frame, text=f"{self.threshold_var.get():.3f}")
        self.threshold_label.grid(row=1, column=2)
        threshold_scale.config(command=self.on_threshold_change)
        
        # Visualization Options
        viz_frame = ttk.LabelFrame(self.root, text="Visualization", padding=10)
        viz_frame.pack(fill="x", padx=10, pady=5)
        
        self.show_skeleton_var = tk.BooleanVar(value=True)
        self.show_angles_var = tk.BooleanVar(value=True)
        self.show_velocity_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(viz_frame, text="Show Skeleton", 
                       variable=self.show_skeleton_var).pack(anchor="w")
        ttk.Checkbutton(viz_frame, text="Show Angles", 
                       variable=self.show_angles_var).pack(anchor="w")
        ttk.Checkbutton(viz_frame, text="Show Velocity", 
                       variable=self.show_velocity_var).pack(anchor="w")
        
        # Algorithm Selection
        algo_frame = ttk.LabelFrame(self.root, text="Detection Algorithms", padding=10)
        algo_frame.pack(fill="x", padx=10, pady=5)
        
        self.algo_autoencoder = tk.BooleanVar(value=True)
        self.algo_isolation = tk.BooleanVar(value=True)
        self.algo_svm = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(algo_frame, text="Deep Autoencoder", 
                       variable=self.algo_autoencoder).pack(anchor="w")
        ttk.Checkbutton(algo_frame, text="Isolation Forest", 
                       variable=self.algo_isolation).pack(anchor="w")
        ttk.Checkbutton(algo_frame, text="One-Class SVM", 
                       variable=self.algo_svm).pack(anchor="w")
        
        # Statistics Display
        stats_frame = ttk.LabelFrame(self.root, text="Live Statistics", padding=10)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=45, state='disabled')
        self.stats_text.pack()
        
        # Control Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Start Training", 
                  command=self.on_train_click).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save Config", 
                  command=self.on_save_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Export Data", 
                  command=self.on_export_click).pack(side="left", padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief="sunken", anchor="w")
        status_bar.pack(fill="x", side="bottom")
    
    def on_sensitivity_change(self, event):
        """Handle sensitivity change"""
        sensitivity = self.sensitivity_var.get()
        self.config.set('detection', 'sensitivity', sensitivity)
        
        # Adjust threshold based on sensitivity
        thresholds = {'low': 0.08, 'medium': 0.05, 'high': 0.03}
        new_threshold = thresholds.get(sensitivity, 0.05)
        self.threshold_var.set(new_threshold)
        self.config.set('detection', 'anomaly_threshold', new_threshold)
        
        if self.detector_callback:
            self.detector_callback('sensitivity', sensitivity)
    
    def on_threshold_change(self, value):
        """Handle threshold change"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.3f}")
        self.config.set('detection', 'anomaly_threshold', threshold)
        
        if self.detector_callback:
            self.detector_callback('threshold', threshold)
    
    def on_train_click(self):
        """Handle train button"""
        file_path = filedialog.askopenfilename(
            title="Select Training Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path and self.detector_callback:
            self.detector_callback('train', file_path)
            self.status_var.set(f"Training on: {os.path.basename(file_path)}")
    
    def on_save_config(self):
        """Save configuration"""
        if self.config.save_config():
            messagebox.showinfo("Success", "Configuration saved successfully!")
        else:
            messagebox.showerror("Error", "Failed to save configuration")
    
    def on_export_click(self):
        """Export data"""
        if self.detector_callback:
            self.detector_callback('export', None)
            messagebox.showinfo("Export", "Data export initiated")
    
    def update_statistics(self, stats_dict):
        """Update statistics display"""
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        for key, value in stats_dict.items():
            self.stats_text.insert(tk.END, f"{key}: {value}\n")
        
        self.stats_text.config(state='disabled')
    
    def run(self):
        """Start GUI main loop"""
        self.root.mainloop()


# ============================================================================
# MAIN DETECTION SYSTEM
# ============================================================================

class ProfessionalGaitDetector:
    """Professional gait analysis system"""
    
    def __init__(self, config_file='gait_config.json', baseline_csv=None, model_path=None):
        # Configuration
        self.config = ConfigManager(config_file)
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config.get('video', 'detection_confidence', 0.5),
            min_tracking_confidence=self.config.get('video', 'tracking_confidence', 0.5),
            model_complexity=2,
            smooth_landmarks=True
        )
        
        # Biomechanics analyzer
        self.analyzer = AdvancedBiomechanicalAnalyzer()
        
        # Visualization
        self.graph_viz = AdvancedGraphVisualizer(self.config)
        
        # Autoencoder Engine
        self.autoencoder = None
        self.autoencoder_enabled = False
        
        if TORCH_AVAILABLE and SKLEARN_AVAILABLE:
            self.autoencoder = AutoencoderEngine(
                input_dim=24,
                sequence_length=30,
                config=self.config
            )
            self.autoencoder_enabled = True
            
            # Load model if provided
            if model_path and os.path.exists(model_path):
                self.autoencoder.load_model(model_path)
                print(f"✓ Loaded model: {model_path}")
            
            # Train on baseline if provided
            if baseline_csv and os.path.exists(baseline_csv):
                print(f"📊 Training on baseline: {baseline_csv}")
                self.train_on_baseline(baseline_csv)
        
        # Ensemble detector
        self.ensemble = EnsembleAnomalyDetector(self.config)
        
        # Statistical baseline for ensemble
        self.baseline_features = []
        self.statistical_mean = None
        self.statistical_std = None
        
        # Data
        self.sequence_buffer = deque(maxlen=30)
        self.session_data = []
        self.frame_count = 0
        self.anomalies = []
        self.consecutive_anomalies = 0
        
        # Threading
        self.running = False
        self.paused = False
        
        print("✓ Professional Gait Detector initialized")
    
    def train_on_baseline(self, csv_path, epochs=15):
        """Train autoencoder on baseline normal gait data"""
        if not self.autoencoder_enabled:
            print("⚠ Autoencoder not available")
            return False
        
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} frames from baseline")
            
            # Get feature columns (skip frame, timestamp, etc.)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in ['frame', 'Unnamed: 0', 'timestamp']]
            
            if len(feature_cols) == 0:
                print("❌ No valid features found in CSV")
                return False
            
            # Ensure we have 24 features
            while len(feature_cols) < 24:
                feature_cols.append(f'pad_{len(feature_cols)}')
                df[f'pad_{len(feature_cols)-1}'] = 0.0
            
            feature_cols = feature_cols[:24]
            data = df[feature_cols].values
            
            # Clean data
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Smooth data
            if len(data) > 5:
                for i in range(data.shape[1]):
                    try:
                        data[:, i] = savgol_filter(data[:, i], min(11, len(data)//2*2-1), 2)
                    except:
                        pass
            
            # Store for statistical analysis
            self.baseline_features = data
            self.statistical_mean = np.mean(data, axis=0)
            self.statistical_std = np.std(data, axis=0)
            
            # Train autoencoder
            success = self.autoencoder.train_on_data(data, feature_cols, epochs=epochs)
            
            if success:
                # Train ensemble models
                print("  Training ensemble models...")
                self.ensemble.fit(data)
                print("✓ Baseline training complete")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_anomaly_comprehensive(self, features):
        """Comprehensive multi-algorithm anomaly detection"""
        
        # Default values
        is_anomaly = False
        error = 0.0
        confidence = 0.0
        algo_votes = {}
        
        # 1. Autoencoder detection (primary)
        if self.autoencoder_enabled and len(self.sequence_buffer) == 30:
            sequence = np.array(list(self.sequence_buffer))
            ae_anomaly, ae_error, ae_confidence = self.autoencoder.predict_anomaly(sequence)
            
            error = ae_error
            confidence = ae_confidence
            is_anomaly = ae_anomaly
            algo_votes['autoencoder'] = ae_anomaly
        
        # 2. Ensemble methods (if trained)
        if len(features) == 24 and self.baseline_features is not None:
            ensemble_votes = self.ensemble.predict(features)
            algo_votes.update(ensemble_votes)
            
            # 3. Statistical Z-score detection
            if self.statistical_mean is not None and self.statistical_std is not None:
                z_scores = np.abs((features - self.statistical_mean) / (self.statistical_std + 1e-8))
                max_z = np.max(z_scores)
                
                # Z-score threshold (3 sigma rule)
                z_anomaly = max_z > 3.0
                algo_votes['zscore'] = z_anomaly
        
        # 4. Ensemble voting
        if len(algo_votes) > 0:
            anomaly_votes = sum(algo_votes.values())
            total_votes = len(algo_votes)
            
            # Majority voting
            is_anomaly = anomaly_votes > (total_votes / 2)
            
            # Adjust confidence based on vote consensus
            vote_consensus = anomaly_votes / total_votes
            if is_anomaly:
                confidence = max(confidence, vote_consensus * 100)
            else:
                confidence = max(confidence, (1 - vote_consensus) * 100)
        
        return is_anomaly, error, confidence, algo_votes
    
    def classify_severity(self, error, threshold):
        """Classify anomaly severity"""
        ratio = error / max(threshold, 0.001)
        
        if ratio < 1.0:
            return 'normal'
        elif ratio < 1.5:
            return 'mild'
        elif ratio < 2.5:
            return 'moderate'
        else:
            return 'severe'
    
    def draw_skeleton_and_overlay(self, image, landmarks):
        """Draw skeleton and visual overlays"""
        h, w = image.shape[:2]
        
        # Draw leg skeleton
        if self.config.get('visualization', 'show_skeleton', True):
            leg_connections = [
                (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
                (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
                (23, 24), (11, 23), (12, 24)             # Hip connections
            ]
            
            for start_idx, end_idx in leg_connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    color = (0, 255, 255) if start_idx % 2 == 1 else (255, 255, 0)
                    cv2.line(image, start_point, end_point, color, 3, cv2.LINE_AA)
            
            # Draw joints
            key_joints = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            for idx in key_joints:
                if idx < len(landmarks):
                    joint = landmarks[idx]
                    point = (int(joint.x * w), int(joint.y * h))
                    color = (0, 255, 0) if idx % 2 == 0 else (0, 255, 255)
                    cv2.circle(image, point, 6, color, -1)
                    cv2.circle(image, point, 6, (255, 255, 255), 2)
        
        # Draw angles
        if self.config.get('visualization', 'show_angles', True):
            # Left knee
            left_knee = landmarks[25]
            left_knee_angle = 180 - self.analyzer.calculate_angle(
                landmarks[23], landmarks[25], landmarks[27]
            )
            pos = (int(left_knee.x * w) + 20, int(left_knee.y * h))
            cv2.putText(image, f"{left_knee_angle:.0f}°", pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Right knee
            right_knee = landmarks[26]
            right_knee_angle = 180 - self.analyzer.calculate_angle(
                landmarks[24], landmarks[26], landmarks[28]
            )
            pos = (int(right_knee.x * w) - 80, int(right_knee.y * h))
            cv2.putText(image, f"{right_knee_angle:.0f}°", pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image
    
    def draw_detection_overlay(self, image, is_anomaly, confidence, severity):
        """Draw detection status overlay"""
        h, w = image.shape[:2]
        
        # Status panel
        panel_h = 180
        panel_w = 400
        panel_x = w - panel_w - 10
        panel_y = 10
        
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (w - 10, panel_y + panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        cv2.rectangle(image, (panel_x, panel_y),
                     (w - 10, panel_y + panel_h), (0, 255, 255), 2)
        
        y = panel_y + 30
        
        # Title
        cv2.putText(image, "GAIT ANALYSIS", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 35
        
        # Status
        if is_anomaly:
            status_text = "⚠ ANOMALY DETECTED"
            status_color = (0, 0, 255)
        else:
            status_text = "✓ NORMAL GAIT"
            status_color = (0, 255, 0)
        
        cv2.putText(image, status_text, (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y += 30
        
        # Confidence
        cv2.putText(image, f"Confidence: {confidence:.1f}%", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        
        # Severity
        severity_colors = {
            'normal': (0, 255, 0),
            'mild': (0, 255, 255),
            'moderate': (0, 165, 255),
            'severe': (0, 0, 255)
        }
        severity_color = severity_colors.get(severity, (255, 255, 255))
        cv2.putText(image, f"Severity: {severity.upper()}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, severity_color, 2)
        y += 30
        
        # Confidence bar
        bar_w = panel_w - 40
        bar_h = 20
        bar_x = panel_x + 20
        bar_y = y
        
        cv2.rectangle(image, (bar_x, bar_y),
                     (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        
        fill = int((confidence / 100) * bar_w)
        bar_color = status_color
        cv2.rectangle(image, (bar_x, bar_y),
                     (bar_x + fill, bar_y + bar_h), bar_color, -1)
        
        cv2.rectangle(image, (bar_x, bar_y),
                     (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        
        # Anomaly border
        if is_anomaly and self.consecutive_anomalies >= 5:
            cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(image, "⚠ SUSTAINED ANOMALY", (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Frame counter
        cv2.putText(image, f"Frame: {self.frame_count} | Anomalies: {len(self.anomalies)}",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def gui_callback(self, action, value):
        """Handle GUI callbacks"""
        if action == 'threshold':
            # Update threshold
            pass
        elif action == 'train':
            # Queue training
            print(f"Training requested: {value}")
        elif action == 'export':
            self.export_session_data()
    
    def export_session_data(self):
        """Export comprehensive session data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'gait_analysis_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        if self.session_data:
            df = pd.DataFrame(self.session_data)
            csv_file = os.path.join(output_dir, f'gait_session_{timestamp}.csv')
            df.to_csv(csv_file, index=False)
            print(f"✓ Session data exported: {csv_file}")
        
        # Save anomalies
        if self.anomalies:
            json_file = os.path.join(output_dir, f'anomalies_{timestamp}.json')
            with open(json_file, 'w') as f:
                json.dump(self.anomalies, f, indent=2)
            print(f"✓ Anomalies exported: {json_file}")
        
        # Generate report
        self.generate_report(output_dir, timestamp)
    
    def generate_report(self, output_dir, timestamp):
        """Generate professional analysis report"""
        report_file = os.path.join(output_dir, f'report_{timestamp}.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PROFESSIONAL GAIT ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Analyzed: {self.frame_count}\n")
            f.write(f"Total Anomalies Detected: {len(self.anomalies)}\n")
            
            if self.frame_count > 0:
                anomaly_rate = (len(self.anomalies) / self.frame_count) * 100
                f.write(f"Anomaly Rate: {anomaly_rate:.2f}%\n\n")
            
            f.write("Severity Breakdown:\n")
            for severity, count in self.graph_viz.severity_counts.items():
                f.write(f"  {severity.capitalize()}: {count}\n")
            
            f.write(f"\nMax Consecutive Anomalies: {self.graph_viz.max_streak}\n")
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"✓ Report generated: {report_file}")
    
    def run(self, video_source=0):
        """Run the detection system with proper anomaly detection"""
        
        # Open video
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Could not open video source: {video_source}")
            return
        
        # Check if trained
        if not self.autoencoder_enabled:
            print("\n" + "="*70)
            print("⚠ WARNING: No ML models enabled!")
            print("  Install PyTorch and scikit-learn for full functionality")
            print("="*70 + "\n")
        elif not self.autoencoder.is_fitted:
            print("\n" + "="*70)
            print("⚠ WARNING: Model not trained!")
            print("  Provide baseline CSV with --baseline flag")
            print("  Or press 'T' during detection to train on live data")
            print("="*70 + "\n")
        
        print("\n" + "=" * 70)
        print("PROFESSIONAL GAIT ANALYSIS SYSTEM - RUNNING")
        print("=" * 70)
        print("✓ Main Window: Real-time detection with overlay")
        print("✓ Dashboard: Multi-panel analytics graphs")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  T - Train on current session data")
        print("  S - Save model")
        print("  R - Reset statistics")
        print("  Q - Quit")
        print("=" * 70 + "\n")
        
        self.running = True
        show_help = True
        
        try:
            while self.running and cap.isOpened():
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Video ended")
                        break
                    
                    # Pose detection
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = self.pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Extract biomechanical features
                        features, biomechanics = self.analyzer.analyze_comprehensive_gait(landmarks)
                        
                        if features and biomechanics:
                            # Add to buffer
                            self.sequence_buffer.append(features)
                            
                            # Save session data
                            self.session_data.append({
                                'frame': self.frame_count,
                                **{f'feature_{i}': v for i, v in enumerate(features)}
                            })
                            
                            # Detect anomalies
                            is_anomaly, error, confidence, algo_votes = self.detect_anomaly_comprehensive(features)
                            
                            # Get threshold
                            threshold = self.autoencoder.anomaly_threshold if self.autoencoder_enabled else 0.05
                            
                            # Classify severity
                            severity = self.classify_severity(error, threshold)
                            
                            # Track consecutive anomalies
                            if is_anomaly:
                                self.consecutive_anomalies += 1
                                self.anomalies.append({
                                    'frame': self.frame_count,
                                    'error': float(error),
                                    'confidence': float(confidence),
                                    'severity': severity,
                                    'algorithm_votes': algo_votes
                                })
                            else:
                                self.consecutive_anomalies = max(0, self.consecutive_anomalies - 1)
                            
                            # Update visualization
                            self.graph_viz.add_data_point(
                                self.frame_count, error, threshold, is_anomaly,
                                confidence, severity, biomechanics, algo_votes
                            )
                            
                            # Draw overlays
                            image = self.draw_skeleton_and_overlay(image, landmarks)
                            image = self.draw_detection_overlay(image, is_anomaly, confidence, severity)
                            
                            self.frame_count += 1
                    else:
                        # No pose detected
                        cv2.putText(image, "⚠ No pose detected - Ensure full body visible",
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Display main window
                    cv2.imshow('Professional Gait Detection', image)
                    
                    # Display dashboard
                    self.graph_viz.show()
                else:
                    # Paused
                    cv2.putText(frame, "⏸ PAUSED - Press SPACE to resume",
                               (frame.shape[1]//2 - 250, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    cv2.imshow('Professional Gait Detection', frame)
                    self.graph_viz.show()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"▶ {status}")
                elif key == ord('t'):
                    # Train on current session
                    if len(self.session_data) >= 100:
                        print("\n📊 Training on current session data...")
                        self.train_on_current_session()
                    else:
                        print(f"\n⚠ Need at least 100 frames, currently: {len(self.session_data)}")
                elif key == ord('s'):
                    # Save model
                    if self.autoencoder_enabled and self.autoencoder.is_fitted:
                        output_dir = 'gait_models'
                        os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = os.path.join(output_dir, f'gait_model_{timestamp}.pth')
                        if self.autoencoder.save_model(model_path):
                            print(f"\n💾 Model saved: {model_path}")
                    else:
                        print("\n⚠ No trained model to save")
                elif key == ord('r'):
                    # Reset statistics
                    self.consecutive_anomalies = 0
                    print("\n🔄 Statistics reset")
                elif key == ord('h'):
                    show_help = not show_help
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Generate final report
            print("\n" + "="*70)
            print("GENERATING SESSION REPORT...")
            print("="*70)
            self.export_session_data()
            
            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Total Frames Analyzed: {self.frame_count}")
            print(f"Total Anomalies Detected: {len(self.anomalies)}")
            if self.frame_count > 0:
                anomaly_rate = (len(self.anomalies) / self.frame_count) * 100
                print(f"Anomaly Rate: {anomaly_rate:.2f}%")
            print(f"Max Consecutive Anomalies: {self.graph_viz.max_streak}")
            print("\nSeverity Breakdown:")
            for severity, count in self.graph_viz.severity_counts.items():
                if count > 0:
                    print(f"  {severity.capitalize()}: {count}")
            print("="*70)
            print("✓ Session complete")
    
    def train_on_current_session(self):
        """Train on data collected during current session"""
        if len(self.session_data) < 100:
            print("⚠ Insufficient data")
            return
        
        # Save session data to temp CSV
        output_dir = 'gait_temp'
        os.makedirs(output_dir, exist_ok=True)
        temp_csv = os.path.join(output_dir, 'session_train.csv')
        
        df = pd.DataFrame(self.session_data)
        df.to_csv(temp_csv, index=False)
        
        # Train
        success = self.train_on_baseline(temp_csv, epochs=10)
        
        if success:
            print("✓ Training complete - Model updated")
        else:
            print("❌ Training failed")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Professional Gait Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with webcam
  python script.py --video 0
  
  # Run with video file and baseline training
  python script.py --video gait.mp4 --baseline normal_gait.csv
  
  # Run with pre-trained model
  python script.py --video 0 --model gait_model.pth
  
  # Train on baseline and save model
  python script.py --video 0 --baseline normal_walk.csv --save-model
        """
    )
    
    parser.add_argument('--video', type=str, default='0',
                       help='Video source (0=webcam, or path to video file)')
    parser.add_argument('--baseline', type=str, default=None,
                       help='CSV file with normal gait data for training')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pre-trained model (.pth file)')
    parser.add_argument('--config', type=str, default='gait_config.json',
                       help='Configuration file path')
    parser.add_argument('--save-model', action='store_true',
                       help='Auto-save trained model at end of session')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Training epochs (default: 15)')
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*70)
        print("PROFESSIONAL GAIT ANOMALY DETECTION SYSTEM")
        print("="*70)
        print("Initializing components...")
        print()
        
        # Check dependencies
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - ML features disabled")
            print("  Install: pip install torch")
        
        if not SKLEARN_AVAILABLE:
            print("⚠ Scikit-learn not available - Ensemble features disabled")
            print("  Install: pip install scikit-learn")
        
        if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
            response = input("\nContinue without ML features? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                exit(0)
        
        print()
        
        # Initialize detector
        video_source = 0 if args.video == '0' else args.video
        
        detector = ProfessionalGaitDetector(
            config_file=args.config,
            baseline_csv=args.baseline,
            model_path=args.model
        )
        
        # Run detection
        detector.run(video_source=video_source)
        
        # Save model if requested
        if args.save_model and detector.autoencoder_enabled and detector.autoencoder.is_fitted:
            output_dir = 'gait_models'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(output_dir, f'gait_model_{timestamp}.pth')
            detector.autoencoder.save_model(model_path)
            print(f"\n💾 Final model saved: {model_path}")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Program finished\n")
