#!/usr/bin/env python3
"""
Baseline Models for SRDMFR Comparison
Implements classical approaches for robot state estimation and repair.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import time

class BaselineModel(ABC):
    """Abstract base class for baseline models"""
    
    @abstractmethod
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Train the baseline model"""
        pass
    
    @abstractmethod
    def predict(self, X_corrupted: torch.Tensor) -> torch.Tensor:
        """Predict clean states from corrupted states"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name for reporting"""
        pass

class KalmanFilterBaseline(BaselineModel):
    """Kalman Filter-based state estimation baseline"""
    
    def __init__(self, state_dim: int, process_noise: float = 0.1, observation_noise: float = 0.1):
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        
        # Initialize Kalman filter parameters
        self.A = torch.eye(state_dim)  # State transition matrix
        self.H = torch.eye(state_dim)  # Observation matrix
        self.Q = torch.eye(state_dim) * process_noise  # Process noise covariance
        self.R = torch.eye(state_dim) * observation_noise  # Observation noise covariance
        self.P = torch.eye(state_dim)  # Error covariance matrix
        self.x = torch.zeros(state_dim)  # Initial state estimate
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Estimate Kalman filter parameters from training data"""
        # Simple approach: estimate noise parameters from data
        batch_size, seq_len, _ = X_train.shape
        
        # Estimate process noise from temporal differences
        if seq_len > 1:
            temporal_diffs = y_train[:, 1:] - y_train[:, :-1]
            process_var = torch.var(temporal_diffs, dim=(0, 1))
            self.Q = torch.diag(process_var.clamp(min=1e-6))
        
        # Estimate observation noise from difference between corrupted and clean
        obs_noise = torch.var(X_train - y_train, dim=(0, 1))
        self.R = torch.diag(obs_noise.clamp(min=1e-6))
    
    def predict(self, X_corrupted: torch.Tensor) -> torch.Tensor:
        """Apply Kalman filtering to corrupted sequences"""
        batch_size, seq_len, state_dim = X_corrupted.shape
        predictions = torch.zeros_like(X_corrupted)
        
        for b in range(batch_size):
            # Reset for each sequence
            x = torch.zeros(state_dim)
            P = torch.eye(state_dim)
            
            for t in range(seq_len):
                # Predict step
                x = self.A @ x
                P = self.A @ P @ self.A.T + self.Q
                
                # Update step
                z = X_corrupted[b, t]  # Observation
                S = self.H @ P @ self.H.T + self.R
                K = P @ self.H.T @ torch.inverse(S)
                
                x = x + K @ (z - self.H @ x)
                P = (torch.eye(state_dim) - K @ self.H) @ P
                
                predictions[b, t] = x
        
        return predictions
    
    def get_name(self) -> str:
        return "Kalman Filter"

class AutoencoderBaseline(BaselineModel):
    """Autoencoder-based reconstruction baseline"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        encoder_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Build decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            out_dim = hidden_dims[i - 1] if i > 0 else state_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i > 0 else nn.Identity(),
                nn.Dropout(0.1) if i > 0 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.model = nn.Sequential(self.encoder, self.decoder)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Train the autoencoder"""
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Flatten sequences for autoencoder training
        batch_size, seq_len, state_dim = X_train.shape
        X_flat = X_train.view(-1, state_dim)
        y_flat = y_train.view(-1, state_dim)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        num_epochs = 100
        batch_size_train = 256
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_flat), batch_size_train):
                batch_X = X_flat[i:i+batch_size_train]
                batch_y = y_flat[i:i+batch_size_train]
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_X)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 20 == 0:
                print(f"Autoencoder Epoch {epoch}/{num_epochs}, Loss: {epoch_loss/num_batches:.6f}")
    
    def predict(self, X_corrupted: torch.Tensor) -> torch.Tensor:
        """Reconstruct corrupted states"""
        X_corrupted = X_corrupted.to(self.device)
        batch_size, seq_len, state_dim = X_corrupted.shape
        
        # Flatten, predict, and reshape
        X_flat = X_corrupted.view(-1, state_dim)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed_flat = self.model(X_flat)
        
        return reconstructed_flat.view(batch_size, seq_len, state_dim)
    
    def get_name(self) -> str:
        return "Autoencoder"

class LSTMBaseline(BaselineModel):
    """LSTM-based sequence model baseline"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.output_layer = nn.Linear(hidden_dim, state_dim)
        self.model = nn.Sequential(self.lstm, self.output_layer)
        
        # Create a simple wrapper to handle LSTM output
        class LSTMWrapper(nn.Module):
            def __init__(self, lstm, output_layer):
                super().__init__()
                self.lstm = lstm
                self.output_layer = output_layer
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.output_layer(lstm_out)
        
        self.model = LSTMWrapper(self.lstm, self.output_layer)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Train the LSTM model"""
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        num_epochs = 50
        batch_size = min(32, X_train.size(0))
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                print(f"LSTM Epoch {epoch}/{num_epochs}, Loss: {epoch_loss/num_batches:.6f}")
    
    def predict(self, X_corrupted: torch.Tensor) -> torch.Tensor:
        """Predict clean sequences from corrupted sequences"""
        X_corrupted = X_corrupted.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_corrupted)
        
        return predictions
    
    def get_name(self) -> str:
        return "LSTM"

class InterpolationBaseline(BaselineModel):
    """Simple interpolation baseline"""
    
    def __init__(self, method: str = 'linear'):
        self.method = method
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """No training needed for interpolation"""
        pass
    
    def predict(self, X_corrupted: torch.Tensor) -> torch.Tensor:
        """Simple interpolation between valid points"""
        batch_size, seq_len, state_dim = X_corrupted.shape
        predictions = X_corrupted.clone()
        
        # For simplicity, just apply moving average smoothing
        window_size = 3
        for b in range(batch_size):
            for d in range(state_dim):
                # Apply simple moving average
                signal = X_corrupted[b, :, d]
                smoothed = torch.zeros_like(signal)
                
                for t in range(seq_len):
                    start_idx = max(0, t - window_size // 2)
                    end_idx = min(seq_len, t + window_size // 2 + 1)
                    smoothed[t] = torch.mean(signal[start_idx:end_idx])
                
                predictions[b, :, d] = smoothed
        
        return predictions
    
    def get_name(self) -> str:
        return "Interpolation"

def create_baseline_models(state_dim: int) -> List[BaselineModel]:
    """Create all baseline models for comparison"""
    return [
        InterpolationBaseline(),
        KalmanFilterBaseline(state_dim),
        AutoencoderBaseline(state_dim),
        LSTMBaseline(state_dim)
    ]

def evaluate_baseline_model(model: BaselineModel, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
    """Evaluate a baseline model and return metrics"""
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Calculate metrics
    mse = torch.mean((predictions - y_test) ** 2).item()
    mae = torch.mean(torch.abs(predictions - y_test)).item()
    
    # Temporal consistency
    if y_test.size(1) > 1:
        pred_diff = predictions[:, 1:] - predictions[:, :-1]
        true_diff = y_test[:, 1:] - y_test[:, :-1]
        temporal_consistency = -torch.mean((pred_diff - true_diff) ** 2).item()
    else:
        temporal_consistency = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'temporal_consistency': temporal_consistency,
        'inference_time_ms': inference_time / X_test.size(0),  # Per sample
        'model_name': model.get_name()
    }

if __name__ == "__main__":
    # Test the baseline models
    state_dim = 72
    seq_len = 50
    batch_size = 10
    
    # Generate dummy data
    X_train = torch.randn(batch_size, seq_len, state_dim)
    y_train = X_train + 0.1 * torch.randn_like(X_train)  # Clean version
    X_test = torch.randn(5, seq_len, state_dim)
    y_test = X_test + 0.1 * torch.randn_like(X_test)
    
    # Test all baselines
    models = create_baseline_models(state_dim)
    
    for model in models:
        print(f"\nTesting {model.get_name()}...")
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_baseline_model(model, X_test, y_test)
            print(f"Results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
                  f"Inference Time={metrics['inference_time_ms']:.2f}ms")
        except Exception as e:
            print(f"Error with {model.get_name()}: {e}")
