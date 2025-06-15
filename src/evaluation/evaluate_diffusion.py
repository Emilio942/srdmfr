#!/usr/bin/env python3
"""
Evaluation script for the Robot State Diffusion Model

This script evaluates trained diffusion models on test data and provides
comprehensive metrics for robotic state repair performance.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_model import (
    RobotStateDiffusionTransformer, 
    ModelConfig, 
    create_diffusion_model,
    RobotType
)
from training.train_diffusion import RobotStateDataset, TrainingConfig


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    mse_per_sensor: Dict[str, float]
    mae_per_sensor: Dict[str, float]
    total_mse: float
    total_mae: float
    temporal_consistency: float
    physics_violation_rate: float
    inference_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'mse_per_sensor': self.mse_per_sensor,
            'mae_per_sensor': self.mae_per_sensor,
            'total_mse': self.total_mse,
            'total_mae': self.total_mae,
            'temporal_consistency': self.temporal_consistency,
            'physics_violation_rate': self.physics_violation_rate,
            'inference_time_ms': self.inference_time_ms
        }


class DiffusionEvaluator:
    """Evaluator for diffusion model performance"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model, self.config = self._load_model(model_path, config_path)
        self.model.eval()
        
        # Define sensor mapping for analysis
        self.sensor_config = {
            'base_angular_velocity': (0, 3),
            'base_linear_velocity': (3, 6), 
            'base_orientation': (6, 10),
            'base_position': (10, 13),
            'battery_voltage': (13, 14),
            'cpu_temperature': (14, 15),
            'force_torque': (15, 21),
            'imu_acceleration': (21, 24),
            'imu_gyroscope': (24, 27),
            'joint_positions': (27, 42),
            'joint_torques': (42, 57),
            'joint_velocities': (57, 72)
        }
    
    def _load_model(self, model_path: str, config_path: Optional[str] = None) -> Tuple[nn.Module, ModelConfig]:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict['model_config'])
        else:
            # Try to get config from checkpoint
            config_dict = checkpoint.get('config', {})
            model_config = config_dict.get('model_config', {})
            if isinstance(model_config, dict):
                config = ModelConfig(**model_config)
            else:
                # Use default config
                config = ModelConfig(max_state_dim=72)
        
        # Create and load model
        model = create_diffusion_model(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from {model_path}")
        print(f"Model config: {config}")
        
        return model, config
    
    def evaluate_dataset(self, data_path: str, num_inference_steps: int = 50) -> EvaluationMetrics:
        """Evaluate model on test dataset"""
        # Load test dataset
        test_dataset = RobotStateDataset(data_path, "test", max_sequence_length=self.config.max_sequence_length)
        
        if len(test_dataset) == 0:
            print("Warning: No test data found, using validation split")
            test_dataset = RobotStateDataset(data_path, "val", max_sequence_length=self.config.max_sequence_length)
        
        print(f"Evaluating on {len(test_dataset)} test samples")
        
        all_predictions = []
        all_ground_truth = []
        all_inference_times = []
        
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
                sample = test_dataset[i]
                states = sample['states'].unsqueeze(0).to(self.device)  # Add batch dim
                robot_type = sample['robot_type'].unsqueeze(0).to(self.device)
                mask = sample['mask'].unsqueeze(0).to(self.device)
                
                # Add noise to create corrupted states
                noise = torch.randn_like(states) * 0.1  # 10% noise
                corrupted_states = states + noise
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Generate repaired states
                repaired_states = self.model.sample(
                    shape=states.shape,
                    robot_types=robot_type,
                    num_inference_steps=num_inference_steps,
                    device=self.device
                )
                
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                
                all_predictions.append(repaired_states.cpu().numpy())
                all_ground_truth.append(states.cpu().numpy())
                all_inference_times.append(inference_time)
        
        # Convert to numpy arrays
        predictions = np.concatenate(all_predictions, axis=0)  # [N, seq_len, state_dim]
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, ground_truth, all_inference_times)
        
        return metrics
    
    def _compute_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                        inference_times: List[float]) -> EvaluationMetrics:
        """Compute comprehensive evaluation metrics"""
        
        # Per-sensor metrics
        mse_per_sensor = {}
        mae_per_sensor = {}
        
        for sensor_name, (start_idx, end_idx) in self.sensor_config.items():
            pred_sensor = predictions[:, :, start_idx:end_idx]
            gt_sensor = ground_truth[:, :, start_idx:end_idx]
            
            mse_per_sensor[sensor_name] = mean_squared_error(
                gt_sensor.reshape(-1), pred_sensor.reshape(-1)
            )
            mae_per_sensor[sensor_name] = mean_absolute_error(
                gt_sensor.reshape(-1), pred_sensor.reshape(-1)
            )
        
        # Total metrics
        total_mse = mean_squared_error(ground_truth.reshape(-1), predictions.reshape(-1))
        total_mae = mean_absolute_error(ground_truth.reshape(-1), predictions.reshape(-1))
        
        # Temporal consistency (how much predictions change between timesteps)
        pred_diff = np.diff(predictions, axis=1)
        gt_diff = np.diff(ground_truth, axis=1)
        temporal_consistency = 1.0 - mean_squared_error(gt_diff.reshape(-1), pred_diff.reshape(-1))
        
        # Physics violation rate (states outside reasonable bounds)
        physics_violations = np.sum(np.abs(predictions) > 2.0) / predictions.size
        
        # Average inference time
        avg_inference_time = np.mean(inference_times)
        
        return EvaluationMetrics(
            mse_per_sensor=mse_per_sensor,
            mae_per_sensor=mae_per_sensor,
            total_mse=total_mse,
            total_mae=total_mae,
            temporal_consistency=temporal_consistency,
            physics_violation_rate=physics_violations,
            inference_time_ms=avg_inference_time
        )
    
    def generate_evaluation_report(self, metrics: EvaluationMetrics, output_dir: str):
        """Generate comprehensive evaluation report with visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        with open(output_path / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Create visualizations
        self._create_sensor_performance_plot(metrics, output_path)
        self._create_overall_metrics_plot(metrics, output_path)
        
        # Generate text report
        self._create_text_report(metrics, output_path)
        
        print(f"Evaluation report saved to {output_path}")
    
    def _create_sensor_performance_plot(self, metrics: EvaluationMetrics, output_path: Path):
        """Create sensor-wise performance visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MSE per sensor
        sensors = list(metrics.mse_per_sensor.keys())
        mse_values = list(metrics.mse_per_sensor.values())
        
        ax1.bar(sensors, mse_values)
        ax1.set_title('MSE per Sensor')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_xticks(range(len(sensors)))
        ax1.set_xticklabels(sensors, rotation=45, ha='right')
        
        # MAE per sensor
        mae_values = list(metrics.mae_per_sensor.values())
        
        ax2.bar(sensors, mae_values)
        ax2.set_title('MAE per Sensor')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_xticks(range(len(sensors)))
        ax2.set_xticklabels(sensors, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / "sensor_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_overall_metrics_plot(self, metrics: EvaluationMetrics, output_path: Path):
        """Create overall performance metrics visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall MSE and MAE
        ax1.bar(['MSE', 'MAE'], [metrics.total_mse, metrics.total_mae])
        ax1.set_title('Overall Error Metrics')
        ax1.set_ylabel('Error Value')
        
        # Temporal consistency
        ax2.bar(['Temporal Consistency'], [metrics.temporal_consistency])
        ax2.set_title('Temporal Consistency')
        ax2.set_ylabel('Consistency Score')
        ax2.set_ylim([0, 1])
        
        # Physics violation rate
        ax3.bar(['Physics Violations'], [metrics.physics_violation_rate * 100])
        ax3.set_title('Physics Violation Rate')
        ax3.set_ylabel('Percentage (%)')
        
        # Inference time
        ax4.bar(['Inference Time'], [metrics.inference_time_ms])
        ax4.set_title('Average Inference Time')
        ax4.set_ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig(output_path / "overall_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_text_report(self, metrics: EvaluationMetrics, output_path: Path):
        """Create detailed text report"""
        report = f"""
# Diffusion Model Evaluation Report

## Overall Performance
- **Total MSE**: {metrics.total_mse:.6f}
- **Total MAE**: {metrics.total_mae:.6f}
- **Temporal Consistency**: {metrics.temporal_consistency:.4f}
- **Physics Violation Rate**: {metrics.physics_violation_rate * 100:.2f}%
- **Average Inference Time**: {metrics.inference_time_ms:.2f} ms

## Per-Sensor Performance

### Mean Squared Error (MSE)
"""
        
        for sensor, mse in metrics.mse_per_sensor.items():
            report += f"- **{sensor}**: {mse:.6f}\n"
        
        report += "\n### Mean Absolute Error (MAE)\n"
        
        for sensor, mae in metrics.mae_per_sensor.items():
            report += f"- **{sensor}**: {mae:.6f}\n"
        
        report += f"""

## Performance Analysis

### Best Performing Sensors
"""
        
        # Find best and worst sensors
        best_mse = min(metrics.mse_per_sensor.items(), key=lambda x: x[1])
        worst_mse = max(metrics.mse_per_sensor.items(), key=lambda x: x[1])
        
        report += f"- **Lowest MSE**: {best_mse[0]} ({best_mse[1]:.6f})\n"
        report += f"- **Highest MSE**: {worst_mse[0]} ({worst_mse[1]:.6f})\n"
        
        # Performance summary
        avg_mse = np.mean(list(metrics.mse_per_sensor.values()))
        report += f"\n### Summary\n"
        report += f"- **Average sensor MSE**: {avg_mse:.6f}\n"
        report += f"- **Performance ratio (best/worst)**: {best_mse[1]/worst_mse[1]:.3f}\n"
        
        if metrics.inference_time_ms < 100:
            report += "- **Real-time capability**: ✅ Suitable for real-time applications\n"
        else:
            report += "- **Real-time capability**: ⚠️ May be too slow for real-time applications\n"
        
        with open(output_path / "evaluation_report.md", 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/raw/medium_dataset_v1",
                        help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Output directory for evaluation results")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model config file")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DiffusionEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # Run evaluation
    print("Starting evaluation...")
    metrics = evaluator.evaluate_dataset(
        data_path=args.data_path,
        num_inference_steps=args.num_inference_steps
    )
    
    # Generate report
    evaluator.generate_evaluation_report(metrics, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total MSE: {metrics.total_mse:.6f}")
    print(f"Total MAE: {metrics.total_mae:.6f}")
    print(f"Temporal Consistency: {metrics.temporal_consistency:.4f}")
    print(f"Physics Violations: {metrics.physics_violation_rate*100:.2f}%")
    print(f"Inference Time: {metrics.inference_time_ms:.2f} ms")
    print(f"Detailed report saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
