#!/usr/bin/env python3
"""
Detailed Physics Analysis Script for SRDMFR
Analyzes physics violations in detail to understand the issues.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.diffusion_model import create_diffusion_model, ModelConfig
from training.train_diffusion import RobotStateDataset
import h5py

def analyze_physics_violations(states, detailed=True):
    """
    Detailed analysis of physics violations in robot states
    
    Args:
        states: Robot states tensor [batch, seq_len, state_dim]
        detailed: Whether to return detailed breakdown
    
    Returns:
        Dictionary of violation statistics
    """
    batch_size, seq_len, state_dim = states.shape
    violations = {}
    total_violations = 0
    
    # Joint position violations (dimensions 0-11, assuming 12 joints)
    joint_positions = states[:, :, :12]
    joint_pos_violations = torch.sum(torch.abs(joint_positions) > 3.14)
    violations['joint_positions'] = {
        'count': int(joint_pos_violations),
        'total_elements': joint_positions.numel(),
        'percentage': float(joint_pos_violations) / joint_positions.numel() * 100
    }
    total_violations += joint_pos_violations
    
    # Joint velocity violations (dimensions 12-23)
    if state_dim > 23:
        joint_velocities = states[:, :, 12:24]
        joint_vel_violations = torch.sum(torch.abs(joint_velocities) > 10.0)
        violations['joint_velocities'] = {
            'count': int(joint_vel_violations),
            'total_elements': joint_velocities.numel(),
            'percentage': float(joint_vel_violations) / joint_velocities.numel() * 100
        }
        total_violations += joint_vel_violations
    
    # IMU acceleration violations (dimensions 48-50)
    if state_dim > 50:
        imu_accel = states[:, :, 48:51]
        imu_accel_violations = torch.sum(torch.abs(imu_accel) > 150.0)
        violations['imu_acceleration'] = {
            'count': int(imu_accel_violations),
            'total_elements': imu_accel.numel(),
            'percentage': float(imu_accel_violations) / imu_accel.numel() * 100
        }
        total_violations += imu_accel_violations
    
    # IMU gyroscope violations (dimensions 51-53)
    if state_dim > 53:
        imu_gyro = states[:, :, 51:54]
        imu_gyro_violations = torch.sum(torch.abs(imu_gyro) > 34.9)
        violations['imu_gyroscope'] = {
            'count': int(imu_gyro_violations),
            'total_elements': imu_gyro.numel(),
            'percentage': float(imu_gyro_violations) / imu_gyro.numel() * 100
        }
        total_violations += imu_gyro_violations
    
    # Battery voltage violations (assuming last dimensions include battery)
    if state_dim > 60:
        battery = states[:, :, 60:61]
        battery_violations = torch.sum((battery < 10.0) | (battery > 25.0))
        violations['battery_voltage'] = {
            'count': int(battery_violations),
            'total_elements': battery.numel(),
            'percentage': float(battery_violations) / battery.numel() * 100
        }
        total_violations += battery_violations
    
    # Overall statistics
    total_elements = states.numel()
    violations['overall'] = {
        'total_violations': int(total_violations),
        'total_elements': total_elements,
        'percentage': float(total_violations) / total_elements * 100
    }
    
    return violations

def analyze_dataset_physics(data_path):
    """Analyze physics violations in the original dataset"""
    print(f"Analyzing dataset at: {data_path}")
    
    dataset = RobotStateDataset(data_path, split='train')
    
    # Sample some data
    all_violations = []
    for i in range(min(10, len(dataset))):
        data = dataset[i]
        states = data['states'].unsqueeze(0)  # Add batch dimension
        violations = analyze_physics_violations(states)
        all_violations.append(violations['overall']['percentage'])
        
        if i == 0:  # Print detailed analysis for first sample
            print(f"\nDetailed analysis for sample {i}:")
            for component, stats in violations.items():
                if component != 'overall':
                    print(f"  {component}: {stats['count']}/{stats['total_elements']} violations ({stats['percentage']:.2f}%)")
    
    avg_violations = np.mean(all_violations)
    print(f"\nDataset physics violation statistics:")
    print(f"  Average violation rate across samples: {avg_violations:.2f}%")
    print(f"  Min violation rate: {min(all_violations):.2f}%")
    print(f"  Max violation rate: {max(all_violations):.2f}%")
    
    return avg_violations

def analyze_model_physics(model_path, data_path):
    """Analyze physics violations in model predictions"""
    print(f"\nAnalyzing model at: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config_dict = checkpoint.get('config', {}).get('model_config', {})
    if isinstance(model_config_dict, dict):
        model_config = ModelConfig(**model_config_dict)
    else:
        model_config = ModelConfig()
    
    model = create_diffusion_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = RobotStateDataset(data_path, split='val')
    
    all_violations = []
    all_reconstructions = []
    
    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            states = data['states'].unsqueeze(0)
            robot_types = data['robot_type'].unsqueeze(0)
            
            # Add noise and denoise
            timesteps = torch.randint(0, model_config.num_diffusion_steps, (1,))
            noise = torch.randn_like(states)
            noisy_states = model.noise_scheduler.add_noise(states, noise, timesteps)
            
            # Predict clean states
            with torch.no_grad():
                predicted_noise = model(noisy_states, timesteps, robot_types)
                reconstructed = model.noise_scheduler.remove_noise(noisy_states, predicted_noise, timesteps)
            
            # Analyze violations in reconstructed states
            violations = analyze_physics_violations(reconstructed)
            all_violations.append(violations['overall']['percentage'])
            
            # Store reconstruction for further analysis
            all_reconstructions.append(reconstructed.cpu().numpy())
            
            if i == 0:  # Print detailed analysis for first sample
                print(f"\nDetailed model analysis for sample {i}:")
                print(f"  Original violations: {analyze_physics_violations(states)['overall']['percentage']:.2f}%")
                print(f"  Reconstructed violations: {violations['overall']['percentage']:.2f}%")
                for component, stats in violations.items():
                    if component != 'overall':
                        print(f"    {component}: {stats['count']}/{stats['total_elements']} violations ({stats['percentage']:.2f}%)")
    
    avg_violations = np.mean(all_violations)
    print(f"\nModel physics violation statistics:")
    print(f"  Average violation rate: {avg_violations:.2f}%")
    print(f"  Min violation rate: {min(all_violations):.2f}%")
    print(f"  Max violation rate: {max(all_violations):.2f}%")
    
    return avg_violations, all_reconstructions

def plot_state_distributions(reconstructions, output_dir):
    """Plot distributions of different state components"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate all reconstructions
    all_states = np.concatenate(reconstructions, axis=0)
    batch_size, seq_len, state_dim = all_states.shape
    
    # Reshape to (total_samples, state_dim)
    flattened_states = all_states.reshape(-1, state_dim)
    
    # Plot distributions for key components
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Joint positions (first 12 dimensions)
    joint_pos = flattened_states[:, :12].flatten()
    axes[0, 0].hist(joint_pos, bins=50, alpha=0.7, label='Joint Positions')
    axes[0, 0].axvline(-3.14, color='r', linestyle='--', label='Lower limit')
    axes[0, 0].axvline(3.14, color='r', linestyle='--', label='Upper limit')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Joint Position Distribution')
    axes[0, 0].legend()
    
    # Joint velocities (dimensions 12-23)
    if state_dim > 23:
        joint_vel = flattened_states[:, 12:24].flatten()
        axes[0, 1].hist(joint_vel, bins=50, alpha=0.7, label='Joint Velocities')
        axes[0, 1].axvline(-10.0, color='r', linestyle='--', label='Lower limit')
        axes[0, 1].axvline(10.0, color='r', linestyle='--', label='Upper limit')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Joint Velocity Distribution')
        axes[0, 1].legend()
    
    # IMU acceleration (dimensions 48-50)
    if state_dim > 50:
        imu_accel = flattened_states[:, 48:51].flatten()
        axes[1, 0].hist(imu_accel, bins=50, alpha=0.7, label='IMU Acceleration')
        axes[1, 0].axvline(-150.0, color='r', linestyle='--', label='Lower limit')
        axes[1, 0].axvline(150.0, color='r', linestyle='--', label='Upper limit')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('IMU Acceleration Distribution')
        axes[1, 0].legend()
    
    # Overall distribution
    all_values = flattened_states.flatten()
    axes[1, 1].hist(all_values, bins=100, alpha=0.7, label='All Values')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Overall State Distribution')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'state_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"State distribution plots saved to {output_dir}/state_distributions.png")

def main():
    parser = argparse.ArgumentParser(description="Detailed physics analysis for SRDMFR")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--data_path", type=str, default="data/raw/medium_dataset_v1",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="physics_analysis",
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    print("SRDMFR Physics Violation Analysis")
    print("=" * 50)
    
    # Analyze dataset physics
    dataset_violations = analyze_dataset_physics(args.data_path)
    
    # Analyze model physics
    model_violations, reconstructions = analyze_model_physics(args.model_path, args.data_path)
    
    # Plot distributions
    plot_state_distributions(reconstructions, args.output_dir)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Dataset average violations: {dataset_violations:.2f}%")
    print(f"Model average violations: {model_violations:.2f}%")
    
    if model_violations > dataset_violations:
        print("⚠️  Model is producing MORE physics violations than the dataset!")
        print("   Recommendations:")
        print("   - Increase physics loss weight further")
        print("   - Use more constrained architecture")
        print("   - Add explicit physics constraint layers")
    else:
        print("✅ Model is producing FEWER physics violations than the dataset")
        print("   This is expected behavior")
    
    # Save detailed results
    results = {
        'dataset_violations': float(dataset_violations),
        'model_violations': float(model_violations),
        'improvement': float(dataset_violations - model_violations),
        'model_path': args.model_path,
        'data_path': args.data_path
    }
    
    output_path = Path(args.output_dir) / 'physics_analysis_results.json'
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
