#!/usr/bin/env python3
"""
Training script for the Robot State Diffusion Model

This script handles training, validation, and evaluation of the diffusion model
on robotic state repair tasks using the collected simulation data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
import wandb
from dataclasses import dataclass, asdict
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_model import (
    RobotStateDiffusionTransformer, 
    ModelConfig, 
    create_diffusion_model,
    RobotType
)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Data
    data_path: str = "data/raw/medium_dataset_v1"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Model
    model_config: ModelConfig = None
    
    # Optimization
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    eval_every: int = 5
    
    # Monitoring
    use_wandb: bool = False
    project_name: str = "srdmfr-diffusion"
    
    # Device
    device: str = "auto"
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig(
                max_state_dim=72  # Unified state dimension (padded to max)
            )
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class RobotStateDataset(Dataset):
    """Dataset for robot state diffusion training"""
    
    def __init__(self, data_path: str, split: str = "train", train_split: float = 0.8, 
                 val_split: float = 0.1, max_sequence_length: int = 50):
        self.data_path = Path(data_path)
        self.split = split
        self.max_sequence_length = max_sequence_length
        
        # Load dataset files
        self.episode_files = sorted(list(self.data_path.glob("episode_*.h5")))
        
        if not self.episode_files:
            raise ValueError(f"No episode files found in {data_path}")
        
        # Split data
        n_files = len(self.episode_files)
        train_end = int(n_files * train_split)
        val_end = int(n_files * (train_split + val_split))
        
        if split == "train":
            self.episode_files = self.episode_files[:train_end]
        elif split == "val":
            self.episode_files = self.episode_files[train_end:val_end]
        elif split == "test":
            self.episode_files = self.episode_files[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Loaded {len(self.episode_files)} files for {split} split")
        
        # Robot type mapping
        self.robot_type_map = {
            RobotType.MANIPULATOR.value: 0,
            RobotType.MOBILE_ROBOT.value: 1
        }
    
    def __len__(self):
        return len(self.episode_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        episode_file = self.episode_files[idx]
        
        with h5py.File(episode_file, 'r') as f:
            # Combine all sensor data into unified state vectors
            healthy_states = self._combine_sensor_data(f['data']['states_healthy'])
            corrupted_states = self._combine_sensor_data(f['data']['states_corrupted'])
            
            # Load metadata from JSON string
            metadata_group = f['metadata']
            faults_data = json.loads(metadata_group['faults_injected'][()])
            
            # Extract robot type from filename or use default
            robot_type = "manipulator"  # Default for now
            if "manipulator" in str(episode_file):
                robot_type = "manipulator"
            elif "mobile" in str(episode_file):
                robot_type = "mobile_robot"
                
            # Convert to tensors
            healthy_states = torch.from_numpy(healthy_states).float()
            corrupted_states = torch.from_numpy(corrupted_states).float()
            robot_type_idx = torch.tensor(self.robot_type_map[robot_type], dtype=torch.long)
            
            # Use healthy states as target for training
            states = healthy_states
            
            # Pad or truncate sequences
            seq_len = min(len(states), self.max_sequence_length)
            if len(states) < self.max_sequence_length:
                # Pad with last state
                padding = states[-1:].repeat(self.max_sequence_length - len(states), 1)
                states = torch.cat([states, padding], dim=0)
            else:
                # Truncate
                states = states[:self.max_sequence_length]
            
            # Create mask for valid timesteps
            mask = torch.ones(self.max_sequence_length, dtype=torch.bool)
            if seq_len < self.max_sequence_length:
                mask[seq_len:] = False
            
            return {
                'states': states,  # [seq_len, state_dim]
                'robot_type': robot_type_idx,  # scalar
                'mask': mask,  # [seq_len]
                # 'metadata': {'robot_type': robot_type, 'faults': faults_data}  # Exclude for batching
            }
    
    def _combine_sensor_data(self, states_group) -> np.ndarray:
        """Combine all sensor data into a unified state vector"""
        all_sensors = []
        
        # Fixed dimensions for each sensor type (padded to max across all robot types)
        sensor_config = {
            'base_angular_velocity': 3,
            'base_linear_velocity': 3, 
            'base_orientation': 4,
            'base_position': 3,
            'battery_voltage': 1,
            'cpu_temperature': 1,
            'force_torque': 6,
            'imu_acceleration': 3,
            'imu_gyroscope': 3,
            'joint_positions': 15,  # Max joints (mobile robot has more)
            'joint_torques': 15,     # Max joints
            'joint_velocities': 15   # Max joints
        }
        
        for sensor, target_dim in sensor_config.items():
            if sensor in states_group:
                data = states_group[sensor][()]
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                # Pad or truncate to target dimension
                current_dim = data.shape[1]
                if current_dim < target_dim:
                    # Pad with zeros
                    padding = np.zeros((data.shape[0], target_dim - current_dim))
                    data = np.concatenate([data, padding], axis=1)
                elif current_dim > target_dim:
                    # Truncate (shouldn't happen with our config)
                    data = data[:, :target_dim]
                
                all_sensors.append(data)
            else:
                # Missing sensor - create zero data
                timesteps = len(states_group[list(states_group.keys())[0]])
                data = np.zeros((timesteps, target_dim))
                all_sensors.append(data)
        
        # Concatenate all sensor data
        combined = np.concatenate(all_sensors, axis=1)
        return combined


class DiffusionTrainer:
    """Trainer for the diffusion model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = create_diffusion_model(config.model_config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.1
        )
        
        # Initialize datasets
        self.train_dataset = RobotStateDataset(
            config.data_path, "train", config.train_split, config.val_split, 
            config.model_config.max_sequence_length
        )
        self.val_dataset = RobotStateDataset(
            config.data_path, "val", config.train_split, config.val_split,
            config.model_config.max_sequence_length
        )
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, 
            shuffle=True, num_workers=0, pin_memory=True  # Disable multiprocessing for debugging
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, 
            shuffle=False, num_workers=0, pin_memory=True  # Disable multiprocessing for debugging
        )
        
        # Initialize tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=asdict(config),
                name=f"diffusion-{int(time.time())}"
            )
            wandb.watch(self.model)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_denoising_loss = 0.0
        total_physics_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            states = batch['states'].to(self.device)  # [batch_size, seq_len, state_dim]
            robot_types = batch['robot_type'].to(self.device)  # [batch_size]
            mask = batch['mask'].to(self.device)  # [batch_size, seq_len]
            
            # Sample random timesteps
            batch_size = states.size(0)
            timesteps = torch.randint(
                0, self.config.model_config.num_diffusion_steps,
                (batch_size,), device=self.device
            )
            
            # Compute loss
            losses = self.model.get_loss(states, timesteps, robot_types, mask=mask)
            loss = losses['total_loss']
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_denoising_loss += losses['denoising_loss'].item()
            total_physics_loss += losses['physics_loss'].item()
            total_reconstruction_loss += losses['reconstruction_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_denoising_loss': total_denoising_loss / num_batches,
            'train_physics_loss': total_physics_loss / num_batches,
            'train_reconstruction_loss': total_reconstruction_loss / num_batches,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_denoising_loss = 0.0
        total_physics_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                states = batch['states'].to(self.device)
                robot_types = batch['robot_type'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Sample random timesteps
                batch_size = states.size(0)
                timesteps = torch.randint(
                    0, self.config.model_config.num_diffusion_steps,
                    (batch_size,), device=self.device
                )
                
                # Compute loss
                losses = self.model.get_loss(states, timesteps, robot_types, mask=mask)
                
                # Accumulate metrics
                total_loss += losses['total_loss'].item()
                total_denoising_loss += losses['denoising_loss'].item()
                total_physics_loss += losses['physics_loss'].item()
                total_reconstruction_loss += losses['reconstruction_loss'].item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_denoising_loss': total_denoising_loss / num_batches,
            'val_physics_loss': total_physics_loss / num_batches,
            'val_reconstruction_loss': total_reconstruction_loss / num_batches,
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        
        # Save best model
        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved new best model with validation loss: {metrics['val_loss']:.4f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.validate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Log metrics
                if self.config.use_wandb:
                    wandb.log(all_metrics, step=epoch)
                
                # Print metrics
                print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch + 1, all_metrics)
            
            # Update scheduler
            self.scheduler.step()
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train robot state diffusion model")
    parser.add_argument("--data_path", type=str, default="data/raw/medium_dataset_v1",
                        help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Checkpoint directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        device=args.device
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
