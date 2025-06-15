#!/usr/bin/env python3
"""
Physics-Focused Training Script for SRDMFR
This script trains the diffusion model with very strong emphasis on physics compliance.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.train_diffusion import DiffusionTrainer, TrainingConfig
from models.diffusion_model import ModelConfig

def create_physics_focused_config():
    """Create configuration with extreme physics focus"""
    model_config = ModelConfig(
        model_dim=256,
        num_layers=8,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_state_dim=72,
        max_sequence_length=50,
        num_diffusion_steps=1000,
        noise_schedule='cosine',
        use_linear_attention=True,
        use_gradient_checkpointing=True,
        mixed_precision=True
    )
    
    training_config = TrainingConfig(
        batch_size=4,
        num_epochs=50,
        learning_rate=5e-5,  # Lower learning rate for stability
        weight_decay=1e-4,   # Higher weight decay for regularization
        gradient_clip_norm=0.5,  # Stricter gradient clipping
        eval_every=5,
        save_every=10,
        early_stopping_patience=20,
        model_config=model_config
    )
    
    return training_config

def main():
    parser = argparse.ArgumentParser(description="Physics-focused training for SRDMFR")
    parser.add_argument("--data_path", type=str, default="data/raw/medium_dataset_v1",
                        help="Path to dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="hyperparameter_experiments/physics_extreme",
                        help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    
    args = parser.parse_args()
    
    # Create physics-focused config
    config = create_physics_focused_config()
    config.num_epochs = args.epochs
    config.data_path = args.data_path
    config.checkpoint_dir = args.checkpoint_dir
    
    print("Starting physics-focused training with extreme physics weight...")
    print(f"Data path: {config.data_path}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    
    # Initialize trainer
    trainer = DiffusionTrainer(config)
    
    # Override physics weight in the model's forward pass
    # We'll do this by monkey-patching the model
    original_forward = trainer.model.forward
    
    def physics_focused_forward(self, noisy_states, timesteps, robot_types, clean_states=None):
        """Modified forward with extreme physics focus"""
        losses = original_forward(noisy_states, timesteps, robot_types, clean_states)
        
        # Dramatically increase physics weight during training
        if clean_states is not None:
            # Recompute total loss with extreme physics focus
            losses['total_loss'] = (
                losses['denoising_loss'] + 
                5.0 * losses['physics_loss'] +  # Extreme physics weight!
                0.1 * losses['reconstruction_loss']
            )
        
        return losses
    
    # Apply the monkey patch
    trainer.model.forward = lambda *args, **kwargs: physics_focused_forward(trainer.model, *args, **kwargs)
    
    # Train the model
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save final model
        final_path = Path(config.checkpoint_dir) / "physics_extreme_final.pt"
        trainer.save_checkpoint(final_path, is_best=True)
        print(f"Final model saved to: {final_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
