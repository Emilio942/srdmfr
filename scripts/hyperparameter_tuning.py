#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for SRDMFR Diffusion Model
Implements systematic grid search and random search for optimal hyperparameters.
"""

import argparse
import json
import os
import sys
import time
import itertools
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.train_diffusion import DiffusionTrainer, ModelConfig
from data_collection.data_pipeline import RobotDataset
from evaluation.evaluate_diffusion import DiffusionEvaluator


class HyperparameterTuner:
    """Hyperparameter tuning for diffusion model"""
    
    def __init__(self, data_path: str, output_dir: str, num_epochs: int = 20):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_score = float('inf')
        
    def define_search_space(self) -> Dict[str, List[Any]]:
        """Define hyperparameter search space"""
        search_space = {
            # Model architecture
            'model_dim': [128, 256, 512],
            'num_layers': [4, 6, 8, 12],
            'num_heads': [4, 8, 16],
            'ff_dim': [512, 1024, 2048],
            'dropout': [0.1, 0.2, 0.3],
            
            # Training parameters
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [8, 16, 32],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            
            # Diffusion parameters
            'num_diffusion_steps': [100, 500, 1000],
            'noise_schedule': ['linear', 'cosine'],
            
            # Loss weights
            'reconstruction_weight': [0.5, 1.0, 2.0],
            'physics_weight': [0.1, 0.5, 1.0],
            'denoising_weight': [1.0, 2.0, 5.0],
            
            # Optimization
            'use_linear_attention': [True, False],
            'gradient_clip': [0.5, 1.0, 2.0],
        }
        return search_space
    
    def sample_random_config(self, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Sample random configuration from search space"""
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        
        # Ensure consistency
        if config['model_dim'] % config['num_heads'] != 0:
            config['model_dim'] = config['num_heads'] * (config['model_dim'] // config['num_heads'])
        
        return config
    
    def evaluate_config(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single hyperparameter configuration"""
        print(f"\\nEvaluating config: {config}")
        
        try:
            # Create model config
            model_config = ModelConfig(
                model_dim=config['model_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                ff_dim=config['ff_dim'],
                dropout=config['dropout'],
                num_diffusion_steps=config['num_diffusion_steps'],
                noise_schedule=config['noise_schedule'],
                use_linear_attention=config['use_linear_attention'],
                mixed_precision=True,
                use_gradient_checkpointing=True
            )
            
            # Setup dataset
            dataset = RobotDataset(self.data_path, split='train')
            dataloader = DataLoader(
                dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                collate_fn=dataset.collate_fn
            )
            
            val_dataset = RobotDataset(self.data_path, split='val')
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                collate_fn=val_dataset.collate_fn
            )
            
            # Setup trainer
            trainer = DiffusionTrainer(
                model_config=model_config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Update trainer parameters
            trainer.learning_rate = config['learning_rate']
            trainer.weight_decay = config['weight_decay']
            trainer.reconstruction_weight = config['reconstruction_weight']
            trainer.physics_weight = config['physics_weight']
            trainer.denoising_weight = config['denoising_weight']
            trainer.gradient_clip = config['gradient_clip']
            
            # Create unique experiment directory
            exp_name = f"exp_{len(self.results):03d}"
            exp_dir = self.output_dir / exp_name
            
            # Train model
            start_time = time.time()
            trainer.train(
                train_dataloader=dataloader,
                val_dataloader=val_dataloader,
                num_epochs=self.num_epochs,
                checkpoint_dir=str(exp_dir / 'checkpoints'),
                save_every=self.num_epochs  # Only save final
            )
            training_time = time.time() - start_time
            
            # Get final metrics
            final_metrics = trainer.get_metrics()
            
            # Quick evaluation
            evaluator = DiffusionEvaluator(
                model_path=str(exp_dir / 'checkpoints' / 'best.pt'),
                device=trainer.device
            )
            
            # Evaluate on small validation set
            val_metrics = evaluator.evaluate_dataset(
                self.data_path, 
                split='val',
                num_samples=10  # Quick evaluation
            )
            
            # Combine metrics
            results = {
                'val_loss': final_metrics.get('val_loss', float('inf')),
                'train_loss': final_metrics.get('train_loss', float('inf')),
                'val_mse': val_metrics.get('mse', float('inf')),
                'val_mae': val_metrics.get('mae', float('inf')),
                'physics_violations': val_metrics.get('physics_violation_rate', 1.0),
                'training_time': training_time,
                'model_params': trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else 0
            }
            
            # Calculate composite score (lower is better)
            score = (
                results['val_loss'] * 0.3 +
                results['val_mse'] * 0.3 +
                results['physics_violations'] * 0.4
            )
            results['composite_score'] = score
            
            # Save detailed results
            config_with_results = {**config, **results}
            with open(exp_dir / 'results.json', 'w') as f:
                json.dump(config_with_results, f, indent=2)
            
            print(f"Config score: {score:.4f}")
            return results
            
        except Exception as e:
            print(f"Error evaluating config: {e}")
            return {
                'val_loss': float('inf'),
                'train_loss': float('inf'),
                'val_mse': float('inf'),
                'val_mae': float('inf'),
                'physics_violations': 1.0,
                'training_time': 0.0,
                'model_params': 0,
                'composite_score': float('inf'),
                'error': str(e)
            }
    
    def run_random_search(self, num_trials: int = 20) -> Dict[str, Any]:
        """Run random search hyperparameter tuning"""
        print(f"Starting random search with {num_trials} trials...")
        
        search_space = self.define_search_space()
        
        for trial in range(num_trials):
            print(f"\\n=== Trial {trial + 1}/{num_trials} ===")
            
            # Sample random configuration
            config = self.sample_random_config(search_space)
            
            # Evaluate configuration
            results = self.evaluate_config(config)
            
            # Store results
            trial_result = {
                'trial': trial,
                'config': config,
                'results': results,
                'timestamp': time.time()
            }
            self.results.append(trial_result)
            
            # Update best configuration
            score = results['composite_score']
            if score < self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                print(f"New best score: {score:.4f}")
            
            # Save intermediate results
            self.save_results()
        
        return self.best_config
    
    def run_grid_search(self, param_subset: Dict[str, List[Any]] = None) -> Dict[str, Any]:
        """Run grid search on subset of parameters"""
        if param_subset is None:
            # Default small grid for quick testing
            param_subset = {
                'learning_rate': [1e-4, 5e-4],
                'batch_size': [16, 32],
                'model_dim': [256, 512],
                'num_layers': [6, 8],
                'reconstruction_weight': [1.0, 2.0],
                'physics_weight': [0.5, 1.0]
            }
        
        print(f"Starting grid search...")
        
        # Generate all combinations
        param_names = list(param_subset.keys())
        param_values = list(param_subset.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Total combinations: {len(combinations)}")
        
        search_space = self.define_search_space()
        base_config = self.sample_random_config(search_space)  # Base configuration
        
        for i, combination in enumerate(combinations):
            print(f"\\n=== Grid Search {i + 1}/{len(combinations)} ===")
            
            # Create configuration
            config = base_config.copy()
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            # Evaluate configuration
            results = self.evaluate_config(config)
            
            # Store results
            trial_result = {
                'trial': f'grid_{i}',
                'config': config,
                'results': results,
                'timestamp': time.time()
            }
            self.results.append(trial_result)
            
            # Update best configuration
            score = results['composite_score']
            if score < self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                print(f"New best score: {score:.4f}")
            
            # Save intermediate results
            self.save_results()
        
        return self.best_config
    
    def save_results(self):
        """Save all results to file"""
        results_file = self.output_dir / 'hyperparameter_results.json'
        
        summary = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'total_trials': len(self.results),
            'all_results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def analyze_results(self):
        """Analyze and summarize tuning results"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\\n" + "="*60)
        print("HYPERPARAMETER TUNING ANALYSIS")
        print("="*60)
        
        # Best configuration
        print(f"\\nBest Configuration (Score: {self.best_score:.4f}):")
        for param, value in self.best_config.items():
            print(f"  {param}: {value}")
        
        # Performance statistics
        scores = [r['results']['composite_score'] for r in self.results if r['results']['composite_score'] != float('inf')]
        if scores:
            print(f"\\nScore Statistics:")
            print(f"  Best: {min(scores):.4f}")
            print(f"  Worst: {max(scores):.4f}")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std: {np.std(scores):.4f}")
        
        # Parameter analysis
        print(f"\\nParameter Analysis:")
        param_performance = {}
        for param in self.best_config.keys():
            param_scores = {}
            for result in self.results:
                if result['results']['composite_score'] != float('inf'):
                    value = result['config'][param]
                    if value not in param_scores:
                        param_scores[value] = []
                    param_scores[value].append(result['results']['composite_score'])
            
            if param_scores:
                avg_scores = {v: np.mean(scores) for v, scores in param_scores.items()}
                best_value = min(avg_scores.keys(), key=avg_scores.get)
                print(f"  {param}: Best value = {best_value} (avg score: {avg_scores[best_value]:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for diffusion model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='hyperparameter_tuning_results', help='Output directory')
    parser.add_argument('--method', choices=['random', 'grid'], default='random', help='Search method')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of trials for random search')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs per trial')
    
    args = parser.parse_args()
    
    # Create tuner
    tuner = HyperparameterTuner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs
    )
    
    # Run tuning
    if args.method == 'random':
        best_config = tuner.run_random_search(num_trials=args.num_trials)
    else:
        best_config = tuner.run_grid_search()
    
    # Analyze results
    tuner.analyze_results()
    
    print(f"\\nHyperparameter tuning completed!")
    print(f"Best configuration saved to: {tuner.output_dir}")


if __name__ == "__main__":
    main()
