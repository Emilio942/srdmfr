#!/usr/bin/env python3
"""
Ablation Study Script for SRDMFR Diffusion Model Architecture
Systematically evaluates impact of different architectural components.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

# Import from our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from training.train_diffusion import DiffusionTrainer, RobotStateDataset
from models.diffusion_model import ModelConfig
from evaluation.evaluate_diffusion import DiffusionEvaluator


class AblationStudy:
    """Systematic ablation study for diffusion model components"""
    
    def __init__(self, data_path: str, output_dir: str, num_epochs: int = 30):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        
        # Base configuration (proven working setup)
        self.base_config = {
            'model_dim': 256,
            'num_layers': 8,
            'num_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1,
            'num_diffusion_steps': 1000,
            'noise_schedule': 'cosine',
            'use_linear_attention': True,
            'mixed_precision': True,
            'use_gradient_checkpointing': True,
            'learning_rate': 1e-4,
            'batch_size': 16,
            'weight_decay': 1e-5,
            'reconstruction_weight': 1.0,
            'physics_weight': 0.5,
            'denoising_weight': 2.0,
            'gradient_clip': 1.0
        }
        
        self.results = {}
    
    def define_ablations(self) -> Dict[str, Dict[str, Any]]:
        """Define specific ablations to study"""
        ablations = {
            # Architecture ablations
            'no_linear_attention': {
                'use_linear_attention': False,
                'description': 'Replace linear attention with standard attention'
            },
            'smaller_model': {
                'model_dim': 128,
                'ff_dim': 512,
                'description': 'Reduce model size by 50%'
            },
            'larger_model': {
                'model_dim': 512,
                'ff_dim': 2048,
                'description': 'Increase model size by 100%'
            },
            'fewer_layers': {
                'num_layers': 4,
                'description': 'Reduce transformer layers from 8 to 4'
            },
            'more_layers': {
                'num_layers': 12,
                'description': 'Increase transformer layers from 8 to 12'
            },
            'fewer_heads': {
                'num_heads': 4,
                'description': 'Reduce attention heads from 8 to 4'
            },
            'more_heads': {
                'num_heads': 16,
                'description': 'Increase attention heads from 8 to 16'
            },
            'no_dropout': {
                'dropout': 0.0,
                'description': 'Remove dropout regularization'
            },
            'high_dropout': {
                'dropout': 0.3,
                'description': 'Increase dropout to 0.3'
            },
            
            # Diffusion process ablations
            'linear_schedule': {
                'noise_schedule': 'linear',
                'description': 'Use linear noise schedule instead of cosine'
            },
            'fewer_diffusion_steps': {
                'num_diffusion_steps': 100,
                'description': 'Reduce diffusion steps from 1000 to 100'
            },
            'more_diffusion_steps': {
                'num_diffusion_steps': 2000,
                'description': 'Increase diffusion steps from 1000 to 2000'
            },
            
            # Loss function ablations
            'no_physics_loss': {
                'physics_weight': 0.0,
                'description': 'Remove physics constraint loss'
            },
            'high_physics_loss': {
                'physics_weight': 2.0,
                'description': 'Increase physics loss weight'
            },
            'no_reconstruction_loss': {
                'reconstruction_weight': 0.0,
                'description': 'Remove reconstruction loss'
            },
            'high_reconstruction_loss': {
                'reconstruction_weight': 5.0,
                'description': 'Increase reconstruction loss weight'
            },
            'equal_loss_weights': {
                'reconstruction_weight': 1.0,
                'physics_weight': 1.0,
                'denoising_weight': 1.0,
                'description': 'Use equal weights for all loss components'
            },
            
            # Training ablations
            'higher_lr': {
                'learning_rate': 5e-4,
                'description': 'Increase learning rate'
            },
            'lower_lr': {
                'learning_rate': 5e-5,
                'description': 'Decrease learning rate'
            },
            'larger_batch': {
                'batch_size': 32,
                'description': 'Increase batch size from 16 to 32'
            },
            'smaller_batch': {
                'batch_size': 8,
                'description': 'Decrease batch size from 16 to 8'
            },
            'no_weight_decay': {
                'weight_decay': 0.0,
                'description': 'Remove weight decay regularization'
            },
            'high_weight_decay': {
                'weight_decay': 1e-4,
                'description': 'Increase weight decay'
            },
            
            # Memory/efficiency ablations
            'no_gradient_checkpointing': {
                'use_gradient_checkpointing': False,
                'description': 'Disable gradient checkpointing'
            },
            'no_mixed_precision': {
                'mixed_precision': False,
                'description': 'Disable mixed precision training'
            }
        }
        
        return ablations
    
    def run_single_ablation(self, name: str, changes: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Run a single ablation experiment"""
        print(f"\\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"Description: {description}")
        print(f"Changes: {changes}")
        print('='*60)
        
        # Create modified config
        config = self.base_config.copy()
        config.update(changes)
        
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
                mixed_precision=config['mixed_precision'],
                use_gradient_checkpointing=config['use_gradient_checkpointing']
            )
            
            # Setup dataset
            dataset = RobotStateDataset(self.data_path, split='train')
            dataloader = DataLoader(
                dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                collate_fn=dataset.collate_fn
            )
            
            val_dataset = RobotStateDataset(self.data_path, split='val')
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
            
            # Create experiment directory
            exp_dir = self.output_dir / name
            
            # Train model
            start_time = time.time()
            trainer.train(
                train_dataloader=dataloader,
                val_dataloader=val_dataloader,
                num_epochs=self.num_epochs,
                checkpoint_dir=str(exp_dir / 'checkpoints'),
                save_every=10
            )
            training_time = time.time() - start_time
            
            # Get final metrics
            final_metrics = trainer.get_metrics()
            
            # Detailed evaluation
            evaluator = DiffusionEvaluator(
                model_path=str(exp_dir / 'checkpoints' / 'best.pt'),
                device=trainer.device
            )
            
            eval_metrics = evaluator.evaluate_dataset(
                self.data_path, 
                split='test',
                output_dir=str(exp_dir / 'evaluation')
            )
            
            # Compile results
            results = {
                'config': config,
                'description': description,
                'changes': changes,
                'final_train_loss': final_metrics.get('train_loss', float('inf')),
                'final_val_loss': final_metrics.get('val_loss', float('inf')),
                'eval_mse': eval_metrics.get('mse', float('inf')),
                'eval_mae': eval_metrics.get('mae', float('inf')),
                'physics_violation_rate': eval_metrics.get('physics_violation_rate', 1.0),
                'temporal_consistency': eval_metrics.get('temporal_consistency', 0.0),
                'inference_time_ms': eval_metrics.get('inference_time_ms', float('inf')),
                'training_time_sec': training_time,
                'model_parameters': trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else 0
            }
            
            # Calculate performance score
            score = (
                results['final_val_loss'] * 0.3 +
                results['eval_mse'] * 0.2 +
                results['physics_violation_rate'] * 0.3 +
                results['inference_time_ms'] / 1000 * 0.2  # Normalize to seconds
            )
            results['performance_score'] = score
            
            # Save results
            with open(exp_dir / 'ablation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Ablation {name} completed. Score: {score:.4f}")
            return results
            
        except Exception as e:
            print(f"Error in ablation {name}: {e}")
            return {
                'config': config,
                'description': description,
                'changes': changes,
                'error': str(e),
                'performance_score': float('inf')
            }
    
    def run_all_ablations(self, skip_existing: bool = True) -> Dict[str, Any]:
        """Run all defined ablations"""
        ablations = self.define_ablations()
        
        print(f"Starting ablation study with {len(ablations)} experiments...")
        print(f"Base configuration: {self.base_config}")
        
        # Run baseline first
        if not skip_existing or not (self.output_dir / 'baseline').exists():
            baseline_results = self.run_single_ablation(
                'baseline', {}, 'Baseline configuration'
            )
            self.results['baseline'] = baseline_results
        
        # Run each ablation
        for name, ablation in ablations.items():
            if skip_existing and (self.output_dir / name).exists():
                print(f"Skipping existing ablation: {name}")
                continue
                
            changes = {k: v for k, v in ablation.items() if k != 'description'}
            results = self.run_single_ablation(name, changes, ablation['description'])
            self.results[name] = results
            
            # Save intermediate results
            self.save_results()
        
        return self.results
    
    def run_selected_ablations(self, ablation_names: List[str]) -> Dict[str, Any]:
        """Run only selected ablations"""
        ablations = self.define_ablations()
        
        for name in ablation_names:
            if name not in ablations:
                print(f"Warning: Unknown ablation '{name}', skipping...")
                continue
            
            ablation = ablations[name]
            changes = {k: v for k, v in ablation.items() if k != 'description'}
            results = self.run_single_ablation(name, changes, ablation['description'])
            self.results[name] = results
            
            # Save intermediate results
            self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save all ablation results"""
        results_file = self.output_dir / 'ablation_study_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def analyze_results(self):
        """Analyze and compare ablation results"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\\n" + "="*80)
        print("ABLATION STUDY ANALYSIS")
        print("="*80)
        
        # Sort by performance score
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1].get('performance_score', float('inf'))
        )
        
        print(f"\\nRanked Results (lower score = better):")
        print("-" * 80)
        for i, (name, results) in enumerate(sorted_results[:10]):  # Top 10
            score = results.get('performance_score', float('inf'))
            val_loss = results.get('final_val_loss', 'N/A')
            mse = results.get('eval_mse', 'N/A')
            physics_viol = results.get('physics_violation_rate', 'N/A')
            
            print(f"{i+1:2d}. {name:25s} | Score: {score:8.4f} | "
                  f"Val Loss: {val_loss:8.4f} | MSE: {mse:12.2e} | Physics Viol: {physics_viol:6.2%}")
        
        # Baseline comparison
        if 'baseline' in self.results:
            baseline_score = self.results['baseline']['performance_score']
            print(f"\\nBaseline Performance: {baseline_score:.4f}")
            print("\\nImprovements over baseline:")
            
            improvements = []
            for name, results in self.results.items():
                if name != 'baseline':
                    score = results.get('performance_score', float('inf'))
                    if score < baseline_score:
                        improvement = (baseline_score - score) / baseline_score * 100
                        improvements.append((name, improvement, score))
            
            improvements.sort(key=lambda x: x[1], reverse=True)
            
            for name, improvement, score in improvements[:5]:
                print(f"  {name:25s}: {improvement:6.2f}% improvement (score: {score:.4f})")
        
        # Component analysis
        print(f"\\nComponent Impact Analysis:")
        component_impacts = {
            'Architecture': ['smaller_model', 'larger_model', 'fewer_layers', 'more_layers', 'fewer_heads', 'more_heads'],
            'Attention': ['no_linear_attention'],
            'Regularization': ['no_dropout', 'high_dropout', 'no_weight_decay', 'high_weight_decay'],
            'Diffusion Process': ['linear_schedule', 'fewer_diffusion_steps', 'more_diffusion_steps'],
            'Loss Functions': ['no_physics_loss', 'high_physics_loss', 'no_reconstruction_loss', 'equal_loss_weights'],
            'Training': ['higher_lr', 'lower_lr', 'larger_batch', 'smaller_batch']
        }
        
        for category, ablations in component_impacts.items():
            category_scores = []
            for ablation in ablations:
                if ablation in self.results:
                    score = self.results[ablation].get('performance_score', float('inf'))
                    if score != float('inf'):
                        category_scores.append(score)
            
            if category_scores:
                avg_score = sum(category_scores) / len(category_scores)
                baseline_score = self.results.get('baseline', {}).get('performance_score', float('inf'))
                if baseline_score != float('inf'):
                    impact = (avg_score - baseline_score) / baseline_score * 100
                    print(f"  {category:15s}: {impact:+6.2f}% average impact")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for diffusion model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='ablation_study_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs per ablation')
    parser.add_argument('--ablations', nargs='*', help='Specific ablations to run (default: all)')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing ablation experiments')
    
    args = parser.parse_args()
    
    # Create ablation study
    study = AblationStudy(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs
    )
    
    # Run ablations
    if args.ablations:
        results = study.run_selected_ablations(args.ablations)
    else:
        results = study.run_all_ablations(skip_existing=args.skip_existing)
    
    # Analyze results
    study.analyze_results()
    
    print(f"\\nAblation study completed!")
    print(f"Results saved to: {study.output_dir}")


if __name__ == "__main__":
    main()
