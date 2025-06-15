#!/usr/bin/env python3
"""
Focused Hyperparameter Tuning for Physics Compliance
Targets the high physics loss issue identified in v3 training.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_training_experiment(name, config, base_data_path, base_output_dir):
    """Run a single training experiment with specific hyperparameters"""
    
    output_dir = Path(base_output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build training command
    cmd = [
        "python", "src/training/train_diffusion.py",
        "--data_path", base_data_path,
        "--checkpoint_dir", str(output_dir),
        "--num_epochs", str(config.get("epochs", 50)),
        "--batch_size", str(config.get("batch_size", 4)),
        "--learning_rate", str(config.get("learning_rate", 0.0001)),
    ]
    
    print(f"Starting experiment: {name}")
    print(f"Config: {config}")
    
    try:
        # Run training
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✅ Experiment {name} completed successfully")
            
            # Run quick evaluation
            eval_cmd = [
                "python", "src/evaluation/evaluate_diffusion.py",
                "--model_path", str(output_dir / "best.pt"),
                "--data_path", base_data_path,
                "--output_dir", str(output_dir),
            ]
            
            eval_result = subprocess.run(eval_cmd, cwd=".", capture_output=True, text=True)
            if eval_result.returncode == 0:
                print(f"✅ Evaluation for {name} completed")
            else:
                print(f"⚠️ Evaluation failed for {name}")
            
        else:
            print(f"❌ Experiment {name} failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Experiment {name} timed out after 1 hour")
    except Exception as e:
        print(f"❌ Experiment {name} error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Physics-focused hyperparameter tuning")
    parser.add_argument("--data_path", default="data/raw/medium_dataset_v1", help="Dataset path")
    parser.add_argument("--output_dir", default="hyperparameter_experiments", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick experiments (fewer epochs)")
    
    args = parser.parse_args()
    
    base_epochs = 30 if args.quick else 50
    
    # Define experiments focused on physics compliance
    experiments = [
        {
            "name": "physics_focused_v1",
            "config": {
                "epochs": base_epochs,
                "batch_size": 4,
                "learning_rate": 0.00005,  # Lower LR for more stable physics learning
                "description": "Lower learning rate for better physics convergence"
            }
        },
        {
            "name": "physics_focused_v2", 
            "config": {
                "epochs": base_epochs,
                "batch_size": 8,  # Larger batch for more stable gradients
                "learning_rate": 0.0001,
                "description": "Larger batch size for gradient stability"
            }
        },
        {
            "name": "physics_focused_v3",
            "config": {
                "epochs": base_epochs,
                "batch_size": 4,
                "learning_rate": 0.0002,  # Higher LR to escape local minima
                "description": "Higher learning rate to escape physics loss plateau"
            }
        },
        {
            "name": "longer_training",
            "config": {
                "epochs": base_epochs * 2,  # Longer training
                "batch_size": 4,
                "learning_rate": 0.0001,
                "description": "Extended training for better convergence"
            }
        }
    ]
    
    print(f"🔬 Starting Physics-Focused Hyperparameter Tuning")
    print(f"📁 Data path: {args.data_path}")
    print(f"📂 Output dir: {args.output_dir}")
    print(f"🎯 Focus: Reducing physics loss (current: ~18-21)")
    print(f"⚡ Quick mode: {args.quick}")
    print(f"🧪 Experiments: {len(experiments)}")
    print()
    
    # Run experiments
    results = []
    for exp in experiments:
        print(f"{'='*50}")
        start_time = datetime.now()
        
        run_training_experiment(
            exp["name"], 
            exp["config"], 
            args.data_path, 
            args.output_dir
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        results.append({
            "name": exp["name"],
            "config": exp["config"],
            "duration_minutes": duration,
            "completed": True
        })
        
        print(f"⏱️ Duration: {duration:.1f} minutes")
        print()
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "focus": "Physics compliance improvement",
        "baseline_physics_loss": {"train": 17.99, "val": 21.56},
        "experiments": results
    }
    
    summary_file = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"🎯 Physics-focused tuning completed!")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"🔍 Check individual experiment folders for detailed results")

if __name__ == "__main__":
    main()
