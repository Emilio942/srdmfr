#!/usr/bin/env python3
"""
Monitor training progress and dataset generation
"""

import os
import time
import psutil
from pathlib import Path

def check_training_progress():
    """Check training progress"""
    checkpoint_dir = Path("checkpoints/diffusion_v1")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"Latest checkpoint: {latest.name} ({latest.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print("No checkpoints found yet")
    else:
        print("Checkpoint directory doesn't exist")

def check_dataset_progress():
    """Check dataset generation progress"""
    dataset_dir = Path("data/raw/medium_dataset_v1")
    if dataset_dir.exists():
        episodes = list(dataset_dir.glob("episode_*.h5"))
        total_size = sum(f.stat().st_size for f in episodes)
        print(f"Dataset: {len(episodes)} episodes, {total_size / (1024*1024):.1f} MB total")
    else:
        print("Dataset directory doesn't exist")

def check_processes():
    """Check running Python processes"""
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        if proc.info['name'] == 'python' and proc.info['cmdline']:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'train_diffusion' in cmdline or 'generate_medium_dataset' in cmdline:
                python_procs.append({
                    'pid': proc.info['pid'],
                    'cmd': cmdline.split('/')[-1],
                    'cpu': proc.info['cpu_percent'],
                    'mem': proc.info['memory_percent']
                })
    
    if python_procs:
        print("Running processes:")
        for proc in python_procs:
            print(f"  PID {proc['pid']}: {proc['cmd']} (CPU: {proc['cpu']:.1f}%, MEM: {proc['mem']:.1f}%)")
    else:
        print("No training/generation processes running")

def main():
    print("=== SRDMFR Project Status ===")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("1. Dataset Generation Progress:")
    check_dataset_progress()
    print()
    
    print("2. Training Progress:")
    check_training_progress()
    print()
    
    print("3. Active Processes:")
    check_processes()
    print()

if __name__ == "__main__":
    main()
