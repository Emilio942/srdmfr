#!/usr/bin/env python3
"""
Dataset Analysis for SRDMFR Project

This script analyzes the generated datasets and provides statistical insights,
quality metrics, and visualizations of the collected data.
"""

import os
import sys
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def analyze_episode_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a single episode HDF5 file"""
    stats = {}
    
    with h5py.File(file_path, 'r') as f:
        # Basic file info
        stats['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
        stats['file_path'] = str(file_path)
        
        # Metadata analysis
        metadata = f['metadata']
        stats['robot_name'] = metadata.attrs.get('robot_name', 'unknown')
        stats['simulation_duration'] = metadata.attrs.get('simulation_duration', 0)
        stats['total_samples'] = metadata.attrs.get('total_samples', 0)
        stats['episode_id'] = metadata.attrs.get('episode_id', -1)
        
        # Fault analysis
        if 'faults_injected' in metadata:
            faults_data = json.loads(metadata['faults_injected'][()])
            stats['num_faults'] = len(faults_data)
            stats['fault_types'] = [fault['fault_type'] for fault in faults_data]
            stats['fault_severities'] = [fault['severity'] for fault in faults_data]
        else:
            stats['num_faults'] = 0
            stats['fault_types'] = []
            stats['fault_severities'] = []
        
        # Data shape analysis
        data_group = f['data']
        if 'states_healthy' in data_group:
            healthy_group = data_group['states_healthy']
            stats['num_state_variables'] = len(healthy_group.keys())
            stats['state_variables'] = list(healthy_group.keys())
            
            # Sample one variable to get data shape
            if 'joint_positions' in healthy_group:
                joint_pos_shape = healthy_group['joint_positions'].shape
                stats['data_shape'] = joint_pos_shape
                stats['num_joints'] = joint_pos_shape[1] if len(joint_pos_shape) > 1 else 0
        
    return stats

def analyze_dataset_directory(dataset_dir: Path) -> Dict[str, Any]:
    """Analyze all episodes in a dataset directory"""
    episode_files = list(dataset_dir.glob("episode_*.h5"))
    
    if not episode_files:
        return {"error": "No episode files found"}
    
    dataset_stats = {
        "dataset_dir": str(dataset_dir),
        "total_episodes": len(episode_files),
        "total_size_mb": 0,
        "episodes": [],
        "summary": defaultdict(list)
    }
    
    print(f"Analyzing {len(episode_files)} episodes...")
    
    for i, episode_file in enumerate(sorted(episode_files)):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(episode_files)}")
        
        episode_stats = analyze_episode_file(episode_file)
        dataset_stats["episodes"].append(episode_stats)
        dataset_stats["total_size_mb"] += episode_stats["file_size_mb"]
        
        # Collect summary statistics
        dataset_stats["summary"]["robot_names"].append(episode_stats["robot_name"])
        dataset_stats["summary"]["file_sizes"].append(episode_stats["file_size_mb"])
        dataset_stats["summary"]["simulation_durations"].append(episode_stats["simulation_duration"])
        dataset_stats["summary"]["total_samples"].append(episode_stats["total_samples"])
        dataset_stats["summary"]["num_faults"].append(episode_stats["num_faults"])
        dataset_stats["summary"]["fault_types"].extend(episode_stats["fault_types"])
        dataset_stats["summary"]["fault_severities"].extend(episode_stats["fault_severities"])
    
    return dataset_stats

def generate_report(dataset_stats: Dict[str, Any], output_dir: Path):
    """Generate comprehensive analysis report"""
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Basic statistics
    summary = dataset_stats["summary"]
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"  Directory: {dataset_stats['dataset_dir']}")
    print(f"  Total Episodes: {dataset_stats['total_episodes']}")
    print(f"  Total Size: {dataset_stats['total_size_mb']:.2f} MB ({dataset_stats['total_size_mb']/1024:.2f} GB)")
    
    # Robot distribution
    robot_counts = Counter(summary["robot_names"])
    print(f"\nROBOT DISTRIBUTION:")
    for robot, count in robot_counts.items():
        percentage = (count / len(summary["robot_names"])) * 100
        print(f"  {robot}: {count} episodes ({percentage:.1f}%)")
    
    # File size statistics
    file_sizes = np.array(summary["file_sizes"])
    print(f"\nFILE SIZE STATISTICS:")
    print(f"  Mean: {np.mean(file_sizes):.2f} MB")
    print(f"  Median: {np.median(file_sizes):.2f} MB")
    print(f"  Std: {np.std(file_sizes):.2f} MB")
    print(f"  Min: {np.min(file_sizes):.2f} MB")
    print(f"  Max: {np.max(file_sizes):.2f} MB")
    
    # Simulation duration statistics
    durations = np.array(summary["simulation_durations"])
    print(f"\nSIMULATION DURATION STATISTICS:")
    print(f"  Mean: {np.mean(durations):.2f} seconds")
    print(f"  Total simulated time: {np.sum(durations):.2f} seconds ({np.sum(durations)/3600:.2f} hours)")
    
    # Sample statistics
    samples = np.array(summary["total_samples"])
    print(f"\nSAMPLE STATISTICS:")
    print(f"  Total samples: {np.sum(samples):,}")
    print(f"  Mean samples per episode: {np.mean(samples):.0f}")
    print(f"  Sampling frequency: ~{np.mean(samples)/np.mean(durations):.1f} Hz")
    
    # Fault statistics
    fault_counts = np.array(summary["num_faults"])
    print(f"\nFAULT INJECTION STATISTICS:")
    print(f"  Episodes with faults: {np.sum(fault_counts > 0)} / {len(fault_counts)} ({100*np.sum(fault_counts > 0)/len(fault_counts):.1f}%)")
    print(f"  Total faults injected: {np.sum(fault_counts)}")
    print(f"  Mean faults per episode: {np.mean(fault_counts):.2f}")
    
    if summary["fault_types"]:
        fault_type_counts = Counter(summary["fault_types"])
        print(f"\nFAULT TYPE DISTRIBUTION:")
        for fault_type, count in fault_type_counts.most_common():
            percentage = (count / len(summary["fault_types"])) * 100
            print(f"  {fault_type}: {count} occurrences ({percentage:.1f}%)")
        
        fault_severity_counts = Counter(summary["fault_severities"])
        print(f"\nFAULT SEVERITY DISTRIBUTION:")
        for severity, count in fault_severity_counts.most_common():
            percentage = (count / len(summary["fault_severities"])) * 100
            print(f"  {severity}: {count} occurrences ({percentage:.1f}%)")
    
    # Save detailed statistics to JSON
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {
            "dataset_overview": {
                "directory": dataset_stats["dataset_dir"],
                "total_episodes": dataset_stats["total_episodes"],
                "total_size_mb": dataset_stats["total_size_mb"],
                "robot_distribution": dict(robot_counts),
                "file_size_stats": {
                    "mean": float(np.mean(file_sizes)),
                    "median": float(np.median(file_sizes)),
                    "std": float(np.std(file_sizes)),
                    "min": float(np.min(file_sizes)),
                    "max": float(np.max(file_sizes))
                },
                "duration_stats": {
                    "mean_duration": float(np.mean(durations)),
                    "total_simulated_time": float(np.sum(durations))
                },
                "sample_stats": {
                    "total_samples": int(np.sum(samples)),
                    "mean_samples_per_episode": float(np.mean(samples)),
                    "estimated_sampling_frequency": float(np.mean(samples)/np.mean(durations))
                },
                "fault_stats": {
                    "episodes_with_faults": int(np.sum(fault_counts > 0)),
                    "fault_percentage": float(100*np.sum(fault_counts > 0)/len(fault_counts)),
                    "total_faults": int(np.sum(fault_counts)),
                    "mean_faults_per_episode": float(np.mean(fault_counts)),
                    "fault_type_distribution": dict(fault_type_counts) if summary["fault_types"] else {},
                    "fault_severity_distribution": dict(fault_severity_counts) if summary["fault_severities"] else {}
                }
            }
        }
        json.dump(json_stats, f, indent=2)
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    
    # Generate visualizations if matplotlib is available
    try:
        generate_visualizations(dataset_stats, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

def generate_visualizations(dataset_stats: Dict[str, Any], output_dir: Path):
    """Generate visualization plots"""
    summary = dataset_stats["summary"]
    
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Analysis Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Robot distribution
    robot_counts = Counter(summary["robot_names"])
    axes[0, 0].pie(robot_counts.values(), labels=robot_counts.keys(), autopct='%1.1f%%')
    axes[0, 0].set_title('Robot Type Distribution')
    
    # 2. File size distribution
    axes[0, 1].hist(summary["file_sizes"], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('File Size (MB)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('File Size Distribution')
    
    # 3. Simulation duration distribution
    axes[0, 2].hist(summary["simulation_durations"], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Simulation Duration (seconds)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Simulation Duration Distribution')
    
    # 4. Samples per episode
    axes[1, 0].hist(summary["total_samples"], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Samples per Episode')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Samples per Episode Distribution')
    
    # 5. Fault count distribution
    axes[1, 1].hist(summary["num_faults"], bins=max(summary["num_faults"])+1, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Number of Faults per Episode')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Faults per Episode Distribution')
    
    # 6. Fault type distribution (if faults exist)
    if summary["fault_types"]:
        fault_type_counts = Counter(summary["fault_types"])
        axes[1, 2].bar(range(len(fault_type_counts)), fault_type_counts.values())
        axes[1, 2].set_xticks(range(len(fault_type_counts)))
        axes[1, 2].set_xticklabels(fault_type_counts.keys(), rotation=45, ha='right')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Fault Type Distribution')
    else:
        axes[1, 2].text(0.5, 0.5, 'No faults in dataset', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Fault Type Distribution')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = output_dir / "dataset_visualizations.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {viz_file}")

def main():
    """Main analysis function"""
    
    # Available dataset directories
    data_dir = Path("data/raw")
    available_datasets = [d for d in data_dir.iterdir() if d.is_dir() and list(d.glob("episode_*.h5"))]
    
    if not available_datasets:
        print("No datasets found in data/raw/")
        return
    
    print("Available datasets:")
    for i, dataset in enumerate(available_datasets):
        episode_count = len(list(dataset.glob("episode_*.h5")))
        print(f"  {i+1}. {dataset.name} ({episode_count} episodes)")
    
    # Analyze all datasets or specific one
    choice = input(f"\nEnter dataset number to analyze (1-{len(available_datasets)}) or 'all' for all datasets: ").strip()
    
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    if choice.lower() == 'all':
        for dataset_dir in available_datasets:
            print(f"\n{'='*60}")
            print(f"ANALYZING DATASET: {dataset_dir.name}")
            print(f"{'='*60}")
            
            dataset_stats = analyze_dataset_directory(dataset_dir)
            output_dir = analysis_dir / dataset_dir.name
            generate_report(dataset_stats, output_dir)
    else:
        try:
            dataset_idx = int(choice) - 1
            if 0 <= dataset_idx < len(available_datasets):
                dataset_dir = available_datasets[dataset_idx]
                print(f"\nAnalyzing dataset: {dataset_dir.name}")
                
                dataset_stats = analyze_dataset_directory(dataset_dir)
                output_dir = analysis_dir / dataset_dir.name
                generate_report(dataset_stats, output_dir)
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")

if __name__ == "__main__":
    main()
