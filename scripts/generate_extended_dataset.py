#!/usr/bin/env python3
"""
Large Dataset Generation Script
Generates extended datasets for comprehensive training and evaluation.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.real_simulation_runner import RealSimulationRunner

def generate_large_dataset(output_dir: str, num_episodes: int, episode_duration: int = 60):
    """Generate a large dataset with specified number of episodes"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting large dataset generation: {num_episodes} episodes")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Episode duration: {episode_duration} seconds")
    logger.info(f"Estimated total time: {num_episodes * episode_duration / 60:.1f} minutes")
    
    # Create simulation runner
    runner = RealSimulationRunner(
        robots=['kuka_arm', 'mobile_robot'],
        episode_duration=episode_duration,
        sampling_frequency=60,  # 60Hz sampling
        output_dir=output_dir,
        max_episodes=num_episodes
    )
    
    try:
        # Generate the dataset
        runner.generate_dataset()
        logger.info("Dataset generation completed successfully!")
        
        # Print summary
        runner.print_dataset_summary()
        
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
        logger.info("Partial dataset saved")
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate large datasets for SRDMFR training")
    parser.add_argument("--output_dir", required=True, help="Output directory for dataset")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to generate")
    parser.add_argument("--episode_duration", type=int, default=60, help="Duration of each episode in seconds")
    parser.add_argument("--validate", action="store_true", help="Validate dataset after generation")
    
    args = parser.parse_args()
    
    # Estimate dataset size
    estimated_size_mb = args.num_episodes * 3.35  # ~3.35MB per episode
    estimated_size_gb = estimated_size_mb / 1024
    
    print(f"Dataset Generation Plan:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Duration: {args.episode_duration}s each")
    print(f"  Estimated size: {estimated_size_mb:.1f} MB ({estimated_size_gb:.2f} GB)")
    print(f"  Estimated time: {args.num_episodes * 0.5:.1f} minutes")
    print()
    
    if estimated_size_gb > 5:
        response = input(f"This will generate {estimated_size_gb:.2f} GB of data. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Generate dataset
    generate_large_dataset(args.output_dir, args.num_episodes, args.episode_duration)
    
    # Optional validation
    if args.validate:
        print("Validating generated dataset...")
        os.system(f"python scripts/analyze_dataset.py --dataset_path {args.output_dir}")

if __name__ == "__main__":
    main()
