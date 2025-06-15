#!/usr/bin/env python3
"""
Large Dataset Generation for SRDMFR Project

This script generates a substantial dataset for training the diffusion model.
Target: 10GB+ of simulation data with comprehensive fault scenarios.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.real_simulation_runner import RealSimulationRunner, RealSimulationConfig

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dataset_generation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def estimate_dataset_size(num_episodes: int, episode_duration: float, 
                         sample_rate: float) -> float:
    """Estimate dataset size in GB"""
    samples_per_episode = episode_duration * sample_rate
    
    # Rough estimation: 
    # - ~50 float values per sample (robot state)
    # - 4 bytes per float
    # - 2 copies (healthy + corrupted state)
    # - metadata overhead ~20%
    
    bytes_per_sample = 50 * 4 * 2 * 1.2
    total_bytes = num_episodes * samples_per_episode * bytes_per_sample
    return total_bytes / (1024**3)  # Convert to GB

def main():
    logger = setup_logging()
    
    # Configuration for large dataset
    config = RealSimulationConfig(
        simulation_duration=60.0,      # 1 minute per episode
        control_frequency=100.0,       # 100 Hz control
        data_logging_frequency=50.0,   # 50 Hz data logging
        fault_probability=0.5,         # 50% episodes have faults
        fault_duration_range=(2.0, 30.0),  # Longer fault durations
        output_dir="data/raw/large_dataset",
        max_file_size_mb=200
    )
    
    # Calculate episodes needed for ~10GB
    target_size_gb = 10.0
    estimated_size_per_episode = estimate_dataset_size(1, config.simulation_duration, 
                                                      config.data_logging_frequency)
    episodes_needed = int(target_size_gb / estimated_size_per_episode) + 50  # Add buffer
    
    logger.info(f"Target dataset size: {target_size_gb:.1f} GB")
    logger.info(f"Estimated size per episode: {estimated_size_per_episode*1000:.1f} MB")
    logger.info(f"Episodes needed: {episodes_needed}")
    logger.info(f"Total estimated runtime: {episodes_needed * 60 / 60:.1f} hours")
    
    # Create runner
    runner = RealSimulationRunner(config)
    
    try:
        logger.info("Starting large dataset generation...")
        start_time = time.time()
        
        # Generate dataset in batches to monitor progress
        batch_size = 50
        completed_episodes = 0
        
        while completed_episodes < episodes_needed:
            batch_episodes = min(batch_size, episodes_needed - completed_episodes)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {completed_episodes//batch_size + 1}: "
                       f"Episodes {completed_episodes+1} to {completed_episodes + batch_episodes}")
            logger.info(f"Progress: {completed_episodes}/{episodes_needed} "
                       f"({100*completed_episodes/episodes_needed:.1f}%)")
            logger.info(f"{'='*60}\n")
            
            batch_start = time.time()
            
            # Generate batch
            for i in range(batch_episodes):
                episode_num = completed_episodes + i
                
                # Select robot configuration
                robot_config = runner.config.robot_configs[episode_num % len(runner.config.robot_configs)]
                
                # Run episode with progress logging
                logger.info(f"Episode {episode_num+1}/{episodes_needed}: {robot_config['name']}")
                episode_data = runner.run_single_episode(robot_config)
                file_path = runner.save_episode_data(episode_data)
                
                # Log progress every 10 episodes
                if (episode_num + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_episode = elapsed / (episode_num + 1)
                    eta_seconds = avg_time_per_episode * (episodes_needed - episode_num - 1)
                    eta_hours = eta_seconds / 3600
                    
                    logger.info(f"Progress: {episode_num+1}/{episodes_needed} episodes completed")
                    logger.info(f"ETA: {eta_hours:.2f} hours")
            
            completed_episodes += batch_episodes
            batch_time = time.time() - batch_start
            
            logger.info(f"\nBatch completed in {batch_time/60:.1f} minutes")
            logger.info(f"Average time per episode: {batch_time/batch_episodes:.2f} seconds")
            
            # Check current dataset size
            dataset_path = Path(config.output_dir)
            if dataset_path.exists():
                total_size = sum(f.stat().st_size for f in dataset_path.glob("*.h5"))
                size_gb = total_size / (1024**3)
                logger.info(f"Current dataset size: {size_gb:.2f} GB")
                
                # Check if we've reached target size
                if size_gb >= target_size_gb:
                    logger.info(f"Target size reached! Stopping at {completed_episodes} episodes.")
                    break
        
        # Create final manifest
        logger.info("Creating final dataset manifest...")
        manifest_path = runner.generate_dataset(num_episodes=0)  # Just create manifest
        
        total_time = time.time() - start_time
        
        # Final statistics
        logger.info(f"\n{'='*80}")
        logger.info("DATASET GENERATION COMPLETED!")
        logger.info(f"{'='*80}")
        logger.info(f"Total episodes: {completed_episodes}")
        logger.info(f"Total runtime: {total_time/3600:.2f} hours")
        logger.info(f"Average time per episode: {total_time/completed_episodes:.2f} seconds")
        logger.info(f"Dataset location: {config.output_dir}")
        logger.info(f"Manifest: {manifest_path}")
        
        # Calculate final dataset size
        dataset_path = Path(config.output_dir)
        total_size = sum(f.stat().st_size for f in dataset_path.glob("*.h5"))
        size_gb = total_size / (1024**3)
        logger.info(f"Final dataset size: {size_gb:.2f} GB")
        
        logger.info(f"{'='*80}")
        
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()
