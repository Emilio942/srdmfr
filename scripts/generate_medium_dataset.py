#!/usr/bin/env python3
"""
Medium Dataset Generation for SRDMFR Project

This script generates a medium-sized dataset (500 episodes, ~1.5GB) 
for initial development and testing of the diffusion model.
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
    log_file = log_dir / f"medium_dataset_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Generate medium-sized dataset"""
    
    logger = setup_logging()
    
    # Configuration for medium dataset
    target_episodes = 500
    target_size_gb = 1.5
    estimated_size_per_episode_mb = 3.0  # Based on current observations
    
    logger.info(f"Target episodes: {target_episodes}")
    logger.info(f"Target dataset size: {target_size_gb} GB")
    logger.info(f"Estimated size per episode: {estimated_size_per_episode_mb} MB")
    logger.info(f"Estimated total size: {target_episodes * estimated_size_per_episode_mb / 1024:.2f} GB")
    
    # Create configuration
    config = RealSimulationConfig(
        simulation_duration=60.0,  # 1 minute per episode
        control_frequency=240.0,   # 240 Hz control
        data_logging_frequency=50.0,  # 50 Hz data logging
        output_dir="data/raw/medium_dataset_v1",
        fault_probability=0.4,  # 40% chance of faults
        fault_duration_range=(5.0, 30.0)  # 5-30 seconds
    )
    
    # Initialize runner
    runner = RealSimulationRunner(config)
    logger.info("Medium dataset generation initialized")
    
    # Batch processing
    batch_size = 25
    num_batches = (target_episodes + batch_size - 1) // batch_size
    
    start_time = time.time()
    episode_count = 0
    
    for batch in range(num_batches):
        batch_start = batch * batch_size + 1
        batch_end = min((batch + 1) * batch_size, target_episodes)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {batch + 1}/{num_batches}: Episodes {batch_start} to {batch_end}")
        logger.info(f"Progress: {episode_count}/{target_episodes} ({100*episode_count/target_episodes:.1f}%)")
        logger.info(f"{'='*60}")
        
        for episode in range(batch_start, batch_end + 1):
            episode_start_time = time.time()
            
            # Alternate between robots
            robot_idx = (episode - 1) % len(config.robot_configs)
            robot_config = config.robot_configs[robot_idx]
            robot_name = robot_config["name"]
            
            logger.info(f"Episode {episode}/{target_episodes}: {robot_name}")
            
            try:
                # Generate episode
                episode_data = runner.run_single_episode(robot_config)
                
                # Save episode
                file_path = runner.save_episode_data(episode_data)
                
                episode_count += 1
                episode_duration = time.time() - episode_start_time
                
                # Log progress
                if episode % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_episode = elapsed_time / episode_count
                    remaining_episodes = target_episodes - episode_count
                    eta_seconds = avg_time_per_episode * remaining_episodes
                    eta_hours = eta_seconds / 3600
                    
                    logger.info(f"Progress: {episode_count}/{target_episodes} episodes completed")
                    logger.info(f"ETA: {eta_hours:.1f} hours")
                
            except Exception as e:
                logger.error(f"Error in episode {episode}: {e}")
                continue
        
        # Batch summary
        elapsed_time = time.time() - start_time
        logger.info(f"Batch {batch + 1} completed. Total time: {elapsed_time/60:.1f} minutes")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"MEDIUM DATASET GENERATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Episodes generated: {episode_count}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Average time per episode: {total_time/episode_count:.1f} seconds")
    
    # Generate final manifest
    runner.generate_dataset(num_episodes=0)  # Just create manifest
    
    # Run analysis
    logger.info("Running dataset analysis...")
    try:
        from scripts.analyze_dataset import analyze_dataset_directory, generate_report
        dataset_stats = analyze_dataset_directory(Path(config.output_dir))
        analysis_dir = Path("analysis") / "medium_dataset_v1"
        generate_report(dataset_stats, analysis_dir)
        logger.info(f"Analysis complete. Report saved to {analysis_dir}")
    except Exception as e:
        logger.warning(f"Could not run analysis: {e}")
    
    logger.info("Medium dataset generation complete!")

if __name__ == "__main__":
    main()
