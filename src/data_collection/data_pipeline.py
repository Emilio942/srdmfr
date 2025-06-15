"""
SRDMFR Data Collection Pipeline
==============================

Automated data collection system for robotics state estimation.
Integrates simulation, fault injection, and data storage.

Author: SRDMFR Team
Date: June 2025
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import logging
from pathlib import Path
import hashlib
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Local imports (will be available when modules are in place)
# from .robotics_simulator import RoboticsSimulator, RobotState, RobotType, SimulationConfig
# from .fault_injection import FaultInjectionFramework, FaultParameters, FaultType, FaultSeverity

logger = logging.getLogger(__name__)


@dataclass
class DataCollectionConfig:
    """Configuration for data collection pipeline"""
    
    # Output settings
    output_dir: str = "./data/raw"
    dataset_name: str = "srdmfr_dataset"
    max_file_size_mb: int = 1000  # Split files when exceeding this size
    
    # Sampling settings
    sampling_rate_hz: int = 100  # Data collection frequency
    episode_duration_sec: float = 30.0  # Length of each episode
    episodes_per_robot: int = 100  # Number of episodes per robot type
    
    # Fault injection settings
    fault_probability: float = 0.3  # Probability of fault in episode
    max_concurrent_faults: int = 3  # Maximum simultaneous faults
    
    # Data quality settings
    min_state_changes: int = 10  # Minimum state variations to keep episode
    max_corruption_ratio: float = 0.8  # Maximum fraction of corrupted data
    
    # Validation settings
    enable_validation: bool = True
    validation_split: float = 0.2  # Fraction for validation set


@dataclass
class EpisodeMetadata:
    """Metadata for a single data collection episode"""
    
    episode_id: str
    robot_name: str
    robot_type: str
    start_time: datetime
    duration_sec: float
    
    # Fault information
    faults_injected: List[Dict]
    fault_severity_levels: List[str]
    
    # Data statistics
    total_samples: int
    corrupted_samples: int
    healthy_samples: int
    
    # Quality metrics
    data_completeness: float  # Fraction of expected samples collected
    state_variance: float     # Measure of state dynamics
    sensor_coverage: List[str]  # Which sensors provided data
    
    # File information
    file_path: str
    file_size_bytes: int
    checksum_md5: str


class DataCollectionPipeline:
    """
    Comprehensive data collection pipeline for SRDMFR project.
    
    Features:
    - Multi-robot automated data collection
    - Integrated fault injection
    - Real-time data validation
    - Hierarchical data storage (HDF5)
    - Metadata tracking and quality control
    """
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.episodes_metadata: List[EpisodeMetadata] = []
        self.current_file_index = 0
        
        # Threading for concurrent collection
        self.data_queue = queue.Queue(maxsize=1000)
        self.collection_active = False
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'total_samples': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'total_faults_injected': 0,
            'collection_start_time': None
        }
        
        logger.info(f"Data collection pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        
    def collect_robot_dataset(self, robot_configs: Dict[str, Tuple[str, str]]) -> str:
        """
        Collect complete dataset for multiple robot types.
        
        Args:
            robot_configs: Dict mapping robot_name -> (robot_type, urdf_path)
            
        Returns:
            Path to created dataset
        """
        
        self.stats['collection_start_time'] = datetime.now()
        dataset_path = self._create_dataset_structure()
        
        logger.info(f"Starting data collection for {len(robot_configs)} robot types")
        
        for robot_name, (robot_type, urdf_path) in robot_configs.items():
            
            logger.info(f"Collecting data for {robot_name} ({robot_type})")
            
            try:
                robot_episodes = self._collect_robot_episodes(
                    robot_name, robot_type, urdf_path
                )
                
                self.stats['successful_episodes'] += len(robot_episodes)
                logger.info(f"Successfully collected {len(robot_episodes)} episodes for {robot_name}")
                
            except Exception as e:
                logger.error(f"Failed to collect data for {robot_name}: {e}")
                self.stats['failed_episodes'] += self.config.episodes_per_robot
                continue
        
        # Finalize dataset
        self._finalize_dataset(dataset_path)
        
        return dataset_path
    
    def _create_dataset_structure(self) -> str:
        """Create hierarchical dataset directory structure"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"{self.config.dataset_name}_{timestamp}"
        dataset_path = self.output_dir / dataset_name
        
        # Create subdirectories
        subdirs = ['train', 'validation', 'test', 'metadata', 'config']
        for subdir in subdirs:
            (dataset_path / subdir).mkdir(parents=True, exist_ok=True)
            
        # Save configuration
        config_file = dataset_path / 'config' / 'collection_config.json'
        with open(config_file, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2, default=str)
            
        logger.info(f"Created dataset structure at {dataset_path}")
        return str(dataset_path)
    
    def _collect_robot_episodes(self, robot_name: str, robot_type: str, 
                               urdf_path: str) -> List[EpisodeMetadata]:
        """Collect episodes for a specific robot"""
        
        episodes = []
        
        # Initialize simulation (mock for now - would use actual simulator)
        # sim = RoboticsSimulator(SimulationConfig(gui_enabled=False))
        # fault_injector = FaultInjectionFramework()
        
        for episode_idx in range(self.config.episodes_per_robot):
            
            episode_id = f"{robot_name}_episode_{episode_idx:04d}"
            
            try:
                episode_metadata = self._collect_single_episode(
                    episode_id, robot_name, robot_type, urdf_path
                )
                
                if self._validate_episode(episode_metadata):
                    episodes.append(episode_metadata)
                    self.stats['total_episodes'] += 1
                    
                    if episode_idx % 10 == 0:
                        logger.info(f"Completed episode {episode_idx}/{self.config.episodes_per_robot} for {robot_name}")
                        
                else:
                    logger.warning(f"Episode {episode_id} failed validation")
                    
            except Exception as e:
                logger.error(f"Failed to collect episode {episode_id}: {e}")
                continue
                
        return episodes
    
    def _collect_single_episode(self, episode_id: str, robot_name: str, 
                               robot_type: str, urdf_path: str) -> EpisodeMetadata:
        """Collect data for a single episode"""
        
        start_time = datetime.now()
        episode_data = []
        faults_injected = []
        
        # Determine if faults should be injected
        inject_faults = np.random.random() < self.config.fault_probability
        
        if inject_faults:
            # Generate random fault scenario
            faults_injected = self._generate_random_faults()
            
        # Mock data collection (would be replaced with actual simulation)
        total_samples = int(self.config.episode_duration_sec * self.config.sampling_rate_hz)
        
        for sample_idx in range(total_samples):
            timestamp = sample_idx / self.config.sampling_rate_hz
            
            # Generate mock robot state
            mock_state = self._generate_mock_robot_state(timestamp, robot_type)
            
            # Apply faults if any
            if inject_faults:
                mock_state = self._apply_mock_faults(mock_state, faults_injected, timestamp)
                
            episode_data.append(mock_state)
            
        # Calculate statistics
        corrupted_samples = sum(1 for state in episode_data if state.get('is_corrupted', False))
        healthy_samples = total_samples - corrupted_samples
        
        # Save episode data
        file_path = self._save_episode_data(episode_id, episode_data)
        
        # Create metadata
        metadata = EpisodeMetadata(
            episode_id=episode_id,
            robot_name=robot_name,
            robot_type=robot_type,
            start_time=start_time,
            duration_sec=self.config.episode_duration_sec,
            faults_injected=faults_injected,
            fault_severity_levels=[f.get('severity', 'none') for f in faults_injected],
            total_samples=total_samples,
            corrupted_samples=corrupted_samples,
            healthy_samples=healthy_samples,
            data_completeness=1.0,  # Mock - would calculate actual completeness
            state_variance=self._calculate_state_variance(episode_data),
            sensor_coverage=['imu', 'encoders', 'force_torque'],  # Mock
            file_path=file_path,
            file_size_bytes=os.path.getsize(file_path),
            checksum_md5=self._calculate_file_checksum(file_path)
        )
        
        self.stats['total_samples'] += total_samples
        self.stats['total_faults_injected'] += len(faults_injected)
        
        return metadata
    
    def _generate_mock_robot_state(self, timestamp: float, robot_type: str) -> Dict:
        """Generate mock robot state data (placeholder for actual simulation)"""
        
        # Base state dimensions based on robot type
        if robot_type == "mobile":
            joint_dim = 2  # wheel joints
            base_pose_dim = 7  # x,y,z,qw,qx,qy,qz
        elif robot_type == "manipulator":
            joint_dim = 6  # 6-DOF arm
            base_pose_dim = 0  # fixed base
        elif robot_type == "humanoid":
            joint_dim = 30  # full humanoid
            base_pose_dim = 7
        else:
            joint_dim = 6
            base_pose_dim = 7
            
        # Generate realistic-looking data with temporal correlations
        state = {
            'timestamp': timestamp,
            'robot_type': robot_type,
            'is_corrupted': False,
            
            # Joint states
            'joint_positions': np.sin(timestamp * 2 * np.pi * 0.1) * np.random.uniform(-1, 1, joint_dim),
            'joint_velocities': np.cos(timestamp * 2 * np.pi * 0.1) * np.random.uniform(-0.5, 0.5, joint_dim),
            'joint_torques': np.random.normal(0, 1, joint_dim),
            
            # IMU data
            'imu_acceleration': np.array([0, 0, 9.81]) + np.random.normal(0, 0.01, 3),
            'imu_gyroscope': np.random.normal(0, 0.001, 3),
            
            # Force/Torque
            'force_torque': np.random.normal(0, 0.1, 6),
            
            # System status
            'battery_voltage': 24.0 - np.random.exponential(0.05),
            'motor_temperatures': 25.0 + np.abs(np.random.normal(0, 5, joint_dim)).clip(0, 40),
            'cpu_temperature': 45.0 + np.random.normal(0, 2)
        }
        
        if base_pose_dim > 0:
            state['base_position'] = np.random.normal(0, 0.1, 3)
            state['base_orientation'] = np.array([1, 0, 0, 0]) + np.random.normal(0, 0.01, 4)
            state['base_orientation'] /= np.linalg.norm(state['base_orientation'])  # Normalize quaternion
            
        return state
    
    def _generate_random_faults(self) -> List[Dict]:
        """Generate random fault scenarios for an episode"""
        
        num_faults = np.random.randint(1, self.config.max_concurrent_faults + 1)
        faults = []
        
        fault_types = ['sensor_noise', 'sensor_bias', 'sensor_drift', 'actuator_backlash', 'power_drop']
        severity_levels = ['minimal', 'mild', 'moderate', 'severe']
        
        for i in range(num_faults):
            fault = {
                'fault_id': f"fault_{i}",
                'fault_type': np.random.choice(fault_types),
                'severity': np.random.choice(severity_levels),
                'start_time': np.random.uniform(0, self.config.episode_duration_sec * 0.8),
                'duration': np.random.uniform(1.0, self.config.episode_duration_sec * 0.5),
                'affected_joints': np.random.choice(6, size=np.random.randint(1, 4), replace=False).tolist(),
                'parameters': {
                    'noise_std': np.random.uniform(0.001, 0.1),
                    'bias_value': np.random.uniform(-0.5, 0.5),
                    'drift_rate': np.random.uniform(0.001, 0.01)
                }
            }
            faults.append(fault)
            
        return faults
    
    def _apply_mock_faults(self, state: Dict, faults: List[Dict], timestamp: float) -> Dict:
        """Apply mock fault effects to state (placeholder for actual fault injection)"""
        
        corrupted_state = state.copy()
        corrupted_state['is_corrupted'] = False
        
        for fault in faults:
            if fault['start_time'] <= timestamp <= fault['start_time'] + fault['duration']:
                corrupted_state['is_corrupted'] = True
                
                # Apply simple fault effects based on type
                if fault['fault_type'] == 'sensor_noise':
                    noise_std = fault['parameters']['noise_std']
                    corrupted_state['imu_acceleration'] += np.random.normal(0, noise_std, 3)
                    
                elif fault['fault_type'] == 'sensor_bias':
                    bias = fault['parameters']['bias_value']
                    for joint_idx in fault['affected_joints']:
                        if joint_idx < len(corrupted_state['joint_positions']):
                            corrupted_state['joint_positions'][joint_idx] += bias
                            
                elif fault['fault_type'] == 'power_drop':
                    voltage_drop = 2.0 if fault['severity'] == 'severe' else 1.0
                    corrupted_state['battery_voltage'] -= voltage_drop
                    
        return corrupted_state
    
    def _calculate_state_variance(self, episode_data: List[Dict]) -> float:
        """Calculate variance metric for episode data quality"""
        
        if not episode_data:
            return 0.0
            
        # Calculate variance of joint positions as proxy for dynamics
        joint_positions = np.array([state['joint_positions'] for state in episode_data])
        return float(np.mean(np.var(joint_positions, axis=0)))
    
    def _save_episode_data(self, episode_id: str, episode_data: List[Dict]) -> str:
        """Save episode data to HDF5 file"""
        
        file_path = self.output_dir / f"{episode_id}.h5"
        
        with h5py.File(file_path, 'w') as f:
            # Save metadata
            f.attrs['episode_id'] = episode_id
            f.attrs['num_samples'] = len(episode_data)
            f.attrs['collection_time'] = datetime.now().isoformat()
            
            # Save time series data
            if episode_data:
                sample_state = episode_data[0]
                
                # Create datasets for each data field
                for key, value in sample_state.items():
                    if isinstance(value, np.ndarray):
                        # Stack all timesteps for array data
                        data_stack = np.array([state[key] for state in episode_data])
                        f.create_dataset(key, data=data_stack, compression='gzip')
                    elif isinstance(value, (int, float, bool, str)):
                        # Store scalar time series
                        data_array = np.array([state[key] for state in episode_data])
                        f.create_dataset(key, data=data_array, compression='gzip')
                        
        return str(file_path)
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _validate_episode(self, metadata: EpisodeMetadata) -> bool:
        """Validate episode quality and completeness"""
        
        if not self.config.enable_validation:
            return True
            
        # Check data completeness
        if metadata.data_completeness < 0.95:
            logger.warning(f"Episode {metadata.episode_id} has low completeness: {metadata.data_completeness}")
            return False
            
        # Check corruption ratio
        corruption_ratio = metadata.corrupted_samples / metadata.total_samples
        if corruption_ratio > self.config.max_corruption_ratio:
            logger.warning(f"Episode {metadata.episode_id} has high corruption: {corruption_ratio}")
            return False
            
        # Check state dynamics
        if metadata.state_variance < 1e-6:
            logger.warning(f"Episode {metadata.episode_id} has low state variance: {metadata.state_variance}")
            return False
            
        # Check file integrity
        if not os.path.exists(metadata.file_path):
            logger.error(f"Episode {metadata.episode_id} file not found: {metadata.file_path}")
            return False
            
        return True
    
    def _finalize_dataset(self, dataset_path: str):
        """Finalize dataset with metadata and statistics"""
        
        # Save episode metadata
        metadata_file = Path(dataset_path) / 'metadata' / 'episodes_metadata.json'
        with open(metadata_file, 'w') as f:
            # Convert metadata to serializable format
            metadata_dicts = []
            for meta in self.episodes_metadata:
                meta_dict = asdict(meta)
                meta_dict['start_time'] = meta_dict['start_time'].isoformat()
                metadata_dicts.append(meta_dict)
            json.dump(metadata_dicts, f, indent=2)
            
        # Save collection statistics
        stats_file = Path(dataset_path) / 'metadata' / 'collection_stats.json'
        final_stats = self.stats.copy()
        final_stats['collection_end_time'] = datetime.now().isoformat()
        final_stats['collection_start_time'] = final_stats['collection_start_time'].isoformat()
        final_stats['total_duration_minutes'] = (
            datetime.now() - datetime.fromisoformat(final_stats['collection_start_time'].replace('T', ' '))
        ).total_seconds() / 60.0
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
            
        # Create dataset summary
        summary = self._create_dataset_summary()
        summary_file = Path(dataset_path) / 'metadata' / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Dataset finalized at {dataset_path}")
        logger.info(f"Total episodes: {self.stats['total_episodes']}")
        logger.info(f"Total samples: {self.stats['total_samples']}")
        logger.info(f"Success rate: {self.stats['successful_episodes']/(self.stats['successful_episodes']+self.stats['failed_episodes'])*100:.1f}%")
    
    def _create_dataset_summary(self) -> Dict:
        """Create comprehensive dataset summary"""
        
        # Analyze episode metadata
        robot_types = {}
        fault_types = {}
        severity_distribution = {}
        
        for meta in self.episodes_metadata:
            # Robot type distribution
            robot_types[meta.robot_type] = robot_types.get(meta.robot_type, 0) + 1
            
            # Fault analysis
            for fault in meta.faults_injected:
                fault_type = fault.get('fault_type', 'none')
                fault_types[fault_type] = fault_types.get(fault_type, 0) + 1
                
            for severity in meta.fault_severity_levels:
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
                
        summary = {
            'dataset_info': {
                'total_episodes': len(self.episodes_metadata),
                'total_samples': sum(meta.total_samples for meta in self.episodes_metadata),
                'total_corrupted_samples': sum(meta.corrupted_samples for meta in self.episodes_metadata),
                'average_episode_duration': np.mean([meta.duration_sec for meta in self.episodes_metadata]),
                'total_dataset_size_gb': sum(meta.file_size_bytes for meta in self.episodes_metadata) / 1e9
            },
            'robot_distribution': robot_types,
            'fault_distribution': fault_types,
            'severity_distribution': severity_distribution,
            'quality_metrics': {
                'average_completeness': np.mean([meta.data_completeness for meta in self.episodes_metadata]),
                'average_state_variance': np.mean([meta.state_variance for meta in self.episodes_metadata]),
                'corruption_ratio': sum(meta.corrupted_samples for meta in self.episodes_metadata) / 
                                 sum(meta.total_samples for meta in self.episodes_metadata)
            }
        }
        
        return summary


# Configuration templates for different use cases
def get_rapid_prototyping_config() -> DataCollectionConfig:
    """Configuration for rapid prototyping with smaller dataset"""
    return DataCollectionConfig(
        dataset_name="srdmfr_prototype",
        sampling_rate_hz=50,
        episode_duration_sec=10.0,
        episodes_per_robot=20,
        fault_probability=0.5,
        max_concurrent_faults=2
    )


def get_production_config() -> DataCollectionConfig:
    """Configuration for production-quality dataset"""
    return DataCollectionConfig(
        dataset_name="srdmfr_production",
        sampling_rate_hz=100,
        episode_duration_sec=30.0,
        episodes_per_robot=500,
        fault_probability=0.3,
        max_concurrent_faults=3,
        max_file_size_mb=2000
    )


def get_edge_testing_config() -> DataCollectionConfig:
    """Configuration optimized for edge device testing"""
    return DataCollectionConfig(
        dataset_name="srdmfr_edge",
        sampling_rate_hz=50,  # Lower rate for edge constraints
        episode_duration_sec=15.0,
        episodes_per_robot=100,
        fault_probability=0.4,
        max_concurrent_faults=2,
        max_file_size_mb=500  # Smaller files for edge storage
    )


# Example usage
if __name__ == "__main__":
    
    # Create data collection pipeline
    config = get_rapid_prototyping_config()
    pipeline = DataCollectionPipeline(config)
    
    # Define robot configurations (mock for now)
    robot_configs = {
        "turtlebot3": ("mobile", "turtlebot3.urdf"),
        "ur5_arm": ("manipulator", "ur5.urdf"),
        "atlas_humanoid": ("humanoid", "atlas.urdf")
    }
    
    # Start data collection
    logger.info("Starting SRDMFR data collection...")
    
    try:
        dataset_path = pipeline.collect_robot_dataset(robot_configs)
        logger.info(f"Data collection completed successfully!")
        logger.info(f"Dataset saved to: {dataset_path}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
