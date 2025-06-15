"""
SRDMFR Data Collection Integration & Test
========================================

Integration script that brings together simulation, fault injection,
and data collection for the SRDMFR project.

Author: SRDMFR Team
Date: June 2025
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('srdmfr_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MockDataCollectionSystem:
    """
    Mock implementation of complete data collection system.
    
    This serves as a placeholder until actual simulation libraries are installed.
    Demonstrates the full pipeline with realistic data generation.
    """
    
    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.collection_stats = {
            'start_time': datetime.now(),
            'episodes_collected': 0,
            'total_samples': 0,
            'faults_injected': 0,
            'robots_processed': 0
        }
        
        logger.info("Mock SRDMFR data collection system initialized")
        
    def run_data_collection(self, duration_minutes: int = 5) -> Dict:
        """
        Run complete data collection pipeline.
        
        Args:
            duration_minutes: How long to run collection
            
        Returns:
            Collection summary statistics
        """
        
        logger.info(f"Starting data collection for {duration_minutes} minutes")
        
        # Define robot configurations
        robot_configs = {
            "turtlebot3": {
                "type": "mobile_robot",
                "joints": 2,
                "sensors": ["imu", "encoders", "battery"],
                "urdf": "turtlebot3.urdf"
            },
            "ur5_arm": {
                "type": "manipulator", 
                "joints": 6,
                "sensors": ["encoders", "force_torque", "temperature"],
                "urdf": "ur5.urdf"
            },
            "atlas_v4": {
                "type": "humanoid",
                "joints": 30,
                "sensors": ["imu", "encoders", "force_torque", "temperature"],
                "urdf": "atlas.urdf"
            }
        }
        
        # Collection parameters
        episodes_per_robot = 20  # Reduced for demo
        episode_duration = 10.0  # seconds
        sampling_rate = 100  # Hz
        
        # Process each robot type
        for robot_name, robot_config in robot_configs.items():
            
            logger.info(f"Processing robot: {robot_name}")
            
            robot_data = self._collect_robot_data(
                robot_name, 
                robot_config, 
                episodes_per_robot,
                episode_duration,
                sampling_rate
            )
            
            self._save_robot_dataset(robot_name, robot_data)
            
            self.collection_stats['robots_processed'] += 1
            
            # Check time limit
            elapsed = (datetime.now() - self.collection_stats['start_time']).total_seconds() / 60
            if elapsed >= duration_minutes:
                logger.info(f"Time limit reached ({duration_minutes} min)")
                break
                
        # Generate final report
        summary = self._generate_collection_summary()
        self._save_collection_report(summary)
        
        return summary
    
    def _collect_robot_data(self, robot_name: str, robot_config: Dict,
                           num_episodes: int, episode_duration: float,
                           sampling_rate: int) -> Dict:
        """Collect data for a specific robot type"""
        
        robot_data = {
            'robot_name': robot_name,
            'robot_config': robot_config,
            'episodes': [],
            'collection_time': datetime.now().isoformat()
        }
        
        for episode_idx in range(num_episodes):
            
            episode_id = f"{robot_name}_ep_{episode_idx:03d}"
            
            logger.info(f"  Collecting episode {episode_idx+1}/{num_episodes}: {episode_id}")
            
            # Generate episode data
            episode_data = self._generate_episode_data(
                episode_id, robot_config, episode_duration, sampling_rate
            )
            
            robot_data['episodes'].append(episode_data)
            self.collection_stats['episodes_collected'] += 1
            
            if episode_idx % 5 == 0:
                logger.info(f"  Progress: {episode_idx+1}/{num_episodes} episodes")
                
        return robot_data
    
    def _generate_episode_data(self, episode_id: str, robot_config: Dict,
                              duration: float, sampling_rate: int) -> Dict:
        """Generate realistic episode data with fault injection"""
        
        import numpy as np  # Mock import
        
        num_samples = int(duration * sampling_rate)
        time_vector = np.linspace(0, duration, num_samples)
        
        # Robot parameters
        num_joints = robot_config['joints']
        sensors = robot_config['sensors']
        
        # Generate base trajectories
        joint_positions = []
        joint_velocities = []
        joint_torques = []
        
        for t in time_vector:
            # Sinusoidal motion with some randomness
            pos = np.sin(2 * np.pi * 0.1 * t + np.random.uniform(0, 2*np.pi, num_joints)) * 0.5
            vel = 0.1 * 2 * np.pi * np.cos(2 * np.pi * 0.1 * t + np.random.uniform(0, 2*np.pi, num_joints))
            torque = np.random.normal(0, 1, num_joints)
            
            joint_positions.append(pos)
            joint_velocities.append(vel)
            joint_torques.append(torque)
            
        # Convert to arrays
        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)
        joint_torques = np.array(joint_torques)
        
        # Generate sensor data
        sensor_data = {}
        
        if 'imu' in sensors:
            # IMU data (acceleration + gyroscope)
            sensor_data['imu_acceleration'] = np.random.normal([0, 0, 9.81], 0.01, (num_samples, 3))
            sensor_data['imu_gyroscope'] = np.random.normal(0, 0.001, (num_samples, 3))
            
        if 'force_torque' in sensors:
            # 6-DOF force/torque sensor
            sensor_data['force_torque'] = np.random.normal(0, 0.1, (num_samples, 6))
            
        if 'temperature' in sensors:
            # Motor temperatures
            base_temp = 25.0
            load_effect = np.abs(joint_torques) * 2.0  # Temperature rises with load
            thermal_noise = np.random.normal(0, 1, (num_samples, num_joints))
            sensor_data['motor_temperatures'] = base_temp + load_effect + thermal_noise
            
        if 'battery' in sensors:
            # Battery voltage (slowly decreasing)
            initial_voltage = 24.0
            discharge_rate = 0.1 / 3600  # 0.1V per hour
            sensor_data['battery_voltage'] = initial_voltage - time_vector * discharge_rate + np.random.normal(0, 0.01, num_samples)
            
        # Fault injection decision
        fault_probability = 0.3
        inject_fault = np.random.random() < fault_probability
        
        fault_info = []
        if inject_fault:
            fault_info = self._inject_mock_faults(
                joint_positions, joint_velocities, joint_torques, 
                sensor_data, time_vector
            )
            
        # Create episode data structure
        episode_data = {
            'episode_id': episode_id,
            'duration_sec': duration,
            'sampling_rate_hz': sampling_rate,
            'num_samples': num_samples,
            'robot_config': robot_config,
            
            # Joint data
            'joint_positions': joint_positions.tolist(),
            'joint_velocities': joint_velocities.tolist(), 
            'joint_torques': joint_torques.tolist(),
            
            # Sensor data
            'sensor_data': {k: v.tolist() for k, v in sensor_data.items()},
            
            # Fault information
            'faults_injected': fault_info,
            'has_faults': len(fault_info) > 0,
            
            # Metadata
            'collection_timestamp': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(joint_positions, sensor_data)
        }
        
        # Update statistics
        self.collection_stats['total_samples'] += num_samples
        self.collection_stats['faults_injected'] += len(fault_info)
        
        return episode_data
    
    def _inject_mock_faults(self, joint_pos, joint_vel, joint_torques,
                           sensor_data, time_vector) -> List[Dict]:
        """Apply mock fault injection to data"""
        
        import numpy as np
        
        faults = []
        
        # Random fault selection
        fault_types = ['sensor_noise', 'sensor_bias', 'actuator_degradation', 'power_drop']
        num_faults = np.random.randint(1, 3)
        
        for i in range(num_faults):
            fault_type = np.random.choice(fault_types)
            start_time = np.random.uniform(0, len(time_vector) * 0.7)
            duration = np.random.uniform(len(time_vector) * 0.1, len(time_vector) * 0.5)
            
            start_idx = int(start_time)
            end_idx = int(min(start_idx + duration, len(time_vector)))
            
            fault_info = {
                'fault_type': fault_type,
                'start_sample': start_idx,
                'end_sample': end_idx,
                'affected_components': [],
                'severity': np.random.choice(['mild', 'moderate', 'severe'])
            }
            
            # Apply fault effects
            if fault_type == 'sensor_noise':
                # Add noise to IMU
                if 'imu_acceleration' in sensor_data:
                    noise_level = 0.1 if fault_info['severity'] == 'severe' else 0.05
                    sensor_data['imu_acceleration'][start_idx:end_idx] += np.random.normal(0, noise_level, (end_idx-start_idx, 3))
                    fault_info['affected_components'].append('imu_acceleration')
                    
            elif fault_type == 'sensor_bias':
                # Add bias to joint encoders
                bias_level = 0.1 if fault_info['severity'] == 'severe' else 0.05
                affected_joints = np.random.choice(joint_pos.shape[1], size=np.random.randint(1, 3), replace=False)
                for joint_idx in affected_joints:
                    joint_pos[start_idx:end_idx, joint_idx] += bias_level
                fault_info['affected_components'] = [f'joint_{j}' for j in affected_joints]
                
            elif fault_type == 'actuator_degradation':
                # Reduce torque output
                reduction_factor = 0.5 if fault_info['severity'] == 'severe' else 0.8
                affected_joints = np.random.choice(joint_torques.shape[1], size=np.random.randint(1, 2), replace=False)
                for joint_idx in affected_joints:
                    joint_torques[start_idx:end_idx, joint_idx] *= reduction_factor
                fault_info['affected_components'] = [f'actuator_{j}' for j in affected_joints]
                
            elif fault_type == 'power_drop':
                # Reduce battery voltage
                if 'battery_voltage' in sensor_data:
                    voltage_drop = 2.0 if fault_info['severity'] == 'severe' else 1.0
                    sensor_data['battery_voltage'][start_idx:end_idx] -= voltage_drop
                    fault_info['affected_components'].append('battery_voltage')
                    
            faults.append(fault_info)
            
        return faults
    
    def _assess_data_quality(self, joint_positions, sensor_data) -> Dict:
        """Assess quality metrics for collected data"""
        
        import numpy as np
        
        quality = {
            'joint_motion_variance': float(np.var(joint_positions)),
            'data_completeness': 1.0,  # Mock - assume complete
            'signal_to_noise_ratio': 20.0 + np.random.uniform(-5, 5),  # Mock SNR
            'temporal_consistency': 0.95 + np.random.uniform(-0.05, 0.05)
        }
        
        return quality
    
    def _save_robot_dataset(self, robot_name: str, robot_data: Dict):
        """Save robot dataset to file"""
        
        # Create robot-specific directory
        robot_dir = self.output_dir / robot_name
        robot_dir.mkdir(exist_ok=True)
        
        # Save raw data
        data_file = robot_dir / f"{robot_name}_episodes.json"
        with open(data_file, 'w') as f:
            json.dump(robot_data, f, indent=2, default=str)
            
        # Save metadata
        metadata = {
            'robot_name': robot_name,
            'total_episodes': len(robot_data['episodes']),
            'total_samples': sum(ep['num_samples'] for ep in robot_data['episodes']),
            'episodes_with_faults': sum(1 for ep in robot_data['episodes'] if ep['has_faults']),
            'collection_time': robot_data['collection_time'],
            'file_size_bytes': data_file.stat().st_size
        }
        
        metadata_file = robot_dir / f"{robot_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved {robot_name} dataset: {metadata['total_episodes']} episodes, "
                   f"{metadata['total_samples']} samples")
    
    def _generate_collection_summary(self) -> Dict:
        """Generate comprehensive collection summary"""
        
        end_time = datetime.now()
        duration = (end_time - self.collection_stats['start_time']).total_seconds()
        
        summary = {
            'collection_info': {
                'start_time': self.collection_stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': duration / 60.0,
                'success': True
            },
            'data_statistics': {
                'robots_processed': self.collection_stats['robots_processed'],
                'total_episodes': self.collection_stats['episodes_collected'],
                'total_samples': self.collection_stats['total_samples'],
                'faults_injected': self.collection_stats['faults_injected'],
                'samples_per_second': self.collection_stats['total_samples'] / duration,
                'fault_injection_rate': self.collection_stats['faults_injected'] / self.collection_stats['episodes_collected']
            },
            'output_info': {
                'output_directory': str(self.output_dir),
                'files_created': len(list(self.output_dir.rglob('*.json'))),
                'total_size_mb': sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file()) / 1e6
            }
        }
        
        return summary
    
    def _save_collection_report(self, summary: Dict):
        """Save final collection report"""
        
        report_file = self.output_dir / 'collection_report.json'
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Create human-readable summary
        txt_report = self.output_dir / 'collection_summary.txt'
        with open(txt_report, 'w') as f:
            f.write("SRDMFR Data Collection Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Collection completed: {summary['collection_info']['end_time']}\n")
            f.write(f"Duration: {summary['collection_info']['duration_minutes']:.1f} minutes\n\n")
            
            f.write("Data Statistics:\n")
            f.write(f"  - Robots processed: {summary['data_statistics']['robots_processed']}\n")
            f.write(f"  - Total episodes: {summary['data_statistics']['total_episodes']}\n")
            f.write(f"  - Total samples: {summary['data_statistics']['total_samples']:,}\n")
            f.write(f"  - Faults injected: {summary['data_statistics']['faults_injected']}\n")
            f.write(f"  - Fault rate: {summary['data_statistics']['fault_injection_rate']:.2f} faults/episode\n\n")
            
            f.write("Output Information:\n")
            f.write(f"  - Output directory: {summary['output_info']['output_directory']}\n")
            f.write(f"  - Files created: {summary['output_info']['files_created']}\n")
            f.write(f"  - Total size: {summary['output_info']['total_size_mb']:.1f} MB\n")
            
        logger.info(f"Collection report saved to {report_file}")


def run_srdmfr_data_collection():
    """Main function to run SRDMFR data collection"""
    
    try:
        # Check if numpy is available (mock import)
        try:
            import numpy as np
            logger.info("NumPy available - using realistic data generation")
        except ImportError:
            logger.warning("NumPy not available - using simplified mock data")
            # Create a simple numpy substitute for basic operations
            class MockNumPy:
                @staticmethod
                def random():
                    import random
                    return random.random()
                @staticmethod
                def linspace(start, stop, num):
                    step = (stop - start) / (num - 1)
                    return [start + i * step for i in range(num)]
                @staticmethod 
                def sin(x):
                    import math
                    if isinstance(x, list):
                        return [math.sin(xi) for xi in x]
                    return math.sin(x)
                @staticmethod
                def array(data):
                    return data
            
            # Monkey patch for demo
            import sys
            sys.modules['numpy'] = MockNumPy()
            np = MockNumPy()
        
        # Initialize data collection system
        logger.info("Initializing SRDMFR Data Collection System...")
        
        data_collector = MockDataCollectionSystem()
        
        # Run data collection
        logger.info("Starting data collection pipeline...")
        start_time = time.time()
        
        summary = data_collector.run_data_collection(duration_minutes=2)  # Short demo run
        
        end_time = time.time()
        
        # Report results
        logger.info("=" * 50)
        logger.info("SRDMFR DATA COLLECTION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Duration: {end_time - start_time:.1f} seconds")
        logger.info(f"Robots processed: {summary['data_statistics']['robots_processed']}")
        logger.info(f"Episodes collected: {summary['data_statistics']['total_episodes']}")
        logger.info(f"Total samples: {summary['data_statistics']['total_samples']:,}")
        logger.info(f"Faults injected: {summary['data_statistics']['faults_injected']}")
        logger.info(f"Output size: {summary['output_info']['total_size_mb']:.1f} MB")
        logger.info(f"Output directory: {summary['output_info']['output_directory']}")
        logger.info("=" * 50)
        
        return summary
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


if __name__ == "__main__":
    print("🤖 SRDMFR - Self-Repairing Diffusion Models für Robotikzustände")
    print("📊 Data Collection Pipeline Demo")
    print("-" * 60)
    
    summary = run_srdmfr_data_collection()
    
    print("\n✅ Data collection demo completed successfully!")
    print(f"📁 Check output directory: {summary['output_info']['output_directory']}")
    print(f"📈 Collected {summary['data_statistics']['total_samples']} samples across {summary['data_statistics']['total_episodes']} episodes")
