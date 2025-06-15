#!/usr/bin/env python3
"""
Real Simulation Runner for SRDMFR Project

This module integrates the robotics simulator and fault injection framework
to generate realistic training data instead of mock data.
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import h5py
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulation.robotics_simulator import (
    RoboticsSimulator, SimulationConfig, RobotType, RobotState
)
from src.simulation.fault_injection import (
    FaultInjectionFramework, FaultConfig, FaultType, FaultSeverity
)

logger = logging.getLogger(__name__)


@dataclass
class RealSimulationConfig:
    """Configuration for real simulation data collection"""
    # Simulation parameters
    simulation_duration: float = 30.0  # seconds
    control_frequency: float = 100.0   # Hz
    data_logging_frequency: float = 50.0  # Hz
    
    # Robot configurations
    robot_configs: List[Dict] = None
    
    # Fault injection parameters
    fault_probability: float = 0.3  # 30% chance of fault per episode
    fault_duration_range: Tuple[float, float] = (1.0, 10.0)  # seconds
    
    # Output parameters
    output_dir: str = "data/raw/real_simulation"
    max_file_size_mb: int = 100
    
    def __post_init__(self):
        if self.robot_configs is None:
            self.robot_configs = [
                {
                    "name": "kuka_arm",
                    "type": RobotType.MANIPULATOR,
                    "urdf_path": "kuka_iiwa/model.urdf",
                    "start_position": [0, 0, 0],
                    "start_orientation": [0, 0, 0, 1]
                },
                {
                    "name": "mobile_robot",
                    "type": RobotType.MOBILE_ROBOT,
                    "urdf_path": "r2d2.urdf",
                    "start_position": [2, 0, 0],
                    "start_orientation": [0, 0, 0, 1]
                }
            ]


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class RealSimulationRunner:
    """
    Runs real physics simulations with integrated fault injection
    to generate training data for the SRDMFR diffusion model.
    """
    
    def __init__(self, config: RealSimulationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simulator
        sim_config = SimulationConfig(
            gui_enabled=False,
            gravity=(0, 0, -9.81),
            timestep=1.0 / 240.0,  # 240 Hz physics
            real_time=False
        )
        self.simulator = RoboticsSimulator(sim_config)
        
        # Initialize fault injector
        self.fault_injector = FaultInjectionFramework()
        
        # Data collection
        self.current_episode_data = []
        self.episode_count = 0
        
        logger.info("Real simulation runner initialized")
    
    def run_single_episode(self, robot_config: Dict, 
                          inject_faults: bool = True) -> Dict:
        """
        Run a single simulation episode and collect data
        
        Returns:
            Dict containing episode data and metadata
        """
        logger.info(f"Starting episode {self.episode_count} with robot {robot_config['name']}")
        
        # Reset simulation
        self.simulator.reset()
        
        # Add robot to simulation
        robot_name = robot_config['name']
        try:
            self.simulator.add_robot(
                robot_name=robot_name,
                robot_type=robot_config['type'],
                urdf_path=robot_config['urdf_path'],
                start_position=robot_config['start_position'],
                start_orientation=robot_config['start_orientation']
            )
        except Exception as e:
            logger.warning(f"Could not load URDF {robot_config['urdf_path']}: {e}")
            logger.info("Creating simple test robot instead")
            self.simulator.add_simple_test_robot(robot_name)
        
        # Configure fault injection if enabled
        faults_injected = []
        if inject_faults and np.random.random() < self.config.fault_probability:
            fault_config = self._generate_random_fault_config()
            faults_injected = [fault_config]
            self.fault_injector.configure_fault(fault_config)
            logger.info(f"Configured fault: {fault_config}")
        
        # Run simulation and collect data
        episode_data = self._run_simulation_loop(robot_name, faults_injected)
        
        self.episode_count += 1
        return episode_data
    
    def _generate_random_fault_config(self) -> FaultConfig:
        """Generate a random fault configuration for this episode"""
        fault_types = list(FaultType)
        severity_levels = list(FaultSeverity)
        
        fault_type = np.random.choice(fault_types)
        severity = np.random.choice(severity_levels)
        
        # Random timing
        start_time = np.random.uniform(2.0, self.config.simulation_duration / 2)
        duration = np.random.uniform(*self.config.fault_duration_range)
        duration = min(duration, self.config.simulation_duration - start_time)
        
        return FaultConfig(
            fault_type=fault_type,
            severity=severity,
            start_time=start_time,
            duration=duration,
            affected_joints=self._get_random_joints(fault_type),
            parameters=self._get_fault_parameters(fault_type, severity)
        )
    
    def _get_random_joints(self, fault_type: FaultType) -> List[int]:
        """Get random joint indices for fault injection"""
        if fault_type in [FaultType.ACTUATOR_BIAS, FaultType.ACTUATOR_BACKLASH, 
                         FaultType.ACTUATOR_SATURATION, FaultType.ACTUATOR_FRICTION]:
            # Affect 1-3 joints for actuator faults
            num_joints = np.random.randint(1, 4)
            return list(np.random.choice(range(7), size=num_joints, replace=False))
        else:
            # Sensor faults might affect all joints
            return list(range(7))
    
    def _get_fault_parameters(self, fault_type: FaultType, 
                            severity: FaultSeverity) -> Dict:
        """Generate fault-specific parameters"""
        base_intensity = {
            FaultSeverity.MINIMAL: 0.1,
            FaultSeverity.MILD: 0.3,
            FaultSeverity.MODERATE: 0.5,
            FaultSeverity.SEVERE: 0.7,
            FaultSeverity.CRITICAL: 0.9
        }[severity]
        
        if fault_type == FaultType.SENSOR_NOISE:
            return {
                "noise_std": base_intensity * 0.5,
                "bias_drift": base_intensity * 0.2
            }
        elif fault_type == FaultType.ACTUATOR_BIAS:
            return {
                "power_reduction": base_intensity,
                "response_delay": base_intensity * 0.1
            }
        elif fault_type == FaultType.SENSOR_DRIFT:
            return {
                "drift_rate": base_intensity * 0.01,
                "max_drift": base_intensity * 1.0
            }
        else:
            return {"intensity": base_intensity}
    
    def _run_simulation_loop(self, robot_name: str, 
                           faults_injected: List[FaultConfig]) -> Dict:
        """Run the main simulation loop and collect data"""
        
        # Timing setup
        total_steps = int(self.config.simulation_duration * self.config.control_frequency)
        log_every_n_steps = int(self.config.control_frequency / self.config.data_logging_frequency)
        
        # Data storage
        states_healthy = []
        states_corrupted = []
        timestamps = []
        fault_labels = []
        
        start_time = time.time()
        
        for step in range(total_steps):
            current_time = step / self.config.control_frequency
            
            # Apply fault injection
            current_faults = []
            for fault in faults_injected:
                if fault.start_time <= current_time <= (fault.start_time + fault.duration):
                    current_faults.append(fault)
            
            # Step simulation
            self.simulator.step()
            
            # Get robot state
            if robot_name in self.simulator.robots:
                robot_state = self.simulator.get_robot_state(robot_name)
                
                # Apply faults to get corrupted state
                corrupted_state = robot_state
                if current_faults:
                    corrupted_state = self.fault_injector.apply_faults(
                        robot_state, current_faults
                    )
                
                # Log data at specified frequency
                if step % log_every_n_steps == 0:
                    states_healthy.append(self._robot_state_to_dict(robot_state))
                    states_corrupted.append(self._robot_state_to_dict(corrupted_state))
                    timestamps.append(current_time)
                    fault_labels.append(len(current_faults) > 0)
                
                # Simple control - just try to keep robot stable
                self._apply_simple_control(robot_name, robot_state)
        
        execution_time = time.time() - start_time
        
        # Compile episode data
        episode_data = {
            'metadata': {
                'episode_id': self.episode_count,
                'robot_name': robot_name,
                'simulation_duration': self.config.simulation_duration,
                'total_samples': len(timestamps),
                'faults_injected': [self._fault_config_to_dict(f) for f in faults_injected],
                'execution_time': execution_time,
                'timestamp': time.time()
            },
            'data': {
                'timestamps': timestamps,
                'states_healthy': states_healthy,
                'states_corrupted': states_corrupted,
                'fault_labels': fault_labels
            }
        }
        
        logger.info(f"Episode completed: {len(timestamps)} samples collected in {execution_time:.2f}s")
        return episode_data
    
    def _robot_state_to_dict(self, state: RobotState) -> Dict:
        """Convert RobotState to dictionary for serialization"""
        return {
            'joint_positions': state.joint_positions.tolist(),
            'joint_velocities': state.joint_velocities.tolist(),
            'joint_torques': state.joint_torques.tolist(),
            'base_position': state.base_position.tolist(),
            'base_orientation': state.base_orientation.tolist(),
            'base_linear_velocity': state.base_linear_velocity.tolist(),
            'base_angular_velocity': state.base_angular_velocity.tolist(),
            'imu_acceleration': state.imu_acceleration.tolist(),
            'imu_gyroscope': state.imu_gyroscope.tolist(),
            'force_torque': state.force_torque.tolist(),
            'battery_voltage': state.battery_voltage,
            'cpu_temperature': state.cpu_temperature
        }
    
    def _fault_config_to_dict(self, fault_config: FaultConfig) -> Dict:
        """Convert FaultConfig to dictionary for serialization"""
        return {
            'fault_type': fault_config.fault_type.value,
            'severity': fault_config.severity.value,
            'start_time': fault_config.start_time,
            'duration': fault_config.duration,
            'affected_joints': fault_config.affected_joints,
            'parameters': fault_config.parameters
        }
    
    def _apply_simple_control(self, robot_name: str, state: RobotState):
        """Apply simple stabilizing control to the robot"""
        # Simple PD control to zero position
        if robot_name in self.simulator.robots:
            robot = self.simulator.robots[robot_name]
            
            # Target zero position with small random variations
            target_positions = np.zeros_like(state.joint_positions)
            target_positions += np.random.normal(0, 0.1, size=target_positions.shape)
            
            # PD gains
            kp = 100.0
            kd = 10.0
            
            # Compute control torques
            position_error = target_positions - state.joint_positions
            velocity_error = -state.joint_velocities  # target velocity is 0
            
            control_torques = kp * position_error + kd * velocity_error
            
            # Apply torque limits
            max_torque = 100.0
            control_torques = np.clip(control_torques, -max_torque, max_torque)
            
            # Apply control (this would be implemented in the robot class)
            try:
                robot.apply_joint_torques(control_torques)
            except AttributeError:
                # Method not implemented yet, skip
                pass
    
    def save_episode_data(self, episode_data: Dict, file_path: Optional[str] = None) -> str:
        """Save episode data to HDF5 file"""
        if file_path is None:
            timestamp = int(time.time())
            file_path = self.output_dir / f"episode_{self.episode_count:06d}_{timestamp}.h5"
        
        with h5py.File(file_path, 'w') as f:
            # Save metadata (convert numpy types to JSON-serializable types)
            metadata_group = f.create_group('metadata')
            for key, value in episode_data['metadata'].items():
                # Convert to JSON-serializable format
                serializable_value = convert_to_json_serializable(value)
                if isinstance(serializable_value, list):
                    metadata_group.create_dataset(key, data=json.dumps(serializable_value))
                else:
                    metadata_group.attrs[key] = serializable_value
            
            # Save data arrays
            data_group = f.create_group('data')
            for key, value in episode_data['data'].items():
                if key in ['states_healthy', 'states_corrupted']:
                    # Handle list of dictionaries
                    state_group = data_group.create_group(key)
                    if value:  # Check if list is not empty
                        # Convert list of dicts to dict of arrays
                        state_dict = {}
                        for state_key in value[0].keys():
                            state_dict[state_key] = [state[state_key] for state in value]
                        
                        for state_key, state_values in state_dict.items():
                            state_group.create_dataset(state_key, data=np.array(state_values))
                else:
                    data_group.create_dataset(key, data=value)
        
        logger.info(f"Episode data saved to {file_path}")
        return str(file_path)
    
    def generate_dataset(self, num_episodes: int = 10) -> str:
        """Generate a complete dataset with multiple episodes"""
        logger.info(f"Starting dataset generation with {num_episodes} episodes")
        
        episode_files = []
        
        for i in range(num_episodes):
            # Randomly select robot configuration
            robot_config = np.random.choice(self.config.robot_configs)
            
            # Run episode
            episode_data = self.run_single_episode(robot_config)
            
            # Save episode
            file_path = self.save_episode_data(episode_data)
            episode_files.append(file_path)
            
            # Progress logging
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i + 1}/{num_episodes} episodes")
        
        # Create dataset manifest
        manifest_path = self.output_dir / "dataset_manifest.json"
        manifest = {
            "dataset_info": {
                "total_episodes": num_episodes,
                "generation_time": time.time(),
                "config": convert_to_json_serializable(self.config.__dict__)
            },
            "episode_files": episode_files
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Dataset generation complete! {num_episodes} episodes saved.")
        logger.info(f"Dataset manifest: {manifest_path}")
        
        return str(manifest_path)
    
    def cleanup(self):
        """Clean up simulation resources"""
        if hasattr(self.simulator, 'physics_client') and self.simulator.physics_client is not None:
            self.simulator.disconnect()
        logger.info("Simulation resources cleaned up")


def main():
    """Example usage of RealSimulationRunner"""
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = RealSimulationConfig(
        simulation_duration=20.0,
        control_frequency=50.0,
        data_logging_frequency=25.0,
        fault_probability=0.4,
        output_dir="data/raw/real_simulation_test"
    )
    
    # Create runner
    runner = RealSimulationRunner(config)
    
    try:
        # Generate small test dataset
        manifest_path = runner.generate_dataset(num_episodes=5)
        print(f"Dataset generated successfully: {manifest_path}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
