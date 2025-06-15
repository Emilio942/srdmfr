"""
SRDMFR Robotics Simulator Environment
====================================

Multi-robot simulation environment for Self-Repairing Diffusion Models.
Supports various robot types: Mobile robots, Manipulators, Humanoids.

Author: SRDMFR Team
Date: June 2025
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotType(Enum):
    """Supported robot types"""
    MOBILE_ROBOT = "mobile"
    MANIPULATOR = "manipulator" 
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"


@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""
    timestep: float = 1/240  # 240 Hz simulation
    gravity: Tuple[float, float, float] = (0, 0, -9.81)
    gui_enabled: bool = True
    real_time: bool = False
    physics_engine: str = "bullet"  # bullet, ode, dart
    

@dataclass 
class RobotState:
    """Complete robot state representation"""
    timestamp: float
    # Joint states
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    joint_temperatures: np.ndarray
    
    # Base pose (for mobile robots)
    base_position: np.ndarray
    base_orientation: np.ndarray  # quaternion
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    
    # Sensor data
    imu_acceleration: np.ndarray
    imu_gyroscope: np.ndarray
    force_torque: np.ndarray  # 6DOF F/T sensor
    
    # Environmental sensors
    camera_image: Optional[np.ndarray] = None
    lidar_points: Optional[np.ndarray] = None
    proximity_distances: Optional[np.ndarray] = None
    
    # System status
    battery_voltage: float = 24.0
    cpu_temperature: float = 45.0
    motor_temperatures: Optional[np.ndarray] = None


class RoboticsSimulator:
    """
    Multi-robot physics simulator for SRDMFR project.
    
    Features:
    - Multiple robot types support
    - Systematic fault injection
    - High-frequency state logging
    - Realistic sensor simulation
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.physics_client = None
        self.robots: Dict[str, 'Robot'] = {}
        self.simulation_time = 0.0
        self.step_count = 0
        
        # State logging
        self.state_history: List[Dict] = []
        self.max_history_length = 10000
        
        self._initialize_physics()
        
    def _initialize_physics(self):
        """Initialize PyBullet physics simulation"""
        if self.config.gui_enabled:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # Set physics parameters
        p.setGravity(*self.config.gravity)
        p.setTimeStep(self.config.timestep)
        p.setRealTimeSimulation(self.config.real_time)
        
        # Load ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF("plane.urdf")
        
        logger.info("Physics simulation initialized")
        
    def add_robot(self, robot_name: str, robot_type: RobotType, 
                  urdf_path: str, start_position: Tuple[float, float, float] = (0, 0, 0),
                  start_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Add a robot to the simulation"""
        
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=start_position,
            baseOrientation=start_orientation
        )
        
        # Create robot wrapper
        robot = Robot(
            robot_id=robot_id,
            name=robot_name,
            robot_type=robot_type,
            physics_client=self.physics_client
        )
        
        self.robots[robot_name] = robot
        logger.info(f"Added {robot_type.value} robot '{robot_name}' at {start_position}")
        
        return robot
    
    def add_simple_test_robot(self, robot_name: str):
        """Add a simple test robot for cases where URDF loading fails"""
        # Create a simple box robot
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], 
                                        rgbaColor=[1, 0, 0, 1])
        
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=[0, 0, 1]
        )
        
        # Create robot wrapper
        robot = Robot(
            robot_id=robot_id,
            name=robot_name,
            robot_type=RobotType.MOBILE_ROBOT,
            physics_client=self.physics_client
        )
        
        self.robots[robot_name] = robot
        logger.info(f"Added simple test robot '{robot_name}'")
        return robot
    
    def step(self, num_steps: int = 1):
        """Step the simulation forward"""
        for _ in range(num_steps):
            p.stepSimulation()
            self.simulation_time += self.config.timestep
            self.step_count += 1
            
            # Collect states from all robots
            if self.step_count % 10 == 0:  # Log every 10 steps (100Hz @ 1000Hz sim)
                self._log_states()
    
    def _log_states(self):
        """Log current states of all robots"""
        timestamp = self.simulation_time
        states = {}
        
        for name, robot in self.robots.items():
            states[name] = robot.get_state()
            
        self.state_history.append({
            'timestamp': timestamp,
            'robots': states
        })
        
        # Limit history length to prevent memory overflow
        if len(self.state_history) > self.max_history_length:
            self.state_history = self.state_history[-self.max_history_length//2:]
    
    def get_simulation_data(self) -> Dict:
        """Get all collected simulation data"""
        return {
            'config': self.config,
            'total_time': self.simulation_time,
            'total_steps': self.step_count,
            'state_history': self.state_history,
            'robots': {name: robot.get_info() for name, robot in self.robots.items()}
        }
    
    def reset(self):
        """Reset simulation to initial state"""
        for robot in self.robots.values():
            robot.reset()
        self.simulation_time = 0.0
        self.step_count = 0
        self.state_history.clear()
        logger.info("Simulation reset")
    
    def close(self):
        """Clean up and close simulation"""
        p.disconnect(self.physics_client)
        logger.info("Simulation closed")
    
    def get_robot_state(self, robot_name: str) -> Optional[RobotState]:
        """Get current state of a specific robot"""
        if robot_name in self.robots:
            return self.robots[robot_name].get_state()
        else:
            logger.warning(f"Robot '{robot_name}' not found")
            return None
    
    def disconnect(self):
        """Disconnect from physics simulation"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            logger.info("Disconnected from physics simulation")


class Robot:
    """
    Individual robot wrapper for state management and control.
    """
    
    def __init__(self, robot_id: int, name: str, robot_type: RobotType, physics_client):
        self.robot_id = robot_id
        self.name = name
        self.robot_type = robot_type
        self.physics_client = physics_client
        
        # Get robot joint information
        self.num_joints = p.getNumJoints(robot_id)
        self.joint_info = self._get_joint_info()
        self.controllable_joints = [i for i in range(self.num_joints) 
                                   if self.joint_info[i]['joint_type'] != p.JOINT_FIXED]
        
        # Initial state
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(robot_id)
        
        logger.info(f"Robot {name} initialized with {self.num_joints} joints, "
                   f"{len(self.controllable_joints)} controllable")
    
    def _get_joint_info(self) -> List[Dict]:
        """Get detailed information about all joints"""
        joint_info = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_info.append({
                'joint_index': i,
                'joint_name': info[1].decode('utf-8'),
                'joint_type': info[2],
                'q_index': info[3],
                'u_index': info[4],
                'flags': info[5],
                'joint_damping': info[6],
                'joint_friction': info[7],
                'joint_lower_limit': info[8],
                'joint_upper_limit': info[9],
                'joint_max_force': info[10],
                'joint_max_velocity': info[11],
                'link_name': info[12].decode('utf-8')
            })
        return joint_info
    
    def get_state(self) -> RobotState:
        """Get current complete robot state"""
        
        # Base pose and velocity
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel_linear, base_vel_angular = p.getBaseVelocity(self.robot_id)
        
        # Joint states
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_torques = np.array([state[3] for state in joint_states])
        
        # Simulate additional sensor readings
        imu_accel = np.random.normal(0, 0.01, 3)  # Simulated IMU noise
        imu_gyro = np.random.normal(0, 0.001, 3)
        force_torque = np.random.normal(0, 0.1, 6)  # Simulated F/T sensor
        
        # Simulate motor temperatures (based on torque)
        motor_temps = 25.0 + np.abs(joint_torques) * 2.0 + np.random.normal(0, 1, len(joint_torques))
        joint_temperatures = motor_temps.clip(20, 80)  # Realistic temperature range
        
        return RobotState(
            timestamp=time.time(),
            joint_positions=joint_positions,
            joint_velocities=joint_velocities, 
            joint_torques=joint_torques,
            joint_temperatures=joint_temperatures,
            base_position=np.array(base_pos),
            base_orientation=np.array(base_orn),
            base_linear_velocity=np.array(base_vel_linear),
            base_angular_velocity=np.array(base_vel_angular),
            imu_acceleration=imu_accel,
            imu_gyroscope=imu_gyro,
            force_torque=force_torque,
            battery_voltage=24.0 - np.random.exponential(0.1),  # Battery discharge simulation
            cpu_temperature=45.0 + np.random.normal(0, 2),
            motor_temperatures=joint_temperatures
        )
    
    def apply_joint_control(self, target_positions: np.ndarray, 
                           target_velocities: Optional[np.ndarray] = None,
                           target_torques: Optional[np.ndarray] = None):
        """Apply control commands to robot joints"""
        
        if target_velocities is None:
            target_velocities = np.zeros(len(self.controllable_joints))
        if target_torques is None:
            target_torques = np.zeros(len(self.controllable_joints))
            
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(target_positions):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_positions[i],
                    targetVelocity=target_velocities[i],
                    force=target_torques[i] if i < len(target_torques) else 100.0
                )
    
    def apply_joint_torques(self, torques: np.ndarray):
        """Apply torques directly to robot joints"""
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(torques):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.TORQUE_CONTROL,
                    force=torques[i]
                )

    def get_info(self) -> Dict:
        """Get robot information summary"""
        return {
            'name': self.name,
            'type': self.robot_type.value,
            'num_joints': self.num_joints,
            'controllable_joints': len(self.controllable_joints),
            'joint_names': [info['joint_name'] for info in self.joint_info]
        }
    
    def reset(self):
        """Reset robot to initial state"""
        p.resetBasePositionAndOrientation(
            self.robot_id, 
            self.initial_position, 
            self.initial_orientation
        )
        
        # Reset joint positions to zero
        for joint_idx in self.controllable_joints:
            p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0)


def create_standard_robots() -> Dict[str, Tuple[RobotType, str]]:
    """
    Define standard robot configurations for the SRDMFR project.
    
    Returns:
        Dict mapping robot names to (robot_type, urdf_path) tuples
    """
    
    # Note: These URDF paths would need to be actual files
    # For now, we'll use PyBullet's built-in robots where available
    
    robots = {
        # Mobile robots
        "turtlebot3": (RobotType.MOBILE_ROBOT, "turtlebot.urdf"),
        "husky": (RobotType.MOBILE_ROBOT, "husky/husky.urdf"),
        
        # Manipulator arms  
        "ur5": (RobotType.MANIPULATOR, "ur5_robot.urdf"),
        "panda": (RobotType.MANIPULATOR, "franka_panda/panda.urdf"),
        "kuka_iiwa": (RobotType.MANIPULATOR, "kuka_iiwa/model.urdf"),
        
        # Humanoid robots
        "atlas": (RobotType.HUMANOID, "atlas/atlas_v4_with_multisense.urdf"),
        "nao": (RobotType.HUMANOID, "nao.urdf"),
        
        # Quadruped robots
        "a1": (RobotType.QUADRUPED, "a1/a1.urdf"),
        "anymal": (RobotType.QUADRUPED, "anymal_c/anymal.urdf")
    }
    
    return robots


# Example usage and testing
if __name__ == "__main__":
    
    # Create simulation
    config = SimulationConfig(
        timestep=1/1000,  # 1000 Hz for high fidelity
        gui_enabled=True,
        real_time=False
    )
    
    sim = RoboticsSimulator(config)
    
    try:
        # Add a simple test robot (using PyBullet's built-in)
        # Note: This would be replaced with actual robot URDFs
        test_robot = sim.add_robot(
            robot_name="test_mobile", 
            robot_type=RobotType.MOBILE_ROBOT,
            urdf_path="r2d2.urdf",  # PyBullet built-in
            start_position=(0, 0, 0.5)
        )
        
        logger.info("Starting simulation...")
        
        # Run simulation for 5 seconds
        for i in range(5000):  # 5 seconds @ 1000Hz
            # Apply some simple control (oscillating motion)
            if len(test_robot.controllable_joints) > 0:
                targets = np.sin(sim.simulation_time * 2 * np.pi * 0.5) * 0.5
                targets = np.array([targets] * len(test_robot.controllable_joints))
                test_robot.apply_joint_control(targets)
            
            sim.step()
            
            if i % 1000 == 0:
                state = test_robot.get_state()
                logger.info(f"Time: {sim.simulation_time:.2f}s, "
                           f"Base pos: {state.base_position}, "
                           f"Joint pos: {state.joint_positions[:3]}")
        
        # Get simulation data
        data = sim.get_simulation_data()
        logger.info(f"Collected {len(data['state_history'])} state samples")
        
        # Keep simulation running for manual inspection
        if config.gui_enabled:
            logger.info("Simulation running. Press Enter to close...")
            input()
            
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        sim.close()
