"""
SRDMFR Fault Injection Framework
===============================

Systematic fault injection for robotics sensors and actuators.
Enables controlled generation of corrupted robot states for training.

Author: SRDMFR Team  
Date: June 2025
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import copy
import logging

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of faults that can be injected"""
    # Sensor faults
    SENSOR_NOISE = "sensor_noise"
    SENSOR_BIAS = "sensor_bias" 
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_SCALE = "sensor_scale"
    
    # Actuator faults
    ACTUATOR_BIAS = "actuator_bias"
    ACTUATOR_SATURATION = "actuator_saturation"
    ACTUATOR_BACKLASH = "actuator_backlash"
    ACTUATOR_FRICTION = "actuator_friction"
    ACTUATOR_DEADZONE = "actuator_deadzone"
    
    # System faults
    POWER_DROP = "power_drop"
    THERMAL_FAULT = "thermal_fault"
    COMMUNICATION_LOSS = "comm_loss"
    TIMING_FAULT = "timing_fault"


class FaultSeverity(Enum):
    """Severity levels for fault injection"""
    MINIMAL = 0.1      # Barely noticeable
    MILD = 0.3         # Noticeable but manageable
    MODERATE = 0.5     # Significant impact
    SEVERE = 0.7       # Major impact
    CRITICAL = 0.9     # System-threatening


@dataclass
class FaultParameters:
    """Parameters for a specific fault"""
    fault_type: FaultType
    severity: FaultSeverity
    affected_sensors: List[str] = field(default_factory=list)
    affected_joints: List[int] = field(default_factory=list)
    
    # Timing parameters
    start_time: float = 0.0
    duration: Optional[float] = None  # None = permanent
    intermittent: bool = False
    intermittent_period: float = 1.0
    
    # Fault-specific parameters
    noise_std: float = 0.1
    bias_value: float = 0.0
    drift_rate: float = 0.01  # per second
    dropout_probability: float = 0.1
    stuck_value: Optional[float] = None
    scale_factor: float = 1.0
    
    # Environmental parameters
    temperature_effect: bool = False
    vibration_effect: bool = False
    electromagnetic_interference: bool = False


@dataclass
class FaultState:
    """Current state of an active fault"""
    parameters: FaultParameters
    start_time: float
    current_time: float
    is_active: bool = True
    accumulated_drift: float = 0.0
    last_output: Optional[np.ndarray] = None


@dataclass
class FaultConfig:
    """Simplified fault configuration for external use"""
    fault_type: FaultType
    severity: FaultSeverity
    start_time: float
    duration: float
    affected_joints: List[int]
    parameters: Dict


class FaultInjectionFramework:
    """
    Comprehensive fault injection system for robotics simulation.
    
    Features:
    - Multiple fault types (sensor, actuator, system)
    - Configurable severity levels
    - Temporal fault patterns (permanent, intermittent, progressive)
    - Realistic fault modeling based on physical principles
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        self.active_faults: Dict[str, FaultState] = {}
        self.fault_history: List[Dict] = []
        self.simulation_time = 0.0
        
        # Fault libraries - realistic parameter ranges
        self._initialize_fault_libraries()
        
    def _initialize_fault_libraries(self):
        """Initialize libraries of realistic fault parameters"""
        
        # Sensor noise characteristics (based on real sensor specs)
        self.sensor_noise_profiles = {
            'imu_accelerometer': {'std': 0.02, 'bias_stability': 0.001},  # m/s²
            'imu_gyroscope': {'std': 0.001, 'bias_stability': 0.0001},   # rad/s
            'encoder': {'std': 0.0001, 'bias_stability': 0.00001},       # rad
            'force_torque': {'std': 0.1, 'bias_stability': 0.01},        # N, Nm
            'temperature': {'std': 0.5, 'bias_stability': 0.1},          # °C
            'voltage': {'std': 0.01, 'bias_stability': 0.001},           # V
        }
        
        # Environmental effect models
        self.environmental_effects = {
            'temperature': {
                'coeff_drift': 0.001,  # %/°C
                'threshold_high': 60.0,  # °C
                'threshold_low': -10.0   # °C
            },
            'vibration': {
                'frequency_range': (10, 1000),  # Hz
                'amplitude_factor': 2.0
            },
            'emi': {
                'frequency_range': (1e6, 100e6),  # Hz
                'interference_level': 0.05
            }
        }
    
    def add_fault(self, fault_id: str, fault_params: FaultParameters):
        """Add a new fault to the injection system"""
        
        fault_state = FaultState(
            parameters=fault_params,
            start_time=self.simulation_time,
            current_time=self.simulation_time
        )
        
        self.active_faults[fault_id] = fault_state
        
        logger.info(f"Added fault '{fault_id}': {fault_params.fault_type.value} "
                   f"(severity: {fault_params.severity.value})")
    
    def remove_fault(self, fault_id: str):
        """Remove an active fault"""
        if fault_id in self.active_faults:
            del self.active_faults[fault_id]
            logger.info(f"Removed fault '{fault_id}'")
    
    def update_time(self, simulation_time: float):
        """Update simulation time and manage fault lifecycles"""
        self.simulation_time = simulation_time
        
        # Update active faults
        expired_faults = []
        for fault_id, fault_state in self.active_faults.items():
            fault_state.current_time = simulation_time
            
            # Check if fault should expire
            if (fault_state.parameters.duration is not None and 
                simulation_time - fault_state.start_time > fault_state.parameters.duration):
                expired_faults.append(fault_id)
                continue
            
            # Update accumulated effects (e.g., drift)
            dt = 0.001  # Assume 1ms timestep
            if fault_state.parameters.fault_type == FaultType.SENSOR_DRIFT:
                fault_state.accumulated_drift += fault_state.parameters.drift_rate * dt
        
        # Remove expired faults
        for fault_id in expired_faults:
            self.remove_fault(fault_id)
    
    def inject_faults(self, robot_state, sensor_name: str = None) -> 'RobotState':
        """
        Apply all active faults to a robot state.
        
        Args:
            robot_state: Clean robot state
            sensor_name: Specific sensor to target (None = all applicable)
            
        Returns:
            Corrupted robot state
        """
        
        # Create a copy to avoid modifying original
        corrupted_state = copy.deepcopy(robot_state)
        
        for fault_id, fault_state in self.active_faults.items():
            if not self._is_fault_active(fault_state):
                continue
                
            fault_params = fault_state.parameters
            
            # Apply fault based on type
            if fault_params.fault_type == FaultType.SENSOR_NOISE:
                corrupted_state = self._apply_sensor_noise(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.SENSOR_BIAS:
                corrupted_state = self._apply_sensor_bias(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.SENSOR_DRIFT:
                corrupted_state = self._apply_sensor_drift(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.SENSOR_DROPOUT:
                corrupted_state = self._apply_sensor_dropout(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.SENSOR_STUCK:
                corrupted_state = self._apply_sensor_stuck(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.ACTUATOR_BACKLASH:
                corrupted_state = self._apply_actuator_backlash(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.POWER_DROP:
                corrupted_state = self._apply_power_drop(
                    corrupted_state, fault_params, fault_state)
                
            elif fault_params.fault_type == FaultType.THERMAL_FAULT:
                corrupted_state = self._apply_thermal_fault(
                    corrupted_state, fault_params, fault_state)
        
        return corrupted_state
    
    def _is_fault_active(self, fault_state: FaultState) -> bool:
        """Check if a fault should be active at current time"""
        
        # Check if fault has started
        if self.simulation_time < fault_state.parameters.start_time:
            return False
            
        # Check intermittent behavior
        if fault_state.parameters.intermittent:
            cycle_time = (self.simulation_time - fault_state.start_time) % (
                fault_state.parameters.intermittent_period * 2)
            return cycle_time < fault_state.parameters.intermittent_period
            
        return True
    
    def _apply_sensor_noise(self, state, fault_params: FaultParameters, 
                           fault_state: FaultState):
        """Apply additive noise to sensor readings"""
        
        severity_scale = fault_params.severity.value
        noise_std = fault_params.noise_std * severity_scale
        
        # IMU acceleration noise
        if 'imu_acceleration' in fault_params.affected_sensors or not fault_params.affected_sensors:
            noise = np.random.normal(0, noise_std, state.imu_acceleration.shape)
            state.imu_acceleration += noise
            
        # IMU gyroscope noise  
        if 'imu_gyroscope' in fault_params.affected_sensors or not fault_params.affected_sensors:
            noise = np.random.normal(0, noise_std * 0.1, state.imu_gyroscope.shape)
            state.imu_gyroscope += noise
            
        # Joint encoder noise
        if 'joint_encoders' in fault_params.affected_sensors or not fault_params.affected_sensors:
            if len(fault_params.affected_joints) > 0:
                for joint_idx in fault_params.affected_joints:
                    if joint_idx < len(state.joint_positions):
                        noise = np.random.normal(0, noise_std * 0.01)
                        state.joint_positions[joint_idx] += noise
            else:
                noise = np.random.normal(0, noise_std * 0.01, state.joint_positions.shape)
                state.joint_positions += noise
        
        # Force/Torque sensor noise
        if 'force_torque' in fault_params.affected_sensors or not fault_params.affected_sensors:
            noise = np.random.normal(0, noise_std * 10, state.force_torque.shape)
            state.force_torque += noise
            
        return state
    
    def _apply_sensor_bias(self, state, fault_params: FaultParameters, 
                          fault_state: FaultState):
        """Apply constant bias to sensor readings"""
        
        bias_magnitude = fault_params.bias_value * fault_params.severity.value
        
        # Apply bias to specified sensors
        if 'imu_acceleration' in fault_params.affected_sensors:
            state.imu_acceleration += bias_magnitude
            
        if 'joint_encoders' in fault_params.affected_sensors:
            for joint_idx in fault_params.affected_joints:
                if joint_idx < len(state.joint_positions):
                    state.joint_positions[joint_idx] += bias_magnitude * 0.1
                    
        return state
    
    def _apply_sensor_drift(self, state, fault_params: FaultParameters,
                           fault_state: FaultState):
        """Apply time-varying drift to sensor readings"""
        
        drift_value = (fault_state.accumulated_drift * 
                      fault_params.severity.value)
        
        if 'imu_acceleration' in fault_params.affected_sensors:
            state.imu_acceleration += drift_value
            
        if 'joint_encoders' in fault_params.affected_sensors:
            for joint_idx in fault_params.affected_joints:
                if joint_idx < len(state.joint_positions):
                    state.joint_positions[joint_idx] += drift_value * 0.01
                    
        return state
    
    def _apply_sensor_dropout(self, state, fault_params: FaultParameters,
                             fault_state: FaultState):
        """Apply random sensor dropouts"""
        
        dropout_prob = (fault_params.dropout_probability * 
                       fault_params.severity.value)
        
        if np.random.random() < dropout_prob:
            if 'imu_acceleration' in fault_params.affected_sensors:
                # Use last known value or zero
                if fault_state.last_output is not None:
                    state.imu_acceleration = fault_state.last_output.copy()
                else:
                    state.imu_acceleration = np.zeros_like(state.imu_acceleration)
        else:
            # Store current value for potential future dropouts
            fault_state.last_output = state.imu_acceleration.copy()
            
        return state
    
    def _apply_sensor_stuck(self, state, fault_params: FaultParameters,
                           fault_state: FaultState):
        """Apply stuck sensor values"""
        
        if fault_params.stuck_value is not None:
            stuck_val = fault_params.stuck_value
            
            if 'joint_encoders' in fault_params.affected_sensors:
                for joint_idx in fault_params.affected_joints:
                    if joint_idx < len(state.joint_positions):
                        state.joint_positions[joint_idx] = stuck_val
                        
        return state
    
    def _apply_actuator_backlash(self, state, fault_params: FaultParameters,
                               fault_state: FaultState):
        """Apply backlash to joint positions"""
        
        backlash_amount = 0.05 * fault_params.severity.value  # 5 degrees max
        
        for joint_idx in fault_params.affected_joints:
            if joint_idx < len(state.joint_positions):
                # Simulate backlash as hysteresis in position
                velocity = state.joint_velocities[joint_idx] if joint_idx < len(state.joint_velocities) else 0
                if abs(velocity) > 0.01:  # Moving
                    backlash_error = backlash_amount * np.sign(velocity) * np.random.uniform(0.5, 1.0)
                    state.joint_positions[joint_idx] += backlash_error
                    
        return state
    
    def _apply_power_drop(self, state, fault_params: FaultParameters,
                         fault_state: FaultState):
        """Apply power supply issues"""
        
        voltage_drop = 2.0 * fault_params.severity.value  # Up to 2V drop
        state.battery_voltage -= voltage_drop
        
        # Reduce maximum torques proportionally
        voltage_ratio = state.battery_voltage / 24.0  # Nominal 24V
        state.joint_torques *= max(0.1, voltage_ratio)  # Minimum 10% torque
        
        return state
    
    def _apply_thermal_fault(self, state, fault_params: FaultParameters,
                           fault_state: FaultState):
        """Apply thermal effects on sensors and actuators"""
        
        temp_increase = 20.0 * fault_params.severity.value  # Up to 20°C increase
        
        # Increase motor temperatures
        if state.motor_temperatures is not None:
            state.motor_temperatures += temp_increase
            
            # Add thermal noise to sensors
            thermal_noise_std = 0.001 * temp_increase
            state.imu_acceleration += np.random.normal(0, thermal_noise_std, 3)
            
            # Reduce actuator performance at high temperatures
            for i, temp in enumerate(state.motor_temperatures):
                if temp > 70 and i < len(state.joint_torques):  # High temperature threshold
                    reduction_factor = max(0.5, 1.0 - (temp - 70) / 50.0)
                    state.joint_torques[i] *= reduction_factor
                    
        return state
    
    def create_fault_scenario(self, scenario_name: str) -> List[FaultParameters]:
        """Create predefined fault scenarios for testing"""
        
        scenarios = {
            'sensor_degradation': [
                FaultParameters(
                    fault_type=FaultType.SENSOR_NOISE,
                    severity=FaultSeverity.MILD,
                    affected_sensors=['imu_acceleration', 'imu_gyroscope'],
                    start_time=1.0,
                    duration=5.0
                ),
                FaultParameters(
                    fault_type=FaultType.SENSOR_DRIFT,
                    severity=FaultSeverity.MODERATE,
                    affected_sensors=['joint_encoders'],
                    affected_joints=[0, 1, 2],
                    start_time=2.0,
                    drift_rate=0.005
                )
            ],
            
            'actuator_problems': [
                FaultParameters(
                    fault_type=FaultType.ACTUATOR_BACKLASH,
                    severity=FaultSeverity.MODERATE,
                    affected_joints=[1, 2],
                    start_time=0.5
                ),
                FaultParameters(
                    fault_type=FaultType.POWER_DROP,
                    severity=FaultSeverity.SEVERE,
                    start_time=3.0,
                    duration=2.0
                )
            ],
            
            'intermittent_faults': [
                FaultParameters(
                    fault_type=FaultType.SENSOR_DROPOUT,
                    severity=FaultSeverity.MODERATE,
                    affected_sensors=['imu_acceleration'],
                    intermittent=True,
                    intermittent_period=0.5,
                    dropout_probability=0.3,
                    start_time=1.0
                )
            ],
            
            'critical_failure': [
                FaultParameters(
                    fault_type=FaultType.SENSOR_STUCK,
                    severity=FaultSeverity.CRITICAL,
                    affected_sensors=['joint_encoders'],
                    affected_joints=[0],
                    stuck_value=0.0,
                    start_time=2.0
                ),
                FaultParameters(
                    fault_type=FaultType.THERMAL_FAULT,
                    severity=FaultSeverity.SEVERE,
                    start_time=2.5
                )
            ]
        }
        
        return scenarios.get(scenario_name, [])
    
    def get_fault_statistics(self) -> Dict:
        """Get statistics about active and historical faults"""
        
        active_count = len(self.active_faults)
        fault_types = [fs.parameters.fault_type.value for fs in self.active_faults.values()]
        
        return {
            'active_faults': active_count,
            'fault_types_active': fault_types,
            'total_fault_history': len(self.fault_history),
            'simulation_time': self.simulation_time
        }
    
    def configure_fault(self, fault_config):
        """Configure a fault for injection (compatible with RealSimulationRunner)"""
        fault_id = f"fault_{len(self.active_faults)}"
        
        # Convert fault_config to FaultParameters
        fault_params = FaultParameters(
            fault_type=fault_config.fault_type,
            severity=fault_config.severity,
            affected_joints=fault_config.affected_joints,
            start_time=fault_config.start_time,
            duration=fault_config.duration
        )
        
        # Apply specific parameters
        for key, value in fault_config.parameters.items():
            if hasattr(fault_params, key):
                setattr(fault_params, key, value)
        
        self.add_fault(fault_id, fault_params)
        return fault_id
    
    def apply_faults(self, robot_state, fault_configs):
        """Apply faults to robot state (compatible with RealSimulationRunner)"""
        # Update fault timing
        for fault in fault_configs:
            self.update_time(fault.start_time)
        
        # Apply all active faults
        return self.inject_faults(robot_state)

# Example usage and testing
if __name__ == "__main__":
    
    # Create fault injection framework
    fault_injector = FaultInjectionFramework(random_seed=42)
    
    # Create a mock robot state for testing
    class MockRobotState:
        def __init__(self):
            self.timestamp = time.time()
            self.joint_positions = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
            self.joint_velocities = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.joint_torques = np.array([1.0, 2.0, 1.5, 0.5, 0.8, 0.3])
            self.imu_acceleration = np.array([0.0, 0.0, 9.81])
            self.imu_gyroscope = np.array([0.0, 0.0, 0.0])
            self.force_torque = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
            self.battery_voltage = 24.0
            self.motor_temperatures = np.array([25.0, 26.0, 27.0, 25.5, 26.5, 25.8])
    
    # Test different fault scenarios
    scenarios = ['sensor_degradation', 'actuator_problems', 'critical_failure']
    
    for scenario_name in scenarios:
        print(f"\n=== Testing {scenario_name} scenario ===")
        
        # Create fresh fault injector for each scenario
        fault_injector = FaultInjectionFramework(random_seed=42)
        
        # Add scenario faults
        faults = fault_injector.create_fault_scenario(scenario_name)
        for i, fault in enumerate(faults):
            fault_injector.add_fault(f"{scenario_name}_{i}", fault)
        
        # Simulate for 10 seconds
        clean_state = MockRobotState()
        
        for t in np.arange(0, 10, 0.1):  # 10Hz sampling
            fault_injector.update_time(t)
            
            # Create fresh clean state for each timestep
            clean_state = MockRobotState()
            corrupted_state = fault_injector.inject_faults(clean_state)
            
            if t in [0.0, 1.0, 2.0, 3.0, 5.0]:  # Log at key times
                print(f"Time {t:.1f}s:")
                print(f"  Clean joint pos: {clean_state.joint_positions[:3]}")
                print(f"  Corrupted joint pos: {corrupted_state.joint_positions[:3]}")
                print(f"  Clean IMU accel: {clean_state.imu_acceleration}")
                print(f"  Corrupted IMU accel: {corrupted_state.imu_acceleration}")
                print(f"  Battery: {clean_state.battery_voltage:.2f}V -> {corrupted_state.battery_voltage:.2f}V")
                
        stats = fault_injector.get_fault_statistics()
        print(f"Final stats: {stats}")
