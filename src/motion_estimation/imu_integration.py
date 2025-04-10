# IMU integration module
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial.transform import Rotation

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import ImuData, VehiclePose

class ImuIntegrator:
    """
    Integrates IMU measurements to estimate vehicle motion.
    
    This class implements methods to integrate angular velocity and linear acceleration
    measurements from an IMU to estimate changes in orientation and position.
    """
    
    def __init__(self, imu_params: Dict = None):
        """
        Initialize the IMU integrator.
        
        Args:
            imu_params: Dictionary containing IMU parameters:
                - 'gyro_bias': Initial gyroscope bias [bx, by, bz] in rad/s
                - 'accel_bias': Initial accelerometer bias [bx, by, bz] in m/s²
                - 'gravity': Gravity vector [gx, gy, gz] in m/s²
        """
        self.imu_params = imu_params or {}
        
        # Initialize biases (can be updated during calibration)
        self.gyro_bias = np.array(self.imu_params.get('gyro_bias', [0.0, 0.0, 0.0]))
        self.accel_bias = np.array(self.imu_params.get('accel_bias', [0.0, 0.0, 0.0]))
        
        # Gravity vector in world frame (default: [0, 0, 9.81] for Z-up)
        self.gravity = np.array(self.imu_params.get('gravity', [0.0, 0.0, 9.81]))
        
        # State variables
        self.velocity = np.zeros(3)  # Current velocity in world frame
        
    def correct_measurements(self, imu_data: ImuData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply bias corrections to raw IMU measurements.
        
        Args:
            imu_data: Raw IMU measurements
            
        Returns:
            Tuple of (corrected_angular_velocity, corrected_linear_acceleration)
        """
        # Correct for biases
        corrected_angular_vel = imu_data.angular_velocity - self.gyro_bias
        corrected_linear_acc = imu_data.linear_acceleration - self.accel_bias
        
        return corrected_angular_vel, corrected_linear_acc
    
    def integrate_orientation(self, current_rotation: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate angular velocity to update orientation.
        
        Args:
            current_rotation: Current rotation matrix (3x3)
            angular_velocity: Angular velocity vector [wx, wy, wz] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated rotation matrix
        """
        # Convert current rotation matrix to scipy Rotation object
        current_rot = Rotation.from_matrix(current_rotation)
        
        # Calculate rotation increment (small angle approximation)
        angle = np.linalg.norm(angular_velocity * dt)
        if angle < 1e-10:
            # No significant rotation
            return current_rotation
        
        # Create rotation increment
        axis = angular_velocity / np.linalg.norm(angular_velocity)
        delta_rot = Rotation.from_rotvec(axis * angle)
        
        # Apply rotation increment (right multiplication for body-fixed frame)
        new_rot = current_rot * delta_rot
        
        # Return as rotation matrix
        return new_rot.as_matrix()
    
    def integrate_position(self, current_pose: VehiclePose, imu_data: ImuData, dt: float) -> VehiclePose:
        """
        Integrate IMU measurements to update vehicle pose.
        
        Args:
            current_pose: Current vehicle pose
            imu_data: Current IMU measurements
            dt: Time step in seconds
            
        Returns:
            Updated vehicle pose
        """
        # Correct IMU measurements
        angular_vel, linear_acc = self.correct_measurements(imu_data)
        
        # Update orientation
        new_rotation = self.integrate_orientation(current_pose.rotation, angular_vel, dt)
        
        # Rotate acceleration to world frame and compensate for gravity
        R_world_imu = new_rotation  # Assuming IMU frame is aligned with vehicle frame
        acc_world = R_world_imu @ linear_acc - self.gravity
        
        # Update velocity (first-order integration)
        new_velocity = self.velocity + acc_world * dt
        
        # Update position (first-order integration)
        new_translation = current_pose.translation + self.velocity * dt + 0.5 * acc_world * dt**2
        
        # Update internal velocity state
        self.velocity = new_velocity
        
        # Create new pose
        new_pose = VehiclePose(imu_data.timestamp, new_rotation, new_translation)
        
        return new_pose
    
    def reset_velocity(self, velocity: np.ndarray = None):
        """
        Reset the internal velocity state.
        
        Args:
            velocity: New velocity vector. If None, resets to zero.
        """
        self.velocity = velocity if velocity is not None else np.zeros(3)
    
    def process_imu_sequence(self, initial_pose: VehiclePose, imu_data_sequence: List[ImuData]) -> List[VehiclePose]:
        """
        Process a sequence of IMU measurements to estimate vehicle trajectory.
        
        Args:
            initial_pose: Initial vehicle pose
            imu_data_sequence: List of IMU measurements in chronological order
            
        Returns:
            List of estimated vehicle poses
        """
        if not imu_data_sequence:
            return [initial_pose]
        
        poses = [initial_pose]
        current_pose = initial_pose
        
        # Reset velocity at the start
        self.reset_velocity()
        
        # Process each IMU measurement
        for i in range(1, len(imu_data_sequence)):
            # Calculate time step
            dt = imu_data_sequence[i].timestamp - imu_data_sequence[i-1].timestamp
            if dt <= 0:
                # Skip invalid time steps
                continue
                
            # Integrate IMU measurement
            current_pose = self.integrate_position(current_pose, imu_data_sequence[i], dt)
            poses.append(current_pose)
        
        return poses
