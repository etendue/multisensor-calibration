# Wheel odometry calculation module
from typing import List, Tuple
import numpy as np

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import WheelEncoderData, VehiclePose

class WheelOdometry:
    """
    Calculates vehicle motion from wheel encoder data.
    
    This class implements different vehicle models for converting wheel encoder data
    (speeds or positions) into vehicle motion estimates (dx, dy, dtheta).
    """
    
    def __init__(self, vehicle_params: dict):
        """
        Initialize the wheel odometry calculator.
        
        Args:
            vehicle_params: Dictionary containing vehicle parameters:
                - 'wheel_radius': Wheel radius in meters
                - 'track_width': Distance between left and right wheels in meters
                - 'wheelbase': Distance between front and rear axles in meters
                - 'encoder_ticks_per_revolution': Number of encoder ticks per wheel revolution (if using position data)
                - 'model': Vehicle model to use ('differential', 'ackermann', etc.)
        """
        self.wheel_radius = vehicle_params.get('wheel_radius', 0.3)  # meters
        self.track_width = vehicle_params.get('track_width', 1.5)    # meters
        self.wheelbase = vehicle_params.get('wheelbase', 2.7)        # meters
        self.ticks_per_rev = vehicle_params.get('encoder_ticks_per_revolution', 131072)  # ticks/revolution
        self.model = vehicle_params.get('model', 'differential')
        
        # For position-based encoders, we need to track previous positions
        self.prev_wheel_positions = None
        
        # Flag to determine if we're using wheel speeds or positions
        self.using_positions = vehicle_params.get('using_positions', True)
        
    def _calculate_wheel_speeds(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData = None) -> np.ndarray:
        """
        Calculate wheel speeds in meters per second.
        
        Args:
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            
        Returns:
            Array of wheel speeds [v_fl, v_fr, v_rl, v_rr] in m/s
        """
        if self.using_positions and prev_wheel_data is not None:
            # Calculate speed from position difference
            dt = wheel_data.timestamp - prev_wheel_data.timestamp
            if dt <= 0:
                return np.zeros(4)
                
            # Calculate position differences (handle wrap-around for position encoders)
            position_diff = wheel_data.wheel_speeds - prev_wheel_data.wheel_speeds
            
            # Handle wrap-around (assuming encoder values are in range [0, ticks_per_rev])
            for i in range(len(position_diff)):
                if position_diff[i] > self.ticks_per_rev / 2:
                    position_diff[i] -= self.ticks_per_rev
                elif position_diff[i] < -self.ticks_per_rev / 2:
                    position_diff[i] += self.ticks_per_rev
            
            # Convert ticks to radians and then to linear speed
            angular_speed = position_diff / self.ticks_per_rev * 2 * np.pi / dt
            linear_speed = angular_speed * self.wheel_radius
            
            return linear_speed
        else:
            # Assuming wheel_speeds already contains speeds in appropriate units
            # Convert to m/s if needed (implementation depends on data format)
            return wheel_data.wheel_speeds
    
    def update_differential_drive(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData = None) -> Tuple[float, float, float]:
        """
        Calculate vehicle motion using differential drive model.
        
        Args:
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            
        Returns:
            Tuple of (dx, dy, dtheta) representing vehicle motion in vehicle frame
        """
        # Get wheel speeds
        wheel_speeds = self._calculate_wheel_speeds(wheel_data, prev_wheel_data)
        
        # For differential drive, we average the left and right sides
        v_left = (wheel_speeds[0] + wheel_speeds[2]) / 2.0  # Average of front-left and rear-left
        v_right = (wheel_speeds[1] + wheel_speeds[3]) / 2.0  # Average of front-right and rear-right
        
        # Calculate linear and angular velocity
        v = (v_left + v_right) / 2.0  # Linear velocity
        omega = (v_right - v_left) / self.track_width  # Angular velocity
        
        # Calculate time difference
        dt = 0.01  # Default small value
        if prev_wheel_data is not None:
            dt = wheel_data.timestamp - prev_wheel_data.timestamp
            if dt <= 0:
                return 0.0, 0.0, 0.0
        
        # Calculate displacement
        dx = v * dt  # Forward displacement
        dy = 0.0     # No lateral displacement in differential drive
        dtheta = omega * dt  # Angular displacement
        
        return dx, dy, dtheta
    
    def update_ackermann(self, wheel_data: WheelEncoderData, steering_angle: float, prev_wheel_data: WheelEncoderData = None) -> Tuple[float, float, float]:
        """
        Calculate vehicle motion using Ackermann steering model.
        
        Args:
            wheel_data: Current wheel encoder data
            steering_angle: Current steering angle in radians
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            
        Returns:
            Tuple of (dx, dy, dtheta) representing vehicle motion in vehicle frame
        """
        # Get wheel speeds
        wheel_speeds = self._calculate_wheel_speeds(wheel_data, prev_wheel_data)
        
        # For Ackermann, we average the rear wheels for velocity
        v_rear = (wheel_speeds[2] + wheel_speeds[3]) / 2.0  # Average of rear wheels
        
        # Calculate time difference
        dt = 0.01  # Default small value
        if prev_wheel_data is not None:
            dt = wheel_data.timestamp - prev_wheel_data.timestamp
            if dt <= 0:
                return 0.0, 0.0, 0.0
        
        # Calculate displacement based on bicycle model approximation
        if abs(steering_angle) < 1e-3:  # Straight line motion
            dx = v_rear * dt
            dy = 0.0
            dtheta = 0.0
        else:
            # Calculate turning radius
            R = self.wheelbase / np.tan(abs(steering_angle))
            
            # Calculate angular displacement
            dtheta = v_rear * dt / R
            if steering_angle < 0:
                dtheta = -dtheta  # Adjust sign based on steering direction
                
            # Calculate linear displacements
            dx = R * np.sin(dtheta)
            dy = R * (1 - np.cos(dtheta))
            if steering_angle < 0:
                dy = -dy  # Adjust sign based on steering direction
        
        return dx, dy, dtheta
    
    def update(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData = None, steering_angle: float = 0.0) -> Tuple[float, float, float]:
        """
        Calculate vehicle motion based on the selected model.
        
        Args:
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            steering_angle: Current steering angle in radians (for Ackermann model)
            
        Returns:
            Tuple of (dx, dy, dtheta) representing vehicle motion in vehicle frame
        """
        if self.model == 'differential':
            return self.update_differential_drive(wheel_data, prev_wheel_data)
        elif self.model == 'ackermann':
            return self.update_ackermann(wheel_data, steering_angle, prev_wheel_data)
        else:
            raise ValueError(f"Unsupported vehicle model: {self.model}")
    
    def integrate_pose(self, current_pose: VehiclePose, wheel_data: WheelEncoderData, 
                      prev_wheel_data: WheelEncoderData = None, steering_angle: float = 0.0) -> VehiclePose:
        """
        Integrate wheel odometry to update vehicle pose.
        
        Args:
            current_pose: Current vehicle pose
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            steering_angle: Current steering angle in radians (for Ackermann model)
            
        Returns:
            Updated vehicle pose
        """
        # Calculate motion in vehicle frame
        dx, dy, dtheta = self.update(wheel_data, prev_wheel_data, steering_angle)
        
        # Convert to world frame
        cos_theta = current_pose.rotation[0, 0]  # Assuming rotation is a proper rotation matrix
        sin_theta = current_pose.rotation[1, 0]
        
        # Rotate displacement to world frame
        dx_world = dx * cos_theta - dy * sin_theta
        dy_world = dx * sin_theta + dy * cos_theta
        
        # Update position
        new_translation = current_pose.translation.copy()
        new_translation[0] += dx_world
        new_translation[1] += dy_world
        
        # Update orientation (simple 2D rotation for now)
        new_rotation = current_pose.rotation.copy()
        cos_dtheta = np.cos(dtheta)
        sin_dtheta = np.sin(dtheta)
        rot_dtheta = np.array([
            [cos_dtheta, -sin_dtheta, 0],
            [sin_dtheta, cos_dtheta, 0],
            [0, 0, 1]
        ])
        new_rotation = np.dot(rot_dtheta, new_rotation)
        
        # Create new pose
        new_pose = VehiclePose(wheel_data.timestamp, new_rotation, new_translation)
        
        return new_pose
