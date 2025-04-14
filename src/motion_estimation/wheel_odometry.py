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

    def update_ackermann(self, wheel_data: WheelEncoderData, steering_angle: float = None, prev_wheel_data: WheelEncoderData = None) -> Tuple[float, float, float]:
        """
        Calculate vehicle motion using Ackermann steering model.

        Args:
            wheel_data: Current wheel encoder data
            steering_angle: Current steering angle in radians (optional if wheel_angles are available)
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

        # If wheel_angles are available, use them instead of the provided steering_angle
        if hasattr(wheel_data, 'wheel_angles') and wheel_data.wheel_angles is not None:
            # Use the average of front wheel angles as the effective steering angle
            front_left_angle = wheel_data.wheel_angles[0]
            front_right_angle = wheel_data.wheel_angles[1]
            effective_steering_angle = (front_left_angle + front_right_angle) / 2.0

            # Print debug info if the angles are significantly different
            if abs(front_left_angle - front_right_angle) > 0.1:  # More than ~5.7 degrees difference
                print(f"Warning: Front wheel angles differ significantly: left={front_left_angle:.3f}, right={front_right_angle:.3f}")

            # Use the effective steering angle for calculations
            steering_angle = effective_steering_angle
        elif steering_angle is None:
            # If no wheel angles and no steering angle provided, assume straight line
            steering_angle = 0.0
            print("Warning: No steering angle available, assuming straight line motion")

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

    def update_ackermann_advanced(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData = None) -> Tuple[float, float, float]:
        """
        Calculate vehicle motion using an advanced Ackermann steering model.

        This model accounts for the different turning radii of each wheel based on their measured angles.
        Requires wheel_angles to be available in the wheel_data.

        Args:
            wheel_data: Current wheel encoder data with wheel angles
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)

        Returns:
            Tuple of (dx, dy, dtheta) representing vehicle motion in vehicle frame
        """
        # Check if wheel angles are available
        if not hasattr(wheel_data, 'wheel_angles') or wheel_data.wheel_angles is None:
            print("Warning: No wheel angle data available for advanced Ackermann model. Falling back to standard model.")
            return self.update_ackermann(wheel_data, None, prev_wheel_data)

        # Get wheel speeds
        wheel_speeds = self._calculate_wheel_speeds(wheel_data, prev_wheel_data)

        # Calculate time difference
        dt = 0.01  # Default small value
        if prev_wheel_data is not None:
            dt = wheel_data.timestamp - prev_wheel_data.timestamp
            if dt <= 0:
                return 0.0, 0.0, 0.0

        # Extract wheel angles
        fl_angle = wheel_data.wheel_angles[0]  # Front left
        fr_angle = wheel_data.wheel_angles[1]  # Front right
        rl_angle = wheel_data.wheel_angles[2]  # Rear left (typically ~0)
        rr_angle = wheel_data.wheel_angles[3]  # Rear right (typically ~0)

        # If all angles are very small, treat as straight-line motion
        if abs(fl_angle) < 1e-3 and abs(fr_angle) < 1e-3:
            # Average all wheel speeds for more robust estimation
            v_avg = np.mean(wheel_speeds)
            dx = v_avg * dt
            dy = 0.0
            dtheta = 0.0
            return dx, dy, dtheta

        # Calculate the instantaneous center of rotation (ICR)
        # This is a more advanced calculation based on the wheel angles
        # For a proper Ackermann steering, the ICR should be on the extension of the rear axle

        # Simplified approach: use the average of front wheel angles to determine ICR
        effective_angle = (fl_angle + fr_angle) / 2.0

        # Calculate turning radius to the center of the rear axle
        R = self.wheelbase / np.tan(abs(effective_angle))
        print(f"Turning radius: {R:.2f} meters, effective_angle: {effective_angle:.6f} radians")

        # Calculate angular velocity around ICR
        v_rear = (wheel_speeds[2] + wheel_speeds[3]) / 2.0  # Average of rear wheels
        omega = v_rear / R
        if effective_angle < 0:
            omega = -omega  # Adjust sign based on steering direction
        print(f"v_rear: {v_rear:.2f} m/s, omega: {omega:.6f} rad/s")

        # Calculate displacement
        dtheta = omega * dt
        print(f"dt: {dt:.6f} seconds, dtheta: {dtheta:.6f} radians")

        # For small angular displacements, we can approximate:
        if abs(dtheta) < 1e-3:
            dx = v_rear * dt
            dy = 0.0
            print(f"Small angular displacement: dx={dx:.6f}, dy={dy:.6f}, dtheta={dtheta:.6f}")
        else:
            # For larger angular displacements, calculate the chord length
            dx = R * np.sin(dtheta)
            dy = R * (1 - np.cos(dtheta))
            if effective_angle < 0:
                dy = -dy  # Adjust sign based on steering direction
            print(f"Large angular displacement: dx={dx:.6f}, dy={dy:.6f}, dtheta={dtheta:.6f}, R={R:.2f}, v_rear={v_rear:.2f}, dt={dt:.4f}")
            print(f"sin(dtheta): {np.sin(dtheta):.10f}, 1-cos(dtheta): {1-np.cos(dtheta):.10f}")

        return dx, dy, dtheta

    def update(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData = None, steering_angle: float = None) -> Tuple[float, float, float]:
        """
        Calculate vehicle motion based on the selected model.

        Args:
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            steering_angle: Current steering angle in radians (for Ackermann model, optional if wheel_angles are available)

        Returns:
            Tuple of (dx, dy, dtheta) representing vehicle motion in vehicle frame
        """
        print(f"Using model: {self.model}")
        if self.model == 'differential':
            print("Using differential drive model")
            return self.update_differential_drive(wheel_data, prev_wheel_data)
        elif self.model == 'ackermann':
            print("Using Ackermann model")
            # Check if wheel angle data is available for advanced Ackermann model
            if hasattr(wheel_data, 'wheel_angles') and wheel_data.wheel_angles is not None:
                # Use the advanced model that leverages wheel angle data
                print("Using advanced Ackermann model with wheel angles")
                return self.update_ackermann_advanced(wheel_data, prev_wheel_data)
            else:
                # Fall back to the standard model with provided steering angle
                print("Using standard Ackermann model with steering angle")
                return self.update_ackermann(wheel_data, steering_angle, prev_wheel_data)
        elif self.model == 'ackermann_advanced':
            # Explicitly request the advanced model
            print("Using advanced Ackermann model explicitly")
            return self.update_ackermann_advanced(wheel_data, prev_wheel_data)
        else:
            raise ValueError(f"Unsupported vehicle model: {self.model}")

    def integrate_pose(self, current_pose: VehiclePose, wheel_data: WheelEncoderData,
                      prev_wheel_data: WheelEncoderData = None, steering_angle: float = None) -> VehiclePose:
        """
        Integrate wheel odometry to update vehicle pose.

        Args:
            current_pose: Current vehicle pose
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
            steering_angle: Current steering angle in radians (for Ackermann model, optional if wheel_angles are available)

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

        # Print debug information
        print(f"Vehicle frame: dx={dx:.6f}, dy={dy:.6f}, dtheta={dtheta:.6f}")
        print(f"World frame: dx_world={dx_world:.6f}, dy_world={dy_world:.6f}")
        print(f"Current position: [{current_pose.translation[0]:.6f}, {current_pose.translation[1]:.6f}, {current_pose.translation[2]:.6f}]")

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
