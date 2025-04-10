# Extended Kalman Filter for sensor fusion
from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import ImuData, WheelEncoderData, VehiclePose

class EKF:
    """
    Extended Kalman Filter for fusing IMU and wheel odometry data.
    
    This class implements an EKF to estimate vehicle state (position, velocity, orientation, IMU biases)
    by fusing IMU measurements (prediction step) and wheel odometry (update step).
    """
    
    def __init__(self, initial_state: Dict, initial_covariance: np.ndarray, noise_params: Dict, vehicle_params: Dict):
        """
        Initialize the EKF.
        
        Args:
            initial_state: Dictionary containing initial state values:
                - 'position': Initial position [x, y, z] in world frame
                - 'velocity': Initial velocity [vx, vy, vz] in world frame
                - 'orientation': Initial orientation as quaternion [qw, qx, qy, qz]
                - 'gyro_bias': Initial gyroscope bias [bx, by, bz]
                - 'accel_bias': Initial accelerometer bias [bx, by, bz]
            initial_covariance: Initial state covariance matrix (16x16)
            noise_params: Dictionary containing noise parameters:
                - 'gyro_noise': Gyroscope measurement noise (std dev) in rad/s
                - 'accel_noise': Accelerometer measurement noise (std dev) in m/s²
                - 'gyro_bias_noise': Gyroscope bias random walk noise (std dev) in rad/s/√s
                - 'accel_bias_noise': Accelerometer bias random walk noise (std dev) in m/s²/√s
                - 'wheel_speed_noise': Wheel speed measurement noise (std dev) in m/s
            vehicle_params: Dictionary containing vehicle parameters:
                - 'wheel_radius': Wheel radius in meters
                - 'track_width': Distance between left and right wheels in meters
                - 'wheelbase': Distance between front and rear axles in meters
                - 'model': Vehicle model to use ('differential', 'ackermann', etc.)
        """
        # State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bgx, bgy, bgz, bax, bay, baz]
        # where p is position, v is velocity, q is orientation quaternion, bg is gyro bias, ba is accel bias
        self.state = np.zeros(16)
        
        # Initialize state from provided values
        self.state[0:3] = initial_state.get('position', np.zeros(3))
        self.state[3:6] = initial_state.get('velocity', np.zeros(3))
        self.state[6:10] = initial_state.get('orientation', np.array([1.0, 0.0, 0.0, 0.0]))  # Default: identity quaternion
        self.state[10:13] = initial_state.get('gyro_bias', np.zeros(3))
        self.state[13:16] = initial_state.get('accel_bias', np.zeros(3))
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(self.state[6:10])
        if quat_norm > 0:
            self.state[6:10] /= quat_norm
        
        # State covariance matrix
        self.P = initial_covariance
        
        # Store noise parameters
        self.noise_params = noise_params
        self.gyro_noise = noise_params.get('gyro_noise', 0.01)  # rad/s
        self.accel_noise = noise_params.get('accel_noise', 0.1)  # m/s²
        self.gyro_bias_noise = noise_params.get('gyro_bias_noise', 0.0001)  # rad/s/√s
        self.accel_bias_noise = noise_params.get('accel_bias_noise', 0.001)  # m/s²/√s
        self.wheel_speed_noise = noise_params.get('wheel_speed_noise', 0.1)  # m/s
        
        # Store vehicle parameters
        self.vehicle_params = vehicle_params
        self.wheel_radius = vehicle_params.get('wheel_radius', 0.3)  # meters
        self.track_width = vehicle_params.get('track_width', 1.5)  # meters
        self.wheelbase = vehicle_params.get('wheelbase', 2.7)  # meters
        self.model = vehicle_params.get('model', 'differential')
        
        # Gravity vector in world frame
        self.gravity = np.array([0.0, 0.0, 9.81])  # m/s²
        
        # Last timestamp for dt calculation
        self.last_timestamp = None
        
    def quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion [qw, qx, qy, qz]
            
        Returns:
            3x3 rotation matrix
        """
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy uses [x,y,z,w] order
    
    def rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        q = Rotation.from_matrix(R).as_quat()  # Returns [x,y,z,w]
        return np.array([q[3], q[0], q[1], q[2]])  # Convert to [w,x,y,z]
    
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.
        
        Args:
            q1: First quaternion [qw, qx, qy, qz]
            q2: Second quaternion [qw, qx, qy, qz]
            
        Returns:
            Result quaternion [qw, qx, qy, qz]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def small_angle_quaternion(self, angle_vec: np.ndarray) -> np.ndarray:
        """
        Create quaternion from small angle rotation vector.
        
        Args:
            angle_vec: Small angle rotation vector [wx, wy, wz] * dt
            
        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        angle = np.linalg.norm(angle_vec)
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = angle_vec / angle
        return np.array([np.cos(angle/2), axis[0]*np.sin(angle/2), axis[1]*np.sin(angle/2), axis[2]*np.sin(angle/2)])
    
    def predict(self, imu_data: ImuData, dt: float):
        """
        Perform EKF prediction step using IMU measurements.
        
        Args:
            imu_data: IMU measurement
            dt: Time step in seconds
        """
        # Extract current state
        p = self.state[0:3]  # Position
        v = self.state[3:6]  # Velocity
        q = self.state[6:10]  # Orientation quaternion
        bg = self.state[10:13]  # Gyroscope bias
        ba = self.state[13:16]  # Accelerometer bias
        
        # Correct IMU measurements
        gyro = imu_data.angular_velocity - bg
        accel = imu_data.linear_acceleration - ba
        
        # Orientation update
        dq = self.small_angle_quaternion(gyro * dt)
        q_new = self.quaternion_multiply(q, dq)
        
        # Normalize quaternion
        q_new = q_new / np.linalg.norm(q_new)
        
        # Rotate acceleration to world frame and remove gravity
        R = self.quaternion_to_rotation(q)
        accel_world = R @ accel - self.gravity
        
        # Update velocity and position
        v_new = v + accel_world * dt
        p_new = p + v * dt + 0.5 * accel_world * dt**2
        
        # Bias random walk (no change in mean, only in covariance)
        bg_new = bg
        ba_new = ba
        
        # Update state vector
        self.state[0:3] = p_new
        self.state[3:6] = v_new
        self.state[6:10] = q_new
        self.state[10:13] = bg_new
        self.state[13:16] = ba_new
        
        # Compute Jacobian of state transition function
        F = np.eye(16)
        
        # Position update Jacobians
        F[0:3, 3:6] = np.eye(3) * dt  # ∂p/∂v
        
        # Velocity update Jacobians
        F[3:6, 6:10] = self._compute_velocity_quaternion_jacobian(q, accel, dt)  # ∂v/∂q
        F[3:6, 13:16] = -R * dt  # ∂v/∂ba
        
        # Orientation update Jacobians
        F[6:10, 6:10] = self._compute_quaternion_update_jacobian(q, gyro, dt)  # ∂q/∂q
        F[6:10, 10:13] = self._compute_quaternion_gyro_bias_jacobian(q, dt)  # ∂q/∂bg
        
        # Process noise covariance
        Q = np.zeros((16, 16))
        
        # Gyroscope noise contribution
        Q[6:10, 6:10] = self._compute_gyro_noise_covariance(q, dt)
        
        # Accelerometer noise contribution
        Q[3:6, 3:6] = R @ np.diag([self.accel_noise**2, self.accel_noise**2, self.accel_noise**2]) @ R.T * dt**2
        
        # Bias random walk noise
        Q[10:13, 10:13] = np.eye(3) * (self.gyro_bias_noise**2 * dt)
        Q[13:16, 13:16] = np.eye(3) * (self.accel_bias_noise**2 * dt)
        
        # Update covariance
        self.P = F @ self.P @ F.T + Q
        
    def _compute_velocity_quaternion_jacobian(self, q: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of velocity update with respect to quaternion.
        
        This is a simplified implementation. In practice, you would compute the full Jacobian.
        
        Args:
            q: Orientation quaternion [qw, qx, qy, qz]
            accel: Acceleration in body frame [ax, ay, az]
            dt: Time step
            
        Returns:
            3x4 Jacobian matrix
        """
        # Placeholder for the Jacobian computation
        # In a real implementation, you would compute the derivative of R*accel with respect to q
        return np.zeros((3, 4))
    
    def _compute_quaternion_update_jacobian(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of quaternion update with respect to quaternion.
        
        This is a simplified implementation. In practice, you would compute the full Jacobian.
        
        Args:
            q: Orientation quaternion [qw, qx, qy, qz]
            gyro: Angular velocity [wx, wy, wz]
            dt: Time step
            
        Returns:
            4x4 Jacobian matrix
        """
        # Placeholder for the Jacobian computation
        # In a real implementation, you would compute the derivative of q ⊗ dq with respect to q
        return np.eye(4)
    
    def _compute_quaternion_gyro_bias_jacobian(self, q: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of quaternion update with respect to gyro bias.
        
        This is a simplified implementation. In practice, you would compute the full Jacobian.
        
        Args:
            q: Orientation quaternion [qw, qx, qy, qz]
            dt: Time step
            
        Returns:
            4x3 Jacobian matrix
        """
        # Placeholder for the Jacobian computation
        # In a real implementation, you would compute the derivative of q ⊗ dq with respect to bg
        return np.zeros((4, 3))
    
    def _compute_gyro_noise_covariance(self, q: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute quaternion process noise covariance due to gyroscope noise.
        
        This is a simplified implementation. In practice, you would compute the full covariance.
        
        Args:
            q: Orientation quaternion [qw, qx, qy, qz]
            dt: Time step
            
        Returns:
            4x4 covariance matrix
        """
        # Placeholder for the covariance computation
        # In a real implementation, you would propagate gyro noise through the quaternion update
        return np.eye(4) * (self.gyro_noise**2 * dt**2)
    
    def update(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData = None):
        """
        Perform EKF update step using wheel odometry measurements.
        
        Args:
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data (needed for position encoders)
        """
        # Skip update if no previous data (needed for delta calculation)
        if prev_wheel_data is None:
            return
        
        # Calculate time difference
        dt = wheel_data.timestamp - prev_wheel_data.timestamp
        if dt <= 0:
            return
        
        # Calculate wheel speeds
        wheel_speeds = self._calculate_wheel_speeds(wheel_data, prev_wheel_data)
        
        # For differential drive, we average the left and right sides
        v_left = (wheel_speeds[0] + wheel_speeds[2]) / 2.0  # Average of front-left and rear-left
        v_right = (wheel_speeds[1] + wheel_speeds[3]) / 2.0  # Average of front-right and rear-right
        
        # Calculate linear and angular velocity
        v_measured = (v_left + v_right) / 2.0  # Linear velocity
        omega_measured = (v_right - v_left) / self.track_width  # Angular velocity
        
        # Measurement vector: [v, omega]
        z = np.array([v_measured, omega_measured])
        
        # Extract current state
        v = self.state[3:6]  # Velocity
        q = self.state[6:10]  # Orientation quaternion
        
        # Predict measurement from current state
        # For simplicity, we assume the vehicle moves in the x-direction in body frame
        R = self.quaternion_to_rotation(q)
        v_body = R.T @ v  # Rotate velocity to body frame
        
        # Predicted measurement
        v_predicted = v_body[0]  # Forward velocity in body frame
        
        # For omega, we can use the z-component of angular velocity in body frame
        # In a real implementation, you would derive this from the quaternion rate
        omega_predicted = 0.0  # Simplified - would need proper computation
        
        h = np.array([v_predicted, omega_predicted])
        
        # Measurement Jacobian H
        H = np.zeros((2, 16))
        
        # Derivative of v_body[0] with respect to velocity (simplified)
        H[0, 3:6] = R.T[0, :]  # First row of R.T
        
        # Derivative of v_body[0] with respect to quaternion (simplified)
        # In practice, you would compute the full Jacobian
        H[0, 6:10] = np.zeros(4)
        
        # Derivative of omega with respect to quaternion (simplified)
        # In practice, you would compute the full Jacobian
        H[1, 6:10] = np.zeros(4)
        
        # Measurement noise covariance
        R_mat = np.diag([self.wheel_speed_noise**2, (self.wheel_speed_noise / self.track_width)**2])
        
        # Innovation
        y = z - h
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_mat
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Normalize quaternion
        self.state[6:10] = self.state[6:10] / np.linalg.norm(self.state[6:10])
        
        # Update covariance
        self.P = (np.eye(16) - K @ H) @ self.P
    
    def _calculate_wheel_speeds(self, wheel_data: WheelEncoderData, prev_wheel_data: WheelEncoderData) -> np.ndarray:
        """
        Calculate wheel speeds in meters per second.
        
        Args:
            wheel_data: Current wheel encoder data
            prev_wheel_data: Previous wheel encoder data
            
        Returns:
            Array of wheel speeds [v_fl, v_fr, v_rl, v_rr] in m/s
        """
        # Check if we're using position encoders or speed encoders
        using_positions = self.vehicle_params.get('using_positions', True)
        ticks_per_rev = self.vehicle_params.get('encoder_ticks_per_revolution', 131072)
        
        if using_positions:
            # Calculate speed from position difference
            dt = wheel_data.timestamp - prev_wheel_data.timestamp
            if dt <= 0:
                return np.zeros(4)
                
            # Calculate position differences (handle wrap-around for position encoders)
            position_diff = wheel_data.wheel_speeds - prev_wheel_data.wheel_speeds
            
            # Handle wrap-around (assuming encoder values are in range [0, ticks_per_rev])
            for i in range(len(position_diff)):
                if position_diff[i] > ticks_per_rev / 2:
                    position_diff[i] -= ticks_per_rev
                elif position_diff[i] < -ticks_per_rev / 2:
                    position_diff[i] += ticks_per_rev
            
            # Convert ticks to radians and then to linear speed
            angular_speed = position_diff / ticks_per_rev * 2 * np.pi / dt
            linear_speed = angular_speed * self.wheel_radius
            
            return linear_speed
        else:
            # Assuming wheel_speeds already contains speeds in appropriate units
            # Convert to m/s if needed (implementation depends on data format)
            return wheel_data.wheel_speeds
    
    def get_vehicle_pose(self) -> VehiclePose:
        """
        Get the current vehicle pose from the EKF state.
        
        Returns:
            VehiclePose object representing the current vehicle pose
        """
        # Extract position and orientation from state
        position = self.state[0:3]
        q = self.state[6:10]
        
        # Convert quaternion to rotation matrix
        rotation = self.quaternion_to_rotation(q)
        
        # Create VehiclePose object
        # Use the last timestamp if available, otherwise use 0.0
        timestamp = self.last_timestamp if self.last_timestamp is not None else 0.0
        
        return VehiclePose(timestamp, rotation, position)
    
    def process_measurements(self, imu_data_sequence: List[ImuData], wheel_data_sequence: List[WheelEncoderData]) -> List[VehiclePose]:
        """
        Process sequences of IMU and wheel encoder measurements to estimate vehicle trajectory.
        
        Args:
            imu_data_sequence: List of IMU measurements in chronological order
            wheel_data_sequence: List of wheel encoder measurements in chronological order
            
        Returns:
            List of estimated vehicle poses
        """
        if not imu_data_sequence:
            return [self.get_vehicle_pose()]
        
        poses = []
        prev_wheel_data = None
        wheel_idx = 0
        
        # Process each IMU measurement
        for imu_data in imu_data_sequence:
            # Calculate time step
            if self.last_timestamp is not None:
                dt = imu_data.timestamp - self.last_timestamp
                if dt > 0:
                    # Prediction step with IMU data
                    self.predict(imu_data, dt)
            
            # Update last timestamp
            self.last_timestamp = imu_data.timestamp
            
            # Find wheel data with timestamp closest to current IMU timestamp
            while wheel_idx < len(wheel_data_sequence) - 1 and wheel_data_sequence[wheel_idx + 1].timestamp <= imu_data.timestamp:
                wheel_idx += 1
            
            # Update step with wheel data if available
            if wheel_idx < len(wheel_data_sequence):
                current_wheel_data = wheel_data_sequence[wheel_idx]
                if prev_wheel_data is not None and current_wheel_data.timestamp <= imu_data.timestamp:
                    self.update(current_wheel_data, prev_wheel_data)
                    prev_wheel_data = current_wheel_data
                elif prev_wheel_data is None:
                    prev_wheel_data = current_wheel_data
            
            # Add current pose to trajectory
            poses.append(self.get_vehicle_pose())
        
        return poses
