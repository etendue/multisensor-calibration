import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_structures import ImuData, WheelEncoderData, VehiclePose
from src.motion_estimation.ekf import EKF

class TestEKF(unittest.TestCase):
    def setUp(self):
        # Set up initial state
        self.initial_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            'gyro_bias': np.zeros(3),
            'accel_bias': np.zeros(3)
        }
        
        # Set up initial covariance
        self.initial_covariance = np.eye(16)
        
        # Set up noise parameters
        self.noise_params = {
            'gyro_noise': 0.01,  # rad/s
            'accel_noise': 0.1,  # m/s²
            'gyro_bias_noise': 0.0001,  # rad/s/√s
            'accel_bias_noise': 0.001,  # m/s²/√s
            'wheel_speed_noise': 0.1  # m/s
        }
        
        # Set up vehicle parameters
        self.vehicle_params = {
            'wheel_radius': 0.3,  # meters
            'track_width': 1.5,   # meters
            'wheelbase': 2.7,     # meters
            'model': 'differential',
            'using_positions': True,
            'encoder_ticks_per_revolution': 131072
        }
        
        # Create EKF
        self.ekf = EKF(self.initial_state, self.initial_covariance, self.noise_params, self.vehicle_params)
    
    def test_quaternion_operations(self):
        # Test quaternion to rotation matrix conversion
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        R = self.ekf.quaternion_to_rotation(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))
        
        # Test rotation matrix to quaternion conversion
        R = np.eye(3)
        q = self.ekf.rotation_to_quaternion(R)
        np.testing.assert_array_almost_equal(q, np.array([1.0, 0.0, 0.0, 0.0]))
        
        # Test quaternion multiplication
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        q_result = self.ekf.quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(q_result, q2)
    
    def test_predict_no_motion(self):
        # Test prediction step with no motion
        imu_data = ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))  # Only gravity
        dt = 1.0
        
        # Initial state
        initial_state = self.ekf.state.copy()
        
        # Perform prediction
        self.ekf.predict(imu_data, dt)
        
        # State should be mostly unchanged (except for numerical errors)
        # Position, velocity, orientation should be the same
        np.testing.assert_array_almost_equal(self.ekf.state[0:3], initial_state[0:3])  # Position
        np.testing.assert_array_almost_equal(self.ekf.state[3:6], initial_state[3:6])  # Velocity
        np.testing.assert_array_almost_equal(self.ekf.state[6:10], initial_state[6:10])  # Orientation
    
    def test_predict_with_acceleration(self):
        # Test prediction step with forward acceleration
        imu_data = ImuData(0.0, np.zeros(3), np.array([1.0, 0.0, 9.81]))  # 1 m/s² forward + gravity
        dt = 1.0
        
        # Perform prediction
        self.ekf.predict(imu_data, dt)
        
        # Check velocity and position updates
        expected_velocity = np.array([1.0, 0.0, 0.0])
        expected_position = np.array([0.5, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(self.ekf.state[3:6], expected_velocity)  # Velocity
        np.testing.assert_array_almost_equal(self.ekf.state[0:3], expected_position)  # Position
    
    def test_predict_with_rotation(self):
        # Test prediction step with rotation
        imu_data = ImuData(0.0, np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 9.81]))  # 0.1 rad/s around z-axis
        dt = 1.0
        
        # Perform prediction
        self.ekf.predict(imu_data, dt)
        
        # Check orientation update
        # Expected quaternion for 0.1 rad rotation around z-axis
        expected_qw = np.cos(0.05)
        expected_qz = np.sin(0.05)
        expected_quaternion = np.array([expected_qw, 0.0, 0.0, expected_qz])
        
        np.testing.assert_array_almost_equal(self.ekf.state[6:10], expected_quaternion)
    
    def test_update_with_wheel_data(self):
        # Test update step with wheel data
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        wheel_data = WheelEncoderData(1.0, np.array([13107, 13107, 13107, 13107]))  # 10% of a revolution in 1 second
        
        # Set up initial state with some velocity
        self.ekf.state[3:6] = np.array([0.5, 0.0, 0.0])  # 0.5 m/s forward
        
        # Perform update
        self.ekf.update(wheel_data, prev_wheel_data)
        
        # Velocity should be updated towards the wheel measurement
        # Expected wheel speed: 0.1 * 2π * 0.3 ≈ 0.188 m/s
        # The update will move the velocity towards this value
        self.assertLess(self.ekf.state[3], 0.5)  # Should decrease towards wheel measurement
        self.assertGreater(self.ekf.state[3], 0.188)  # But not reach it completely due to Kalman gain
    
    def test_get_vehicle_pose(self):
        # Test getting vehicle pose from EKF state
        self.ekf.state[0:3] = np.array([1.0, 2.0, 3.0])  # Position
        self.ekf.state[6:10] = np.array([0.707, 0.707, 0.0, 0.0])  # 90° rotation around x-axis
        self.ekf.last_timestamp = 1.0
        
        # Get pose
        pose = self.ekf.get_vehicle_pose()
        
        # Check pose properties
        self.assertEqual(pose.timestamp, 1.0)
        np.testing.assert_array_almost_equal(pose.translation, np.array([1.0, 2.0, 3.0]))
        
        # Expected rotation matrix for 90° around x-axis
        expected_rotation = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        
        np.testing.assert_array_almost_equal(pose.rotation, expected_rotation, decimal=5)
    
    def test_process_measurements(self):
        # Test processing a sequence of measurements
        imu_sequence = [
            ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),  # Initial state
            ImuData(1.0, np.zeros(3), np.array([1.0, 0.0, 9.81])),  # 1 m/s² forward
            ImuData(2.0, np.zeros(3), np.array([1.0, 0.0, 9.81])),  # 1 m/s² forward
            ImuData(3.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))   # No acceleration
        ]
        
        wheel_sequence = [
            WheelEncoderData(0.0, np.array([0, 0, 0, 0])),
            WheelEncoderData(1.0, np.array([6554, 6554, 6554, 6554])),  # 5% of a revolution
            WheelEncoderData(2.0, np.array([19661, 19661, 19661, 19661])),  # 15% of a revolution
            WheelEncoderData(3.0, np.array([26214, 26214, 26214, 26214]))   # 20% of a revolution
        ]
        
        # Process measurements
        poses = self.ekf.process_measurements(imu_sequence, wheel_sequence)
        
        # Should have 4 poses
        self.assertEqual(len(poses), 4)
        
        # Check timestamps
        for i, pose in enumerate(poses):
            self.assertEqual(pose.timestamp, float(i))
        
        # Final pose should show forward movement
        self.assertGreater(poses[-1].translation[0], 0)

if __name__ == '__main__':
    unittest.main()
