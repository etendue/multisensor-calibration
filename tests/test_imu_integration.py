import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_structures import ImuData, VehiclePose
from src.motion_estimation.imu_integration import ImuIntegrator

class TestImuIntegration(unittest.TestCase):
    def setUp(self):
        # Set up IMU parameters for testing
        self.imu_params = {
            'gyro_bias': [0.0, 0.0, 0.0],
            'accel_bias': [0.0, 0.0, 0.0],
            'gravity': [0.0, 0.0, 9.81]
        }
        
        # Create IMU integrator
        self.imu_integrator = ImuIntegrator(self.imu_params)
        
        # Create initial pose
        self.initial_pose = VehiclePose(0.0, np.eye(3), np.zeros(3))
        
    def test_correct_measurements(self):
        # Test bias correction
        imu_data = ImuData(0.0, np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
        
        # Set biases
        self.imu_integrator.gyro_bias = np.array([0.01, 0.02, 0.03])
        self.imu_integrator.accel_bias = np.array([0.1, 0.2, 0.3])
        
        # Correct measurements
        corrected_gyro, corrected_accel = self.imu_integrator.correct_measurements(imu_data)
        
        # Check that biases were subtracted
        np.testing.assert_array_almost_equal(corrected_gyro, np.array([0.09, 0.18, 0.27]))
        np.testing.assert_array_almost_equal(corrected_accel, np.array([0.9, 1.8, 2.7]))
    
    def test_integrate_orientation_no_rotation(self):
        # Test orientation integration with zero angular velocity
        current_rotation = np.eye(3)
        angular_velocity = np.zeros(3)
        dt = 1.0
        
        # Integrate orientation
        new_rotation = self.imu_integrator.integrate_orientation(current_rotation, angular_velocity, dt)
        
        # Rotation should be unchanged
        np.testing.assert_array_almost_equal(new_rotation, current_rotation)
    
    def test_integrate_orientation_z_rotation(self):
        # Test orientation integration with rotation around z-axis
        current_rotation = np.eye(3)
        angular_velocity = np.array([0.0, 0.0, 0.1])  # 0.1 rad/s around z-axis
        dt = 1.0
        
        # Integrate orientation
        new_rotation = self.imu_integrator.integrate_orientation(current_rotation, angular_velocity, dt)
        
        # Expected rotation: 0.1 rad around z-axis
        cos_theta = np.cos(0.1)
        sin_theta = np.sin(0.1)
        expected_rotation = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        np.testing.assert_array_almost_equal(new_rotation, expected_rotation)
    
    def test_integrate_position_no_acceleration(self):
        # Test position integration with zero acceleration
        imu_data = ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))  # Only gravity
        dt = 1.0
        
        # Reset velocity to zero
        self.imu_integrator.reset_velocity()
        
        # Integrate position
        new_pose = self.imu_integrator.integrate_position(self.initial_pose, imu_data, dt)
        
        # Position should be unchanged (gravity is compensated)
        np.testing.assert_array_almost_equal(new_pose.translation, self.initial_pose.translation)
        
        # Velocity should remain zero
        np.testing.assert_array_almost_equal(self.imu_integrator.velocity, np.zeros(3))
    
    def test_integrate_position_with_acceleration(self):
        # Test position integration with forward acceleration
        imu_data = ImuData(0.0, np.zeros(3), np.array([1.0, 0.0, 9.81]))  # 1 m/s² forward + gravity
        dt = 1.0
        
        # Reset velocity to zero
        self.imu_integrator.reset_velocity()
        
        # Integrate position
        new_pose = self.imu_integrator.integrate_position(self.initial_pose, imu_data, dt)
        
        # Expected position: p = p0 + v0*t + 0.5*a*t²
        # With v0 = 0, a = 1 m/s², t = 1s: p = p0 + 0.5 m in x direction
        expected_position = np.array([0.5, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(new_pose.translation, expected_position)
        
        # Expected velocity: v = v0 + a*t
        # With v0 = 0, a = 1 m/s², t = 1s: v = 1 m/s in x direction
        expected_velocity = np.array([1.0, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(self.imu_integrator.velocity, expected_velocity)
    
    def test_process_imu_sequence(self):
        # Test processing a sequence of IMU measurements
        imu_sequence = [
            ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),  # Initial state
            ImuData(1.0, np.zeros(3), np.array([1.0, 0.0, 9.81])),  # 1 m/s² forward
            ImuData(2.0, np.zeros(3), np.array([1.0, 0.0, 9.81])),  # 1 m/s² forward
            ImuData(3.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))   # No acceleration
        ]
        
        # Process sequence
        poses = self.imu_integrator.process_imu_sequence(self.initial_pose, imu_sequence)
        
        # Should have 4 poses
        self.assertEqual(len(poses), 4)
        
        # Check timestamps
        for i, pose in enumerate(poses):
            self.assertEqual(pose.timestamp, float(i))
        
        # Check final position
        # After 2 seconds of 1 m/s² acceleration and 1 second of constant velocity:
        # Position at t=1: 0.5 m
        # Position at t=2: 0.5 + 1.5 = 2.0 m
        # Position at t=3: 2.0 + 2.0 = 4.0 m
        expected_final_position = np.array([4.0, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(poses[-1].translation, expected_final_position)

if __name__ == '__main__':
    unittest.main()
