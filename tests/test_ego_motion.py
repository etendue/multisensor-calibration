import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_structures import ImuData, WheelEncoderData, VehiclePose
from src.motion_estimation.ego_motion import estimate_initial_ego_motion, rotation_matrix_to_quaternion

class TestEgoMotion(unittest.TestCase):
    def setUp(self):
        # Create initial pose
        self.initial_pose = VehiclePose(0.0, np.eye(3), np.zeros(3))

        # Create sample IMU data
        self.imu_data = [
            ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),  # Initial state
            ImuData(1.0, np.zeros(3), np.array([1.0, 0.0, 9.81])),  # 1 m/s² forward
            ImuData(2.0, np.zeros(3), np.array([1.0, 0.0, 9.81])),  # 1 m/s² forward
            ImuData(3.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))   # No acceleration
        ]

        # Create sample wheel encoder data
        self.wheel_data = [
            WheelEncoderData(0.0, np.array([0, 0, 0, 0]), np.zeros(4)),
            WheelEncoderData(1.0, np.array([6554, 6554, 6554, 6554]), np.zeros(4)),  # 5% of a revolution
            WheelEncoderData(2.0, np.array([19661, 19661, 19661, 19661]), np.zeros(4)),  # 15% of a revolution
            WheelEncoderData(3.0, np.array([26214, 26214, 26214, 26214]), np.zeros(4))   # 20% of a revolution
        ]

        # Set axle length
        self.axle_length = 1.5  # meters

    def test_rotation_matrix_to_quaternion(self):
        # Test identity rotation
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)
        np.testing.assert_array_almost_equal(q, np.array([1.0, 0.0, 0.0, 0.0]))

        # Test 90° rotation around z-axis
        theta = np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        q = rotation_matrix_to_quaternion(R)
        expected_q = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
        np.testing.assert_array_almost_equal(q, expected_q)

    def test_estimate_initial_ego_motion_with_ekf(self):
        # Test ego motion estimation with EKF
        poses = estimate_initial_ego_motion(self.imu_data, self.wheel_data, self.initial_pose, self.axle_length, use_ekf=True)

        # Check number of poses
        self.assertEqual(len(poses), len(self.imu_data))

        # Check timestamps
        for i, pose in enumerate(poses):
            self.assertEqual(pose.timestamp, float(i))

        # Check that there is forward motion
        self.assertGreater(poses[-1].translation[0], 0)

    def test_estimate_initial_ego_motion_without_ekf(self):
        # Test ego motion estimation without EKF (simple integration)
        poses = estimate_initial_ego_motion(self.imu_data, self.wheel_data, self.initial_pose, self.axle_length, use_ekf=False)

        # Check number of poses
        self.assertEqual(len(poses), len(self.imu_data))

        # Check timestamps
        for i, pose in enumerate(poses):
            self.assertEqual(pose.timestamp, float(i))

        # Check that there is forward motion
        self.assertGreater(poses[-1].translation[0], 0)

    def test_estimate_initial_ego_motion_imu_only(self):
        # Test ego motion estimation with IMU data only
        poses = estimate_initial_ego_motion(self.imu_data, [], self.initial_pose, self.axle_length)

        # Check number of poses
        self.assertEqual(len(poses), len(self.imu_data))

        # Check timestamps
        for i, pose in enumerate(poses):
            self.assertEqual(pose.timestamp, float(i))

        # Check that there is forward motion
        self.assertGreater(poses[-1].translation[0], 0)

    def test_estimate_initial_ego_motion_no_data(self):
        # Test ego motion estimation with no data
        poses = estimate_initial_ego_motion([], [], self.initial_pose, self.axle_length)

        # Should return just the initial pose
        self.assertEqual(len(poses), 1)
        np.testing.assert_array_equal(poses[0].translation, self.initial_pose.translation)
        np.testing.assert_array_equal(poses[0].rotation, self.initial_pose.rotation)

if __name__ == '__main__':
    unittest.main()
