import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_structures import WheelEncoderData, VehiclePose
from src.motion_estimation.wheel_odometry import WheelOdometry

class TestWheelOdometry(unittest.TestCase):
    def setUp(self):
        # Set up vehicle parameters for testing
        self.vehicle_params = {
            'wheel_radius': 0.3,  # meters
            'track_width': 1.5,   # meters
            'wheelbase': 2.7,     # meters
            'model': 'differential',
            'using_positions': True,
            'encoder_ticks_per_revolution': 131072
        }

        # Create wheel odometry calculator
        self.wheel_odom = WheelOdometry(self.vehicle_params)

        # Create initial pose
        self.initial_pose = VehiclePose(0.0, np.eye(3), np.zeros(3))

    def test_calculate_wheel_speeds_from_positions(self):
        # Test calculating wheel speeds from position encoders
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        wheel_data = WheelEncoderData(1.0, np.array([13107, 13107, 13107, 13107]))  # 10% of a revolution in 1 second

        # Calculate wheel speeds
        wheel_speeds = self.wheel_odom._calculate_wheel_speeds(wheel_data, prev_wheel_data)

        # Expected speed: 10% of circumference per second = 0.1 * 2π * 0.3 ≈ 0.188 m/s
        expected_speed = 0.1 * 2 * np.pi * 0.3

        # Check that all wheels have the expected speed
        for speed in wheel_speeds:
            self.assertAlmostEqual(speed, expected_speed, places=3)

    def test_differential_drive_straight(self):
        # Test straight-line motion with differential drive
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        wheel_data = WheelEncoderData(1.0, np.array([13107, 13107, 13107, 13107]))  # 10% of a revolution in 1 second

        # Calculate motion
        dx, dy, dtheta = self.wheel_odom.update_differential_drive(wheel_data, prev_wheel_data)

        # Expected motion: forward movement, no lateral movement, no rotation
        expected_dx = 0.1 * 2 * np.pi * 0.3  # 0.188 m
        expected_dy = 0.0
        expected_dtheta = 0.0

        self.assertAlmostEqual(dx, expected_dx, places=3)
        self.assertAlmostEqual(dy, expected_dy, places=3)
        self.assertAlmostEqual(dtheta, expected_dtheta, places=3)

    def test_differential_drive_turn(self):
        # Test turning motion with differential drive
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        # Left wheels slower than right wheels
        wheel_data = WheelEncoderData(1.0, np.array([6554, 19661, 6554, 19661]))  # 5% and 15% of a revolution

        # Calculate motion
        dx, dy, dtheta = self.wheel_odom.update_differential_drive(wheel_data, prev_wheel_data)

        # Expected motion: forward movement and rotation
        expected_dx = 0.1 * 2 * np.pi * 0.3  # Average speed: 0.188 m
        expected_dy = 0.0
        # Expected rotation: (right_speed - left_speed) / track_width
        expected_dtheta = (0.15 - 0.05) * 2 * np.pi * 0.3 / 1.5  # 0.063 rad

        self.assertAlmostEqual(dx, expected_dx, places=3)
        self.assertAlmostEqual(dy, expected_dy, places=3)
        self.assertAlmostEqual(dtheta, expected_dtheta, places=3)

    def test_ackermann_straight(self):
        # Test straight-line motion with Ackermann steering
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        wheel_data = WheelEncoderData(1.0, np.array([13107, 13107, 13107, 13107]))  # 10% of a revolution in 1 second
        steering_angle = 0.0  # Straight ahead

        # Calculate motion
        dx, dy, dtheta = self.wheel_odom.update_ackermann(wheel_data, steering_angle, prev_wheel_data)

        # Expected motion: forward movement, no lateral movement, no rotation
        expected_dx = 0.1 * 2 * np.pi * 0.3  # 0.188 m
        expected_dy = 0.0
        expected_dtheta = 0.0

        self.assertAlmostEqual(dx, expected_dx, places=3)
        self.assertAlmostEqual(dy, expected_dy, places=3)
        self.assertAlmostEqual(dtheta, expected_dtheta, places=3)

    def test_ackermann_turn(self):
        # Test turning motion with Ackermann steering
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        wheel_data = WheelEncoderData(1.0, np.array([13107, 13107, 13107, 13107]))  # 10% of a revolution in 1 second

        # Create wheel angles directly in the wheel_data
        wheel_angles = np.array([0.2, 0.2, 0.0, 0.0])  # Front wheels at 0.2 rad, rear wheels at 0
        wheel_data.wheel_angles = wheel_angles

        # Set vehicle model to ackermann to ensure the test uses the correct method
        self.wheel_odom.model = 'ackermann'

        # Calculate motion using the update method
        dx, dy, dtheta = self.wheel_odom.update(wheel_data, prev_wheel_data)

        # Expected motion: curved path
        # For small angles, dx ≈ v*dt, dtheta ≈ v*dt/R where R = wheelbase/tan(steering_angle)
        v = 0.1 * 2 * np.pi * 0.3  # 0.188 m/s
        R = 2.7 / np.tan(0.2)  # turning radius
        expected_dtheta = v / R  # angular displacement

        # Check that motion is reasonable (exact values depend on implementation details)
        self.assertGreater(dx, 0)  # Should move forward

        # For our implementation, we expect dtheta to be close to the expected value
        self.assertAlmostEqual(dtheta, expected_dtheta, places=3)

    def test_integrate_pose_straight(self):
        # Test pose integration for straight-line motion
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))
        wheel_data = WheelEncoderData(1.0, np.array([13107, 13107, 13107, 13107]))  # 10% of a revolution in 1 second

        # Integrate pose
        new_pose = self.wheel_odom.integrate_pose(self.initial_pose, wheel_data, prev_wheel_data)

        # Expected pose: moved forward in x direction
        expected_x = 0.1 * 2 * np.pi * 0.3  # 0.188 m

        self.assertAlmostEqual(new_pose.translation[0], expected_x, places=3)
        self.assertAlmostEqual(new_pose.translation[1], 0.0, places=3)
        self.assertAlmostEqual(new_pose.translation[2], 0.0, places=3)

        # Rotation should be unchanged
        np.testing.assert_array_almost_equal(new_pose.rotation, self.initial_pose.rotation)

    def test_ackermann_with_wheel_angles(self):
        # Test Ackermann steering using wheel angle data instead of steering angle
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))

        # Create wheel data with wheel angles
        wheel_speeds = np.array([13107, 13107, 13107, 13107])  # 10% of a revolution in 1 second
        wheel_angles = np.array([0.2, 0.2, 0.0, 0.0])  # Front wheels at 0.2 rad, rear wheels at 0
        wheel_data = WheelEncoderData(1.0, wheel_speeds, wheel_angles)

        # Calculate motion using the update method (should automatically use wheel angles)
        # Set model to ackermann
        self.wheel_odom.model = 'ackermann'
        dx, dy, dtheta = self.wheel_odom.update(wheel_data, prev_wheel_data)

        # Expected motion: curved path similar to test_ackermann_turn
        v = 0.1 * 2 * np.pi * 0.3  # 0.188 m/s
        R = 2.7 / np.tan(0.2)  # turning radius
        expected_dtheta = v / R  # angular displacement

        # Check that motion is reasonable (exact values depend on implementation details)
        self.assertGreater(dx, 0)  # Should move forward
        self.assertAlmostEqual(dtheta, expected_dtheta, places=3)  # Angular displacement should match expected value

    def test_ackermann_advanced(self):
        # Test the advanced Ackermann model
        prev_wheel_data = WheelEncoderData(0.0, np.array([0, 0, 0, 0]))

        # Create wheel data with wheel angles
        wheel_speeds = np.array([13107, 13107, 13107, 13107])  # 10% of a revolution in 1 second
        wheel_angles = np.array([0.2, 0.2, 0.0, 0.0])  # Front wheels at 0.2 rad, rear wheels at 0
        wheel_data = WheelEncoderData(1.0, wheel_speeds, wheel_angles)

        # Calculate motion using the advanced Ackermann model
        dx, dy, dtheta = self.wheel_odom.update_ackermann_advanced(wheel_data, prev_wheel_data)

        # Expected motion: curved path
        v = 0.1 * 2 * np.pi * 0.3  # 0.188 m/s
        R = 2.7 / np.tan(0.2)  # turning radius
        expected_dtheta = v / R  # angular displacement

        # Check that motion is reasonable (exact values depend on implementation details)
        self.assertGreater(dx, 0)  # Should move forward
        self.assertAlmostEqual(dtheta, expected_dtheta, places=3)  # Angular displacement should match expected value

if __name__ == '__main__':
    unittest.main()
