# Initial ego-motion estimation module
from typing import List, Dict
import numpy as np
import time
from scipy.spatial.transform import Rotation

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import ImuData, WheelEncoderData, VehiclePose

# Import motion estimation modules
from motion_estimation.wheel_odometry import WheelOdometry
from motion_estimation.imu_integration import ImuIntegrator
from motion_estimation.ekf import EKF

def estimate_initial_ego_motion(imu_data: List[ImuData], wheel_data: List[WheelEncoderData],
                               initial_pose: VehiclePose, axle_length: float,
                               use_ekf: bool = True) -> List[VehiclePose]:
    """
    Estimates an initial vehicle trajectory using IMU and wheel encoder data.

    Args:
        imu_data: List of synchronized IMU measurements.
        wheel_data: List of synchronized wheel encoder measurements.
        initial_pose: The starting pose of the vehicle.
        axle_length: The distance between the front/rear axles (or track width depending on model).
        use_ekf: Whether to use EKF for sensor fusion (True) or simple integration (False).

    Returns:
        A list of estimated VehiclePose objects representing the initial trajectory.
        This involves:
        - Wheel Odometry: Calculating dx, dy, dyaw from wheel speeds/ticks.
        - IMU Integration: Integrating angular velocities for orientation and
            double integrating accelerations (gravity compensated) for position.
        - Sensor Fusion (EKF): Combining odometry and IMU to mitigate drift
            and improve accuracy.
    """
    print("Estimating initial ego-motion...")

    if len(imu_data) == 0:
        print("No IMU data available. Cannot estimate ego-motion.")
        return [initial_pose]

    if len(wheel_data) == 0:
        print("No wheel encoder data available. Using IMU-only integration.")
        # Use IMU-only integration
        imu_integrator = ImuIntegrator()
        return imu_integrator.process_imu_sequence(initial_pose, imu_data)

    # Set up vehicle parameters
    vehicle_params = {
        'wheel_radius': 0.3,  # meters (example value, should be configured)
        'track_width': axle_length,  # meters
        'wheelbase': 2.7,  # meters (example value, should be configured)
        'using_positions': False,  # we're using wheel speeds, not positions
        'encoder_ticks_per_revolution': 131072  # encoder resolution
    }

    # Check if wheel angle data is available
    if hasattr(wheel_data[0], 'wheel_angles') and wheel_data[0].wheel_angles is not None:
        print("Wheel angle data is available, using Ackermann advanced model")
        vehicle_params['model'] = 'ackermann_advanced'
    else:
        print("No wheel angle data available, using differential model")
        vehicle_params['model'] = 'differential'

    if use_ekf:
        # Use EKF for sensor fusion
        print("Using EKF for sensor fusion...")

        # Initialize EKF state
        initial_state = {
            'position': initial_pose.translation,
            'velocity': np.zeros(3),
            'orientation': rotation_matrix_to_quaternion(initial_pose.rotation),
            'gyro_bias': np.zeros(3),
            'accel_bias': np.zeros(3)
        }

        # Initialize covariance matrix
        initial_covariance = np.eye(16)
        initial_covariance[0:3, 0:3] *= 0.1**2  # position uncertainty
        initial_covariance[3:6, 3:6] *= 0.5**2  # velocity uncertainty
        initial_covariance[6:10, 6:10] *= 0.1**2  # orientation uncertainty
        initial_covariance[10:13, 10:13] *= 0.01**2  # gyro bias uncertainty
        initial_covariance[13:16, 13:16] *= 0.1**2  # accel bias uncertainty

        # Noise parameters
        noise_params = {
            'gyro_noise': 0.01,  # rad/s
            'accel_noise': 0.1,  # m/s²
            'gyro_bias_noise': 0.0001,  # rad/s/√s
            'accel_bias_noise': 0.001,  # m/s²/√s
            'wheel_speed_noise': 0.1  # m/s
        }

        # Create EKF and process measurements
        ekf = EKF(initial_state, initial_covariance, noise_params, vehicle_params)
        poses = ekf.process_measurements(imu_data, wheel_data)

    else:
        # Use simple integration (wheel odometry + IMU)
        print("Using simple integration (wheel odometry + IMU)...")

        # Create wheel odometry calculator
        wheel_odom = WheelOdometry(vehicle_params)

        # Create IMU integrator
        imu_integrator = ImuIntegrator()

        # Initialize trajectory
        poses = [initial_pose]
        current_pose = initial_pose
        prev_wheel_data = None

        # Process each IMU measurement
        for i in range(1, len(imu_data)):
            # Calculate time step
            dt = imu_data[i].timestamp - imu_data[i-1].timestamp
            if dt <= 0:
                continue

            # Find closest wheel data
            closest_wheel_data = None
            for wd in wheel_data:
                if abs(wd.timestamp - imu_data[i].timestamp) < 0.05:  # 50ms threshold
                    closest_wheel_data = wd
                    break

            # Update pose using wheel odometry if available
            if closest_wheel_data is not None and prev_wheel_data is not None:
                # Use wheel odometry for position update
                wheel_pose = wheel_odom.integrate_pose(current_pose, closest_wheel_data, prev_wheel_data)
                prev_wheel_data = closest_wheel_data

                # Use IMU for orientation update
                imu_pose = imu_integrator.integrate_position(current_pose, imu_data[i], dt)

                # Combine the two (simple fusion - use wheel odometry for position, IMU for orientation)
                new_pose = VehiclePose(
                    imu_data[i].timestamp,
                    imu_pose.rotation,
                    wheel_pose.translation
                )
            else:
                # Use IMU-only integration
                new_pose = imu_integrator.integrate_position(current_pose, imu_data[i], dt)
                if prev_wheel_data is None and closest_wheel_data is not None:
                    prev_wheel_data = closest_wheel_data

            poses.append(new_pose)
            current_pose = new_pose

    print(f"Initial ego-motion estimated for {len(poses)} poses.")
    return poses


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [qw, qx, qy, qz].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [qw, qx, qy, qz]
    """
    # Use scipy's Rotation class
    quat = Rotation.from_matrix(R).as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
