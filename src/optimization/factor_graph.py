# Factor graph construction module
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import time
from collections import defaultdict

# Import data structures
from src.data_structures import VehiclePose, Landmark, Feature, CameraIntrinsics, Extrinsics, ImuData, WheelEncoderData

# Import GTSAM utilities and factors
from src.optimization.gtsam_utils import (
    check_gtsam_availability,
    pose_to_gtsam_pose,
    intrinsics_to_gtsam_calibration,
    create_noise_model,
    create_between_factor_noise_model,
    create_projection_factor_noise_model,
    create_imu_noise_model,
    create_imu_bias
)
from src.optimization.factors import (
    create_wheel_odometry_factor,
    create_reprojection_factor,
    create_imu_factor,
    create_prior_factor_pose,
    create_prior_factor_point,
    create_prior_factor_calibration,
    create_prior_factor_bias
)
from src.optimization.variables import VariableIndex, create_initial_values

# Try to import GTSAM
try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("Warning: GTSAM not available. Install with 'pip install gtsam' for optimization.")

def build_factor_graph(poses: List[VehiclePose],
                      landmarks: Dict[int, Landmark],
                      features: Dict[Tuple[float, str], List[Feature]],
                      intrinsics: Dict[str, CameraIntrinsics],
                      extrinsics_guess: Dict[str, Extrinsics],
                      imu_data: List[ImuData],
                      wheel_data: List[WheelEncoderData],
                      config: Dict = None) -> Tuple[Any, Any, VariableIndex]:
    """
    Constructs the factor graph representing the optimization problem.

    Args:
        poses: Current estimates of vehicle poses (variables).
        landmarks: Current estimates of 3D landmark positions (variables).
        features: Detected 2D features (used for reprojection errors).
        intrinsics: Current estimates of camera intrinsics (variables).
        extrinsics_guess: Initial guess for camera extrinsics (variables).
        imu_data: IMU measurements (for IMU factors).
        wheel_data: Wheel encoder measurements (for odometry factors).
        config: Configuration parameters for the optimization.

    Returns:
        Tuple containing:
        - GTSAM factor graph
        - GTSAM initial values
        - Variable index mapping
    """
    print("Building the factor graph...")

    # Check if GTSAM is available
    if not check_gtsam_availability():
        print("GTSAM is not available. Using placeholder implementation.")
        # Placeholder implementation (same as before)
        graph = {"variables": [], "factors": []}

        # Add variables (poses, landmarks, intrinsics, extrinsics, biases)
        graph["variables"].extend([f"Pose_{i}" for i in range(len(poses))])
        graph["variables"].extend([f"Landmark_{j}" for j in landmarks.keys()])
        graph["variables"].extend([f"Intrinsics_{cam_id}" for cam_id in intrinsics.keys()])
        graph["variables"].extend([f"Extrinsics_{cam_id}" for cam_id in extrinsics_guess.keys()])

        # Add factors (Reprojection, IMU, Odometry, Priors)
        num_reprojection_factors = sum(len(obs) for lm in landmarks.values() for obs in [lm.observations])
        num_imu_factors = len(poses) - 1
        num_odom_factors = len(poses) - 1

        graph["factors"].append(f"{num_reprojection_factors} Reprojection Factors")
        graph["factors"].append(f"{num_imu_factors} IMU Factors")
        graph["factors"].append(f"{num_odom_factors} Odometry Factors")

        print(f"Factor graph built with {len(graph['variables'])} variables and {len(graph['factors'])} factor types.")
        return graph, None, None

    # Initialize configuration with defaults if not provided
    if config is None:
        config = {}

    # Get optimization parameters from config
    use_robust_kernel = config.get('robust_kernel', True)
    robust_kernel_threshold = config.get('robust_kernel_threshold', 1.0)
    pixel_sigma = config.get('pixel_sigma', 1.0)
    translation_sigma = config.get('translation_sigma', 0.1)
    rotation_sigma = config.get('rotation_sigma', 0.05)
    use_imu = config.get('use_imu', True)
    use_wheel_odometry = config.get('use_wheel_odometry', True)

    # Create variable index for mapping between our data structures and GTSAM variables
    variable_index = VariableIndex()

    # Create GTSAM factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Add prior factor on the first pose (anchor the system)
    first_pose = poses[0]
    first_pose_key = variable_index.add_pose(first_pose.timestamp)
    gtsam_first_pose = pose_to_gtsam_pose(first_pose.rotation, first_pose.translation)
    graph.add(create_prior_factor_pose(
        first_pose_key,
        gtsam_first_pose,
        translation_sigma=0.01,  # Strong prior on first pose
        rotation_sigma=0.01
    ))

    # Add prior factors on intrinsics and extrinsics
    for camera_id, intr in intrinsics.items():
        # Add intrinsics prior
        intrinsics_key = variable_index.add_camera_intrinsics(camera_id)
        gtsam_calibration = intrinsics_to_gtsam_calibration(
            intr.fx, intr.fy, intr.cx, intr.cy, intr.distortion_coeffs
        )
        graph.add(create_prior_factor_calibration(
            intrinsics_key,
            gtsam_calibration,
            focal_sigma=10.0,  # Moderate prior on focal length
            principal_point_sigma=10.0  # Moderate prior on principal point
        ))

        # Add extrinsics prior
        extr = extrinsics_guess[camera_id]
        extrinsics_key = variable_index.add_camera_extrinsics(camera_id)
        gtsam_extrinsics = pose_to_gtsam_pose(extr.rotation, extr.translation)
        graph.add(create_prior_factor_pose(
            extrinsics_key,
            gtsam_extrinsics,
            translation_sigma=0.1,  # Moderate prior on extrinsics
            rotation_sigma=0.1
        ))

    # Add reprojection factors for each landmark observation
    for landmark_id, landmark in landmarks.items():
        landmark_key = variable_index.add_landmark(landmark_id)

        # Add reprojection factors for each observation of this landmark
        for (timestamp, camera_id), feature_idx in landmark.observations.items():
            # Get the corresponding pose and feature
            pose_idx = next((i for i, p in enumerate(poses) if abs(p.timestamp - timestamp) < 1e-6), None)
            if pose_idx is None:
                continue  # Skip if no matching pose found

            pose = poses[pose_idx]
            pose_key = variable_index.add_pose(pose.timestamp)

            # Get the feature observation
            feature_list = features.get((timestamp, camera_id), [])
            if not feature_list or feature_idx >= len(feature_list):
                continue  # Skip if feature not found

            feature = feature_list[feature_idx]
            measurement = feature.uv

            # Get camera intrinsics and extrinsics keys
            camera_intrinsics_key = variable_index.get_camera_intrinsics_key(camera_id)
            camera_extrinsics_key = variable_index.get_camera_extrinsics_key(camera_id)

            # Get calibration object
            calibration = intrinsics_to_gtsam_calibration(
                intrinsics[camera_id].fx,
                intrinsics[camera_id].fy,
                intrinsics[camera_id].cx,
                intrinsics[camera_id].cy,
                intrinsics[camera_id].distortion_coeffs
            )

            # Add reprojection factor
            graph.add(create_reprojection_factor(
                landmark_key,
                pose_key,
                None,  # camera_key is not used in the updated implementation
                calibration,
                measurement,
                pixel_sigma=pixel_sigma,
                robust=use_robust_kernel,
                robust_threshold=robust_kernel_threshold
            ))

    # Add wheel odometry factors between consecutive poses
    if use_wheel_odometry and len(wheel_data) > 1:
        # Create a mapping from timestamp to wheel data for easier lookup
        wheel_data_map = {wd.timestamp: wd for wd in wheel_data}

        # Process consecutive poses
        for i in range(len(poses) - 1):
            pose1 = poses[i]
            pose2 = poses[i + 1]
            pose1_key = variable_index.get_pose_key(pose1.timestamp)
            pose2_key = variable_index.get_pose_key(pose2.timestamp)

            # Find wheel data closest to these poses
            wheel1 = min(wheel_data, key=lambda wd: abs(wd.timestamp - pose1.timestamp))
            wheel2 = min(wheel_data, key=lambda wd: abs(wd.timestamp - pose2.timestamp))

            # Skip if the wheel data is too far from the poses
            if abs(wheel1.timestamp - pose1.timestamp) > 0.1 or abs(wheel2.timestamp - pose2.timestamp) > 0.1:
                continue

            # Calculate relative motion from wheel odometry
            # This is a simplified calculation - in a real implementation, you would use the wheel odometry model
            dt = wheel2.timestamp - wheel1.timestamp
            if dt <= 0:
                continue

            # Calculate average wheel speeds (assuming differential drive model)
            v_left = (wheel1.wheel_speeds[0] + wheel1.wheel_speeds[2]) / 2.0
            v_right = (wheel1.wheel_speeds[1] + wheel1.wheel_speeds[3]) / 2.0
            v = (v_left + v_right) / 2.0  # Linear velocity
            omega = (v_right - v_left) / 1.5  # Angular velocity (assuming track width of 1.5m)

            # Calculate displacement
            dx = v * dt
            dy = 0.0  # No lateral displacement in differential drive
            dtheta = omega * dt

            # Add wheel odometry factor
            graph.add(create_wheel_odometry_factor(
                pose1_key,
                pose2_key,
                dx,
                dy,
                dtheta,
                translation_sigma=translation_sigma,
                rotation_sigma=rotation_sigma,
                robust=use_robust_kernel,
                robust_threshold=robust_kernel_threshold
            ))

    # Add IMU factors if IMU data is available
    if use_imu and len(imu_data) > 1:
        # Create IMU bias variable
        bias_key = variable_index.add_imu_bias()
        initial_bias = create_imu_bias(np.zeros(3), np.zeros(3))

        # Add prior on IMU bias
        graph.add(create_prior_factor_bias(
            bias_key,
            initial_bias,
            accel_sigma=0.1,
            gyro_sigma=0.01
        ))

        # Create IMU preintegration parameters
        imu_params = create_imu_noise_model(
            accel_sigma=0.1,
            gyro_sigma=0.01,
            accel_bias_sigma=0.01,
            gyro_bias_sigma=0.001
        )

        # Create IMU preintegration
        imu_preintegrated = gtsam.PreintegratedImuMeasurements(imu_params, initial_bias)

        # Process consecutive poses
        for i in range(len(poses) - 1):
            pose1 = poses[i]
            pose2 = poses[i + 1]
            pose1_key = variable_index.get_pose_key(pose1.timestamp)
            pose2_key = variable_index.get_pose_key(pose2.timestamp)

            # Add velocity variables if not already added
            vel1_key = variable_index.add_velocity(pose1.timestamp)
            vel2_key = variable_index.add_velocity(pose2.timestamp)

            # Find IMU measurements between these poses
            imu_between = [imu for imu in imu_data if pose1.timestamp <= imu.timestamp < pose2.timestamp]

            # Skip if no IMU data between poses
            if not imu_between:
                continue

            # Reset preintegration
            imu_preintegrated.resetIntegration()

            # Integrate IMU measurements
            for i, imu in enumerate(imu_between):
                if i == 0:
                    dt = imu.timestamp - pose1.timestamp
                else:
                    dt = imu.timestamp - imu_between[i-1].timestamp

                # Ensure dt is positive
                if dt > 0:
                    imu_preintegrated.integrateMeasurement(
                        imu.angular_velocity,
                        imu.linear_acceleration,
                        dt
                    )

            # Add IMU factor
            graph.add(create_imu_factor(
                pose1_key,
                vel1_key,
                pose2_key,
                vel2_key,
                bias_key,
                imu_preintegrated
            ))

    # Create initial values for the optimizer
    initial_values = create_initial_values(
        poses,
        landmarks,
        intrinsics,
        extrinsics_guess,
        variable_index,
        initial_velocity=np.zeros(3) if use_imu else None,
        initial_bias=(np.zeros(3), np.zeros(3)) if use_imu else None
    )

    print(f"Factor graph built with {graph.size()} factors and {initial_values.size()} variables.")
    return graph, initial_values, variable_index
