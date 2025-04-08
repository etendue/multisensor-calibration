# Factor graph construction module
from typing import List, Dict, Tuple, Any
import numpy as np
import time

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import VehiclePose, Landmark, Feature, CameraIntrinsics, Extrinsics, ImuData, WheelEncoderData

def build_factor_graph(poses: List[VehiclePose], landmarks: Dict[int, Landmark], features: Dict[Tuple[float, str], List[Feature]], intrinsics: Dict[str, CameraIntrinsics], extrinsics_guess: Dict[str, Extrinsics], imu_data: List[ImuData], wheel_data: List[WheelEncoderData]) -> Any:
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

    Returns:
        A representation of the factor graph (e.g., using libraries like GTSAM or Ceres Solver).
        This involves creating:
        - Variable Nodes: For each pose, landmark, intrinsic set, extrinsic set, IMU bias.
        - Factor Nodes:
            - Reprojection Error Factors: Connect pose, landmark, intrinsics, extrinsics.
              Error = || project(Pose * Extrinsic * Landmark_pos) - feature_pos ||^2
            - IMU Preintegration Factors: Connect consecutive poses and IMU biases.
              Error based on integrated IMU measurements vs. relative pose change.
            - Wheel Odometry Factors: Connect consecutive poses.
              Error based on odometry measurements vs. relative pose change.
            - Prior Factors: On initial poses, intrinsics, extrinsics, biases.
    """
    print("Building the factor graph...")
    time.sleep(1.0) # Simulate work
    # Placeholder: In reality, this uses a specific library (GTSAM, Ceres)
    # to define variables and factors.
    graph = {"variables": [], "factors": []}

    # Add variables (poses, landmarks, intrinsics, extrinsics, biases)
    graph["variables"].extend([f"Pose_{i}" for i in range(len(poses))])
    graph["variables"].extend([f"Landmark_{j}" for j in landmarks.keys()])
    graph["variables"].extend([f"Intrinsics_{cam_id}" for cam_id in intrinsics.keys()])
    graph["variables"].extend([f"Extrinsics_{cam_id}" for cam_id in extrinsics_guess.keys()])
    # graph["variables"].append("ImuBiases") # Gyro + Accel biases

    # Add factors (Reprojection, IMU, Odometry, Priors)
    num_reprojection_factors = sum(len(obs) for lm in landmarks.values() for obs in [lm.observations])
    num_imu_factors = len(poses) - 1
    num_odom_factors = len(poses) - 1

    graph["factors"].append(f"{num_reprojection_factors} Reprojection Factors")
    graph["factors"].append(f"{num_imu_factors} IMU Factors")
    graph["factors"].append(f"{num_odom_factors} Odometry Factors")
    # graph["factors"].append("Prior Factors")

    print(f"Factor graph built with {len(graph['variables'])} variables and {len(graph['factors'])} factor types.")
    return graph
