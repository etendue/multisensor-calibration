#!/usr/bin/env python3
# Simplified test script for optimization backend integration
import os
import sys
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import modules
from data_structures import VehiclePose, Landmark, Feature, CameraIntrinsics, Extrinsics, ImuData, WheelEncoderData
from optimization.factor_graph import build_factor_graph
from optimization.bundle_adjustment import run_bundle_adjustment
from optimization.gtsam_utils import check_gtsam_availability

def main():
    """Test the optimization backend integration."""
    print("--- Testing Optimization Backend Integration ---")

    # Create dummy data for testing
    # 1. Create some poses
    poses = [
        VehiclePose(0.0, np.eye(3), np.zeros(3)),
        VehiclePose(1.0, np.eye(3), np.array([1.0, 0.0, 0.0])),
        VehiclePose(2.0, np.eye(3), np.array([2.0, 0.0, 0.0]))
    ]

    # 2. Create some landmarks
    landmarks = {
        0: Landmark(np.array([1.0, 1.0, 5.0]), {}),
        1: Landmark(np.array([2.0, -1.0, 5.0]), {})
    }

    # Add observations to landmarks
    landmarks[0].observations = {
        (0.0, "cam0"): 0,  # Index of feature in features[(0.0, "cam0")]
        (1.0, "cam0"): 0   # Index of feature in features[(1.0, "cam0")]
    }
    landmarks[1].observations = {
        (1.0, "cam0"): 1,  # Index of feature in features[(1.0, "cam0")]
        (2.0, "cam0"): 0   # Index of feature in features[(2.0, "cam0")]
    }

    # 3. Create features
    features = {
        (0.0, "cam0"): [Feature(u=320.0, v=240.0)],
        (1.0, "cam0"): [Feature(u=310.0, v=240.0), Feature(u=330.0, v=240.0)],
        (2.0, "cam0"): [Feature(u=320.0, v=240.0)]
    }

    # 4. Create camera intrinsics
    intrinsics = {
        "cam0": CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    }

    # 5. Create camera extrinsics
    extrinsics = {
        "cam0": Extrinsics(rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
    }

    # 6. Create IMU data
    imu_data = [
        ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),
        ImuData(0.5, np.zeros(3), np.array([0.0, 0.0, 9.81])),
        ImuData(1.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),
        ImuData(1.5, np.zeros(3), np.array([0.0, 0.0, 9.81])),
        ImuData(2.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))
    ]

    # 7. Create wheel encoder data
    wheel_data = [
        WheelEncoderData(0.0, np.array([0.0, 0.0, 0.0, 0.0])),
        WheelEncoderData(1.0, np.array([10.0, 10.0, 10.0, 10.0])),
        WheelEncoderData(2.0, np.array([20.0, 20.0, 20.0, 20.0]))
    ]

    # 8. Create optimization config
    optimization_config = {
        'robust_kernel': True,
        'robust_kernel_threshold': 1.0,
        'pixel_sigma': 1.0,
        'translation_sigma': 0.1,
        'rotation_sigma': 0.05,
        'use_imu': True,
        'use_wheel_odometry': True,
        'max_iterations': 10,
        'convergence_delta': 1e-6,
        'verbose': True
    }

    # 9. Build factor graph
    print("\n--- Building Factor Graph ---")
    factor_graph, initial_values, variable_index = build_factor_graph(
        poses, landmarks, features, intrinsics, extrinsics, imu_data, wheel_data, config=optimization_config
    )

    # 10. Run bundle adjustment
    print("\n--- Running Bundle Adjustment ---")
    if check_gtsam_availability():
        optimized_values = run_bundle_adjustment(
            factor_graph, initial_values, variable_index, config=optimization_config
        )
        print("\nOptimization successful!")
    else:
        print("GTSAM is not available. Skipping optimization.")
        optimized_values = initial_values

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()
