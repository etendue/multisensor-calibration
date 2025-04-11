#!/usr/bin/env python3
# Test script for optimization backend integration
import os
import sys
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data_structures import VehiclePose, Landmark, Feature, CameraIntrinsics, Extrinsics, ImuData, WheelEncoderData
from src.optimization.factor_graph import build_factor_graph
from src.optimization.bundle_adjustment import run_bundle_adjustment, extract_calibration_results
from src.optimization.gtsam_utils import check_gtsam_availability
from src.validation.metrics import visualize_results

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
        optimized_values, optimization_progress = run_bundle_adjustment(
            factor_graph, initial_values, variable_index, config=optimization_config
        )
        print("\nOptimization successful!")
    else:
        print("GTSAM is not available. Skipping optimization.")
        optimized_values = initial_values
        optimization_progress = None

    # 11. Extract results
    print("\n--- Extracting Calibration Results ---")
    camera_ids = list(intrinsics.keys())
    final_intrinsics, final_extrinsics, final_biases = extract_calibration_results(
        optimized_values, variable_index, camera_ids
    )

    # 12. Print results
    print("\n--- Final Calibration Results ---")
    print("Optimized Intrinsics:")
    for cam_id, intr in final_intrinsics.items():
        if intr is not None:
            print(f"  {cam_id}: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.cx:.2f}, cy={intr.cy:.2f}")
        else:
            print(f"  {cam_id}: None")

    print("\nOptimized Extrinsics:")
    for cam_id, extr in final_extrinsics.items():
        if extr is not None:
            print(f"  {cam_id}:")
            print(f"    Translation: {np.round(extr.translation.flatten(), 3)}")
            print(f"    Rotation:\n{np.round(extr.rotation, 3)}")
        else:
            print(f"  {cam_id}: None")

    print("\nOptimized IMU Biases:")
    print(f"  {final_biases}")

    # 13. Visualize results
    print("\n--- Visualizing Results ---")
    visualize_results(landmarks, poses, final_intrinsics, final_extrinsics, optimization_progress)

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()
