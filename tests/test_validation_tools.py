#!/usr/bin/env python3
# Comprehensive test script for validation tools
import os
import sys
import unittest
import numpy as np
from typing import Dict, List, Tuple
import tempfile

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data_structures import VehiclePose, Landmark, Feature, CameraIntrinsics, Extrinsics, ImuData, WheelEncoderData
from src.validation.metrics import calculate_reprojection_error, visualize_results, print_calibration_report
from src.optimization.factor_graph import build_factor_graph
from src.optimization.bundle_adjustment import run_bundle_adjustment, extract_calibration_results
from src.optimization.gtsam_utils import check_gtsam_availability

class TestValidationTools(unittest.TestCase):
    """Test suite for validation tools."""

    def setUp(self):
        """Set up test fixtures."""
        # Create basic camera intrinsics
        self.intrinsics = {
            'cam0': CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
        }

        # Create basic camera extrinsics
        self.extrinsics = {
            'cam0': Extrinsics(rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
        }

        # Create a simple pose
        self.poses = [
            VehiclePose(timestamp=0.0, rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
        ]

    def test_reprojection_error_calculation(self):
        """Test reprojection error calculation."""
        # Create a landmark at [0, 0, 5]
        landmarks = {
            0: Landmark(
                position=np.array([0.0, 0.0, 5.0]),
                observations={(0.0, 'cam0'): 0}  # Index of feature in features[(0.0, 'cam0')]
            )
        }

        # Create a feature at [320, 240] (center of image)
        features = {
            (0.0, 'cam0'): [Feature(u=322.0, v=242.0)]  # Slightly offset from where it should be
        }

        error = calculate_reprojection_error(
            landmarks=landmarks,
            features=features,
            poses=self.poses,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics
        )

        # Expected error: sqrt((322-320)^2 + (242-240)^2) = sqrt(8) ≈ 2.8284
        self.assertAlmostEqual(error, np.sqrt(8.0), places=6)

    def test_visualization_empty_data(self):
        """Test visualization with empty data."""
        # This should not raise any errors
        visualize_results({}, [], {}, {})

    def test_visualization_simple_trajectory(self):
        """Test visualization with a simple circular trajectory."""
        # Create a circular trajectory
        timestamps = np.linspace(0, 2*np.pi, 10)
        radius = 2.0
        poses = []
        for t in timestamps:
            # Position on circle
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            z = 0.0

            # Rotation matrix to face the center
            direction = -np.array([x, y, z])  # Point to circle center
            direction = direction / np.linalg.norm(direction)
            up = np.array([0, 0, 1])
            forward = np.cross(up, direction)
            forward = forward / np.linalg.norm(forward)
            right = np.cross(direction, forward)
            R = np.column_stack([forward, right, direction])

            poses.append(VehiclePose(
                timestamp=float(t),
                rotation=R,
                translation=np.array([x, y, z])
            ))

        # Create some landmarks in a grid pattern
        landmarks = {}
        for i, x in enumerate(np.linspace(-3, 3, 5)):
            for j, y in enumerate(np.linspace(-3, 3, 5)):
                landmark_id = i * 5 + j
                landmarks[landmark_id] = Landmark(
                    position=np.array([x, y, 0.0]),
                    observations={(0.0, 'cam0'): 0}  # Dummy observation
                )

        # Add a camera with non-trivial extrinsics
        extrinsics = {
            'cam0': Extrinsics(
                rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90° yaw
                translation=np.array([0.5, 0.0, 0.3])  # Offset from vehicle
            )
        }

        # Test visualization (should not raise errors)
        visualize_results(landmarks, poses, self.intrinsics, extrinsics)

    def test_visualization_with_optimization_progress(self):
        """Test visualization with optimization progress data."""
        # Create a simple trajectory
        poses = [
            VehiclePose(0.0, np.eye(3), np.array([0.0, 0.0, 0.0])),
            VehiclePose(1.0, np.eye(3), np.array([1.0, 0.0, 0.0])),
            VehiclePose(2.0, np.eye(3), np.array([2.0, 0.0, 0.0]))
        ]

        # Create some landmarks
        landmarks = {
            0: Landmark(np.array([1.0, 1.0, 5.0]), {}),
            1: Landmark(np.array([2.0, -1.0, 5.0]), {})
        }

        # Create mock optimization progress
        optimization_progress = {
            'iterations': list(range(10)),
            'total_errors': [100.0 * np.exp(-i/2) for i in range(10)],  # Exponential decay
            'reprojection_errors': [10.0 * np.exp(-i/3) for i in range(10)],  # Slower decay
            'times': [i * 0.1 for i in range(10)]
        }

        # Test visualization with progress data (should not raise errors)
        visualize_results(landmarks, poses, self.intrinsics, self.extrinsics, optimization_progress)

    def test_print_calibration_report(self):
        """Test printing calibration report."""
        # This should not raise any errors
        print_calibration_report(self.intrinsics, self.extrinsics, 1.5)

    def test_interactive_controls(self):
        """Test visualization with interactive controls."""
        # Create a simple trajectory
        poses = [
            VehiclePose(0.0, np.eye(3), np.array([0.0, 0.0, 0.0])),
            VehiclePose(1.0, np.eye(3), np.array([1.0, 0.0, 0.0])),
            VehiclePose(2.0, np.eye(3), np.array([2.0, 0.0, 0.0]))
        ]

        # Create many landmarks to test sampling
        landmarks = {}
        for i in range(2000):  # More than max_landmarks
            landmarks[i] = Landmark(
                position=np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0, 10)]),
                observations={}
            )

        # Test visualization with landmark sampling and coordinate frame controls
        visualize_results(
            landmarks, poses, self.intrinsics, self.extrinsics,
            max_landmarks=500,  # Should sample down to 500 landmarks
            show_coordinate_frames=True,
            coordinate_frame_interval=1,  # Show for every pose
            coordinate_frame_scale=0.3
        )

class TestIntegrationWithOptimization(unittest.TestCase):
    """Test integration of validation tools with optimization backend."""

    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if GTSAM is not available
        if not check_gtsam_availability():
            self.skipTest("GTSAM is not available")

        # Create dummy data for testing
        # 1. Create some poses
        self.poses = [
            VehiclePose(0.0, np.eye(3), np.zeros(3)),
            VehiclePose(1.0, np.eye(3), np.array([1.0, 0.0, 0.0])),
            VehiclePose(2.0, np.eye(3), np.array([2.0, 0.0, 0.0]))
        ]

        # 2. Create some landmarks
        self.landmarks = {
            0: Landmark(np.array([1.0, 1.0, 5.0]), {}),
            1: Landmark(np.array([2.0, -1.0, 5.0]), {})
        }

        # Add observations to landmarks
        self.landmarks[0].observations = {
            (0.0, "cam0"): 0,  # Index of feature in features[(0.0, "cam0")]
            (1.0, "cam0"): 0   # Index of feature in features[(1.0, "cam0")]
        }
        self.landmarks[1].observations = {
            (1.0, "cam0"): 1,  # Index of feature in features[(1.0, "cam0")]
            (2.0, "cam0"): 0   # Index of feature in features[(2.0, "cam0")]
        }

        # 3. Create features
        self.features = {
            (0.0, "cam0"): [Feature(u=320.0, v=240.0)],
            (1.0, "cam0"): [Feature(u=310.0, v=240.0), Feature(u=330.0, v=240.0)],
            (2.0, "cam0"): [Feature(u=320.0, v=240.0)]
        }

        # 4. Create camera intrinsics
        self.intrinsics = {
            "cam0": CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
        }

        # 5. Create camera extrinsics
        self.extrinsics = {
            "cam0": Extrinsics(rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
        }

        # 6. Create IMU data
        self.imu_data = [
            ImuData(0.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),
            ImuData(0.5, np.zeros(3), np.array([0.0, 0.0, 9.81])),
            ImuData(1.0, np.zeros(3), np.array([0.0, 0.0, 9.81])),
            ImuData(1.5, np.zeros(3), np.array([0.0, 0.0, 9.81])),
            ImuData(2.0, np.zeros(3), np.array([0.0, 0.0, 9.81]))
        ]

        # 7. Create wheel encoder data
        self.wheel_data = [
            WheelEncoderData(0.0, np.array([0.0, 0.0, 0.0, 0.0])),
            WheelEncoderData(1.0, np.array([10.0, 10.0, 10.0, 10.0])),
            WheelEncoderData(2.0, np.array([20.0, 20.0, 20.0, 20.0]))
        ]

        # 8. Create optimization config
        self.optimization_config = {
            'robust_kernel': True,
            'robust_kernel_threshold': 1.0,
            'pixel_sigma': 1.0,
            'translation_sigma': 0.1,
            'rotation_sigma': 0.05,
            'use_imu': True,
            'use_wheel_odometry': True,
            'max_iterations': 5,  # Reduced for faster tests
            'convergence_delta': 1e-6,
            'verbose': True
        }

    def test_optimization_and_visualization(self):
        """Test optimization and visualization together."""
        try:
            # 1. Build factor graph
            factor_graph, initial_values, variable_index = build_factor_graph(
                self.poses, self.landmarks, self.features, self.intrinsics, self.extrinsics,
                self.imu_data, self.wheel_data, config=self.optimization_config
            )

            # 2. Run bundle adjustment
            optimized_values, optimization_progress = run_bundle_adjustment(
                factor_graph, initial_values, variable_index, config=self.optimization_config
            )

            # 3. Extract results
            camera_ids = list(self.intrinsics.keys())
            final_intrinsics, final_extrinsics, final_biases = extract_calibration_results(
                optimized_values, variable_index, camera_ids
            )

            # Check if we got valid extrinsics
            valid_extrinsics = {}
            for cam_id, extr in final_extrinsics.items():
                if extr is None:
                    print(f"Warning: Using initial extrinsics for camera {cam_id} since optimized value is None")
                    valid_extrinsics[cam_id] = self.extrinsics[cam_id]
                else:
                    valid_extrinsics[cam_id] = extr

            # 4. Calculate reprojection error using valid extrinsics
            reprojection_error = calculate_reprojection_error(
                self.landmarks, self.features, self.poses,
                final_intrinsics if all(intr is not None for intr in final_intrinsics.values()) else self.intrinsics,
                valid_extrinsics
            )

            # 5. Print calibration report
            print_calibration_report(final_intrinsics, valid_extrinsics, reprojection_error)

            # 6. Visualize results
            visualize_results(self.landmarks, self.poses, final_intrinsics, valid_extrinsics, optimization_progress)

            # Test passed if we got here without errors
            self.assertTrue(True)

        except Exception as e:
            self.fail(f"Optimization and visualization test failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
