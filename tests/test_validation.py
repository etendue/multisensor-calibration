import unittest
import numpy as np
from typing import Dict, List, Tuple
import tempfile
import os

from src.data_structures import CameraIntrinsics, Extrinsics, Landmark, Feature, VehiclePose
from src.validation.metrics import calculate_reprojection_error, visualize_results

class TestValidationMetrics(unittest.TestCase):
    def setUp(self):
        # Create test data
        # Camera intrinsics (simple pinhole model)
        self.intrinsics = {
            'cam0': CameraIntrinsics(
                fx=500.0, fy=500.0,  # focal length
                cx=320.0, cy=240.0,  # principal point
                distortion_coeffs=np.zeros(5)  # no distortion
            )
        }
        
        # Camera extrinsics (identity transformation)
        self.extrinsics = {
            'cam0': Extrinsics(
                rotation=np.eye(3),
                translation=np.zeros(3)
            )
        }
        
        # Vehicle poses (single pose at origin)
        self.poses = [
            VehiclePose(
                timestamp=0.0,
                rotation=np.eye(3),
                translation=np.zeros(3)
            )
        ]
        
    def test_perfect_projection(self):
        """Test reprojection error calculation with perfect projection."""
        # Create a landmark at (0, 0, 5) in world frame
        landmarks = {
            0: Landmark(
                position=np.array([0.0, 0.0, 5.0]),
                observations={(0.0, 'cam0'): 0}  # Maps to first feature
            )
        }
        
        # Project the landmark to image plane manually
        # With identity transforms and z=5, u=x/z, v=y/z
        # After applying intrinsics: pixel_x = fx*u + cx, pixel_y = fy*v + cy
        pixel_x = self.intrinsics['cam0'].fx * 0.0 + self.intrinsics['cam0'].cx
        pixel_y = self.intrinsics['cam0'].fy * 0.0 + self.intrinsics['cam0'].cy
        
        # Create a perfect feature observation
        features = {
            (0.0, 'cam0'): [
                Feature(u=pixel_x, v=pixel_y)  # No descriptor needed for this test
            ]
        }
        
        error = calculate_reprojection_error(
            landmarks=landmarks,
            features=features,
            poses=self.poses,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics
        )
        
        self.assertAlmostEqual(error, 0.0, places=6)
        
    def test_known_error(self):
        """Test reprojection error calculation with known error."""
        # Create a landmark at (1, 2, 5) in world frame
        landmarks = {
            0: Landmark(
                position=np.array([1.0, 2.0, 5.0]),
                observations={(0.0, 'cam0'): 0}  # Maps to first feature
            )
        }
        
        # Project the landmark to image plane manually
        # u = x/z = 0.2, v = y/z = 0.4
        pixel_x = self.intrinsics['cam0'].fx * 0.2 + self.intrinsics['cam0'].cx
        pixel_y = self.intrinsics['cam0'].fy * 0.4 + self.intrinsics['cam0'].cy
        
        # Add a known error of 2 pixels in both x and y
        features = {
            (0.0, 'cam0'): [
                Feature(u=pixel_x + 2.0, v=pixel_y + 2.0)  # No descriptor needed for this test
            ]
        }
        
        error = calculate_reprojection_error(
            landmarks=landmarks,
            features=features,
            poses=self.poses,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics
        )
        
        # Expected RMS error = sqrt((2^2 + 2^2)/1) = 2*sqrt(2) ≈ 2.8284
        self.assertAlmostEqual(error, 2.0 * np.sqrt(2.0), places=6)

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
        
        # Create mock optimization progress
        optimization_progress = {
            'iterations': list(range(10)),
            'total_errors': [100.0 * np.exp(-i/2) for i in range(10)],  # Exponential decay
            'reprojection_errors': [10.0 * np.exp(-i/3) for i in range(10)],  # Slower decay
            'times': [i * 0.1 for i in range(10)]
        }
        
        # Test visualization with progress data (should not raise errors)
        visualize_results(landmarks, poses, self.intrinsics, extrinsics, optimization_progress)

if __name__ == '__main__':
    unittest.main()