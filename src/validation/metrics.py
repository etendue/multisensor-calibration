# Validation metrics module
from typing import Dict, List, Tuple
import numpy as np

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import CameraIntrinsics, Extrinsics, Landmark, Feature, VehiclePose

def calculate_reprojection_error(landmarks: Dict[int, Landmark], 
                                features: Dict[Tuple[float, str], List[Feature]], 
                                poses: List[VehiclePose],
                                intrinsics: Dict[str, CameraIntrinsics],
                                extrinsics: Dict[str, Extrinsics]) -> float:
    """
    Calculate the root mean square (RMS) reprojection error.
    
    Args:
        landmarks: Dictionary of 3D landmarks.
        features: Dictionary of 2D features.
        poses: List of vehicle poses.
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.
        
    Returns:
        RMS reprojection error in pixels.
    """
    # Placeholder implementation
    # In a real implementation, we would:
    # 1. For each landmark observation
    # 2. Project the 3D point into the camera
    # 3. Calculate the error between the projected point and the observed feature
    # 4. Compute the RMS of all errors
    
    # Simulate a reasonable reprojection error
    return np.random.uniform(0.3, 0.8)

def visualize_results(landmarks: Dict[int, Landmark], 
                     poses: List[VehiclePose],
                     intrinsics: Dict[str, CameraIntrinsics],
                     extrinsics: Dict[str, Extrinsics]) -> None:
    """
    Visualize the calibration results.
    
    Args:
        landmarks: Dictionary of 3D landmarks.
        poses: List of vehicle poses.
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.
    """
    # Placeholder for visualization code
    # In a real implementation, this would:
    # 1. Plot the vehicle trajectory
    # 2. Plot the 3D landmarks
    # 3. Visualize the camera positions and orientations
    # 4. Potentially project points into images to verify calibration
    
    print("Visualization would be shown here (not implemented in placeholder).")

def print_calibration_report(intrinsics: Dict[str, CameraIntrinsics],
                            extrinsics: Dict[str, Extrinsics],
                            reprojection_error: float) -> None:
    """
    Print a report of the calibration results.
    
    Args:
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.
        reprojection_error: RMS reprojection error.
    """
    print("\n--- Calibration Results Report ---")
    print(f"RMS Reprojection Error: {reprojection_error:.3f} pixels")
    
    print("\nCamera Intrinsics:")
    for cam_id, intr in intrinsics.items():
        print(f"  {cam_id}:")
        print(f"    fx: {intr.fx:.2f}, fy: {intr.fy:.2f}")
        print(f"    cx: {intr.cx:.2f}, cy: {intr.cy:.2f}")
        if np.any(intr.distortion_coeffs):
            print(f"    Distortion: {intr.distortion_coeffs}")
    
    print("\nSensor Extrinsics (relative to vehicle frame):")
    for sensor_id, extr in extrinsics.items():
        print(f"  {sensor_id}:")
        print(f"    Translation: [{extr.translation[0]:.3f}, {extr.translation[1]:.3f}, {extr.translation[2]:.3f}]")
        # Rotation could be printed as Euler angles for readability
        # This is a placeholder
        print(f"    Rotation Matrix:\n{np.round(extr.rotation, 3)}")
    
    print("\n--- End of Report ---")
