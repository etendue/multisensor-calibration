# Configuration file parser
import yaml
import os
import numpy as np

# Import data structures from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import CameraIntrinsics, Extrinsics

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        A dictionary containing the configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def parse_intrinsics(config: dict) -> dict:
    """
    Parse camera intrinsics from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary mapping camera IDs to CameraIntrinsics objects.
    """
    intrinsics = {}
    
    if 'cameras' not in config:
        return intrinsics
    
    for cam_id, cam_config in config['cameras'].items():
        if 'intrinsics' in cam_config:
            intr = cam_config['intrinsics']
            distortion = np.array(intr.get('distortion', [0, 0, 0, 0, 0]))
            intrinsics[cam_id] = CameraIntrinsics(
                fx=intr['fx'],
                fy=intr['fy'],
                cx=intr['cx'],
                cy=intr['cy'],
                distortion_coeffs=distortion
            )
    
    return intrinsics

def parse_extrinsics(config: dict) -> dict:
    """
    Parse sensor extrinsics from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary mapping sensor IDs to Extrinsics objects.
    """
    extrinsics = {}
    
    if 'sensors' not in config:
        return extrinsics
    
    for sensor_id, sensor_config in config['sensors'].items():
        if 'extrinsics' in sensor_config:
            ext = sensor_config['extrinsics']
            # Parse rotation (could be matrix, quaternion, or Euler angles)
            if 'rotation_matrix' in ext:
                rotation = np.array(ext['rotation_matrix'])
            elif 'quaternion' in ext:
                # Implement quaternion to rotation matrix conversion
                # This is a placeholder
                rotation = np.eye(3)
            elif 'euler' in ext:
                # Implement Euler angles to rotation matrix conversion
                # This is a placeholder
                rotation = np.eye(3)
            else:
                rotation = np.eye(3)
            
            # Parse translation
            if 'translation' in ext:
                translation = np.array(ext['translation'])
            else:
                translation = np.zeros(3)
            
            extrinsics[sensor_id] = Extrinsics(rotation=rotation, translation=translation)
    
    return extrinsics

def get_vehicle_parameters(config: dict) -> dict:
    """
    Get vehicle parameters from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary of vehicle parameters.
    """
    vehicle_params = {}
    
    if 'vehicle' in config:
        vehicle_params = config['vehicle']
    
    # Set default values if not specified
    if 'axle_length' not in vehicle_params:
        vehicle_params['axle_length'] = 1.6  # Default value in meters
    
    return vehicle_params
