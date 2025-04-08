# Data structures for the multisensor calibration system
from typing import List, Dict, Tuple, Any
import numpy as np

class TimestampedData:
    """Base class for timestamped sensor data."""
    def __init__(self, timestamp: float):
        self.timestamp = timestamp

class ImageData(TimestampedData):
    """Represents image data from a single camera at a specific time."""
    def __init__(self, timestamp: float, camera_id: str, image: np.ndarray):
        super().__init__(timestamp)
        self.camera_id = camera_id
        self.image = image # Placeholder for actual image data (e.g., path or numpy array)

class ImuData(TimestampedData):
    """Represents IMU data (angular velocity and linear acceleration)."""
    def __init__(self, timestamp: float, angular_velocity: np.ndarray, linear_acceleration: np.ndarray):
        super().__init__(timestamp)
        # omega = [omega_x, omega_y, omega_z]
        self.angular_velocity = angular_velocity
        # a = [a_x, a_y, a_z]
        self.linear_acceleration = linear_acceleration

class WheelEncoderData(TimestampedData):
    """Represents wheel encoder data (e.g., ticks or speed) for all wheels."""
    def __init__(self, timestamp: float, wheel_speeds: np.ndarray):
        super().__init__(timestamp)
        # Example: [speed_fl, speed_fr, speed_rl, speed_rr]
        self.wheel_speeds = wheel_speeds

class CameraIntrinsics:
    """Represents camera intrinsic parameters."""
    def __init__(self, fx: float, fy: float, cx: float, cy: float, distortion_coeffs: np.ndarray = None):
        self.fx = fx # Focal length x
        self.fy = fy # Focal length y
        self.cx = cx # Principal point x
        self.cy = cy # Principal point y
        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5) # k1, k2, p1, p2, k3
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # Intrinsic matrix

class Extrinsics:
    """Represents extrinsic parameters (pose) of a sensor relative to the vehicle frame."""
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        # rotation: 3x3 Rotation matrix (SO(3)) or quaternion
        # translation: 3x1 Translation vector
        self.rotation = rotation
        self.translation = translation
        # Transformation matrix T = [R | t]
        #                         [0 | 1]
        self.T = np.eye(4)
        self.T[0:3, 0:3] = rotation
        self.T[0:3, 3] = translation.flatten()

class Feature:
    """Represents a detected 2D feature in an image."""
    def __init__(self, u: float, v: float, descriptor: Any = None):
        self.uv = np.array([u, v]) # Pixel coordinates
        self.descriptor = descriptor # Feature descriptor (e.g., SIFT, ORB)

class Match:
    """Represents a match between features in different images or times."""
    def __init__(self, feature1_idx: int, feature2_idx: int):
        self.idx1 = feature1_idx
        self.idx2 = feature2_idx

class Landmark:
    """Represents a 3D landmark in the vehicle coordinate frame."""
    def __init__(self, position: np.ndarray, observations: Dict[Tuple[float, str], int]):
        # position: [X, Y, Z] in vehicle frame
        self.position = position
        # observations: {(timestamp, camera_id): feature_index}
        self.observations = observations

class VehiclePose(TimestampedData):
    """Represents the vehicle's pose (position and orientation) at a specific time."""
    def __init__(self, timestamp: float, rotation: np.ndarray, translation: np.ndarray):
        super().__init__(timestamp)
        # rotation: 3x3 Rotation matrix (SO(3)) or quaternion
        # translation: 3x1 Translation vector [x, y, z] in the world/start frame
        self.rotation = rotation
        self.translation = translation
        self.T = np.eye(4) # Transformation matrix from vehicle frame to world frame
        self.T[0:3, 0:3] = rotation
        self.T[0:3, 3] = translation.flatten()
