# Variable creation and mapping for GTSAM optimization
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

# Try to import GTSAM
try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("Warning: GTSAM not available. Install with 'pip install gtsam' for optimization.")

# Import utility functions
from src.optimization.gtsam_utils import (
    check_gtsam_availability,
    pose_to_gtsam_pose,
    intrinsics_to_gtsam_calibration,
    translation_to_gtsam_point,
    create_imu_bias
)

# Import data structures
from src.data_structures import (
    VehiclePose,
    Landmark,
    CameraIntrinsics,
    Extrinsics
)

class VariableIndex:
    """
    Manages the mapping between our data structures and GTSAM variables.
    """

    def __init__(self):
        """Initialize the variable index."""
        if not GTSAM_AVAILABLE:
            raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

        # Initialize symbol generators for different variable types
        self.pose_symbol = gtsam.symbol_shorthand.X
        self.velocity_symbol = gtsam.symbol_shorthand.V
        self.landmark_symbol = gtsam.symbol_shorthand.L
        self.camera_extrinsics_symbol = gtsam.symbol_shorthand.T
        self.camera_intrinsics_symbol = gtsam.symbol_shorthand.K
        self.imu_bias_symbol = gtsam.symbol_shorthand.B

        # Initialize mappings
        self.pose_indices = {}  # {timestamp: index}
        self.landmark_indices = {}  # {landmark_id: index}
        self.camera_indices = {}  # {camera_id: index}

        # Initialize counters
        self.next_pose_index = 0
        self.next_landmark_index = 0
        self.next_camera_index = 0
        self.next_bias_index = 0

    def add_pose(self, timestamp: float) -> Any:
        """
        Add a pose variable to the index.

        Args:
            timestamp: Timestamp of the pose.

        Returns:
            GTSAM symbol for the pose.
        """
        if timestamp in self.pose_indices:
            return self.pose_symbol(self.pose_indices[timestamp])

        index = self.next_pose_index
        self.pose_indices[timestamp] = index
        self.next_pose_index += 1
        return self.pose_symbol(index)

    def add_velocity(self, timestamp: float) -> Any:
        """
        Add a velocity variable to the index.

        Args:
            timestamp: Timestamp of the velocity.

        Returns:
            GTSAM symbol for the velocity.
        """
        # Velocity indices match pose indices
        if timestamp not in self.pose_indices:
            self.add_pose(timestamp)

        return self.velocity_symbol(self.pose_indices[timestamp])

    def add_landmark(self, landmark_id: int) -> Any:
        """
        Add a landmark variable to the index.

        Args:
            landmark_id: ID of the landmark.

        Returns:
            GTSAM symbol for the landmark.
        """
        if landmark_id in self.landmark_indices:
            return self.landmark_symbol(self.landmark_indices[landmark_id])

        index = self.next_landmark_index
        self.landmark_indices[landmark_id] = index
        self.next_landmark_index += 1
        return self.landmark_symbol(index)

    def add_camera_extrinsics(self, camera_id: str) -> Any:
        """
        Add a camera extrinsics variable to the index.

        Args:
            camera_id: ID of the camera.

        Returns:
            GTSAM symbol for the camera extrinsics.
        """
        if camera_id in self.camera_indices:
            return self.camera_extrinsics_symbol(self.camera_indices[camera_id])

        index = self.next_camera_index
        self.camera_indices[camera_id] = index
        self.next_camera_index += 1
        return self.camera_extrinsics_symbol(index)

    def add_camera_intrinsics(self, camera_id: str) -> Any:
        """
        Add a camera intrinsics variable to the index.

        Args:
            camera_id: ID of the camera.

        Returns:
            GTSAM symbol for the camera intrinsics.
        """
        # Intrinsics indices match extrinsics indices
        if camera_id not in self.camera_indices:
            self.add_camera_extrinsics(camera_id)

        return self.camera_intrinsics_symbol(self.camera_indices[camera_id])

    def add_imu_bias(self, index: int = 0) -> Any:
        """
        Add an IMU bias variable to the index.

        Args:
            index: Index of the IMU bias (default: 0).

        Returns:
            GTSAM symbol for the IMU bias.
        """
        return self.imu_bias_symbol(index)

    def get_pose_key(self, timestamp: float) -> Any:
        """
        Get the GTSAM key for a pose.

        Args:
            timestamp: Timestamp of the pose.

        Returns:
            GTSAM symbol for the pose.
        """
        if timestamp not in self.pose_indices:
            raise KeyError(f"Pose with timestamp {timestamp} not found in index.")

        return self.pose_symbol(self.pose_indices[timestamp])

    def get_velocity_key(self, timestamp: float) -> Any:
        """
        Get the GTSAM key for a velocity.

        Args:
            timestamp: Timestamp of the velocity.

        Returns:
            GTSAM symbol for the velocity.
        """
        if timestamp not in self.pose_indices:
            raise KeyError(f"Velocity with timestamp {timestamp} not found in index.")

        return self.velocity_symbol(self.pose_indices[timestamp])

    def get_landmark_key(self, landmark_id: int) -> Any:
        """
        Get the GTSAM key for a landmark.

        Args:
            landmark_id: ID of the landmark.

        Returns:
            GTSAM symbol for the landmark.
        """
        if landmark_id not in self.landmark_indices:
            raise KeyError(f"Landmark with ID {landmark_id} not found in index.")

        return self.landmark_symbol(self.landmark_indices[landmark_id])

    def get_camera_extrinsics_key(self, camera_id: str) -> Any:
        """
        Get the GTSAM key for camera extrinsics.

        Args:
            camera_id: ID of the camera.

        Returns:
            GTSAM symbol for the camera extrinsics.
        """
        if camera_id not in self.camera_indices:
            raise KeyError(f"Camera with ID {camera_id} not found in index.")

        return self.camera_extrinsics_symbol(self.camera_indices[camera_id])

    def get_camera_intrinsics_key(self, camera_id: str) -> Any:
        """
        Get the GTSAM key for camera intrinsics.

        Args:
            camera_id: ID of the camera.

        Returns:
            GTSAM symbol for the camera intrinsics.
        """
        if camera_id not in self.camera_indices:
            raise KeyError(f"Camera with ID {camera_id} not found in index.")

        return self.camera_intrinsics_symbol(self.camera_indices[camera_id])

    def get_imu_bias_key(self, index: int = 0) -> Any:
        """
        Get the GTSAM key for an IMU bias.

        Args:
            index: Index of the IMU bias (default: 0).

        Returns:
            GTSAM symbol for the IMU bias.
        """
        return self.imu_bias_symbol(index)

def create_initial_values(poses: List[VehiclePose],
                         landmarks: Dict[int, Landmark],
                         intrinsics: Dict[str, CameraIntrinsics],
                         extrinsics: Dict[str, Extrinsics],
                         variable_index: VariableIndex,
                         initial_velocity: Optional[np.ndarray] = None,
                         initial_bias: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Any:
    """
    Create initial values for the optimizer.

    Args:
        poses: List of vehicle poses.
        landmarks: Dictionary of landmarks.
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.
        variable_index: Variable index for mapping.
        initial_velocity: Initial velocity (default: zeros).
        initial_bias: Initial IMU bias (accel_bias, gyro_bias) (default: zeros).

    Returns:
        GTSAM Values object with initial values.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create values container
    values = gtsam.Values()

    # Add poses
    for pose in poses:
        pose_key = variable_index.add_pose(pose.timestamp)
        gtsam_pose = pose_to_gtsam_pose(pose.rotation, pose.translation)
        values.insert(pose_key, gtsam_pose)

        # Add velocity if IMU is used
        if initial_velocity is not None:
            vel_key = variable_index.add_velocity(pose.timestamp)
            # Use numpy array for velocity (GTSAM doesn't have Vector3 in some versions)
            gtsam_vel = np.array([initial_velocity[0], initial_velocity[1], initial_velocity[2]])
            values.insert(vel_key, gtsam_vel)

    # Add landmarks
    for landmark_id, landmark in landmarks.items():
        landmark_key = variable_index.add_landmark(landmark_id)
        gtsam_point = translation_to_gtsam_point(landmark.position)
        values.insert(landmark_key, gtsam_point)

    # Add camera intrinsics and extrinsics
    for camera_id in intrinsics.keys():
        # Add intrinsics
        intr = intrinsics[camera_id]
        intrinsics_key = variable_index.add_camera_intrinsics(camera_id)
        gtsam_calibration = intrinsics_to_gtsam_calibration(
            intr.fx, intr.fy, intr.cx, intr.cy, intr.distortion_coeffs
        )
        values.insert(intrinsics_key, gtsam_calibration)

        # Add extrinsics
        extr = extrinsics[camera_id]
        extrinsics_key = variable_index.add_camera_extrinsics(camera_id)
        gtsam_extrinsics = pose_to_gtsam_pose(extr.rotation, extr.translation)
        values.insert(extrinsics_key, gtsam_extrinsics)

    # Add IMU bias if provided
    if initial_bias is not None:
        accel_bias, gyro_bias = initial_bias
        bias_key = variable_index.add_imu_bias()
        gtsam_bias = create_imu_bias(accel_bias, gyro_bias)
        values.insert(bias_key, gtsam_bias)

    return values

def extract_optimized_values(optimized_values: Any, variable_index: Any, camera_ids: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """Extract optimized values from GTSAM Values object."""
    # Check if GTSAM is available
    if not check_gtsam_availability():
        print("GTSAM is not available. Cannot extract optimized values.")
        return {}, {}, None

    import gtsam  # Import here to avoid issues if GTSAM is not available

    optimized_intrinsics = {}
    optimized_extrinsics = {}
    optimized_biases = None

    # Helper function to convert GTSAM Pose3 to our Extrinsics type
    def convert_pose3_to_extrinsics(pose_value: Any) -> Extrinsics:
        if isinstance(pose_value, gtsam.Pose3):
            return Extrinsics(
                rotation=pose_value.rotation().matrix(),
                translation=np.array([
                    pose_value.translation().x(),
                    pose_value.translation().y(),
                    pose_value.translation().z()
                ])
            )
        elif isinstance(pose_value, np.ndarray):
            # Handle different numpy array shapes
            if pose_value.shape == (4, 4):  # 4x4 transformation matrix
                return Extrinsics(
                    rotation=pose_value[:3, :3],
                    translation=pose_value[:3, 3]
                )
            elif pose_value.shape == (3, 3):  # 3x3 rotation matrix only
                return Extrinsics(
                    rotation=pose_value,
                    translation=np.zeros(3)
                )
            elif pose_value.shape == (3,):  # 3D translation vector only
                return Extrinsics(
                    rotation=np.eye(3),
                    translation=pose_value
                )
            else:
                print(f"Warning: Unexpected numpy array shape: {pose_value.shape}, using identity")
                return Extrinsics(
                    rotation=np.eye(3),
                    translation=np.zeros(3)
                )
        else:
            print(f"Warning: Unexpected pose type: {type(pose_value)}, using identity")
            return Extrinsics(
                rotation=np.eye(3),
                translation=np.zeros(3)
            )

    # Helper function to extract Point3 values safely
    def extract_point3(point_value: Any) -> np.ndarray:
        if isinstance(point_value, gtsam.Point3):
            return np.array([point_value.x(), point_value.y(), point_value.z()])
        elif isinstance(point_value, np.ndarray):
            return point_value
        else:
            print(f"Warning: Unexpected point type: {type(point_value)}, using zeros")
            return np.zeros(3)

    # Helper function to check if a key exists in GTSAM Values
    def has_key(values: gtsam.Values, key: Any) -> bool:
        try:
            return values.exists(key)
        except:
            try:
                # Older GTSAM versions
                return key in [values.keys()[i] for i in range(values.size())]
            except:
                return False

    # Helper function to safely extract calibration parameters
    def extract_calibration(cal_value: Any) -> Tuple[float, float, float, float]:
        try:
            if hasattr(cal_value, 'fx') and hasattr(cal_value, 'fy') and hasattr(cal_value, 'px') and hasattr(cal_value, 'py'):
                # Standard GTSAM Cal3_S2 or similar
                return cal_value.fx(), cal_value.fy(), cal_value.px(), cal_value.py()
            elif isinstance(cal_value, np.ndarray):
                # Numpy array format [fx, fy, cx, cy]
                if len(cal_value) >= 4:
                    return cal_value[0], cal_value[1], cal_value[2], cal_value[3]
                else:
                    print(f"Warning: Calibration array too short: {cal_value}, using defaults")
                    return 500.0, 500.0, 320.0, 240.0
            else:
                print(f"Warning: Unknown calibration type: {type(cal_value)}, using defaults")
                return 500.0, 500.0, 320.0, 240.0
        except Exception as e:
            print(f"Warning: Error extracting calibration: {e}, using defaults")
            return 500.0, 500.0, 320.0, 240.0

    # Extract camera intrinsics
    for cam_id in camera_ids:
        try:
            intrinsics_key = variable_index.get_camera_intrinsics_key(cam_id)
            if has_key(optimized_values, intrinsics_key):
                # Try different methods to extract calibration
                try:
                    cal = optimized_values.atCal3_S2(intrinsics_key)
                except Exception as e1:
                    try:
                        # Try generic retrieval
                        cal = optimized_values.at(intrinsics_key)
                    except Exception as e2:
                        print(f"Warning: Could not extract intrinsics for camera {cam_id}: {e1}, {e2}")
                        cal = None

                if cal is not None:
                    fx, fy, cx, cy = extract_calibration(cal)
                    optimized_intrinsics[cam_id] = CameraIntrinsics(
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        distortion_coeffs=np.zeros(5)  # Placeholder, update if using distortion
                    )
                else:
                    optimized_intrinsics[cam_id] = None
            else:
                optimized_intrinsics[cam_id] = None
        except Exception as e:
            print(f"Warning: Could not extract intrinsics for camera {cam_id}: {e}")
            optimized_intrinsics[cam_id] = None

    # Extract camera extrinsics
    for cam_id in camera_ids:
        try:
            extrinsics_key = variable_index.get_camera_extrinsics_key(cam_id)
            if has_key(optimized_values, extrinsics_key):
                # Try different methods to extract pose
                try:
                    pose_value = optimized_values.atPose3(extrinsics_key)
                except Exception as e1:
                    try:
                        # Try generic retrieval
                        pose_value = optimized_values.at(extrinsics_key)
                    except Exception as e2:
                        print(f"Warning: Could not extract extrinsics for camera {cam_id}: {e1}, {e2}")
                        pose_value = None

                if pose_value is not None:
                    optimized_extrinsics[cam_id] = convert_pose3_to_extrinsics(pose_value)
                else:
                    optimized_extrinsics[cam_id] = None
            else:
                optimized_extrinsics[cam_id] = None
        except Exception as e:
            print(f"Warning: Could not extract extrinsics for camera {cam_id}: {e}")
            optimized_extrinsics[cam_id] = None

    # Extract IMU biases if available
    try:
        bias_key = variable_index.get_imu_bias_key()
        if has_key(optimized_values, bias_key):
            try:
                optimized_biases = optimized_values.atConstantBias(bias_key)
            except Exception as e1:
                try:
                    # Try generic retrieval
                    optimized_biases = optimized_values.at(bias_key)
                except Exception as e2:
                    print(f"Warning: Could not extract IMU biases: {e1}, {e2}")
                    optimized_biases = None
    except Exception as e:
        print(f"Warning: Could not extract IMU biases: {e}")
        optimized_biases = None

    return optimized_intrinsics, optimized_extrinsics, optimized_biases
