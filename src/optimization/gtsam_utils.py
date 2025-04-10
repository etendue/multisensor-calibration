# GTSAM utility functions
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

# Try to import GTSAM
try:
    import gtsam
    import gtsam.utils.plot as gtsam_plot
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("Warning: GTSAM not available. Install with 'pip install gtsam' for optimization.")

def check_gtsam_availability() -> bool:
    """
    Check if GTSAM is available.

    Returns:
        True if GTSAM is available, False otherwise.
    """
    return GTSAM_AVAILABLE

def rotation_matrix_to_gtsam_rotation(rotation_matrix: np.ndarray) -> Any:
    """
    Convert a 3x3 rotation matrix to a GTSAM Rot3.

    Args:
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        GTSAM Rot3 object.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    return gtsam.Rot3(rotation_matrix)

def translation_to_gtsam_point(translation: np.ndarray) -> Any:
    """
    Convert a 3x1 translation vector to a GTSAM Point3.

    Args:
        translation: 3x1 translation vector.

    Returns:
        GTSAM Point3 object.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    return gtsam.Point3(translation[0], translation[1], translation[2])

def pose_to_gtsam_pose(rotation_matrix: np.ndarray, translation: np.ndarray) -> Any:
    """
    Convert a rotation matrix and translation vector to a GTSAM Pose3.

    Args:
        rotation_matrix: 3x3 rotation matrix.
        translation: 3x1 translation vector.

    Returns:
        GTSAM Pose3 object.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    rot = rotation_matrix_to_gtsam_rotation(rotation_matrix)
    trans = translation_to_gtsam_point(translation)
    return gtsam.Pose3(rot, trans)

def intrinsics_to_gtsam_calibration(fx: float, fy: float, cx: float, cy: float,
                                   distortion_coeffs: Optional[np.ndarray] = None) -> Any:
    """
    Convert camera intrinsics to a GTSAM calibration object.

    Args:
        fx: Focal length x.
        fy: Focal length y.
        cx: Principal point x.
        cy: Principal point y.
        distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3].

    Returns:
        GTSAM calibration object (Cal3_S2 or Cal3DS2 depending on distortion).
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    if distortion_coeffs is None or np.all(np.abs(distortion_coeffs) < 1e-10):
        # No distortion or negligible distortion
        return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)  # Assuming no skew
    else:
        # With distortion
        k1, k2 = distortion_coeffs[0], distortion_coeffs[1]
        p1, p2 = distortion_coeffs[2], distortion_coeffs[3]
        k3 = distortion_coeffs[4] if len(distortion_coeffs) > 4 else 0.0
        return gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)

def create_noise_model(sigmas: np.ndarray, robust: bool = False, robust_threshold: float = 1.0) -> Any:
    """
    Create a GTSAM noise model.

    Args:
        sigmas: Standard deviations for each dimension.
        robust: Whether to use a robust noise model.
        robust_threshold: Threshold for robust kernel.

    Returns:
        GTSAM noise model.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create diagonal noise model
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    # Apply robust kernel if requested
    if robust:
        noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(robust_threshold),
            noise_model
        )

    return noise_model

def create_between_factor_noise_model(translation_sigma: float, rotation_sigma: float,
                                     robust: bool = False, robust_threshold: float = 1.0) -> Any:
    """
    Create a noise model for between factors (e.g., odometry).

    Args:
        translation_sigma: Standard deviation for translation components.
        rotation_sigma: Standard deviation for rotation components.
        robust: Whether to use a robust noise model.
        robust_threshold: Threshold for robust kernel.

    Returns:
        GTSAM noise model.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create sigmas for 6 DOF pose (rx, ry, rz, tx, ty, tz)
    sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                       translation_sigma, translation_sigma, translation_sigma])

    return create_noise_model(sigmas, robust, robust_threshold)

def create_projection_factor_noise_model(pixel_sigma: float, robust: bool = True,
                                        robust_threshold: float = 1.0) -> Any:
    """
    Create a noise model for projection factors.

    Args:
        pixel_sigma: Standard deviation in pixels.
        robust: Whether to use a robust noise model.
        robust_threshold: Threshold for robust kernel.

    Returns:
        GTSAM noise model.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create sigmas for 2D projection (u, v)
    sigmas = np.array([pixel_sigma, pixel_sigma])

    return create_noise_model(sigmas, robust, robust_threshold)

def create_imu_noise_model(accel_sigma: float, gyro_sigma: float,
                          accel_bias_sigma: float, gyro_bias_sigma: float) -> Any:
    """
    Create noise parameters for IMU preintegration.

    Args:
        accel_sigma: Standard deviation of accelerometer noise.
        gyro_sigma: Standard deviation of gyroscope noise.
        accel_bias_sigma: Standard deviation of accelerometer bias random walk.
        gyro_bias_sigma: Standard deviation of gyroscope bias random walk.

    Returns:
        GTSAM IMU noise parameters.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create IMU noise parameters
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)  # Assuming standard gravity

    # Set gyroscope and accelerometer noise
    accel_noise = accel_sigma ** 2
    gyro_noise = gyro_sigma ** 2
    imu_params.setAccelerometerCovariance(accel_noise * np.eye(3))
    imu_params.setGyroscopeCovariance(gyro_noise * np.eye(3))

    # Set bias random walk - use the available methods in the current GTSAM version
    # Note: In some versions of GTSAM, these methods might have different names
    try:
        # Try the newer API if available
        accel_bias_rw = accel_bias_sigma ** 2
        gyro_bias_rw = gyro_bias_sigma ** 2
        imu_params.setAccelerometerBiasCovariance(accel_bias_rw * np.eye(3))
        imu_params.setGyroscopeBiasCovariance(gyro_bias_rw * np.eye(3))
    except AttributeError:
        # Fall back to older API or skip if not available
        print("Warning: IMU bias covariance methods not available in this GTSAM version.")
        # Some versions use these methods instead
        try:
            imu_params.setBiasAccCovariance(accel_bias_sigma * np.eye(3))
            imu_params.setBiasOmegaCovariance(gyro_bias_sigma * np.eye(3))
        except AttributeError:
            print("Warning: Alternative IMU bias covariance methods also not available.")
            # If neither is available, we'll proceed without setting bias parameters

    return imu_params

def create_imu_bias(accel_bias: np.ndarray, gyro_bias: np.ndarray) -> Any:
    """
    Create a GTSAM IMU bias object.

    Args:
        accel_bias: Accelerometer bias vector [x, y, z].
        gyro_bias: Gyroscope bias vector [x, y, z].

    Returns:
        GTSAM IMU bias object.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    return gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
