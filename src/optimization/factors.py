# Custom factors for GTSAM optimization
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
from src.optimization.gtsam_utils import check_gtsam_availability, create_noise_model

# We'll use GTSAM's BetweenFactor instead of a custom factor for wheel odometry

def create_wheel_odometry_factor(pose1_key: Any, pose2_key: Any,
                               dx: float, dy: float, dtheta: float,
                               translation_sigma: float = 0.1,
                               rotation_sigma: float = 0.05,
                               robust: bool = True,
                               robust_threshold: float = 1.0) -> Any:
    """
    Create a wheel odometry factor.

    Args:
        pose1_key: Key for the first pose.
        pose2_key: Key for the second pose.
        dx: Measured change in x position.
        dy: Measured change in y position.
        dtheta: Measured change in heading.
        translation_sigma: Standard deviation for translation measurements.
        rotation_sigma: Standard deviation for rotation measurements.
        robust: Whether to use a robust noise model.
        robust_threshold: Threshold for robust kernel.

    Returns:
        GTSAM wheel odometry factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create noise model for the factor
    # For BetweenFactorPose3, we need a 6-dimensional noise model (3 for rotation, 3 for translation)
    sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                      translation_sigma, translation_sigma, translation_sigma])
    noise_model = create_noise_model(sigmas, robust, robust_threshold)

    # Create a relative pose for the odometry measurement
    # Create rotation from the heading change
    rot = gtsam.Rot3.Rz(dtheta)
    # Create translation
    trans = gtsam.Point3(dx, dy, 0.0)
    # Create the relative pose
    relative_pose = gtsam.Pose3(rot, trans)

    # Create and return a BetweenFactor
    return gtsam.BetweenFactorPose3(pose1_key, pose2_key, relative_pose, noise_model)

def create_reprojection_factor(point_key: Any, pose_key: Any, camera_key: Any,
                             calibration: Any, measurement: np.ndarray,
                             pixel_sigma: float = 1.0, robust: bool = True,
                             robust_threshold: float = 1.0) -> Any:
    """
    Create a reprojection factor.

    Args:
        point_key: Key for the 3D landmark.
        pose_key: Key for the vehicle pose.
        camera_key: Key for the camera extrinsics.
        calibration: GTSAM calibration object.
        measurement: 2D feature location [u, v].
        pixel_sigma: Standard deviation in pixels.
        robust: Whether to use a robust noise model.
        robust_threshold: Threshold for robust kernel.

    Returns:
        GTSAM reprojection factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create noise model for the factor
    sigmas = np.array([pixel_sigma, pixel_sigma])
    noise_model = create_noise_model(sigmas, robust, robust_threshold)

    # Create and return the factor
    point2 = gtsam.Point2(measurement[0], measurement[1])

    # Create the camera extrinsics pose
    body_P_sensor = gtsam.Pose3()

    # Use the appropriate projection factor based on the calibration type
    if isinstance(calibration, gtsam.Cal3_S2):
        return gtsam.GenericProjectionFactorCal3_S2(
            point2, noise_model, pose_key, point_key, calibration, body_P_sensor
        )
    elif isinstance(calibration, gtsam.Cal3DS2):
        return gtsam.GenericProjectionFactorCal3DS2(
            point2, noise_model, pose_key, point_key, calibration, body_P_sensor
        )
    else:
        raise ValueError(f"Unsupported calibration type: {type(calibration)}")

def create_imu_factor(pose1_key: Any, vel1_key: Any, pose2_key: Any, vel2_key: Any,
                    bias_key: Any, preintegrated_measurements: Any) -> Any:
    """
    Create an IMU factor.

    Args:
        pose1_key: Key for the first pose.
        vel1_key: Key for the first velocity.
        pose2_key: Key for the second pose.
        vel2_key: Key for the second velocity.
        bias_key: Key for the IMU bias.
        preintegrated_measurements: GTSAM preintegrated IMU measurements.

    Returns:
        GTSAM IMU factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create and return the IMU factor
    return gtsam.ImuFactor(
        pose1_key, vel1_key, pose2_key, vel2_key, bias_key, preintegrated_measurements
    )

def create_prior_factor_pose(key: Any, prior_pose: Any,
                           translation_sigma: float = 0.1,
                           rotation_sigma: float = 0.1) -> Any:
    """
    Create a prior factor for a pose.

    Args:
        key: Key for the pose.
        prior_pose: Prior pose (GTSAM Pose3).
        translation_sigma: Standard deviation for translation.
        rotation_sigma: Standard deviation for rotation.

    Returns:
        GTSAM prior factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create noise model for the factor
    sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                      translation_sigma, translation_sigma, translation_sigma])
    noise_model = create_noise_model(sigmas)

    # Create and return the prior factor
    return gtsam.PriorFactorPose3(key, prior_pose, noise_model)

def create_prior_factor_point(key: Any, prior_point: Any, sigma: float = 0.1) -> Any:
    """
    Create a prior factor for a 3D point.

    Args:
        key: Key for the point.
        prior_point: Prior point (GTSAM Point3).
        sigma: Standard deviation for the point coordinates.

    Returns:
        GTSAM prior factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create noise model for the factor
    sigmas = np.array([sigma, sigma, sigma])
    noise_model = create_noise_model(sigmas)

    # Create and return the prior factor
    return gtsam.PriorFactorPoint3(key, prior_point, noise_model)

def create_prior_factor_calibration(key: Any, prior_calibration: Any,
                                  focal_sigma: float = 10.0,
                                  principal_point_sigma: float = 10.0) -> Any:
    """
    Create a prior factor for camera calibration.

    Args:
        key: Key for the calibration.
        prior_calibration: Prior calibration (GTSAM Cal3_S2 or similar).
        focal_sigma: Standard deviation for focal length.
        principal_point_sigma: Standard deviation for principal point.

    Returns:
        GTSAM prior factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create noise model for the factor
    if isinstance(prior_calibration, gtsam.Cal3_S2):
        # For Cal3_S2: fx, fy, s, cx, cy
        sigmas = np.array([focal_sigma, focal_sigma, 0.0, principal_point_sigma, principal_point_sigma])
        noise_model = create_noise_model(sigmas)
        return gtsam.PriorFactorCal3_S2(key, prior_calibration, noise_model)
    elif isinstance(prior_calibration, gtsam.Cal3DS2):
        # For Cal3DS2: fx, fy, s, cx, cy, k1, k2, p1, p2
        sigmas = np.array([focal_sigma, focal_sigma, 0.0, principal_point_sigma, principal_point_sigma,
                          0.1, 0.1, 0.1, 0.1])  # Distortion parameters with lower confidence
        noise_model = create_noise_model(sigmas)
        return gtsam.PriorFactorCal3DS2(key, prior_calibration, noise_model)
    else:
        raise ValueError(f"Unsupported calibration type: {type(prior_calibration)}")

def create_prior_factor_bias(key: Any, prior_bias: Any,
                           accel_sigma: float = 0.1,
                           gyro_sigma: float = 0.01) -> Any:
    """
    Create a prior factor for IMU bias.

    Args:
        key: Key for the bias.
        prior_bias: Prior bias (GTSAM imuBias.ConstantBias).
        accel_sigma: Standard deviation for accelerometer bias.
        gyro_sigma: Standard deviation for gyroscope bias.

    Returns:
        GTSAM prior factor.
    """
    if not GTSAM_AVAILABLE:
        raise ImportError("GTSAM is not available. Install with 'pip install gtsam'.")

    # Create noise model for the factor
    sigmas = np.array([accel_sigma, accel_sigma, accel_sigma,
                      gyro_sigma, gyro_sigma, gyro_sigma])
    noise_model = create_noise_model(sigmas)

    # Create and return the prior factor
    return gtsam.PriorFactorConstantBias(key, prior_bias, noise_model)
