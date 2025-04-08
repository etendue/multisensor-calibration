Technical Implementation Document
Targetless Multisensor Calibration System
Version: 1.0
Date: 2025-04-08
Reference TDD: multisensor_calib_tdd_v1

1. Introduction

This document provides implementation details for the core algorithms of the multisensor calibration system, building upon the design outlined in the Technical Design Document (multisensor_calib_tdd_v1). It includes Python code skeletons for key modules and the relevant mathematical formulations required for their implementation.

2. Overall Pipeline Structure (Python Skeleton)

# --- Core Data Structures (as defined in TDD/previous code) ---
# TimestampedData, ImageData, ImuData, WheelEncoderData,
# CameraIntrinsics, Extrinsics, Feature, Match, Landmark, VehiclePose
import numpy as np
from typing import List, Dict, Tuple, Any

# --- Modules ---

def synchronize_data(image_streams: Dict[str, List[ImageData]],
                     imu_stream: List[ImuData],
                     wheel_stream: List[WheelEncoderData],
                     sync_tolerance: float,
                     target_rate_source: str = 'imu') -> Tuple[List[Dict[str, ImageData]], List[ImuData], List[WheelEncoderData]]:
    """Aligns sensor data streams based on timestamps."""
    # Implementation: Interpolation or nearest neighbor matching.
    print("Synchronizing data...")
    # ... implementation details ...
    synchronized_images = [] # List of dicts: {cam_id: ImageData} per synced timestamp
    synchronized_imu = []
    synchronized_wheels = []
    return synchronized_images, synchronized_imu, synchronized_wheels

def estimate_initial_ego_motion_ekf(imu_data: List[ImuData],
                                    wheel_data: List[WheelEncoderData],
                                    initial_pose: VehiclePose,
                                    initial_bias: Dict,
                                    noise_params: Dict,
                                    vehicle_params: Dict) -> List[VehiclePose]:
    """Estimates initial trajectory using an Extended Kalman Filter."""
    print("Estimating initial ego-motion (EKF)...")
    # ... EKF implementation details (see Section 3) ...
    poses = [initial_pose]
    return poses

def detect_and_match_features(image_data: Dict[str, ImageData],
                              detector_type: str = 'ORB') -> Dict[str, Tuple[List[Feature], List[Any]]]:
    """Detects features and computes descriptors for images at a timestamp."""
    print(f"Detecting features ({detector_type})...")
    # Implementation: Uses OpenCV detector (cv2.ORB_create(), etc.)
    features_descriptors = {} # {cam_id: (features, descriptors)}
    # ... implementation details ...
    return features_descriptors

def track_and_triangulate(synced_image_data: List[Dict[str, ImageData]],
                          initial_poses: List[VehiclePose],
                          initial_intrinsics: Dict[str, CameraIntrinsics],
                          initial_extrinsics: Dict[str, Extrinsics],
                          feature_params: Dict) -> Tuple[Dict[int, Landmark], Dict[Tuple[float, str], List[Feature]]]:
    """Tracks features, selects keyframes, performs triangulation."""
    print("Tracking features and triangulating landmarks...")
    # Implementation: Feature matching (BFMatcher/FLANN), KLT tracking,
    # Keyframe selection logic, Triangulation (DLT) (see Section 4 & 5)
    landmarks = {} # {landmark_id: Landmark}
    feature_observations = {} # {(timestamp, cam_id): [Feature]}
    # ... implementation details ...
    return landmarks, feature_observations

def run_backend_optimization(initial_poses: List[VehiclePose],
                             initial_landmarks: Dict[int, Landmark],
                             feature_observations: Dict[Tuple[float, str], List[Feature]],
                             initial_intrinsics: Dict[str, CameraIntrinsics],
                             initial_extrinsics: Dict[str, Extrinsics],
                             imu_data: List[ImuData],
                             wheel_data: List[WheelEncoderData],
                             optimizer_params: Dict) -> Dict:
    """Builds and solves the factor graph optimization problem using GTSAM."""
    print("Running backend optimization (GTSAM)...")
    # Implementation: Build GTSAM graph, add factors (see Section 6), run optimizer
    optimized_values = {} # Dictionary mapping GTSAM keys to optimized values
    # ... implementation details ...
    return optimized_values

def extract_results(optimized_values: Dict) -> Tuple[Dict[str, CameraIntrinsics], Dict[str, Extrinsics], Dict]:
    """Extracts final calibration parameters from optimization results."""
    print("Extracting results...")
    # Implementation: Retrieve values for intrinsics, extrinsics, biases from GTSAM result object
    final_intrinsics = {}
    final_extrinsics = {}
    final_biases = {}
    # ... implementation details ...
    return final_intrinsics, final_extrinsics, final_biases

# --- Main Pipeline Execution ---
# if __name__ == "__main__":
#   # Load config
#   # Load raw data
#   # synchronized_data = synchronize_data(...)
#   # initial_poses = estimate_initial_ego_motion_ekf(...)
#   # landmarks, features = track_and_triangulate(...)
#   # optimized_values = run_backend_optimization(...)
#   # final_params = extract_results(...)
#   # Save results

3. Ego-Motion Estimation (EKF)

Purpose: Provide an initial estimate of vehicle trajectory T_W_Vk (pose in World frame at time k) by fusing IMU and Wheel Odometry.

State Vector (x): Typically includes pose and IMU biases. Example:
x = [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
Where p is position, v is velocity (both in World frame W), q is the quaternion representing orientation T_W_V, and b_a, b_g are accelerometer and gyroscope biases (in IMU frame I).

Prediction Step (IMU Measurement): Propagate state from k to k+1 using IMU measurements omega_m, a_m between t_k and t_{k+1} (dt = t_{k+1} - t_k).

Estimate true angular velocity: omega = omega_m - b_g

Update Orientation (Quaternion integration): q_{k+1} = q_k * delta_q(omega * dt) where delta_q converts rotation vector to quaternion.

Estimate true acceleration (compensate bias, gravity g_W, rotate to World frame): a_W = R(q_k) * (a_m - b_a) - g_W (where R(q_k) is rotation matrix from q_k).

Update Velocity: v_{k+1} = v_k + a_W * dt

Update Position: p_{k+1} = p_k + v_k * dt + 0.5 * a_W * dt^2

Biases: Often modeled as random walk: b_{k+1} = b_k + noise.

State Transition: x_{k+1|k} = f(x_{k|k}, u_k) where u_k is the IMU input.

Covariance Prediction: P_{k+1|k} = F_k * P_{k|k} * F_k^T + Q_k where F_k is the Jacobian of f w.r.t x, and Q_k is the process noise covariance derived from IMU noise parameters.

Update Step (Wheel Odometry Measurement): Correct the predicted state using wheel odometry measurement z_k.

Vehicle Motion Model (Example: Differential Drive):
v = (speed_r + speed_l) / 2 (average wheel speed)
omega_z = (speed_r - speed_l) / track_width (yaw rate)
Measurement z_k could be [v, omega_z].

Measurement Prediction: z_hat = h(x_{k+1|k}). Predict expected v and omega_z from the EKF state (e.g., v from v_x, v_y, omega_z from quaternion derivative or gyro bias).

Measurement Jacobian: H_k = dh/dx evaluated at x_{k+1|k}.

Kalman Gain: K_k = P_{k+1|k} * H_k^T * (H_k * P_{k+1|k} * H_k^T + R_k)^{-1} where R_k is the measurement noise covariance.

State Update: x_{k+1|k+1} = x_{k+1|k} + K_k * (z_k - z_hat)

Covariance Update: P_{k+1|k+1} = (I - K_k * H_k) * P_{k+1|k}

Code Skeleton:

class EKF:
    def __init__(self, initial_state, initial_covariance, noise_params, vehicle_params):
        self.x = initial_state
        self.P = initial_covariance
        # ... store noise/vehicle params ...

    def predict(self, imu_measurement: ImuData, dt: float):
        # Implement state propagation f(x, u)
        # Calculate Jacobian F
        # Implement covariance propagation: P = F*P*F' + Q
        pass

    def update(self, wheel_measurement: WheelEncoderData):
        # Implement measurement prediction h(x)
        # Calculate Jacobian H
        # Calculate Kalman Gain K
        # Implement state update: x = x + K*(z - h(x))
        # Implement covariance update: P = (I - K*H)*P
        pass

4. Visual Processing (Features & Matching)

Purpose: Detect stable 2D points in images and find correspondences across time and cameras.

Detection (e.g., ORB): Uses FAST keypoint detector and rotated BRIEF descriptor. OpenCV: orb = cv2.ORB_create(...), keypoints, descriptors = orb.detectAndCompute(image, None).

Matching (e.g., Brute-Force Hamming for ORB): Compares descriptors between two sets. OpenCV: bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True), matches = bf.match(des1, des2). Apply ratio test (Lowe's test) if not using crossCheck for better quality.

Tracking (e.g., KLT): Estimates optical flow for sparse features. OpenCV: p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, p0, None, **lk_params). p0 are feature points in img_prev, p1 are corresponding points in img_curr.

5. Initialization (Triangulation)

Purpose: Estimate initial 3D positions of landmarks observed in at least two views with known relative pose.

Input: Matched 2D feature points p1, p2 (in normalized image coordinates) and corresponding camera projection matrices M1, M2. M = K * [R|t]. For calibration, [R|t] is inv(T_V_Ci) * inv(T_W_Vk).

Method (DLT - Direct Linear Transform): Each match provides 2 linear equations for the 3D point P = [X, Y, Z, 1].
p = [u, v, 1]. Projection equation: p_proj = M * P.
Since p and p_proj are parallel: p x (M * P) = 0.
This gives equations like:
(u * m3 - m1) * P = 0
(v * m3 - m2) * P = 0
(where m1, m2, m3 are rows of M).
Stack equations from both views (M1, p1 and M2, p2) into A * P = 0. Solve for P using Singular Value Decomposition (SVD) of A. P is the last column of V where A = U * S * V^T.

Code Skeleton:

def triangulate_point(kp1_norm: np.ndarray, kp2_norm: np.ndarray, M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Triangulates a 3D point using DLT from two views.

    Args:
        kp1_norm: Normalized coordinates [x, y] in view 1.
        kp2_norm: Normalized coordinates [x, y] in view 2.
        M1: 3x4 Projection matrix for view 1 (K * [R|t]).
        M2: 3x4 Projection matrix for view 2.

    Returns:
        Homogeneous 3D point [X, Y, Z, W] or None if failed.
    """
    # Construct matrix A from kp1, M1, kp2, M2 based on DLT equations
    A = []
    # A.append(kp1_norm[0] * M1[2, :] - M1[0, :])
    # A.append(kp1_norm[1] * M1[2, :] - M1[1, :])
    # A.append(kp2_norm[0] * M2[2, :] - M2[0, :])
    # A.append(kp2_norm[1] * M2[2, :] - M2[1, :])
    A = np.array(A)
    # Solve A*P = 0 using SVD
    # U, S, Vh = np.linalg.svd(A)
    # P = Vh[-1, :]
    # return P / P[3] # Dehomogenize
    return np.zeros(4) # Placeholder

6. Factor Graph Optimization (Factors)

Purpose: Define the cost functions (factors) that link variables based on sensor measurements. Minimize sum of squared errors.

Reprojection Factor: Penalizes difference between observed feature location p_obs = [u, v] and projected landmark position P_j (in World frame W).

Camera i at time k. Pose T_W_Vk, Extrinsics T_V_Ci, Intrinsics K_i.

Transform P_j to camera frame C_i: P_c = T_Ci_V * inv(T_W_Vk) * P_j

Project to pixel coordinates (pinhole model): p_proj = project(K_i, P_c)
[u', v', w']^T = K_i * [I|0] * P_c
p_proj = [u'/w', v'/w'] (Apply distortion model here if needed)

Error: e_reproj = p_obs - p_proj

Cost: ||e_reproj||^2_Sigma (where Sigma is covariance of feature measurement).

GTSAM: gtsam.GenericProjectionFactorCal3_S2 (or similar based on intrinsic model) connecting Pose3(T_W_Vk), Point3(P_j), Pose3(T_V_Ci), Cal3_S2(K_i).

IMU Factor (Preintegration): Penalizes discrepancy between relative motion predicted by IMU integration and relative motion from optimized poses.

Variables: Pose_k, Velocity_k, Bias_k, Pose_{k+1}, Velocity_{k+1}, Bias_{k+1}.

IMU Measurements between k and k+1.

Preintegrated Measurements (delta R_m, delta v_m, delta p_m) calculated from IMU data and current bias estimates.

Predicted Relative Motion from poses:
delta R_pred = inv(R_k) * R_{k+1}
delta v_pred = inv(R_k) * (v_{k+1} - v_k - g_W*dt)
delta p_pred = inv(R_k) * (p_{k+1} - p_k - v_k*dt - 0.5*g_W*dt^2)

Error: e_imu = [ log(delta R_m.T * delta R_pred), delta v_pred - delta v_m, delta p_pred - delta p_m ] (Simplified representation). Correct formulation involves bias correction terms.

Cost: ||e_imu||^2_Sigma (where Sigma is covariance from IMU preintegration).

GTSAM: gtsam.ImuFactor or gtsam.CombinedImuFactor connecting poses, velocities, and biases.

Wheel Odometry Factor: Penalizes discrepancy between relative motion measured by wheel odometry and relative motion from optimized poses.

Variables: Pose_k, Pose_{k+1}.

Odometry Measurement delta_pose_odom (e.g., [dx, dy, dtheta]) between k and k+1.

Predicted Relative Pose: delta_pose_pred = logmap( inv(T_W_Vk) * T_W_V{k+1} ) (Convert SE(3) difference to a vector, e.g., [dx, dy, dz, drx, dry, drz]). Need to select relevant components (e.g., dx, dy, dtheta for planar motion).

Error: e_odom = delta_pose_odom - delta_pose_pred (matching dimensions).

Cost: ||e_odom||^2_Sigma (where Sigma is covariance of odometry measurement).

GTSAM: Requires a custom factor derived from gtsam.NoiseModelFactor.

Prior Factor: Encodes prior knowledge about a variable.

Example: Prior on initial pose Pose_0.

Error: e_prior = variable - prior_mean (or SE(3) logmap difference for poses).

Cost: ||e_prior||^2_Sigma (where Sigma reflects confidence in the prior).

GTSAM: gtsam.PriorFactorPose3, gtsam.PriorFactorPoint3, etc.