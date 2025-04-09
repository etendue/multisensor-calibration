# Structure from Motion (SfM) initialization module
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import time
from collections import defaultdict

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import ImageData, VehiclePose, CameraIntrinsics, Feature, Landmark, Extrinsics, Match

# Note: This implementation requires OpenCV for feature detection and matching
# Uncomment the following line when OpenCV is installed
# import cv2


def detect_features(image: np.ndarray, detector_type: str = 'ORB', max_features: int = 1000) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Detect features in an image using the specified detector.

    Args:
        image: Input image as a numpy array.
        detector_type: Type of feature detector ('ORB', 'SIFT', etc.)
        max_features: Maximum number of features to detect.

    Returns:
        Tuple of (keypoints as numpy arrays [x, y], descriptors)
    """
    # Note: This is a placeholder implementation. In a real implementation, we would use OpenCV.
    # Example OpenCV implementation:
    # if detector_type == 'ORB':
    #     detector = cv2.ORB_create(nfeatures=max_features)
    # elif detector_type == 'SIFT':
    #     detector = cv2.SIFT_create(nfeatures=max_features)
    # else:
    #     raise ValueError(f"Unsupported detector type: {detector_type}")
    #
    # keypoints, descriptors = detector.detectAndCompute(image, None)
    # keypoints_np = np.array([kp.pt for kp in keypoints])
    # return keypoints_np, descriptors

    # Placeholder implementation that generates random features
    height, width = image.shape[:2] if len(image.shape) > 1 else (480, 640)
    num_features = min(max_features, np.random.randint(50, 150))
    keypoints = np.random.rand(num_features, 2) * np.array([width, height])
    descriptors = np.random.rand(num_features, 32)  # Random descriptors
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.8) -> List[Match]:
    """
    Match features between two sets of descriptors using a ratio test.

    Args:
        desc1: First set of descriptors.
        desc2: Second set of descriptors.
        ratio_threshold: Threshold for Lowe's ratio test.

    Returns:
        List of Match objects representing the matches.
    """
    # Note: This is a placeholder implementation. In a real implementation, we would use OpenCV.
    # Example OpenCV implementation:
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # matches = matcher.knnMatch(desc1, desc2, k=2)
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < ratio_threshold * n.distance:
    #         good_matches.append(Match(m.queryIdx, m.trainIdx))
    # return good_matches

    # Placeholder implementation that generates random matches
    num_matches = min(len(desc1), len(desc2)) // 3  # Simulate about 1/3 of features matching
    matches = []
    for i in range(num_matches):
        idx1 = np.random.randint(0, len(desc1))
        idx2 = np.random.randint(0, len(desc2))
        matches.append(Match(idx1, idx2))
    return matches


def select_keyframes(image_data: List[ImageData], min_distance: float = 0.5, min_rotation: float = 0.1) -> List[int]:
    """
    Select keyframes from the image data based on distance traveled and rotation.

    Args:
        image_data: List of image data.
        min_distance: Minimum distance between keyframes (meters).
        min_rotation: Minimum rotation between keyframes (radians).

    Returns:
        List of indices of selected keyframes.
    """
    # Placeholder implementation that selects every 5th frame
    return list(range(0, len(image_data), 5))


def triangulate_point(kp1: np.ndarray, kp2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Triangulate a 3D point from two 2D points and projection matrices using DLT.

    Args:
        kp1: 2D point in the first image [u, v].
        kp2: 2D point in the second image [u, v].
        P1: Projection matrix for the first camera (3x4).
        P2: Projection matrix for the second camera (3x4).

    Returns:
        3D point [X, Y, Z] in world coordinates.
    """
    # Construct the DLT matrix A
    A = np.zeros((4, 4))
    A[0] = kp1[0] * P1[2] - P1[0]
    A[1] = kp1[1] * P1[2] - P1[1]
    A[2] = kp2[0] * P2[2] - P2[0]
    A[3] = kp2[1] * P2[2] - P2[1]

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    # Convert to non-homogeneous coordinates
    X = X[:3] / X[3] if X[3] != 0 else X[:3]

    return X


def compute_projection_matrix(intrinsics: CameraIntrinsics, extrinsics: Extrinsics, vehicle_pose: VehiclePose) -> np.ndarray:
    """
    Compute the projection matrix for a camera at a specific vehicle pose.

    Args:
        intrinsics: Camera intrinsic parameters.
        extrinsics: Camera extrinsic parameters relative to the vehicle.
        vehicle_pose: Vehicle pose in the world frame.

    Returns:
        3x4 projection matrix P = K[R|t].
    """
    # Compute the camera pose in the world frame
    # T_W_C = T_W_V * T_V_C
    R_W_V = vehicle_pose.rotation
    t_W_V = vehicle_pose.translation

    R_V_C = extrinsics.rotation
    t_V_C = extrinsics.translation

    # Compute the camera rotation and translation in the world frame
    R_W_C = R_W_V @ R_V_C
    t_W_C = R_W_V @ t_V_C + t_W_V

    # Construct the projection matrix
    Rt = np.hstack((R_W_C, t_W_C.reshape(3, 1)))
    P = intrinsics.K @ Rt

    return P


def perform_visual_initialization(image_data: List[ImageData], initial_poses: List[VehiclePose], initial_intrinsics: Dict[str, CameraIntrinsics], initial_extrinsics: Optional[Dict[str, Extrinsics]] = None) -> Tuple[Dict[int, Landmark], Dict[Tuple[float, str], List[Feature]], Dict[str, CameraIntrinsics]]:
    """
    Performs Structure from Motion (SfM) initialization using images and initial poses.

    Args:
        image_data: List of synchronized image data.
        initial_poses: Initial vehicle trajectory estimates.
        initial_intrinsics: Initial guess for camera intrinsic parameters.
        initial_extrinsics: Initial guess for camera extrinsic parameters. If None, identity transforms are used.

    Returns:
        A tuple containing:
        - landmarks: Dictionary mapping landmark ID to Landmark object.
        - features: Dictionary mapping (timestamp, camera_id) to list of Feature objects.
        - refined_intrinsics: Potentially refined intrinsics after initial SfM/BA.
        This involves:
        - Feature Detection: Finding keypoints (e.g., SIFT, ORB) in images.
        - Feature Matching/Tracking: Finding corresponding features across time and cameras.
        - Triangulation: Estimating initial 3D positions of landmarks using matched features
            and initial camera poses (derived from vehicle poses and initial extrinsics guess).
            Requires initial guess for camera extrinsics relative to vehicle frame.
        - Local Bundle Adjustment (Optional): Refining poses and structure locally.
    """
    print("Performing visual initialization (SfM)...")

    # Initialize data structures
    features = {}  # (timestamp, camera_id) -> List[Feature]
    landmarks = {}  # landmark_id -> Landmark
    feature_to_landmark = {}  # (timestamp, camera_id, feature_idx) -> landmark_id
    next_landmark_id = 0

    # If no extrinsics are provided, use identity transforms
    if initial_extrinsics is None:
        initial_extrinsics = {}
        for cam_id in initial_intrinsics.keys():
            # Default extrinsics: camera at vehicle origin, facing forward
            initial_extrinsics[cam_id] = Extrinsics(rotation=np.eye(3), translation=np.zeros(3))

    # 1. Detect features in all images
    print("Detecting features in all images...")
    all_keypoints = {}  # (timestamp, camera_id) -> keypoints
    all_descriptors = {}  # (timestamp, camera_id) -> descriptors

    for img in image_data:
        # Detect features
        keypoints, descriptors = detect_features(img.image, detector_type='ORB', max_features=1000)

        # Store keypoints and descriptors
        all_keypoints[(img.timestamp, img.camera_id)] = keypoints
        all_descriptors[(img.timestamp, img.camera_id)] = descriptors

        # Create Feature objects
        img_features = [Feature(kp[0], kp[1], desc) for kp, desc in zip(keypoints, descriptors)]
        features[(img.timestamp, img.camera_id)] = img_features

    # 2. Select keyframes
    print("Selecting keyframes...")
    keyframe_indices = select_keyframes(image_data)
    keyframe_data = [image_data[i] for i in keyframe_indices]
    keyframe_poses = [initial_poses[i] for i in keyframe_indices]

    # 3. Match features between keyframes and triangulate landmarks
    print("Matching features and triangulating landmarks...")

    # For each pair of consecutive keyframes
    for i in range(len(keyframe_data) - 1):
        kf1 = keyframe_data[i]
        kf2 = keyframe_data[i + 1]
        pose1 = keyframe_poses[i]
        pose2 = keyframe_poses[i + 1]

        # For each camera
        for cam_id in initial_intrinsics.keys():
            # Skip if we don't have images from this camera at both keyframes
            if (kf1.timestamp, cam_id) not in all_descriptors or (kf2.timestamp, cam_id) not in all_descriptors:
                continue

            # Get descriptors for this camera at both keyframes
            desc1 = all_descriptors[(kf1.timestamp, cam_id)]
            desc2 = all_descriptors[(kf2.timestamp, cam_id)]

            # Match features
            matches = match_features(desc1, desc2)

            # Compute projection matrices
            P1 = compute_projection_matrix(initial_intrinsics[cam_id], initial_extrinsics[cam_id], pose1)
            P2 = compute_projection_matrix(initial_intrinsics[cam_id], initial_extrinsics[cam_id], pose2)

            # Triangulate matched points
            for match in matches:
                # Get keypoints
                kp1 = all_keypoints[(kf1.timestamp, cam_id)][match.idx1]
                kp2 = all_keypoints[(kf2.timestamp, cam_id)][match.idx2]

                # Triangulate
                point_3d = triangulate_point(kp1, kp2, P1, P2)

                # Create a new landmark
                observations = {
                    (kf1.timestamp, cam_id): match.idx1,
                    (kf2.timestamp, cam_id): match.idx2
                }
                landmarks[next_landmark_id] = Landmark(point_3d, observations)

                # Update feature to landmark mapping
                feature_to_landmark[(kf1.timestamp, cam_id, match.idx1)] = next_landmark_id
                feature_to_landmark[(kf2.timestamp, cam_id, match.idx2)] = next_landmark_id

                next_landmark_id += 1

    # 4. Optional: Local Bundle Adjustment to refine the initial structure
    # This would be implemented in a real system but is beyond the scope of this implementation

    print(f"Visual initialization complete. Found {len(landmarks)} landmarks and {sum(len(f) for f in features.values())} features.")
    return landmarks, features, initial_intrinsics  # Return initial intrinsics for now
