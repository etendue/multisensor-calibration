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

# Import OpenCV for feature detection and matching
import cv2
import matplotlib.pyplot as plt
import os


def detect_features(image: np.ndarray, detector_type: str = 'ORB', max_features: int = 1000) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Detect features in an image using the specified detector.

    Args:
        image: Input image as a numpy array.
        detector_type: Type of feature detector ('ORB', 'SIFT', etc.)
        max_features: Maximum number of features to detect.

    Returns:
        Tuple of (keypoints as numpy arrays [x, y], descriptors)
        If no features are detected, returns (empty array, None)
    """
    # Convert image to grayscale if it's color
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply some preprocessing to improve feature detection
    # Normalize image histogram
    gray = cv2.equalizeHist(gray)

    # Create feature detector
    if detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_features)
    elif detector_type == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif detector_type == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    # Check if any keypoints were detected
    if keypoints is None or len(keypoints) == 0:
        print(f"Warning: No features detected in image")
        return np.array([]), None

    # Convert keypoints to numpy array of coordinates
    keypoints_np = np.array([kp.pt for kp in keypoints])

    return keypoints_np, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, detector_type: str = 'ORB', ratio_threshold: float = 0.8) -> List[Match]:
    """
    Match features between two sets of descriptors using a ratio test.

    Args:
        desc1: First set of descriptors.
        desc2: Second set of descriptors.
        detector_type: Type of detector used ('ORB', 'SIFT', etc.) to determine matching method.
        ratio_threshold: Threshold for Lowe's ratio test.

    Returns:
        List of Match objects representing the matches.
    """
    # Choose the appropriate norm based on the detector type
    if detector_type == 'ORB' or detector_type == 'AKAZE':
        norm_type = cv2.NORM_HAMMING
    else:  # SIFT, SURF
        norm_type = cv2.NORM_L2

    # Create matcher
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    # Check if descriptors are valid
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    # Match descriptors using kNN
    try:
        matches_cv = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        # If kNN matching fails, try simple matching
        simple_matches = matcher.match(desc1, desc2)
        return [Match(m.queryIdx, m.trainIdx) for m in simple_matches]

    # Apply ratio test
    good_matches = []
    for match_pair in matches_cv:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(Match(m.queryIdx, m.trainIdx))

    return good_matches


def select_keyframes(image_data: List[ImageData], initial_poses: List[VehiclePose], min_distance: float = 0.5, min_rotation: float = 0.1) -> List[int]:
    """
    Select keyframes from the image data based on distance traveled and rotation.

    Args:
        image_data: List of image data.
        initial_poses: List of vehicle poses corresponding to the image data.
        min_distance: Minimum distance between keyframes (meters).
        min_rotation: Minimum rotation between keyframes (radians).

    Returns:
        List of indices of selected keyframes.
    """
    if len(image_data) == 0:
        return []

    # Always include the first frame
    keyframes = [0]
    last_kf_pose = initial_poses[0]

    # Check each subsequent frame
    for i in range(1, len(initial_poses)):
        current_pose = initial_poses[i]

        # Calculate distance from last keyframe
        distance = np.linalg.norm(current_pose.translation - last_kf_pose.translation)

        # Calculate rotation difference (simplified - just using Frobenius norm)
        rotation_diff = np.linalg.norm(current_pose.rotation - last_kf_pose.rotation, 'fro')

        # If either distance or rotation exceeds threshold, add as keyframe
        if distance > min_distance or rotation_diff > min_rotation:
            keyframes.append(i)
            last_kf_pose = current_pose

    # Ensure we have at least 2 keyframes for triangulation
    if len(keyframes) < 2 and len(image_data) > 1:
        keyframes = [0, len(image_data) - 1]  # First and last frame

    return keyframes


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


def visualize_matches(img1: np.ndarray, img2: np.ndarray, kp1: np.ndarray, kp2: np.ndarray, matches: List[Match], output_dir: str, name: str):
    """
    Visualize feature matches between two images and save the visualization.

    Args:
        img1: First image.
        img2: Second image.
        kp1: Keypoints in the first image.
        kp2: Keypoints in the second image.
        matches: List of matches between the keypoints.
        output_dir: Directory to save the visualization.
        name: Name for the output file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert keypoints to OpenCV format
    cv_kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) for pt in kp1]
    cv_kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) for pt in kp2]

    # Convert matches to OpenCV format
    cv_matches = [cv2.DMatch(match.idx1, match.idx2, 0) for match in matches]

    # Draw matches
    match_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, cv_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the visualization
    output_path = os.path.join(output_dir, f"{name}.jpg")
    cv2.imwrite(output_path, match_img)
    print(f"Saved match visualization to {output_path}")


def perform_visual_initialization(image_data: List[ImageData], initial_poses: List[VehiclePose], initial_intrinsics: Dict[str, CameraIntrinsics], initial_extrinsics: Optional[Dict[str, Extrinsics]] = None, output_dir: str = "results/sfm") -> Tuple[Dict[int, Landmark], Dict[Tuple[float, str], List[Feature]], Dict[str, CameraIntrinsics]]:
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

        # Create Feature objects if descriptors were found
        if descriptors is not None and len(keypoints) > 0:
            img_features = [Feature(kp[0], kp[1], desc) for kp, desc in zip(keypoints, descriptors)]
            features[(img.timestamp, img.camera_id)] = img_features
            print(f"Detected {len(img_features)} features in image at t={img.timestamp:.2f} for camera {img.camera_id}")
        else:
            # Create empty feature list
            features[(img.timestamp, img.camera_id)] = []
            print(f"No features detected in image at t={img.timestamp:.2f} for camera {img.camera_id}")

    # 2. Select keyframes
    print("Selecting keyframes...")
    keyframe_indices = select_keyframes(image_data, initial_poses)
    keyframe_data = [image_data[i] for i in keyframe_indices]
    keyframe_poses = [initial_poses[i] for i in keyframe_indices]
    print(f"Selected {len(keyframe_indices)} keyframes out of {len(image_data)} frames")

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

            # Skip if we don't have features or descriptors
            if all_descriptors[(kf1.timestamp, cam_id)] is None or all_descriptors[(kf2.timestamp, cam_id)] is None:
                print(f"Skipping keyframe pair ({kf1.timestamp:.2f}, {kf2.timestamp:.2f}) for camera {cam_id} due to missing descriptors")
                continue

            # Skip if we don't have enough keypoints
            if len(all_keypoints[(kf1.timestamp, cam_id)]) == 0 or len(all_keypoints[(kf2.timestamp, cam_id)]) == 0:
                print(f"Skipping keyframe pair ({kf1.timestamp:.2f}, {kf2.timestamp:.2f}) for camera {cam_id} due to insufficient keypoints")
                continue

            # Get descriptors for this camera at both keyframes
            desc1 = all_descriptors[(kf1.timestamp, cam_id)]
            desc2 = all_descriptors[(kf2.timestamp, cam_id)]

            # Match features
            detector_type = 'ORB'  # Use the same detector type as in detect_features
            matches = match_features(desc1, desc2, detector_type)
            print(f"Found {len(matches)} matches between keyframes at t={kf1.timestamp:.2f} and t={kf2.timestamp:.2f} for camera {cam_id}")

            # Visualize matches if we have at least some
            if len(matches) > 0:
                visualize_matches(kf1.image, kf2.image,
                                all_keypoints[(kf1.timestamp, cam_id)],
                                all_keypoints[(kf2.timestamp, cam_id)],
                                matches, output_dir, f"matches_{cam_id}_{i}_{i+1}")

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
