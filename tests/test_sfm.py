#!/usr/bin/env python3
# Test script for SfM initialization
import argparse
import numpy as np
import os
import sys
import cv2
import time
import glob
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import modules
from src.data_structures import ImageData, VehiclePose, CameraIntrinsics, Extrinsics
from src.visual_processing.sfm_initializer import perform_visual_initialization

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test SfM Initialization')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file or directory of images')
    parser.add_argument('--output-dir', type=str, default='results/test_sfm', help='Path to output directory')
    parser.add_argument('--frame-step', type=int, default=5, help='Process every Nth frame')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum number of frames to process')
    parser.add_argument('--detector', type=str, default='ORB', choices=['ORB', 'SIFT', 'AKAZE'], help='Feature detector type')
    parser.add_argument('--synthetic', action='store_true', help='Input is a synthetic sequence with ground truth')
    return parser.parse_args()

def load_frames(input_path, frame_step=5, max_frames=100):
    """
    Load frames from a video file or directory of images.

    Args:
        input_path: Path to the video file or directory of images.
        frame_step: Process every Nth frame.
        max_frames: Maximum number of frames to load.

    Returns:
        List of loaded frames as numpy arrays.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    frames = []

    # Check if input is a directory
    if os.path.isdir(input_path):
        # Get all image files in the directory
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_path, f"*.JPG")))

        # Sort files by name
        image_files.sort()

        # Load images
        for i, file_path in enumerate(image_files):
            if i % frame_step == 0 and len(frames) < max_frames:
                frame = cv2.imread(file_path)
                if frame is not None:
                    frames.append(frame)
                    print(f"Loaded frame {i} from {file_path} (total: {len(frames)})")

        print(f"Loaded {len(frames)} frames from directory {input_path}")

    # Check if input is a video file
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        frame_count = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                frames.append(frame)
                print(f"Extracted frame {frame_count} from video (total: {len(frames)})")

            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames from video {input_path}")

    return frames

def create_test_dataset(frames, input_path=None, is_synthetic=False):
    """
    Create a test dataset from frames.

    Args:
        frames: List of video frames.
        input_path: Path to the input data (used for synthetic data).
        is_synthetic: Whether the input is synthetic data with ground truth.

    Returns:
        Tuple of (image_data, initial_poses, initial_intrinsics, initial_extrinsics)
    """
    # Create image data
    image_data = []
    for i, frame in enumerate(frames):
        # Use frame index as timestamp
        timestamp = float(i)
        # Use a single camera ID for all frames
        camera_id = "cam0"
        image_data.append(ImageData(timestamp, camera_id, frame))

    # For synthetic data, load ground truth camera parameters
    if is_synthetic and input_path is not None:
        try:
            # Load camera parameters
            K = np.load(os.path.join(input_path, 'K.npy'))
            Rs = np.load(os.path.join(input_path, 'Rs.npy'))
            ts = np.load(os.path.join(input_path, 'ts.npy'))

            # Create poses from ground truth
            initial_poses = []
            for i in range(min(len(frames), len(Rs))):
                # Convert from camera-to-world to world-to-camera
                R = Rs[i].T
                t = -Rs[i].T @ ts[i]
                initial_poses.append(VehiclePose(float(i), R, t))

            # Create intrinsics from ground truth
            initial_intrinsics = {
                "cam0": CameraIntrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            }

            print("Loaded ground truth camera parameters from synthetic data")
        except Exception as e:
            print(f"Error loading synthetic data parameters: {e}")
            print("Falling back to dummy parameters")
            is_synthetic = False

    # For non-synthetic data, create dummy parameters
    if not is_synthetic:
        # Create dummy poses (straight line motion)
        initial_poses = []
        for i in range(len(frames)):
            # Simple forward motion (5cm per frame)
            translation = np.array([0.05 * i, 0.0, 0.0])
            # No rotation
            rotation = np.eye(3)
            initial_poses.append(VehiclePose(float(i), rotation, translation))

        # Create dummy intrinsics (based on typical camera parameters)
        height, width = frames[0].shape[:2]
        fx = fy = max(width, height)  # Approximate focal length
        cx, cy = width / 2, height / 2  # Principal point at image center
        initial_intrinsics = {
            "cam0": CameraIntrinsics(fx, fy, cx, cy)
        }

    # Create dummy extrinsics (camera at vehicle origin)
    initial_extrinsics = {
        "cam0": Extrinsics(np.eye(3), np.zeros(3))
    }

    return image_data, initial_poses, initial_intrinsics, initial_extrinsics

def main():
    """Main test function."""
    # Parse arguments
    args = parse_arguments()

    print(f"--- Testing SfM Initialization with {args.detector} detector ---")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load frames from input
    frames = load_frames(args.input, args.frame_step, args.max_frames)

    if len(frames) < 2:
        print("Error: Need at least 2 frames for SfM")
        return

    # Create test dataset
    image_data, initial_poses, initial_intrinsics, initial_extrinsics = create_test_dataset(
        frames, args.input, args.synthetic
    )

    # Save some sample frames for reference
    for i in range(min(5, len(frames))):
        output_path = os.path.join(args.output_dir, f"frame_{i}.jpg")
        cv2.imwrite(output_path, frames[i])
        print(f"Saved sample frame to {output_path}")

    # Save camera parameters
    np.save(os.path.join(args.output_dir, 'initial_intrinsics.npy'),
            initial_intrinsics["cam0"].K)

    # Save poses separately to avoid broadcasting issues
    rotations = np.array([p.rotation for p in initial_poses])
    translations = np.array([p.translation for p in initial_poses])
    np.save(os.path.join(args.output_dir, 'initial_rotations.npy'), rotations)
    np.save(os.path.join(args.output_dir, 'initial_translations.npy'), translations)

    # Run SfM initialization
    start_time = time.time()
    landmarks, features, refined_intrinsics = perform_visual_initialization(
        image_data, initial_poses, initial_intrinsics, initial_extrinsics,
        output_dir=args.output_dir
    )
    end_time = time.time()

    # Print results
    print("\n--- SfM Initialization Results ---")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Number of landmarks: {len(landmarks)}")
    print(f"Number of features: {sum(len(f) for f in features.values())}")

    # Save a visualization of the 3D landmarks (top-down view)
    if len(landmarks) > 0:
        # Create a blank image for the top-down view
        top_down_size = 800
        top_down_view = np.ones((top_down_size, top_down_size, 3), dtype=np.uint8) * 255

        # Compute bounds of landmarks
        landmark_positions = np.array([lm.position for lm in landmarks.values()])
        min_x, min_y = landmark_positions[:, 0].min(), landmark_positions[:, 1].min()
        max_x, max_y = landmark_positions[:, 0].max(), landmark_positions[:, 1].max()

        # Add some margin
        margin = 0.1 * max(max_x - min_x, max_y - min_y)
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin

        # Scale factor to fit in the image
        scale_x = top_down_size / (max_x - min_x) if max_x > min_x else 1.0
        scale_y = top_down_size / (max_y - min_y) if max_y > min_y else 1.0
        scale = min(scale_x, scale_y)

        # Draw landmarks
        for lm in landmarks.values():
            x = int((lm.position[0] - min_x) * scale)
            y = int((lm.position[1] - min_y) * scale)
            # Ensure coordinates are within image bounds
            x = max(0, min(x, top_down_size - 1))
            y = max(0, min(y, top_down_size - 1))
            # Draw landmark as a small circle
            cv2.circle(top_down_view, (x, y), 2, (0, 0, 255), -1)

        # Draw camera trajectory
        for pose in initial_poses:
            x = int((pose.translation[0] - min_x) * scale)
            y = int((pose.translation[1] - min_y) * scale)
            # Ensure coordinates are within image bounds
            x = max(0, min(x, top_down_size - 1))
            y = max(0, min(y, top_down_size - 1))
            # Draw camera position as a small circle
            cv2.circle(top_down_view, (x, y), 3, (0, 255, 0), -1)

        # Connect camera positions with lines
        for i in range(1, len(initial_poses)):
            x1 = int((initial_poses[i-1].translation[0] - min_x) * scale)
            y1 = int((initial_poses[i-1].translation[1] - min_y) * scale)
            x2 = int((initial_poses[i].translation[0] - min_x) * scale)
            y2 = int((initial_poses[i].translation[1] - min_y) * scale)
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, top_down_size - 1))
            y1 = max(0, min(y1, top_down_size - 1))
            x2 = max(0, min(x2, top_down_size - 1))
            y2 = max(0, min(y2, top_down_size - 1))
            # Draw line
            cv2.line(top_down_view, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Add legend
        cv2.putText(top_down_view, "Landmarks", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(top_down_view, "Camera Trajectory", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save the visualization
        output_path = os.path.join(args.output_dir, "landmarks_top_down.jpg")
        cv2.imwrite(output_path, top_down_view)
        print(f"Saved top-down view of landmarks to {output_path}")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()
