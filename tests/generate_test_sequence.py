#!/usr/bin/env python3
# Script to generate a synthetic test sequence for SfM
import os
import sys
import argparse
import numpy as np
import cv2

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a synthetic test sequence for SfM')
    parser.add_argument('--output-dir', type=str, default='data/synthetic', help='Directory to save the sequence')
    parser.add_argument('--num-frames', type=int, default=20, help='Number of frames to generate')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--num-points', type=int, default=100, help='Number of 3D points to generate')
    return parser.parse_args()

def generate_random_points(num_points, bounds=10.0):
    """Generate random 3D points."""
    # Generate points in a cube centered at (0, 0, bounds)
    points = np.random.uniform(-bounds, bounds, (num_points, 3))
    # Ensure all points are in front of the camera (positive Z)
    points[:, 2] = np.abs(points[:, 2]) + bounds/2
    return points

def project_points(points, K, R, t):
    """Project 3D points to 2D using the camera matrix."""
    # Convert to homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Create projection matrix
    P = K @ np.hstack((R, t.reshape(3, 1)))

    # Project points
    points_2d_h = points_h @ P.T

    # Convert from homogeneous to Euclidean coordinates
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:]

    return points_2d

def generate_camera_trajectory(num_frames):
    """Generate a camera trajectory."""
    # Simple circular trajectory
    theta = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    radius = 5.0

    # Camera positions
    tx = radius * np.cos(theta)
    ty = radius * np.sin(theta)
    tz = np.zeros_like(theta) + 2.0  # Fixed height

    # Camera orientations (looking at the origin)
    Rs = []
    ts = []

    for i in range(num_frames):
        # Camera position
        t = np.array([tx[i], ty[i], tz[i]])

        # Camera orientation (looking at the origin)
        z = -t / np.linalg.norm(t)  # Forward direction (normalized)
        x = np.cross(np.array([0, 0, 1]), z)  # Right direction
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1, 0, 0])
        else:
            x = x / np.linalg.norm(x)
        y = np.cross(z, x)  # Up direction

        # Rotation matrix
        R = np.vstack((x, y, z)).T

        Rs.append(R)
        ts.append(t)

    return Rs, ts

def draw_points(image, points, color=(0, 0, 255), radius=3):
    """Draw points on an image."""
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        # Check if point is within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), radius, color, -1)
    return image

def main():
    """Main function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate random 3D points
    points_3d = generate_random_points(args.num_points)

    # Generate camera trajectory
    Rs, ts = generate_camera_trajectory(args.num_frames)

    # Camera intrinsic matrix
    fx = fy = max(args.width, args.height) * 0.8  # Focal length
    cx, cy = args.width / 2, args.height / 2  # Principal point
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Save camera parameters
    np.save(os.path.join(args.output_dir, 'K.npy'), K)
    np.save(os.path.join(args.output_dir, 'Rs.npy'), np.array(Rs))
    np.save(os.path.join(args.output_dir, 'ts.npy'), np.array(ts))
    np.save(os.path.join(args.output_dir, 'points_3d.npy'), points_3d)

    # Generate and save images
    for i in range(args.num_frames):
        # Create a blank image
        image = np.ones((args.height, args.width, 3), dtype=np.uint8) * 255

        # Project 3D points to 2D
        points_2d = project_points(points_3d, K, Rs[i], ts[i])

        # Draw points on the image
        image = draw_points(image, points_2d)

        # Add some noise and texture to make feature detection more realistic
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)

        # Add a grid pattern
        grid_size = 50
        for x in range(0, args.width, grid_size):
            cv2.line(image, (x, 0), (x, args.height), (200, 200, 200), 1)
        for y in range(0, args.height, grid_size):
            cv2.line(image, (0, y), (args.width, y), (200, 200, 200), 1)

        # Save the image
        output_path = os.path.join(args.output_dir, f'frame_{i:04d}.jpg')
        cv2.imwrite(output_path, image)
        print(f"Generated frame {i+1}/{args.num_frames}: {output_path}")

    # Create a metadata file
    with open(os.path.join(args.output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"Number of frames: {args.num_frames}\n")
        f.write(f"Image dimensions: {args.width}x{args.height}\n")
        f.write(f"Number of 3D points: {args.num_points}\n")
        f.write(f"Camera intrinsics:\n{K}\n")

    print("\nSynthetic sequence generation complete!")
    print(f"Generated {args.num_frames} frames in {args.output_dir}")
    print("\nYou can now test the SfM initialization with:")
    print(f"python -m tests.test_sfm --input {args.output_dir} --synthetic")

if __name__ == "__main__":
    main()
