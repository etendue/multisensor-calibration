#!/usr/bin/env python3
# Test script for the platform-independent rosbag reader

import argparse
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import modules
from data_handling.data_loader import load_and_synchronize_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Rosbag Reader')
    parser.add_argument('--input', type=str, required=True, help='Path to input rosbag file or directory')
    return parser.parse_args()

def main():
    """Main function to test rosbag reading."""
    args = parse_arguments()
    
    print(f"Testing rosbag reader with input: {args.input}")
    
    # Load data from the rosbag
    images, imu_data, wheel_data = load_and_synchronize_data(args.input)
    
    # Print summary of loaded data
    print("\nData Summary:")
    print(f"  Images: {len(images)} frames")
    if images:
        print(f"    First timestamp: {images[0].timestamp:.3f}s")
        print(f"    Last timestamp: {images[-1].timestamp:.3f}s")
        print(f"    Camera IDs: {set(img.camera_id for img in images)}")
    
    print(f"  IMU data: {len(imu_data)} samples")
    if imu_data:
        print(f"    First timestamp: {imu_data[0].timestamp:.3f}s")
        print(f"    Last timestamp: {imu_data[-1].timestamp:.3f}s")
        print(f"    Sample angular velocity: {imu_data[0].angular_velocity}")
        print(f"    Sample linear acceleration: {imu_data[0].linear_acceleration}")
    
    print(f"  Wheel encoder data: {len(wheel_data)} samples")
    if wheel_data:
        print(f"    First timestamp: {wheel_data[0].timestamp:.3f}s")
        print(f"    Last timestamp: {wheel_data[-1].timestamp:.3f}s")
        print(f"    Sample wheel speeds: {wheel_data[0].wheel_speeds}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
