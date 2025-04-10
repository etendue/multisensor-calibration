#!/usr/bin/env python3
# Test script for GTSAM installation

import sys
import platform
import os
import numpy as np

def print_system_info():
    print("\nSystem Information:")
    print(f"  Python version: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.architecture()[0]}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Python path: {sys.executable}")
    print(f"  Working directory: {os.getcwd()}")

try:
    import gtsam
    print(f"\nGTSAM is installed successfully!")

    # Print path information
    print(f"  GTSAM path: {gtsam.__file__}")
    # Try to get version information if available
    try:
        print(f"  GTSAM version: {gtsam.__version__}")
    except AttributeError:
        print("  GTSAM version: Not available (no __version__ attribute)")

    # Create a simple factor graph
    print("\nTesting GTSAM functionality:")
    print("  Creating a factor graph...")
    graph = gtsam.NonlinearFactorGraph()

    # Create a simple pose
    print("  Creating a pose...")
    pose1 = gtsam.Pose3()

    # Create a simple point
    print("  Creating a point...")
    point1 = gtsam.Point3(1, 2, 3)

    # Create a simple rotation
    print("  Creating a rotation...")
    rot1 = gtsam.Rot3.Rx(0.1)

    # Create a simple calibration
    print("  Creating a camera calibration...")
    cal1 = gtsam.Cal3_S2(500.0, 500.0, 0.0, 320.0, 240.0)

    # Create a simple noise model
    print("  Creating a noise model...")
    noise1 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))

    print("\nAll GTSAM functionality tests passed!")
    print_system_info()
    sys.exit(0)
except ImportError:
    print("\nGTSAM is not installed. Please run scripts/install_gtsam.sh to install it.")
    print_system_info()
    sys.exit(1)
except Exception as e:
    print(f"\nError testing GTSAM: {e}")
    print_system_info()
    sys.exit(1)
