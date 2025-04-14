#!/usr/bin/env python3
"""
Script to test data loading and motion estimation using a test bag file.
Focuses on IMU and wheel odometry data only.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import required modules
from data_handling.data_loader import load_and_synchronize_data
from motion_estimation.ego_motion import estimate_initial_ego_motion
from data_structures import VehiclePose

def main():
    """Main function to test trajectory estimation."""
    # Path to the bag file
    bag_path = '/home/etendue/repo/cal/data/PL4106_event_dfdi_console_manual_event_20221225-055706_0.bag'

    print(f"Loading data from {bag_path}...")

    # Load data from the bag file (skip loading images to save time)
    images, imu_data, wheel_data = load_and_synchronize_data(bag_path, load_images=False)

    # Print summary of loaded data
    print(f"Loaded {len(imu_data)} IMU samples")
    print(f"Loaded {len(wheel_data)} wheel encoder samples")

    # Check if we have wheel angle data
    if hasattr(wheel_data[0], 'wheel_angles') and wheel_data[0].wheel_angles is not None:
        print(f"Wheel angle data is available")
        print(f"Sample wheel angles: {wheel_data[0].wheel_angles}")
    else:
        print(f"No wheel angle data available")

    # Print timestamp information
    imu_start_time = imu_data[0].timestamp
    imu_end_time = imu_data[-1].timestamp
    wheel_start_time = wheel_data[0].timestamp
    wheel_end_time = wheel_data[-1].timestamp

    print(f"\nTimestamp information:")
    print(f"IMU timestamps: {imu_start_time:.2f} to {imu_end_time:.2f} (duration: {imu_end_time - imu_start_time:.2f} seconds)")
    print(f"Wheel timestamps: {wheel_start_time:.2f} to {wheel_end_time:.2f} (duration: {wheel_end_time - wheel_start_time:.2f} seconds)")

    # Check for large gaps in timestamps
    imu_diffs = [imu_data[i+1].timestamp - imu_data[i].timestamp for i in range(len(imu_data)-1)]
    wheel_diffs = [wheel_data[i+1].timestamp - wheel_data[i].timestamp for i in range(len(wheel_data)-1)]

    max_imu_diff = max(imu_diffs)
    max_wheel_diff = max(wheel_diffs)

    print(f"Max gap between IMU samples: {max_imu_diff:.2f} seconds")
    print(f"Max gap between wheel samples: {max_wheel_diff:.2f} seconds")

    if max_imu_diff > 1.0 or max_wheel_diff > 1.0:
        print("WARNING: Large gaps detected in sensor data!")

        # Find the indices of the large gaps
        large_imu_gaps = [(i, imu_diffs[i]) for i in range(len(imu_diffs)) if imu_diffs[i] > 1.0]
        large_wheel_gaps = [(i, wheel_diffs[i]) for i in range(len(wheel_diffs)) if wheel_diffs[i] > 1.0]

        if large_imu_gaps:
            print(f"Large IMU gaps at indices: {large_imu_gaps[:5]}" + (" ..." if len(large_imu_gaps) > 5 else ""))

        if large_wheel_gaps:
            print(f"Large wheel gaps at indices: {large_wheel_gaps[:5]}" + (" ..." if len(large_wheel_gaps) > 5 else ""))

    # Set up vehicle parameters for motion estimation
    vehicle_params = {
        'model': 'ackermann',  # Use Ackermann steering model
        'wheel_radius': 0.3,   # Wheel radius in meters
        'track_width': 1.5,    # Distance between left and right wheels in meters
        'wheelbase': 2.7,      # Distance between front and rear axles in meters
        'using_positions': False,  # We're using wheel speeds, not positions
    }

    # Set up initial pose
    # Create a rotation matrix from identity quaternion [1, 0, 0, 0]
    from scipy.spatial.transform import Rotation
    initial_rotation = np.eye(3)  # Identity rotation matrix

    # Use the timestamp of the first IMU data point for the initial pose
    initial_timestamp = imu_data[0].timestamp
    initial_pose = VehiclePose(initial_timestamp, initial_rotation, np.zeros(3))

    print("Estimating trajectory...")

    # Estimate trajectory using wheel odometry and IMU data
    # Note: We're only using wheel odometry for this test
    poses = estimate_initial_ego_motion(
        imu_data=imu_data,
        wheel_data=wheel_data,
        initial_pose=initial_pose,
        axle_length=vehicle_params['track_width'],
        use_ekf=False  # Don't use EKF for this test, just wheel odometry
    )

    print(f"Estimated {len(poses)} poses")

    # Extract trajectory for visualization
    timestamps = [pose.timestamp for pose in poses]
    positions = np.array([pose.translation for pose in poses])

    # Normalize timestamps relative to the start time
    start_time = timestamps[0]
    normalized_timestamps = [t - start_time for t in timestamps]

    # Print trajectory statistics
    duration = normalized_timestamps[-1] - normalized_timestamps[0]
    print(f"Trajectory duration: {duration:.2f} seconds")

    # Calculate trajectory length
    traj_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    print(f"Trajectory length: {traj_length:.2f} meters")

    # Calculate average speed
    if duration > 0:
        avg_speed = traj_length / duration
        print(f"Average speed: {avg_speed:.2f} m/s, {avg_speed * 3.6:.2f} km/h")

    # Print position information
    print(f"Start position: {positions[0]}")
    print(f"End position: {positions[-1]}")
    print(f"Total displacement: {np.linalg.norm(positions[-1] - positions[0]):.2f} meters")

    # Print some statistics about the positions
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    z_range = positions[:, 2].max() - positions[:, 2].min()
    print(f"X range: {x_range:.2f} meters")
    print(f"Y range: {y_range:.2f} meters")
    print(f"Z range: {z_range:.2f} meters")

    # Visualize the trajectory
    visualize_trajectory(normalized_timestamps, positions)

    # Save the trajectory to a file
    save_trajectory(normalized_timestamps, positions, original_timestamps=timestamps)

def visualize_trajectory(timestamps, positions):
    """Visualize the estimated trajectory."""
    # Create a figure
    fig = plt.figure(figsize=(12, 8))

    # 2D plot (top-down view)
    ax1 = fig.add_subplot(121)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory (Top-Down View)')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()

    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax2.plot([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 'go', markersize=8, label='Start')
    ax2.plot([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 'ro', markersize=8, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Trajectory (3D View)')
    ax2.legend()

    # Set the same scale for all axes
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('trajectory_plot.png', dpi=300)
    print("Saved trajectory plot to trajectory_plot.png")

    # Create an animation of the trajectory
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111)

    # Plot the full trajectory as a reference
    ax_anim.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=1)
    ax_anim.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
    ax_anim.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
    ax_anim.set_xlabel('X (m)')
    ax_anim.set_ylabel('Y (m)')
    ax_anim.set_title('Trajectory Animation')
    ax_anim.grid(True)
    ax_anim.axis('equal')
    ax_anim.legend()

    # Initialize the point that will move along the trajectory
    point, = ax_anim.plot([], [], 'bo', markersize=10)

    # Set axis limits
    ax_anim.set_xlim(positions[:, 0].min() - 5, positions[:, 0].max() + 5)
    ax_anim.set_ylim(positions[:, 1].min() - 5, positions[:, 1].max() + 5)

    # Animation update function
    def update(frame):
        # Skip frames to make the animation faster
        frame = min(frame * 10, len(positions) - 1)
        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        return point,

    # Create the animation
    anim = FuncAnimation(fig_anim, update, frames=len(positions)//10, interval=50, blit=True)

    # Try to save the animation, but don't fail if it doesn't work
    try:
        anim.save('trajectory_animation.gif', writer='pillow', fps=20)
        print("Saved trajectory animation to trajectory_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")

    # Show the plots
    plt.show()

def save_trajectory(timestamps, positions, original_timestamps=None):
    """Save the trajectory to a CSV file.

    Args:
        timestamps: Normalized timestamps (relative to start time)
        positions: Array of positions
        original_timestamps: Original timestamps (Unix epoch time) if available
    """
    # Create a directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the trajectory to a CSV file
    with open('results/trajectory.csv', 'w') as f:
        if original_timestamps:
            f.write('normalized_timestamp,original_timestamp,x,y,z\n')
            for i in range(len(timestamps)):
                f.write(f"{timestamps[i]},{original_timestamps[i]},{positions[i, 0]},{positions[i, 1]},{positions[i, 2]}\n")
        else:
            f.write('timestamp,x,y,z\n')
            for i in range(len(timestamps)):
                f.write(f"{timestamps[i]},{positions[i, 0]},{positions[i, 1]},{positions[i, 2]}\n")

    print("Saved trajectory to results/trajectory.csv")

if __name__ == "__main__":
    main()
