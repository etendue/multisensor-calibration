# Validation metrics module
# Implements FR8: Validation - Tools to assess the quality of the calibration
from typing import Dict, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import data structures from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import CameraIntrinsics, Extrinsics, Landmark, Feature, VehiclePose

def calculate_reprojection_error(landmarks: Dict[int, Landmark],
                                features: Dict[Tuple[float, str], List[Feature]],
                                poses: List[VehiclePose],
                                intrinsics: Dict[str, CameraIntrinsics],
                                extrinsics: Dict[str, Extrinsics]) -> float:
    """
    Calculate the root mean square (RMS) reprojection error.

    Args:
        landmarks: Dictionary of 3D landmarks.
        features: Dictionary of 2D features.
        poses: List of vehicle poses.
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.

    Returns:
        RMS reprojection error in pixels.
    """
    squared_errors = []

    # For each landmark
    for landmark_id, landmark in landmarks.items():
        # For each observation of this landmark
        for (timestamp, camera_id), feature_idx in landmark.observations.items():
            # Find corresponding vehicle pose
            pose = next((p for p in poses if abs(p.timestamp - timestamp) < 1e-6), None)
            if not pose or camera_id not in intrinsics or camera_id not in extrinsics:
                continue

            # Get the corresponding feature
            feature_list = features.get((timestamp, camera_id), [])
            if feature_idx >= len(feature_list):
                continue

            feature = feature_list[feature_idx]
            camera_intrinsics = intrinsics[camera_id]
            camera_extrinsics = extrinsics[camera_id]

            # Transform landmark from world to camera frame
            # First transform from world to vehicle
            landmark_vehicle = pose.transform_point_inverse(landmark.position)
            # Then transform from vehicle to camera (using the camera extrinsics transform_point_inverse)
            point_h = np.ones(4)
            point_h[:3] = landmark_vehicle
            landmark_camera = camera_extrinsics.T @ point_h
            landmark_camera = landmark_camera[:3]  # Back to 3D coordinates

            # Project 3D point to image plane
            if landmark_camera[2] <= 0:  # Point behind camera
                continue

            # Perspective projection
            u = landmark_camera[0] / landmark_camera[2]
            v = landmark_camera[1] / landmark_camera[2]

            # Apply camera intrinsics
            pixel_x = camera_intrinsics.fx * u + camera_intrinsics.cx
            pixel_y = camera_intrinsics.fy * v + camera_intrinsics.cy

            # Calculate squared error between projected and observed points
            error_x = pixel_x - feature.uv[0]
            error_y = pixel_y - feature.uv[1]
            squared_error = error_x * error_x + error_y * error_y
            squared_errors.append(squared_error)

    if not squared_errors:
        return 0.0

    # Calculate RMS error
    mean_squared_error = np.mean(squared_errors)
    rms_error = np.sqrt(mean_squared_error)

    return rms_error

def visualize_results(landmarks: Dict[int, Landmark],
                     poses: List[VehiclePose],
                     intrinsics: Dict[str, CameraIntrinsics],
                     extrinsics: Dict[str, Extrinsics],
                     optimization_progress: Optional[Dict[str, List[float]]] = None,
                     max_landmarks: int = 1000,
                     show_coordinate_frames: bool = True,
                     coordinate_frame_interval: int = 10,
                     coordinate_frame_scale: float = 0.5) -> None:
    """
    Visualize the calibration results in an interactive 3D plot.

    Args:
        landmarks: Dictionary of 3D landmarks.
        poses: List of vehicle poses.
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.
        optimization_progress: Optional dictionary containing optimization metrics over iterations.
        max_landmarks: Maximum number of landmarks to display (for performance).
        show_coordinate_frames: Whether to show coordinate frames for poses and cameras.
        coordinate_frame_interval: Interval for displaying coordinate frames (e.g., every 10th pose).
        coordinate_frame_scale: Scale factor for coordinate frame axes.
    """
    # Create figure with subplots
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=('3D Visualization', 'Optimization Progress'),
                        column_widths=[0.7, 0.3])

    # Plot landmarks with sampling if needed
    if landmarks:
        landmark_positions = np.array([lm.position for lm in landmarks.values()])

        # Sample landmarks if there are too many
        if len(landmark_positions) > max_landmarks:
            # Random sampling
            indices = np.random.choice(len(landmark_positions), max_landmarks, replace=False)
            landmark_positions = landmark_positions[indices]
            print(f"Sampled {max_landmarks} landmarks from {len(landmarks)} total landmarks")

        # Add color based on height (z-coordinate) for better visualization
        z_min, z_max = np.min(landmark_positions[:, 2]), np.max(landmark_positions[:, 2])
        colors = (landmark_positions[:, 2] - z_min) / (z_max - z_min + 1e-10)  # Normalize to [0,1]

        fig.add_trace(
            go.Scatter3d(
                x=landmark_positions[:, 0],
                y=landmark_positions[:, 1],
                z=landmark_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,  # Use height-based coloring
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name=f'Landmarks ({len(landmark_positions)} points)'
            ),
            row=1, col=1
        )

    # Plot vehicle trajectory
    if poses:
        trajectory = np.array([p.translation for p in poses])
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=4),
                name='Vehicle Trajectory'
            ),
            row=1, col=1
        )

        # Plot coordinate frames for a subset of poses
        if show_coordinate_frames:
            for pose in poses[::coordinate_frame_interval]:
                # Get rotation matrix and position
                R = pose.rotation
                t = pose.translation

                # Define coordinate frame axes (scaled for visualization)
                scale = coordinate_frame_scale
                origin = t
                x_axis = t + scale * R[:, 0]
                y_axis = t + scale * R[:, 1]
                z_axis = t + scale * R[:, 2]

                # Plot coordinate frame
                # X-axis (red)
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], x_axis[0]],
                        y=[origin[1], x_axis[1]],
                        z=[origin[2], x_axis[2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                # Y-axis (green)
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], y_axis[0]],
                        y=[origin[1], y_axis[1]],
                        z=[origin[2], y_axis[2]],
                        mode='lines',
                        line=dict(color='green', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                # Z-axis (blue)
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], z_axis[0]],
                        y=[origin[1], z_axis[1]],
                        z=[origin[2], z_axis[2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )

    # Plot camera positions and orientations
    if poses:  # Only plot cameras if we have at least one vehicle pose
        vehicle_pose = poses[0]  # Use first vehicle pose as reference
        for camera_id, extr in extrinsics.items():
            if extr is None:
                print(f"Warning: Skipping visualization of camera {camera_id} - no extrinsics available")
                continue

            # Transform camera position to world frame
            camera_pos_vehicle = extr.translation
            camera_pos_world = vehicle_pose.transform_point(camera_pos_vehicle)

            # Plot camera position
            fig.add_trace(
                go.Scatter3d(
                    x=[camera_pos_world[0]],
                    y=[camera_pos_world[1]],
                    z=[camera_pos_world[2]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='square'
                    ),
                    name=f'Camera {camera_id}'
                ),
                row=1, col=1
            )

            # Plot camera orientation if coordinate frames are enabled
            if show_coordinate_frames:
                scale = coordinate_frame_scale * 0.6  # Slightly smaller for cameras
                R_world = vehicle_pose.rotation @ extr.rotation

                # Define coordinate frame axes
                origin = camera_pos_world
                x_axis = origin + scale * R_world[:, 0]
                y_axis = origin + scale * R_world[:, 1]
                z_axis = origin + scale * R_world[:, 2]

                # Plot camera coordinate frame
                # X-axis (red)
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], x_axis[0]],
                        y=[origin[1], x_axis[1]],
                        z=[origin[2], x_axis[2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                # Y-axis (green)
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], y_axis[0]],
                        y=[origin[1], y_axis[1]],
                        z=[origin[2], y_axis[2]],
                        mode='lines',
                        line=dict(color='green', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                # Z-axis (blue)
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], z_axis[0]],
                        y=[origin[1], z_axis[1]],
                        z=[origin[2], z_axis[2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )

    # Plot optimization progress if available
    if optimization_progress and all(len(optimization_progress[key]) > 0 for key in ['iterations', 'total_errors', 'reprojection_errors']):
        # Plot total error
        fig.add_trace(
            go.Scatter(
                x=optimization_progress['iterations'],
                y=optimization_progress['total_errors'],
                mode='lines+markers',
                name='Total Error',
                line=dict(color='red')
            ),
            row=1, col=2
        )

        # Plot reprojection error
        fig.add_trace(
            go.Scatter(
                x=optimization_progress['iterations'],
                y=optimization_progress['reprojection_errors'],
                mode='lines+markers',
                name='Reprojection Error (RMS)',
                line=dict(color='blue')
            ),
            row=1, col=2
        )

        # Add optimization time if available
        if 'times' in optimization_progress and len(optimization_progress['times']) > 0:
            # Add secondary y-axis for time
            fig.add_trace(
                go.Scatter(
                    x=optimization_progress['iterations'],
                    y=optimization_progress['times'],
                    mode='lines',
                    name='Optimization Time (s)',
                    line=dict(color='green', dash='dash'),
                    yaxis='y2'
                ),
                row=1, col=2
            )

            # Configure secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Time (seconds)",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )

        # Update y-axis to log scale for better visualization of error reduction
        fig.update_yaxes(type="log", title="Error (log scale)", row=1, col=2)
        fig.update_xaxes(title="Iteration", row=1, col=2)

    # Add buttons for interactive controls
    updatemenus = [
        # Button to toggle coordinate frames
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    args=[{"visible": [True if "lines" not in trace.mode else False
                                     for trace in fig.data]}],
                    label="Hide Coordinate Frames",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True for _ in fig.data]}],
                    label="Show All",
                    method="update"
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
        # Button to reset view
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    args=[{"scene.camera.eye": dict(x=-1.8, y=-1.8, z=1.8)}],
                    label="Reset View",
                    method="relayout"
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.3,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ]

    # Update 3D layout with interactive controls
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # This will maintain the aspect ratio
            camera=dict(
                eye=dict(x=-1.8, y=-1.8, z=1.8)  # Default camera position
            )
        ),
        updatemenus=updatemenus,
        showlegend=True,
        title='Calibration Results Visualization',
        # Improve the layout
        height=800,
        width=1200,
        margin=dict(l=0, r=0, t=50, b=0)  # Increased top margin for buttons
    )

    # Add annotations for controls help
    fig.add_annotation(
        text="Drag to rotate, scroll to zoom, right-click to pan",
        xref="paper", yref="paper",
        x=0.5, y=0.01,
        showarrow=False,
        font=dict(size=12)
    )

    # Show figure
    fig.show()

def print_calibration_report(intrinsics: Dict[str, CameraIntrinsics],
                            extrinsics: Dict[str, Extrinsics],
                            reprojection_error: float) -> None:
    """
    Print a report of the calibration results.

    Args:
        intrinsics: Dictionary of camera intrinsics.
        extrinsics: Dictionary of camera extrinsics.
        reprojection_error: RMS reprojection error.
    """
    print("\n--- Calibration Results Report ---")
    print(f"RMS Reprojection Error: {reprojection_error:.3f} pixels")

    print("\nCamera Intrinsics:")
    for cam_id, intr in intrinsics.items():
        print(f"  {cam_id}:")
        print(f"    fx: {intr.fx:.2f}, fy: {intr.fy:.2f}")
        print(f"    cx: {intr.cx:.2f}, cy: {intr.cy:.2f}")
        if np.any(intr.distortion_coeffs):
            print(f"    Distortion: {intr.distortion_coeffs}")

    print("\nSensor Extrinsics (relative to vehicle frame):")
    for sensor_id, extr in extrinsics.items():
        print(f"  {sensor_id}:")
        print(f"    Translation: [{extr.translation[0]:.3f}, {extr.translation[1]:.3f}, {extr.translation[2]:.3f}]")
        # Rotation could be printed as Euler angles for readability
        # This is a placeholder
        print(f"    Rotation Matrix:\n{np.round(extr.rotation, 3)}")

    print("\n--- End of Report ---")
