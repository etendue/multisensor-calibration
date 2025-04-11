#!/usr/bin/env python3
# Main calibration script
import argparse
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import modules
from data_structures import VehiclePose, CameraIntrinsics, Extrinsics
from data_handling.data_loader import load_and_synchronize_data
from data_handling.config_parser import load_config, parse_intrinsics, parse_extrinsics, get_vehicle_parameters
from motion_estimation.ego_motion import estimate_initial_ego_motion
from visual_processing.sfm_initializer import perform_visual_initialization
from optimization.factor_graph import build_factor_graph
from optimization.bundle_adjustment import run_bundle_adjustment, extract_calibration_results, calculate_reprojection_errors
from optimization.gtsam_utils import check_gtsam_availability
from validation.metrics import calculate_reprojection_error, visualize_results, print_calibration_report

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multisensor Calibration Tool')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Path to output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    return parser.parse_args()

def main():
    """Main calibration pipeline."""
    # Parse arguments
    args = parse_arguments()

    print("--- Starting Multisensor Calibration Pipeline ---")

    # Load configuration
    config = load_config(args.config)
    initial_intrinsics_guess = parse_intrinsics(config)
    initial_extrinsics_guess = parse_extrinsics(config)
    vehicle_params = get_vehicle_parameters(config)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initial vehicle pose (at the start of the dataset)
    rear_axle_center_pose = VehiclePose(0.0, np.eye(3), np.zeros(3))  # Start at origin

    # 1. Load Data
    images, imu, wheels = load_and_synchronize_data(args.data_dir)

    # 2. Initial Ego-Motion
    initial_trajectory = estimate_initial_ego_motion(
        imu, wheels, rear_axle_center_pose, vehicle_params.get('axle_length', 1.6)
    )

    # Create output directory for SfM results
    sfm_output_dir = os.path.join(args.output_dir, "sfm")
    os.makedirs(sfm_output_dir, exist_ok=True)

    # 3. Visual Initialization (SfM)
    landmarks, features, current_intrinsics = perform_visual_initialization(
        images, initial_trajectory, initial_intrinsics_guess, initial_extrinsics_guess,
        output_dir=sfm_output_dir
    )

    # 4. Build Factor Graph
    # Get optimization parameters from config
    optimization_config = config.get('optimization', {})

    # Build the factor graph
    factor_graph, initial_values, variable_index = build_factor_graph(
        initial_trajectory, landmarks, features, current_intrinsics,
        initial_extrinsics_guess, imu, wheels, config=optimization_config
    )

    # 5. Run Bundle Adjustment
    # Check if GTSAM is available
    if check_gtsam_availability():
        print("\n--- Running Bundle Adjustment Optimization ---")
        optimized_values = run_bundle_adjustment(
            factor_graph, initial_values, variable_index, config=optimization_config
        )
    else:
        print("\n--- Skipping Optimization Step (GTSAM not available) ---")
        print("Install GTSAM with 'conda install -c conda-forge gtsam' for full optimization.")
        # Use initial values as a fallback
        optimized_values = initial_values

    # 6. Extract Results
    # Get camera IDs from the intrinsics dictionary
    camera_ids = list(current_intrinsics.keys())
    final_intrinsics, final_extrinsics, final_biases = extract_calibration_results(
        optimized_values, variable_index, camera_ids
    )

    # 7. Validate Results
    # Calculate reprojection errors using the factor graph if GTSAM is available
    if check_gtsam_availability():
        error_stats = calculate_reprojection_errors(factor_graph, optimized_values)
        reprojection_error = error_stats.get('rms', 0.0)
    else:
        # Fall back to the simplified implementation
        reprojection_error = calculate_reprojection_error(
            landmarks, features, initial_trajectory, final_intrinsics, final_extrinsics
        )

    # Print calibration report
    print_calibration_report(final_intrinsics, final_extrinsics, reprojection_error)

    # Visualize results if requested
    if args.visualize:
        visualize_results(landmarks, initial_trajectory, final_intrinsics, final_extrinsics)

    print("\n--- Calibration Pipeline Complete ---")

if __name__ == "__main__":
    main()
