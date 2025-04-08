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
from optimization.bundle_adjustment import run_bundle_adjustment, extract_calibration_results
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
    
    # 3. Visual Initialization (SfM)
    landmarks, features, current_intrinsics = perform_visual_initialization(
        images, initial_trajectory, initial_intrinsics_guess
    )
    
    # 4. Build Factor Graph
    factor_graph = build_factor_graph(
        initial_trajectory, landmarks, features, current_intrinsics,
        initial_extrinsics_guess, imu, wheels
    )
    
    # Prepare initial values dictionary for the optimizer
    # This is a simplified version - in reality, this would be more complex
    initial_values_for_opt = {
        f"Pose_{i}": pose for i, pose in enumerate(initial_trajectory)
    }
    # Add other variables to initial values
    initial_values_for_opt.update({
        f"Intrinsics_{cam_id}": intr for cam_id, intr in current_intrinsics.items()
    })
    initial_values_for_opt.update({
        f"Extrinsics_{cam_id}": extr for cam_id, extr in initial_extrinsics_guess.items()
    })
    
    # 5. Run Bundle Adjustment
    # In a real implementation, this would use a specific optimization library
    # For now, we'll use our placeholder that returns the initial values
    print("\n--- Skipping Optimization Step (Requires specific library) ---")
    # optimized_values = run_bundle_adjustment(factor_graph, initial_values_for_opt)
    
    # Simulate having results for the next step
    simulated_optimized_values = {
        f"Intrinsics_{cam_id}": intr for cam_id, intr in initial_intrinsics_guess.items()
    }
    simulated_optimized_values.update({
        f"Extrinsics_{cam_id}": extr for cam_id, extr in initial_extrinsics_guess.items()
    })
    
    # 6. Extract Results
    final_intrinsics, final_extrinsics, final_biases = extract_calibration_results(simulated_optimized_values)
    
    # 7. Validate Results
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
