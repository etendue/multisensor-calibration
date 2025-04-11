# Bundle adjustment optimization module
from typing import Dict, Any, Tuple, List, Optional
import time
import numpy as np

# Import data structures
from src.data_structures import CameraIntrinsics, Extrinsics

# Import GTSAM utilities
from src.optimization.gtsam_utils import check_gtsam_availability
from src.optimization.variables import extract_optimized_values

# Try to import GTSAM
try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("Warning: GTSAM not available. Install with 'pip install gtsam' for optimization.")

def run_bundle_adjustment(factor_graph: Any, initial_values: Any, variable_index: Any = None, config: Dict = None) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Runs the non-linear least-squares optimization to solve the factor graph.

    Args:
        factor_graph: The constructed factor graph (GTSAM NonlinearFactorGraph).
        initial_values: Initial estimates for all variables (GTSAM Values).
        variable_index: Variable index for mapping between our data structures and GTSAM variables.
        config: Configuration parameters for the optimization.

    Returns:
        Tuple containing:
        - GTSAM Values object containing the optimized values
        - Dictionary with optimization metrics (errors, times) over iterations
    """
    print("Running Bundle Adjustment (Non-linear Optimization)...")

    # Initialize progress tracking
    progress = {
        'total_errors': [],
        'reprojection_errors': [],
        'times': [],
        'iterations': []
    }
    start_time = time.time()

    # Check if GTSAM is available
    if not check_gtsam_availability():
        print("GTSAM is not available. Using placeholder implementation.")
        # Placeholder: Return dummy optimized values
        time.sleep(5.0)  # Simulate heavy computation
        print("Optimization finished.")
        return initial_values, progress

    # Initialize configuration with defaults if not provided
    if config is None:
        config = {}

    # Get optimization parameters from config
    max_iterations = config.get('max_iterations', 100)
    convergence_delta = config.get('convergence_delta', 1e-6)
    verbose = config.get('verbose', True)

    # Create optimizer
    print(f"Creating Levenberg-Marquardt optimizer with {max_iterations} max iterations...")

    # Set parameters - in some GTSAM versions, we need to set parameters before creating the optimizer
    try:
        # Try the newer API if available
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        params.setRelativeErrorTol(convergence_delta)
        params.setVerbosityLM("SUMMARY" if verbose else "NONE")
        optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values, params)
    except (AttributeError, TypeError):
        # Fall back to older API
        try:
            # Some versions require creating the optimizer first, then setting parameters
            optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values)
            # Try to set parameters if the method exists
            if hasattr(optimizer, 'setMaxIterations'):
                optimizer.setMaxIterations(max_iterations)
            if hasattr(optimizer, 'setRelativeErrorTol'):
                optimizer.setRelativeErrorTol(convergence_delta)
        except Exception as e:
            print(f"Warning: Could not set optimizer parameters: {e}")
            # Create basic optimizer without parameters
            optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values)

    # Run optimization
    print("Starting optimization...")
    
    # Record initial errors
    current_values = initial_values
    total_error = factor_graph.error(current_values)
    reproj_errors = calculate_reprojection_errors(factor_graph, current_values)
    
    progress['total_errors'].append(total_error)
    progress['reprojection_errors'].append(reproj_errors['rms'])
    progress['times'].append(0.0)
    progress['iterations'].append(0)

    # Optimization loop with progress tracking
    try:
        for iteration in range(max_iterations):
            # Perform one optimization step
            current_values = optimizer.iterate()
            
            # Record progress
            current_time = time.time() - start_time
            total_error = factor_graph.error(current_values)
            reproj_errors = calculate_reprojection_errors(factor_graph, current_values)
            
            progress['total_errors'].append(total_error)
            progress['reprojection_errors'].append(reproj_errors['rms'])
            progress['times'].append(current_time)
            progress['iterations'].append(iteration + 1)
            
            # Check for convergence
            if optimizer.lambda_() < convergence_delta:
                break
    except Exception as e:
        print(f"Warning: Error during optimization iteration: {e}")
        # Fall back to basic optimize() call
        current_values = optimizer.optimize()

    end_time = time.time()

    # Print optimization statistics
    error_before = progress['total_errors'][0]
    error_after = progress['total_errors'][-1]
    print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    print(f"Initial error: {error_before:.6f}, Final error: {error_after:.6f}")
    print(f"Error reduction: {(1.0 - error_after / error_before) * 100:.2f}%")

    return current_values, progress

def extract_calibration_results(optimized_values: Any, variable_index: Any, camera_ids: List[str]) -> Tuple[Dict[str, CameraIntrinsics], Dict[str, Extrinsics], Any]:
    """
    Extracts the final calibration parameters from the optimized results.

    Args:
        optimized_values: GTSAM Values object with optimization results.
        variable_index: Variable index used for mapping.
        camera_ids: List of camera IDs.

    Returns:
        A tuple containing:
        - Optimized camera intrinsics.
        - Optimized sensor extrinsics (cameras, potentially IMU).
        - Optimized IMU biases (if estimated).
    """
    print("Extracting calibration results...")

    # Check if GTSAM is available
    if not check_gtsam_availability():
        print("GTSAM is not available. Using placeholder implementation.")
        # Placeholder: Return dummy results
        optimized_intrinsics = {f"Intrinsics_{cam_id}": None for cam_id in camera_ids}
        optimized_extrinsics = {f"Extrinsics_{cam_id}": None for cam_id in camera_ids}
        optimized_biases = "Placeholder Biases"
        return optimized_intrinsics, optimized_extrinsics, optimized_biases

    # Extract optimized values using the variable index
    optimized_intrinsics, optimized_extrinsics, optimized_biases = extract_optimized_values(
        optimized_values, variable_index, camera_ids
    )

    print("Calibration results extracted.")
    return optimized_intrinsics, optimized_extrinsics, optimized_biases

def calculate_reprojection_errors(factor_graph: Any, optimized_values: Any) -> Dict[str, float]:
    """
    Calculate reprojection errors from the optimized factor graph.

    Args:
        factor_graph: GTSAM factor graph.
        optimized_values: GTSAM Values object with optimization results.

    Returns:
        Dictionary with error statistics (mean, median, max, etc.).
    """
    if not check_gtsam_availability():
        print("GTSAM is not available. Cannot calculate reprojection errors.")
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}

    # Get all factors from the graph
    errors = []
    for i in range(factor_graph.size()):
        factor = factor_graph.at(i)
        # Check if it's a projection factor (reprojection error)
        if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2) or \
           isinstance(factor, gtsam.GenericProjectionFactorCal3DS2):
            # Calculate error for this factor
            error_vector = factor.unwhitenedError(optimized_values)
            # Compute Euclidean distance (reprojection error in pixels)
            error = np.sqrt(np.sum(np.square(error_vector)))
            errors.append(error)

    if not errors:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "count": 0}

    # Calculate statistics
    errors = np.array(errors)
    stats = {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "min": float(np.min(errors)),
        "std": float(np.std(errors)),
        "rms": float(np.sqrt(np.mean(np.square(errors)))),
        "count": len(errors)
    }

    return stats
