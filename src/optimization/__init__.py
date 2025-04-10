# Optimization package initialization

# Import modules
from src.optimization.gtsam_utils import check_gtsam_availability
from src.optimization.factor_graph import build_factor_graph
from src.optimization.bundle_adjustment import run_bundle_adjustment, extract_calibration_results, calculate_reprojection_errors
from src.optimization.variables import VariableIndex, create_initial_values, extract_optimized_values

# Check if GTSAM is available
GTSAM_AVAILABLE = check_gtsam_availability()
