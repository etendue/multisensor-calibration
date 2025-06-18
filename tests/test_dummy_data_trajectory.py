import unittest
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add the project root to the Python path to allow importing scripts
# Assuming this test file is in tests/ and the scripts directory is in the parent directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts import test_trajectory_estimation

class TestDummyDataTrajectory(unittest.TestCase):

    def setUp(self):
        """Set up for the test."""
        self.original_argv = sys.argv.copy()
        self.original_cwd = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up after the test."""
        sys.argv = self.original_argv
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
        # Clean up any stray __pycache__ directories if they were created by the import
        if Path(project_root / "scripts" / "__pycache__").exists():
            shutil.rmtree(Path(project_root / "scripts" / "__pycache__"))
        if Path(project_root / "tests" / "__pycache__").exists():
            shutil.rmtree(Path(project_root / "tests" / "__pycache__"))


    def test_run_main_with_dummy_data(self):
        """Test that main() from test_trajectory_estimation.py runs without exceptions using dummy data."""

        # Modify sys.argv to ensure no specific bag_path is passed,
        # so the script uses the default "dummy" path.
        sys.argv = [sys.argv[0]] # Keep only the script name

        try:
            print(f"Current working directory for test: {os.getcwd()}")
            print(f"Contents before running main: {os.listdir('.')}")
            # Ensure the script uses "dummy" data by checking if the default path is 'dummy'
            # This was set in a previous subtask.
            # We can also temporarily modify the script's global 'bag_path' if needed,
            # but the current setup should default to "dummy"
            test_trajectory_estimation.main()
            print(f"Contents after running main: {os.listdir('.')}")
            # Check if expected output files were created
            self.assertTrue(Path("trajectory_plot.png").exists(), "trajectory_plot.png was not created.")
            self.assertTrue(Path("trajectory_animation.gif").exists(), "trajectory_animation.gif was not created.")
            self.assertTrue(Path("results/trajectory.csv").exists(), "results/trajectory.csv was not created.")
        except Exception as e:
            self.fail(f"test_trajectory_estimation.main() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
