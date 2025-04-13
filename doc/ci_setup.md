# Continuous Integration (CI) Setup

This document describes the CI setup for the Multisensor Calibration project using GitHub Actions.

> **Note:** The current CI configuration is set to be lenient with code style and linting issues. It will report issues but not fail the build for most code quality checks. This approach allows for gradual improvement of code quality without blocking development.

## Overview

The CI system automatically runs tests and code quality checks whenever code is pushed to the repository or a pull request is created. This helps ensure that the codebase remains stable and maintains high quality standards.

## Workflows

### Main CI Workflow (`.github/workflows/main.yml`)

This workflow runs the test suite and reports code coverage:

- **Trigger**: Runs on push to `main` branch and on pull requests to `main`
- **Python Versions**: Tests on Python 3.8, 3.9, and 3.10
- **Steps**:
  1. Set up Python environment
  2. Set up Conda for GTSAM installation
  3. Install dependencies from `requirements.txt`
  4. Install GTSAM using conda
  5. Run basic linting with flake8
  6. Run tests with pytest and generate coverage report
  7. Upload coverage report to Codecov (if configured)

### Code Quality Workflow (`.github/workflows/code-quality.yml`)

This workflow checks code formatting and quality:

- **Trigger**: Runs on push to `main` branch and on pull requests to `main`
- **Steps**:
  1. Set up Python environment
  2. Install code quality tools (black, isort, mypy, ruff)
  3. Check code formatting with black
  4. Check import sorting with isort
  5. Run static analysis with ruff
  6. Run type checking with mypy

### Documentation Workflow (`.github/workflows/docs.yml`)

This workflow builds and potentially deploys documentation:

- **Trigger**: Runs on push to `main` branch when documentation files are changed
- **Steps**:
  1. Set up Python environment
  2. Install documentation tools (sphinx, sphinx_rtd_theme, myst-parser)
  3. (Commented out) Build documentation with Sphinx
  4. (Commented out) Deploy documentation to GitHub Pages

## Configuration Files

- **pytest.ini**: Configuration for pytest
- **.flake8**: Configuration for flake8 linter
- **pyproject.toml**: Configuration for black, isort, mypy, and ruff

## Local Development

To ensure your code will pass CI checks before pushing, you can run the same checks locally:

```bash
# Install development dependencies
pip install pytest pytest-cov black isort mypy ruff flake8

# Run tests
pytest

# Check code formatting
black --check .
isort --check .

# Run linters
flake8 .
ruff check .

# Run type checker
mypy src/
```

## Adding Codecov Integration

To enable code coverage reporting with Codecov:

1. Sign up for a free account at [codecov.io](https://codecov.io/)
2. Connect your GitHub repository
3. Add the Codecov token to your GitHub repository secrets
4. The CI workflow is already configured to upload coverage reports

## Troubleshooting

If CI checks fail:

1. Check the GitHub Actions logs for detailed error messages
2. Run the same checks locally to reproduce and fix the issues
3. Make sure all dependencies are properly specified in `requirements.txt`
4. Ensure GTSAM is properly installed in the CI environment

## Gradually Improving Code Quality

The current CI configuration is intentionally lenient to allow for gradual improvement of code quality. Here's a recommended approach for improving code quality over time:

1. **Focus on new code**: Apply strict standards to new code while allowing existing code to remain as-is
2. **Incremental improvements**: Fix issues in small batches, focusing on one type of issue at a time
3. **Use pre-commit hooks**: Consider setting up pre-commit hooks to automatically fix formatting issues before committing
4. **Tighten CI rules gradually**: As code quality improves, update the CI configuration to be more strict

To make the CI more strict in the future, edit the workflow files to remove the `|| echo` parts and the `--exit-zero` flags.

## Future Improvements

- Add performance benchmarking
- Implement integration tests with real data
- Add deployment workflow for releasing packages
- Set up matrix testing with different operating systems
