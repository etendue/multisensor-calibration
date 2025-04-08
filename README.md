# Multisensor Calibration

A robust, targetless calibration system for autonomous vehicle sensor suites.

## Overview

This project provides an offline software system designed to calibrate the intrinsic and extrinsic parameters of sensors commonly used in autonomous vehicles (cameras, IMU, wheel encoders). The system achieves accurate calibration without relying on predefined calibration targets (e.g., checkerboards), using data collected during normal vehicle operation.

Accurate sensor calibration is critical for reliable environment perception, localization, and sensor fusion in autonomous driving systems.

## Key Features

- **Targetless Calibration**: Calibrate sensors using data from normal driving, without specialized calibration targets
- **Multi-Sensor Support**: Integrates cameras, IMU, and wheel encoders in a unified optimization framework
- **High Accuracy**: Achieves calibration accuracy comparable to target-based methods under suitable conditions
- **Robust Optimization**: Uses factor graph optimization with GTSAM for joint optimization of all parameters

## System Architecture

The system follows a modular pipeline architecture:

1. **Data Loading & Synchronization**: Reads various input formats and aligns measurements based on timestamps
2. **Initial Ego-Motion Estimation**: Fuses IMU and wheel odometry to provide an initial estimate of the vehicle's trajectory
3. **Visual Initialization / SfM**: Detects/matches features, performs triangulation, and establishes initial 3D landmarks
4. **Factor Graph Construction**: Builds the optimization problem using variables (poses, landmarks, intrinsics, extrinsics, biases) and factors (reprojection, IMU, odometry, priors)
5. **Bundle Adjustment Optimization**: Solves the non-linear least-squares problem using Levenberg-Marquardt
6. **Results Extraction & Validation**: Extracts the final calibrated parameters and computes validation metrics

## Requirements

- Python 3.8+
- NumPy, SciPy
- OpenCV (for image processing)
- GTSAM (for factor graph optimization)
- Additional dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/etendue/multisensor-calibration.git
cd multisensor-calibration

# Install dependencies
pip install -r requirements.txt

# Install GTSAM (follow instructions at https://gtsam.org/get_started/)
```

## Usage

1. Prepare your dataset in a supported format (ROS bags recommended)
2. Configure the calibration parameters in a YAML file
3. Run the calibration pipeline:

```bash
python mcl.py --config config.yaml --data-dir /path/to/your/data
```

## Input Data Requirements

For best results, the input data should include:
- Multiple cameras providing a 360° view
- One 6-axis IMU (angular velocity, linear acceleration)
- Four wheel encoders (wheel speeds or ticks)
- Diverse vehicle motion (translations, rotations, varying speeds)
- Good lighting conditions and feature-rich environments

## Performance Targets

- Reprojection error: < 0.5 pixels (RMS)
- Extrinsic translation error: < 2 cm (relative to ground truth, if available)
- Extrinsic rotation error: < 0.2 degrees (relative to ground truth, if available)

## Project Status

This project is currently in development. See the [task plan](doc/task_plan.md) for current progress and upcoming tasks.

## Documentation

- [Product Requirements Document](doc/prd.md)
- [Technical Design Document](doc/tdd.md)
- [Task Plan](doc/task_plan.md)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
