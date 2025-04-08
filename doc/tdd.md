# Technical Design Document (TDD)
## Targetless Multisensor Calibration System for Autonomous Driving

| Document Information |                                |
|----------------------|--------------------------------|
| **Version**          | 1.0                            |
| **Date**             | 2025-04-08                     |
| **Author**           | Gemini                         |
| **Status**           | Draft                          |

## 1. Introduction

This document details the technical design for the offline, targetless multisensor calibration system outlined in PRD multisensor_calib_prd_v1. It covers the system architecture, algorithms, data structures, and implementation choices necessary to meet the specified requirements. The core approach relies on factor graph optimization, integrating visual Structure-from-Motion (SfM) principles with IMU and wheel odometry measurements.

## 2. System Architecture

The system follows a modular pipeline architecture:

```mermaid
graph LR
    A[Raw Sensor Data] --> B(Data Loader & Synchronizer);
    B --> C{Initial Ego-Motion Estimator};
    C -- Initial Trajectory --> D{Visual Initializer / SfM};
    B -- Images --> D;
    B -- IMU Data --> C;
    B -- Wheel Data --> C;
    D -- Initial Structure & Features --> E(Factor Graph Builder);
    C -- Initial Trajectory --> E;
    B -- IMU Data --> E;
    B -- Wheel Data --> E;
    F[Initial Parameters / Config] --> E;
    E --> G(Bundle Adjustment Optimizer);
    G -- Optimized State --> H(Results Extractor & Validator);
    H --> I[Calibration Results & Metrics];

    subgraph Initialization
        C
        D
    end
    subgraph Optimization
        E
        G
    end
```

### System Components

- **Data Loader & Synchronizer**: Reads various input formats, parses data, and aligns measurements based on timestamps (e.g., using interpolation or nearest-neighbor matching).

- **Initial Ego-Motion Estimator**: Fuses IMU and wheel odometry (e.g., via EKF) to provide an initial estimate of the vehicle's trajectory (poses over time).

- **Visual Initializer / SfM**: Detects/matches features, performs triangulation, and establishes initial 3D landmarks and camera poses relative to the initial ego-motion. Selects keyframes.

- **Factor Graph Builder**: Constructs the optimization problem using a chosen library (GTSAM recommended), defining variables (poses, landmarks, intrinsics, extrinsics, biases) and factors (reprojection, IMU, odometry, priors).

- **Bundle Adjustment Optimizer**: Solves the non-linear least-squares problem defined by the factor graph using an iterative solver (Levenberg-Marquardt).

- **Results Extractor & Validator**: Extracts the final calibrated parameters from the optimized state vector and computes validation metrics.

## 3. Data Management

### Input Formats
Support for common formats like ROS bags (preferred), or sets of CSV/text files and image directories. Parsers will be needed for each supported format.

### Synchronization
Timestamps are critical. Strategy: Resample lower-rate data (e.g., images, wheels) to align with higher-rate data (e.g., IMU) or interpolate high-rate data to match image timestamps. Configurable time tolerance.

### Internal Data Representation
Use Python classes (as defined in the code skeleton) for ImageData, ImuData, WheelEncoderData, CameraIntrinsics, Extrinsics, VehiclePose, Landmark, Feature. Use NumPy arrays for numerical data.

## 4. Core Algorithm Design

### Ego-Motion Estimation

- **State**: EKF state vector to include vehicle pose (position, velocity, orientation as quaternion), IMU biases (gyro, accel).

- **Prediction (IMU)**: Standard IMU state propagation equations, integrating angular velocity and gravity-compensated acceleration.

- **Update (Wheel Odometry)**: Use a suitable vehicle motion model (e.g., differential drive for velocity/yaw rate, or Ackermann if steering angle available) based on wheel speeds. Update step corrects position and yaw.

### Visual Processing

- **Feature Detector/Descriptor**: ORB (good balance of speed and performance, rotation invariant). Alternative: SIFT (more robust, but patented/slower). Use OpenCV implementation.

- **Matching/Tracking**: Brute-force Hamming distance for ORB descriptors with ratio test. Track features across consecutive frames using KLT tracker (OpenCV calcOpticalFlowPyrLK) or descriptor matching. Match features between overlapping cameras at similar timestamps.

- **Keyframe Selection**: Select keyframes based on time interval, distance traveled, rotation angle, and number of tracked features/matches.

### Initialization

- **Triangulation**: Linear triangulation (DLT) initially, potentially refined non-linearly. Use matched features between keyframes or cameras with sufficient baseline.

- **Initial Poses/Structure**: Use poses from ego-motion estimate and triangulated points. Potentially run a small, vision-only BA on initial keyframes/landmarks.

### Optimization (Factor Graph)

- **Framework**: GTSAM (Georgia Tech Smoothing and Mapping library) - well-suited for factor graph optimization in robotics/vision.

- **Variables**: Pose3 (GTSAM type for SE(3) poses), Point3 (landmarks), Cal3_S2 or Cal3_UNKN (intrinsics), imuBias::ConstantBias (IMU biases).

- **Factors**:
  - **GenericProjectionFactor / SmartProjectionFactor**: For visual reprojection errors. Use robust loss function (e.g., Huber, Cauchy) to handle outliers.
  - **ImuFactor / CombinedImuFactor**: Use GTSAM's preintegration implementation for IMU constraints between poses. Requires IMU noise parameters.
  - **Custom Odometry Factor**: Implement a factor based on the chosen wheel odometry model, constraining relative motion between poses (primarily X, Y, Yaw). Use robust loss.
  - **PriorFactor**: Apply priors on initial pose, initial biases (zero-mean), and potentially on initial intrinsics/extrinsics based on configuration confidence.

- **Solver**: Levenberg-Marquardt (gtsam.LevenbergMarquardtOptimizer). Configure convergence criteria (error tolerance, max iterations).

### Coordinate Frames

- **Vehicle Frame (V)**: Origin at rear axle center, X-fwd, Y-left, Z-up. All extrinsics are defined relative to this frame.
- **IMU Frame (I)**: Sensor's native frame. T_V_I is the IMU extrinsic to estimate.
- **Camera Frame (C)**: Sensor's native optical frame (Z-fwd, X-right, Y-down). T_V_Ci is the extrinsic for camera i.
- **World Frame (W)**: Inertial frame, often aligned with the first vehicle pose. Vehicle poses T_W_Vk are estimated during optimization.

## 5. Key Data Structures (Python Classes)

(Refer to classes defined in the initial Python code skeleton: TimestampedData, ImageData, ImuData, WheelEncoderData, CameraIntrinsics, Extrinsics, Feature, Match, Landmark, VehiclePose). Ensure consistent use of NumPy for vector/matrix operations. Use SciPy rotation utilities (scipy.spatial.transform.Rotation) for handling rotations (quaternions, matrices).

## 6. API/Interfaces

- Main pipeline script orchestrates calls to modules.
- Functions within modules should have clear inputs/outputs (as sketched in the Python code).
- Configuration loaded from a file (e.g., YAML) specifying paths, initial guesses, algorithm parameters (feature type, solver settings, noise models).

## 7. Validation & Testing

### Unit Tests
Test individual components (data parsing, synchronization, feature detection, triangulation, EKF steps, factor creation) using pytest. Mock dependencies where necessary.

### Integration Tests
Test interactions between modules (e.g., full initialization pipeline).

### Validation Strategy

- Use datasets with known ground truth (e.g., simulation, VICON/motion capture) if available.
- **Metrics**: Compare estimated parameters to ground truth. Calculate RMS reprojection error on validation data. Visualize alignment of data (e.g., project LiDAR points into calibrated cameras, check consistency of triangulated points).
- **Dataset**: Include diverse scenarios (straight, turns, varying speeds, different lighting).

## 8. Deployment/Execution

- Provide requirements.txt or environment file for dependencies (NumPy, SciPy, OpenCV, GTSAM Python wrapper, PyYAML, etc.).
- Command-line script calibrate.py taking configuration file path as input.
- Clear instructions in README.md on installation and usage.

## 9. Assumptions & Risks

- **Assumption**: Sensor data synchronization is achievable within acceptable tolerance.
  - **Risk**: Poor sync leads to bad calibration.
  - **Mitigation**: Implement robust sync methods, report sync errors.

- **Assumption**: Initial parameter guesses are reasonably close for convergence.
  - **Risk**: Optimization diverges or converges to local minimum.
  - **Mitigation**: Improve initialization steps, allow configuration of solver parameters.

- **Assumption**: Sufficient motion excitation exists in the dataset.
  - **Risk**: Some parameters become unobservable.
  - **Mitigation**: Provide guidelines for data collection, implement checks for observability (e.g., matrix condition number).

- **Assumption**: Rigid mounting of sensors.
  - **Risk**: Non-rigid motion violates model.
  - **Mitigation**: Ensure hardware rigidity.

- **Dependency**: Availability and stability of external libraries (OpenCV, GTSAM).