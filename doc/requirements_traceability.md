# Requirements Traceability Matrix

## Overview

This document establishes traceability between requirements defined in the Product Requirements Document (PRD) and their implementation in the codebase. The purpose is to ensure that all requirements are properly implemented and tested, and to facilitate impact analysis when requirements change.

## Traceability Matrix

### Functional Requirements

| Requirement ID | Requirement Description | Implementation Artifacts | Test Artifacts | Status |
|----------------|-------------------------|--------------------------|----------------|--------|
| FR1 | Input Data | src/data_handling/data_loader.py<br>src/data_handling/config_parser.py | tests/test_data_loader.py | Implemented |
| FR2 | Calibration Scope | src/data_structures.py<br>src/optimization/bundle_adjustment.py | tests/test_optimization.py | Implemented |
| FR3 | Targetless Operation | src/visual_processing/sfm_initializer.py | tests/test_sfm.py | Implemented |
| FR4 | Offline Processing | calibrate.py | tests/test_optimization_integration.py | Implemented |
| FR5 | Coordinate System | src/data_structures.py | tests/test_data_structures.py | Implemented |
| FR6 | Input Assumptions | src/data_handling/config_parser.py | tests/test_config_parser.py | Implemented |
| FR7 | Output | src/optimization/bundle_adjustment.py<br>src/validation/metrics.py | tests/test_optimization.py | Implemented |
| FR8 | Validation | src/validation/metrics.py<br>src/validation/__init__.py | tests/test_validation.py | Implemented |

### Non-Functional Requirements

| Requirement ID | Requirement Description | Implementation Artifacts | Test Artifacts | Status |
|----------------|-------------------------|--------------------------|----------------|--------|
| NFR1 | Accuracy | src/optimization/bundle_adjustment.py<br>src/validation/metrics.py | tests/test_validation.py | Implemented |
| NFR2 | Performance | src/optimization/bundle_adjustment.py | tests/test_optimization_integration.py | Implemented |
| NFR3 | Robustness | src/optimization/factor_graph.py<br>src/data_handling/data_loader.py | tests/test_optimization.py | Implemented |
| NFR4 | Usability | calibrate.py<br>config.yaml | N/A | Implemented |
| NFR5 | Maintainability | All source files | All test files | Implemented |

## Traceability Details

### FR1: Input Data

**Requirement**: The system must accept synchronized time-stamped data streams from multiple cameras, one 6-axis IMU, and four wheel encoders.

**Implementation Artifacts**:
- `src/data_handling/data_loader.py`: Implements data loading from ROS bags and other formats
- `src/data_handling/config_parser.py`: Parses configuration files for initial parameters

**Test Artifacts**:
- `tests/test_data_loader.py`: Tests for data loading functionality

**Tasks**:
- T1.3: Implement data loader for ROS bags (Completed)
- T1.3.2.1: Implement MLAImu coordinate system conversion (Completed)
- T1.3.3: Parse Wheel Encoder messages (Completed)

**Status**: Implemented

**Notes**: The data loader supports ROS bag files and can extract camera images, IMU data, and wheel encoder data (including wheel speeds, positions, and angles). It also handles data synchronization. For MLAImu data, a coordinate system conversion is applied to transform from the legacy right-handed system (forward Y, right X, up Z) to the project's left-handed system (X forward, Y left, Z up). For wheel encoder data, the system supports both wheel speed and wheel angle data, enabling advanced vehicle motion models.

### FR2: Calibration Scope

**Requirement**: The system must estimate intrinsic parameters for each camera and extrinsic parameters for each camera and the IMU relative to the vehicle reference frame.

**Implementation Artifacts**:
- `src/data_structures.py`: Defines data structures for intrinsic and extrinsic parameters
- `src/optimization/bundle_adjustment.py`: Implements the optimization of intrinsic and extrinsic parameters

**Test Artifacts**:
- `tests/test_optimization.py`: Tests for optimization functionality

**Tasks**:
- T1.2: Define core data structures (Completed)
- T4.1: Implement bundle adjustment (Completed)

**Status**: Implemented

**Notes**: The system estimates camera intrinsics (focal length, principal point, distortion) and extrinsics (6 DoF pose) for all sensors.

### FR3: Targetless Operation

**Requirement**: The calibration process must not require the use of predefined calibration targets or patterns in the environment.

**Implementation Artifacts**:
- `src/visual_processing/sfm_initializer.py`: Implements Structure from Motion for targetless initialization

**Test Artifacts**:
- `tests/test_sfm.py`: Tests for SfM initialization

**Tasks**:
- T3.1: Implement feature detection and matching (Completed)
- T3.5: Implement feature triangulation (Completed)

**Status**: Implemented

**Notes**: The system uses natural features in the environment instead of calibration targets.

### FR4: Offline Processing

**Requirement**: The system is designed to run offline on collected datasets, not in real-time on the vehicle.

**Implementation Artifacts**:
- `calibrate.py`: Main script for offline processing

**Test Artifacts**:
- `tests/test_optimization_integration.py`: Integration tests for the offline processing pipeline

**Status**: Implemented

**Notes**: The system processes data in batch mode, not in real-time.

### FR5: Coordinate System

**Requirement**: The calibration results must be expressed relative to a clearly defined vehicle-centric coordinate system.

**Implementation Artifacts**:
- `src/data_structures.py`: Defines the coordinate system conventions

**Test Artifacts**:
- `tests/test_data_structures.py`: Tests for coordinate system handling

**Status**: Implemented

**Notes**: The system uses a vehicle-centric coordinate system with origin at the center of the rear axle. The coordinate system is left-handed with X forward, Y left, and Z up. Sensor data from different sources (like MLAImu) is converted to this consistent coordinate system.

### FR6: Input Assumptions

**Requirement**: The system requires approximate initial guesses for camera intrinsic and extrinsic parameters, known vehicle parameters, rigidly mounted sensors, and reasonably synchronized data.

**Implementation Artifacts**:
- `src/data_handling/config_parser.py`: Parses configuration files for initial parameters

**Test Artifacts**:
- `tests/test_config_parser.py`: Tests for configuration parsing

**Status**: Implemented

**Notes**: The system reads initial guesses from a configuration file.

### FR7: Output

**Requirement**: The system must output the estimated intrinsic and extrinsic parameters for each camera and the IMU, along with quality metrics.

**Implementation Artifacts**:
- `src/optimization/bundle_adjustment.py`: Extracts and formats calibration results
- `src/validation/metrics.py`: Calculates and reports quality metrics

**Test Artifacts**:
- `tests/test_optimization.py`: Tests for result extraction

**Status**: Implemented

**Notes**: The system outputs calibration results in a structured format with quality metrics.

### FR8: Validation

**Requirement**: The system should provide tools or metrics to assess the quality of the calibration (e.g., visualization of aligned point clouds, reprojection error statistics).

**Implementation Artifacts**:
- `src/validation/metrics.py`: Implements calculation of reprojection error statistics
- `src/validation/__init__.py`: Package initialization for validation tools

**Test Artifacts**:
- `tests/test_validation.py`: Comprehensive tests for validation metrics and visualization tools

**Tasks**:
- T5.2: Develop validation tools (Completed)
  - T5.2.1: Implement calculation of reprojection error statistics
  - T5.2.2: Implement visualization tools for calibration results
  - T5.2.3: Implement real-time optimization progress visualization

**Status**: Implemented

**Notes**: The validation tools include reprojection error calculation, visualization of camera poses and trajectories, and optimization progress visualization. These tools help users evaluate the accuracy and reliability of the calibration.

### Wheel Odometry Models

**Requirement**: The system must support different vehicle models for wheel odometry calculation to accommodate various vehicle types and available sensor data.

**Implementation Artifacts**:
- `src/motion_estimation/wheel_odometry.py`: Implements different vehicle models for wheel odometry
- `src/motion_estimation/ego_motion.py`: Selects the appropriate model based on available data

**Test Artifacts**:
- `scripts/test_trajectory_estimation.py`: Tests for trajectory estimation using wheel odometry

**Tasks**:
- T2.1: Implement wheel odometry calculation module (Completed)
- T5.1.1: Fix result extraction in `extract_optimized_values` (Completed)

**Status**: Implemented

**Notes**: The system supports three vehicle models for wheel odometry:
1. Differential Drive: Uses left and right wheel speeds to calculate vehicle motion
2. Standard Ackermann: Uses wheel speeds and a single steering angle
3. Advanced Ackermann: Uses wheel speeds and individual wheel angles for higher accuracy

The system automatically selects the most appropriate model based on the available data. If wheel angle data is available, the Advanced Ackermann model is used. If only steering angle data is available, the Standard Ackermann model is used. If neither is available, the Differential Drive model is used as a fallback.

## How to Maintain This Document

1. When implementing a new requirement, update this document to link the requirement to its implementation artifacts
2. When adding tests for a requirement, update the test artifacts column
3. Update the status column as requirements progress from "Not Started" to "In Progress" to "Implemented"
4. When refactoring code, ensure that the traceability links are updated accordingly
