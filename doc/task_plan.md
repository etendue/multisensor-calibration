# TODO / Task Plan
## Targetless Multisensor Calibration System

**Project Goal:** Develop an offline, targetless multisensor calibration tool as per PRD/TDD.

## Phases & Tasks

### Phase 1: Setup & Data Handling (~1-2 weeks)

- [ ] **T1.1:** Setup Git repository and project structure.
- [ ] **T1.2:** Define and implement core data structures (Python classes for sensors, parameters, state).
- [ ] **T1.3:** Implement data loader for ROS bags (or other chosen format).
  - [ ] **T1.3.1:** Parse Image messages.
  - [ ] **T1.3.2:** Parse IMU messages.
  - [ ] **T1.3.3:** Parse Wheel Encoder messages (custom format or standard Odom).
- [ ] **T1.4:** Implement sensor data synchronization module (interpolation/nearest neighbor).
- [ ] **T1.5:** Implement configuration file parser (YAML).
- [ ] **T1.6:** Write unit tests for data loading and synchronization.

### Phase 2: Initial Motion Estimation (~2 weeks)

- [ ] **T2.1:** Implement wheel odometry calculation module (using axle lengths, wheel speeds).
- [ ] **T2.2:** Implement basic IMU integration (orientation/position, gravity compensation).
- [ ] **T2.3:** Implement EKF/UKF for IMU + Wheel Odometry fusion.
  - [ ] **T2.3.1:** Define state vector and noise parameters.
  - [ ] **T2.3.2:** Implement prediction step (IMU).
  - [ ] **T2.3.3:** Implement update step (Wheel Odometry).
- [ ] **T2.4:** Write unit tests for odometry, IMU integration, and EKF steps.
- [ ] **T2.5:** Integrate motion estimation into the main pipeline.

### Phase 3: Visual Processing & Initialization (~3 weeks)

- [ ] **T3.1:** Integrate OpenCV for feature detection (e.g., ORB).
- [ ] **T3.2:** Implement feature matching (e.g., BFMatcher/FLANN) and tracking (e.g., KLT).
- [ ] **T3.3:** Implement multi-camera feature matching logic (match features across cameras at similar timestamps).
- [ ] **T3.4:** Implement keyframe selection logic.
- [ ] **T3.5:** Implement feature triangulation function (e.g., DLT).
- [ ] **T3.6:** Implement visual initialization pipeline (coordinate features, poses, landmarks).
- [ ] **T3.7:** Write unit tests for feature detection, matching, triangulation.
- [ ] **T3.8:** Integrate visual initialization into the main pipeline.

### Phase 4: Optimization Backend (~3-4 weeks)

- [ ] **T4.1:** Install and configure GTSAM Python wrapper.
- [ ] **T4.2:** Implement factor graph construction module.
  - [ ] **T4.2.1:** Add variables (Pose3, Point3, Cal3_*, Bias).
  - [ ] **T4.2.2:** Implement creation of reprojection factors (GenericProjectionFactor or SmartProjectionFactor) with robust loss.
  - [ ] **T4.2.3:** Implement creation of IMU preintegration factors (ImuFactor). Configure noise models.
  - [ ] **T4.2.4:** Implement creation of custom wheel odometry factors with robust loss.
  - [ ] **T4.2.5:** Implement creation of prior factors.
- [ ] **T4.3:** Implement logic to populate initial values for the optimizer.
- [ ] **T4.4:** Implement optimization execution logic (calling GTSAM solver).
- [ ] **T4.5:** Implement results extraction module (getting parameters from optimized state).
- [ ] **T4.6:** Write basic tests for factor creation and optimization call.

### Phase 5: Integration, Validation & Documentation (~2-3 weeks)

- [ ] **T5.1:** Integrate optimization backend into the main pipeline script.
- [ ] **T5.2:** Develop validation tools:
  - [ ] **T5.2.1:** Calculate final RMS reprojection error.
  - [ ] **T5.2.2:** Implement visualization (e.g., plot trajectory, project points into images).
- [ ] **T5.3:** Acquire or create suitable test and validation datasets (including diverse motion).
- [ ] **T5.4:** Perform end-to-end testing and debugging on datasets.
- [ ] **T5.5:** Tune algorithm parameters (noise models, keyframe selection, solver settings).
- [ ] **T5.6:** Write README.md with installation and usage instructions.
- [ ] **T5.7:** Refine code comments and generate API documentation (e.g., using Sphinx).

### Phase 6: Refinement & Release (~1 week)

- [ ] **T6.1:** Code review and cleanup.
- [ ] **T6.2:** Performance profiling and optimization (if required based on NFR2).
- [ ] **T6.3:** Finalize PRD and TDD documents based on implementation.
- [ ] **T6.4:** Package code for release/deployment.

*(Estimated Total Duration: ~12-15 weeks for one engineer, subject to complexity and unforeseen issues)*