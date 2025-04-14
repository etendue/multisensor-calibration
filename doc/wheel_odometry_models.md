# Wheel Odometry Models

This document describes the different wheel odometry models implemented in the multisensor calibration system.

## Overview

The system supports multiple vehicle models for wheel odometry calculation:

1. **Differential Drive Model**: Uses left and right wheel speeds to calculate vehicle motion.
2. **Standard Ackermann Model**: Uses wheel speeds and a single steering angle to calculate vehicle motion.
3. **Advanced Ackermann Model**: Uses wheel speeds and individual wheel angles to calculate vehicle motion with higher accuracy.

## Model Selection

The system automatically selects the most appropriate model based on the available data:

- If wheel angle data is available, the **Advanced Ackermann Model** is used.
- If only steering angle data is available, the **Standard Ackermann Model** is used.
- If neither wheel angles nor steering angle is available, the **Differential Drive Model** is used as a fallback.

This selection happens in the `estimate_initial_ego_motion` function in `src/motion_estimation/ego_motion.py`.

## Differential Drive Model

The differential drive model assumes the vehicle has two wheels on a single axis, with each wheel independently driven. The model calculates vehicle motion as follows:

- Linear velocity: $v = \frac{v_{right} + v_{left}}{2}$
- Angular velocity: $\omega = \frac{v_{right} - v_{left}}{track\_width}$
- Forward motion: $dx = v \cdot \cos(\theta) \cdot dt$
- Lateral motion: $dy = v \cdot \sin(\theta) \cdot dt$
- Rotation: $d\theta = \omega \cdot dt$

Where:
- $v_{right}$ and $v_{left}$ are the right and left wheel speeds
- $track\_width$ is the distance between the left and right wheels
- $\theta$ is the current vehicle heading
- $dt$ is the time difference between measurements

## Standard Ackermann Model

The standard Ackermann model assumes the vehicle has four wheels with the front wheels able to turn. The model uses a single steering angle and calculates vehicle motion as follows:

- Forward motion: $dx = v \cdot \cos(\theta) \cdot dt$
- Lateral motion: $dy = v \cdot \sin(\theta) \cdot dt$
- Rotation: $d\theta = \frac{v \cdot \tan(\delta)}{wheelbase} \cdot dt$

Where:
- $v$ is the average wheel speed
- $\delta$ is the steering angle
- $wheelbase$ is the distance between the front and rear axles
- $\theta$ is the current vehicle heading
- $dt$ is the time difference between measurements

## Advanced Ackermann Model

The advanced Ackermann model uses individual wheel angles to calculate vehicle motion with higher accuracy. This model is particularly useful for vehicles with complex steering mechanisms or when precise wheel angle data is available.

The model calculates:

- Effective steering angle based on the average of front wheel angles
- Turning radius based on the effective steering angle and wheelbase
- Forward motion, lateral motion, and rotation based on the turning radius and wheel speeds

Key calculations include:

- Effective angle: Average of front wheel angles
- Turning radius: $R = \frac{wheelbase}{\tan(effective\_angle)}$
- Angular velocity: $\omega = \frac{v}{R}$
- Forward motion: $dx = v \cdot \cos(\theta) \cdot dt$
- Lateral motion: $dy = v \cdot \sin(\theta) \cdot dt$
- Rotation: $d\theta = \omega \cdot dt$

## Implementation

The wheel odometry models are implemented in the `WheelOdometry` class in `src/motion_estimation/wheel_odometry.py`. The class provides methods for each model:

- `update_differential_drive`: Implements the differential drive model
- `update_ackermann`: Implements the standard Ackermann model
- `update_ackermann_advanced`: Implements the advanced Ackermann model

The `update` method selects the appropriate model based on the available data and the configured model.

## Trajectory Estimation

The wheel odometry models are used in the `estimate_initial_ego_motion` function to estimate the vehicle trajectory. The function:

1. Creates a `WheelOdometry` object with the appropriate vehicle parameters
2. Processes each wheel encoder measurement to calculate vehicle motion
3. Integrates the motion to update the vehicle pose
4. Returns a list of vehicle poses representing the trajectory

The trajectory can be visualized using the `visualize_trajectory` function in `scripts/test_trajectory_estimation.py`.

## Performance and Accuracy

The advanced Ackermann model provides the most accurate trajectory estimation when wheel angle data is available. The standard Ackermann model is less accurate but still provides reasonable results when only steering angle data is available. The differential drive model is the least accurate but can be used as a fallback when no steering data is available.

In our testing with real-world data, the advanced Ackermann model produced a realistic trajectory that matched the expected vehicle motion based on the wheel speeds and angles.
