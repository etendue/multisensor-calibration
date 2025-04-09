# Platform-Independent Rosbag Reader

This document describes the implementation of a platform-independent rosbag reader for the multisensor calibration system.

## Overview

The system uses the `rosbags` Python library to read data from ROS bag files without requiring a full ROS installation. This allows the calibration system to be used on various platforms (Windows, macOS, Linux) without the need to install ROS.

## Implementation Details

### Dependencies

- `rosbags`: A pure Python library for reading and writing ROS bag files (both ROS1 and ROS2 formats)
- `opencv-python`: Used for image processing, particularly for decompressing compressed images

These dependencies are listed in `requirements.txt` and can be installed via pip:

```bash
pip install -r requirements.txt
```

### Key Components

1. **Graceful Degradation**: The system checks for the availability of the `rosbags` library and falls back to dummy data if it's not available.

2. **Support for Both ROS1 and ROS2 Bags**: The implementation can read both ROS1 (.bag) and ROS2 (directory with .db3 files) bag formats.

3. **Message Type Detection**: Instead of requiring specific topic names, the reader identifies relevant data based on message types:
   - Images: `sensor_msgs/Image` and `sensor_msgs/CompressedImage`
   - IMU data: `sensor_msgs/Imu`
   - Wheel encoder data: `nav_msgs/Odometry` and `geometry_msgs/TwistStamped`

4. **Image Processing**: The system handles both compressed and uncompressed images, using OpenCV for decompression when available.

5. **Data Synchronization**: After loading the data, the system synchronizes it based on timestamps.

## Usage

The rosbag reader is integrated into the `load_and_synchronize_data` function in `src/data_handling/data_loader.py`. It automatically detects if the input path is a rosbag file or directory and processes it accordingly.

Example usage:

```python
from data_handling.data_loader import load_and_synchronize_data

# Load data from a ROS1 bag
images, imu_data, wheel_data = load_and_synchronize_data('/path/to/your/bagfile.bag')

# Or from a ROS2 bag directory
images, imu_data, wheel_data = load_and_synchronize_data('/path/to/your/ros2_bag_directory')
```

A test script is also provided to verify the functionality:

```bash
python test_rosbag_reader.py --input /path/to/your/bagfile.bag
```

## Limitations and Future Improvements

1. **Wheel Encoder Data Conversion**: The current implementation makes assumptions about how to convert odometry/twist messages to wheel speeds. This may need to be adapted based on the specific robot model.

2. **Advanced Synchronization**: The current synchronization is basic (sorting by timestamp). Future improvements could include interpolation or more sophisticated methods.

3. **Error Handling**: While the implementation includes basic error handling, more robust error handling could be added for production use.

4. **Performance Optimization**: For very large bags, memory usage could be optimized by implementing streaming or chunking approaches.
