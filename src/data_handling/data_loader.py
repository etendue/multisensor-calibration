# Data loading and synchronization module
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import time
import os
import sys
from pathlib import Path

# Import data structures from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_structures import ImageData, ImuData, WheelEncoderData

# Import rosbags for platform-independent rosbag reading
try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    ROSBAGS_AVAILABLE = True
except ImportError:
    print("Warning: rosbags package not found. Install with 'pip install rosbags' for platform-independent rosbag reading.")
    ROSBAGS_AVAILABLE = False

# Import OpenCV for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not found. Image data will be returned as placeholders.")
    CV2_AVAILABLE = False

def read_rosbag(data_path: str) -> Tuple[List[ImageData], List[ImuData], List[WheelEncoderData]]:
    """
    Read sensor data from a rosbag file using the platform-independent rosbags library.

    Args:
        data_path: Path to the rosbag file or directory.

    Returns:
        A tuple containing lists of image, IMU, and wheel encoder data.
    """
    if not ROSBAGS_AVAILABLE:
        print("Error: rosbags package is required for reading rosbag files.")
        return [], [], []

    # Convert string path to Path object
    bag_path = Path(data_path)
    if not bag_path.exists():
        print(f"Error: Bag file or directory not found: {data_path}")
        return [], [], []

    # Initialize empty lists for sensor data
    images = []
    imu_data = []
    wheel_data = []

    # Create a type store for ROS message definitions
    typestore = get_typestore(Stores.ROS1)

    print(f"Reading rosbag data from: {data_path}")
    try:
        # Open the bag file for reading
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            # Get all connections (topics) in the bag
            connections = reader.connections

            # Find relevant topics by message type
            image_connections = [c for c in connections if 'sensor_msgs/Image' in c.msgtype or 'sensor_msgs/CompressedImage' in c.msgtype]
            imu_connections = [c for c in connections if 'sensor_msgs/Imu' in c.msgtype]
            wheel_connections = [c for c in connections if 'nav_msgs/Odometry' in c.msgtype or 'geometry_msgs/TwistStamped' in c.msgtype]

            # Process image messages
            for connection, timestamp, rawdata in reader.messages(connections=image_connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                # Extract camera ID from topic name
                camera_id = connection.topic.split('/')[-1]

                # Convert timestamp to seconds
                ts = timestamp / 1e9  # nanoseconds to seconds

                # Handle compressed and uncompressed images
                if 'CompressedImage' in connection.msgtype:
                    if CV2_AVAILABLE:
                        try:
                            # Decompress image using OpenCV
                            # Convert compressed image data to numpy array
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            # Decode the compressed image
                            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        except Exception as e:
                            print(f"Error decompressing image: {e}")
                            # Fallback to placeholder
                            image = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        # OpenCV not available, use placeholder
                        image = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    # For uncompressed images, extract the data
                    try:
                        # Get image dimensions
                        height = msg.height
                        width = msg.width
                        # Get encoding (e.g., 'rgb8', 'bgr8', 'mono8')
                        encoding = msg.encoding
                        # Get step (row length in bytes)
                        step = msg.step

                        # Convert raw data to numpy array
                        if 'mono' in encoding.lower():
                            # Grayscale image
                            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                        else:
                            # Color image (assuming 3 channels)
                            channels = 3
                            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, channels)

                            # Convert to BGR if needed (OpenCV default format)
                            if CV2_AVAILABLE and 'rgb' in encoding.lower():
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        # Fallback to placeholder
                        image = np.zeros((480, 640, 3), dtype=np.uint8)

                images.append(ImageData(ts, camera_id, image))

            # Process IMU messages
            for connection, timestamp, rawdata in reader.messages(connections=imu_connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                # Convert timestamp to seconds
                ts = timestamp / 1e9  # nanoseconds to seconds

                # Extract angular velocity and linear acceleration
                angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                linear_acceleration = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

                imu_data.append(ImuData(ts, angular_velocity, linear_acceleration))

            # Process wheel encoder messages
            for connection, timestamp, rawdata in reader.messages(connections=wheel_connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                # Convert timestamp to seconds
                ts = timestamp / 1e9  # nanoseconds to seconds

                # Extract wheel speeds based on message type
                if 'Odometry' in connection.msgtype:
                    # For Odometry messages, we might need to convert linear and angular velocities to wheel speeds
                    # This is a simplified example and would need to be adapted to your specific robot model
                    linear_x = msg.twist.twist.linear.x
                    linear_y = msg.twist.twist.linear.y
                    angular_z = msg.twist.twist.angular.z

                    # Placeholder conversion to wheel speeds (this would depend on your robot's kinematics)
                    # For a differential drive robot with 4 wheels, for example
                    wheel_speeds = np.array([linear_x + angular_z, linear_x - angular_z, linear_x + angular_z, linear_x - angular_z])
                elif 'TwistStamped' in connection.msgtype:
                    # Similar conversion for TwistStamped messages
                    linear_x = msg.twist.linear.x
                    linear_y = msg.twist.linear.y
                    angular_z = msg.twist.angular.z

                    # Placeholder conversion
                    wheel_speeds = np.array([linear_x + angular_z, linear_x - angular_z, linear_x + angular_z, linear_x - angular_z])
                else:
                    # Default placeholder
                    wheel_speeds = np.zeros(4)

                wheel_data.append(WheelEncoderData(ts, wheel_speeds))

        print(f"Successfully read {len(images)} images, {len(imu_data)} IMU samples, and {len(wheel_data)} wheel encoder samples.")

    except Exception as e:
        print(f"Error reading rosbag: {e}")

    return images, imu_data, wheel_data


def load_and_synchronize_data(data_path: str) -> Tuple[List[ImageData], List[ImuData], List[WheelEncoderData]]:
    """
    Loads sensor data from storage and aligns it based on timestamps.

    Args:
        data_path: Path to the directory containing sensor data files or a rosbag file.

    Returns:
        A tuple containing lists of synchronized image, IMU, and wheel encoder data.
        Synchronization might involve interpolation or selecting nearest neighbors in time.
    """
    print(f"Loading and synchronizing data from: {data_path}")

    # Check if the path is a file or directory
    path = Path(data_path)

    # If it's a file and has a .bag extension, try to read it as a rosbag
    if path.is_file() and path.suffix == '.bag':
        if ROSBAGS_AVAILABLE:
            images, imu_data, wheel_data = read_rosbag(data_path)
            if images or imu_data or wheel_data:  # If we got any data
                # Perform synchronization
                return synchronize_sensor_data(images, imu_data, wheel_data)
            else:
                print("No data found in rosbag or error reading it. Falling back to dummy data.")
        else:
            print("rosbags package not available. Falling back to dummy data.")

    # If it's a directory, check if it contains a rosbag2 database
    elif path.is_dir() and any(p.suffix == '.db3' for p in path.glob('*')):
        if ROSBAGS_AVAILABLE:
            images, imu_data, wheel_data = read_rosbag(data_path)
            if images or imu_data or wheel_data:  # If we got any data
                # Perform synchronization
                return synchronize_sensor_data(images, imu_data, wheel_data)
            else:
                print("No data found in rosbag2 directory or error reading it. Falling back to dummy data.")
        else:
            print("rosbags package not available. Falling back to dummy data.")

    # Fallback to dummy data for development/testing
    print("Using dummy data for development/testing.")
    # Dummy data for structure illustration
    timestamps = np.linspace(0, 10, 101) # 10 seconds of data at 10 Hz
    images = [ImageData(ts, f"cam{i}", np.zeros((480, 640))) for ts in timestamps for i in range(4)] # 4 cameras
    imu_data = [ImuData(ts, np.random.randn(3)*0.01, np.random.randn(3)*0.1 + np.array([0,0,9.81])) for ts in np.linspace(0, 10, 1001)] # 100 Hz IMU
    wheel_data = [WheelEncoderData(ts, np.random.rand(4)*5 + 10) for ts in np.linspace(0, 10, 501)] # 50 Hz Wheel Encoders

    # Basic synchronization
    return synchronize_sensor_data(images, imu_data, wheel_data)


def synchronize_sensor_data(images: List[ImageData], imu_data: List[ImuData], wheel_data: List[WheelEncoderData]) -> Tuple[List[ImageData], List[ImuData], List[WheelEncoderData]]:
    """
    Synchronize sensor data based on timestamps.

    Args:
        images: List of image data.
        imu_data: List of IMU data.
        wheel_data: List of wheel encoder data.

    Returns:
        Synchronized sensor data.
    """
    # Sort all data by timestamp
    images.sort(key=lambda x: x.timestamp)
    imu_data.sort(key=lambda x: x.timestamp)
    wheel_data.sort(key=lambda x: x.timestamp)

    # For now, we'll just return the sorted data
    # In a real implementation, we might interpolate IMU and wheel data to match image timestamps,
    # or use more sophisticated synchronization methods

    print("Data loaded and synchronized.")
    return images, imu_data, wheel_data
