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

# Import PyAV for H.264 video decoding
try:
    import av
    PYAV_AVAILABLE = True

    def decode_h264(data, debug=False):
        """Decode H.264 video data using PyAV.

        This function is based on a working implementation that successfully decodes
        H.264 data from the specific rosbag format. It handles different frame types
        (I-frames, P-frames, B-frames) and is resilient to frames that can't be decoded.

        Args:
            data: The raw H.264 encoded data
            debug: Whether to print detailed debug information

        Returns:
            A numpy array containing the decoded image, or None if decoding fails
        """
        try:
            if debug:
                print(f"Data length: {len(data)} bytes")
                if len(data) > 20:
                    print(f"First 20 bytes: {[hex(b) for b in data[:20]]}")

            # Create a codec context for H.264 decoding
            codec_ctx = av.codec.CodecContext.create('h264', 'r')

            # Create a packet directly from the data
            packet = av.packet.Packet(data)

            try:
                # Decode the packet - this might fail for non-I-frames
                frames = list(codec_ctx.decode(packet))

                # If we have frames, convert the first one to a numpy array
                if frames:
                    # Get the first frame
                    frame = frames[0]

                    if debug:
                        print(f"Successfully decoded frame")

                    # Convert to numpy array
                    return frame.to_ndarray(format='bgr24')
                elif debug:
                    print("No frames decoded from the packet (might be a P/B-frame without previous I-frame)")
            except Exception as decode_error:
                if debug:
                    print(f"Error decoding packet: {decode_error} (might be a P/B-frame without previous I-frame)")

            # If we reach here, we couldn't decode the frame
            return None
        except Exception as e:
            print(f"Error in decode_h264: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None

except ImportError:
    print("Warning: PyAV not found. H.264 video decoding will not be available.")
    PYAV_AVAILABLE = False

    def decode_h264(data, debug=False):
        """Fallback function when PyAV is not available."""
        if debug:
            print("PyAV is not available for H.264 decoding")
        return None

def read_rosbag(data_path: str, load_images: bool = True) -> Tuple[List[ImageData], List[ImuData], List[WheelEncoderData]]:
    """
    Read sensor data from a rosbag file using the platform-independent rosbags library.

    Args:
        data_path: Path to the rosbag file or directory.
        load_images: Whether to load and process image data. Set to False to skip image processing
                    and save time when only IMU and wheel encoder data are needed.

    Returns:
        A tuple containing lists of image, IMU, and wheel encoder data. If load_images is False,
        the images list will be empty.
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
    try:
        # Try different store types based on the rosbags version
        if hasattr(Stores, 'ROS1'):
            typestore = get_typestore(Stores.ROS1)
        elif hasattr(Stores, 'ROS1_DISTRO'):
            typestore = get_typestore(Stores.ROS1_DISTRO)
        elif hasattr(Stores, 'LATEST_ROS1'):
            typestore = get_typestore(Stores.LATEST_ROS1)
        else:
            # Try to get the first available store
            store_names = [name for name in dir(Stores) if not name.startswith('_')]
            if store_names:
                typestore = get_typestore(getattr(Stores, store_names[0]))
            else:
                # Fallback to None
                typestore = None
    except Exception as e:
        print(f"Warning: Could not create typestore: {e}")
        typestore = None

    print(f"Reading rosbag data from: {data_path}")
    try:
        # Open the bag file for reading
        reader_kwargs = {}
        if typestore is not None:
            reader_kwargs['default_typestore'] = typestore

        with AnyReader([bag_path], **reader_kwargs) as reader:
            # Get all connections (topics) in the bag
            connections = reader.connections

            # Find relevant topics by message type and topic name
            # For standard ROS message types
            image_connections = [c for c in connections if
                               ('sensor_msgs/Image' in c.msgtype or
                                'sensor_msgs/CompressedImage' in c.msgtype or
                                'sensor_interface_msgs/CompressedVideo' in c.msgtype) or
                               ('/sensor/camera' in c.topic and '/video' in c.topic)]

            imu_connections = [c for c in connections if
                              ('sensor_msgs/Imu' in c.msgtype or
                               'gps_imu_msgs/MLAImu' in c.msgtype) or
                              '/sensor/imu' == c.topic]

            wheel_connections = [c for c in connections if
                                ('nav_msgs/Odometry' in c.msgtype or
                                 'geometry_msgs/TwistStamped' in c.msgtype or
                                 'endpoint_msgs/WheelReport' in c.msgtype) or
                                '/vehicle/wheel_report' == c.topic]

            # Print found connections for debugging
            print(f"Found {len(image_connections)} camera topics:")
            for c in image_connections:
                print(f"  - {c.topic} ({c.msgtype})")

            print(f"Found {len(imu_connections)} IMU topics:")
            for c in imu_connections:
                print(f"  - {c.topic} ({c.msgtype})")

            print(f"Found {len(wheel_connections)} wheel encoder topics:")
            for c in wheel_connections:
                print(f"  - {c.topic} ({c.msgtype})")

            # Process image messages (if requested)
            if load_images:
                for connection, timestamp, rawdata in reader.messages(connections=image_connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    # Extract camera ID from topic name
                    camera_id = connection.topic.split('/')[-1]

                    # Convert timestamp to seconds
                    ts = timestamp / 1e9  # nanoseconds to seconds

                    try:
                        # Handle different image message types
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

                        elif 'CompressedVideo' in connection.msgtype:
                            # Handle sensor_interface_msgs/CompressedVideo with H.264 encoding
                            try:
                                # For CompressedVideo, the image data might be in a different field
                                # Adjust this based on the actual message structure
                                if hasattr(msg, 'data'):
                                    data_field = msg.data
                                elif hasattr(msg, 'image'):
                                    data_field = msg.image
                                elif hasattr(msg, 'compressed_data'):
                                    data_field = msg.compressed_data
                                else:
                                    # Try to find any field that might contain image data
                                    found_data = False
                                    for field_name in dir(msg):
                                        if not field_name.startswith('_') and isinstance(getattr(msg, field_name), bytes):
                                            data_field = getattr(msg, field_name)
                                            found_data = True
                                            break

                                    if not found_data:
                                        raise AttributeError("Could not find image data field in message")

                                # Print message structure for debugging on first image
                                if len(images) == 0:
                                    print(f"CompressedVideo message attributes: {dir(msg)}")

                                # Print the first few bytes of the data for debugging
                                if len(images) == 0:
                                    print(f"First 20 bytes of data: {[b for b in data_field[:20]]}")
                                    print(f"Data length: {len(data_field)} bytes")

                                # Try to decode H.264 video using our specialized function
                                # Enable debug mode for the first few frames
                                debug_mode = len(images) < 3  # Only debug the first 3 frames
                                decoded_image = decode_h264(data_field, debug=debug_mode)

                                if decoded_image is not None:
                                    # Successfully decoded the image
                                    image = decoded_image

                                    # For debugging, save the first frame as JPEG
                                    if len(images) < 3 and CV2_AVAILABLE:
                                        debug_path = f"debug_frame_{len(images)}.jpg"
                                        print(f"Saving debug frame to {debug_path}")
                                        _, jpeg_data = cv2.imencode('.jpg', image)
                                        with open(debug_path, 'wb') as f:
                                            f.write(jpeg_data)

                                    # Print success message for the first few frames
                                    if len(images) < 10:
                                        print(f"Successfully decoded frame {len(images)}")
                                else:
                                    # Decoding failed, try with OpenCV as fallback
                                    if CV2_AVAILABLE:
                                        # Try to decode with OpenCV (might not work for H.264)
                                        np_arr = np.frombuffer(data_field, np.uint8)
                                        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                        if image is None:
                                            # Fallback to placeholder - use a small image to save memory
                                            image = np.zeros((120, 160, 3), dtype=np.uint8)

                                            # Skip this frame if we're just collecting placeholders
                                            # This helps us get to I-frames faster
                                            if len(images) > 0 and len(images) % 10 != 0:
                                                # Skip this frame (only keep 1 in 10 placeholders)
                                                continue
                                    else:
                                        # Fallback to placeholder - use a small image to save memory
                                        image = np.zeros((120, 160, 3), dtype=np.uint8)

                                        # Skip this frame if we're just collecting placeholders
                                        # This helps us get to I-frames faster
                                        if len(images) > 0 and len(images) % 10 != 0:
                                            # Skip this frame (only keep 1 in 10 placeholders)
                                            continue
                            except Exception as e:
                                print(f"Error processing CompressedVideo: {e}")
                                # Print message structure for debugging
                                print(f"Message attributes: {dir(msg)}")
                                # Fallback to placeholder
                                image = np.zeros((480, 640, 3), dtype=np.uint8)

                        else:
                            # For uncompressed images, extract the data
                            try:
                                # Get image dimensions
                                height = msg.height
                                width = msg.width
                                # Get encoding (e.g., 'rgb8', 'bgr8', 'mono8')
                                encoding = msg.encoding

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
                    except Exception as e:
                        print(f"Error processing image message: {e}")
                        # Fallback to placeholder
                        image = np.zeros((480, 640, 3), dtype=np.uint8)

                images.append(ImageData(ts, camera_id, image))

            # Process IMU messages
            for connection, timestamp, rawdata in reader.messages(connections=imu_connections):
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)

                    # Convert timestamp to seconds
                    ts = timestamp / 1e9  # nanoseconds to seconds

                    # Extract angular velocity and linear acceleration based on message type
                    if 'MLAImu' in connection.msgtype:
                        # Handle gps_imu_msgs/MLAImu
                        try:
                            # Print message structure for debugging on first message
                            if len(imu_data) == 0:
                                print(f"MLAImu message attributes: {dir(msg)}")

                            # Based on the provided message structure, we need to access the info.imu_data_basic fields
                            if hasattr(msg, 'info') and hasattr(msg.info, 'imu_data_basic'):
                                imu_basic = msg.info.imu_data_basic

                                # Extract gyro (angular velocity) data
                                if hasattr(imu_basic, 'gyrox') and hasattr(imu_basic, 'gyroy') and hasattr(imu_basic, 'gyroz'):
                                    # MLAImu uses right-handed coordinate system (forward Y, right X, up Z)
                                    # Need to convert to left-handed system (X forward, Y left, Z up)
                                    # Conversion: X_new = Y_old, Y_new = -X_old, Z_new = Z_old
                                    angular_velocity = np.array([imu_basic.gyroy, -imu_basic.gyrox, imu_basic.gyroz])
                                else:
                                    print("Warning: Could not find gyro data in imu_data_basic")
                                    angular_velocity = np.zeros(3)

                                # Extract accelerometer data
                                if hasattr(imu_basic, 'accx') and hasattr(imu_basic, 'accy') and hasattr(imu_basic, 'accz'):
                                    # Apply the same coordinate transformation for accelerometer data
                                    linear_acceleration = np.array([imu_basic.accy, -imu_basic.accx, imu_basic.accz])
                                else:
                                    print("Warning: Could not find acceleration data in imu_data_basic")
                                    linear_acceleration = np.zeros(3)
                            elif hasattr(msg, 'angular_velocity') and hasattr(msg, 'linear_acceleration'):
                                # Standard field names
                                ang_vel = msg.angular_velocity
                                lin_acc = msg.linear_acceleration

                                # Apply coordinate transformation for MLAImu
                                angular_velocity = np.array([ang_vel.y, -ang_vel.x, ang_vel.z])
                                linear_acceleration = np.array([lin_acc.y, -lin_acc.x, lin_acc.z])
                            elif hasattr(msg, 'gyro') and hasattr(msg, 'accel'):
                                # Alternative field names
                                ang_vel = msg.gyro
                                lin_acc = msg.accel

                                # Apply coordinate transformation for MLAImu
                                angular_velocity = np.array([ang_vel.y, -ang_vel.x, ang_vel.z])
                                linear_acceleration = np.array([lin_acc.y, -lin_acc.x, lin_acc.z])
                            else:
                                # Try to find any vector3 fields that might contain the data
                                ang_vel = None
                                lin_acc = None
                                for field_name in dir(msg):
                                    if field_name.startswith('_'):
                                        continue
                                    field = getattr(msg, field_name)
                                    # Check if field has x, y, z attributes
                                    if hasattr(field, 'x') and hasattr(field, 'y') and hasattr(field, 'z'):
                                        if 'gyro' in field_name.lower() or 'angular' in field_name.lower():
                                            ang_vel = field
                                        elif 'accel' in field_name.lower() or 'linear' in field_name.lower():
                                            lin_acc = field

                                # If we found the fields, extract the data
                                if ang_vel is not None and lin_acc is not None:
                                    # Apply coordinate transformation for MLAImu
                                    angular_velocity = np.array([ang_vel.y, -ang_vel.x, ang_vel.z])
                                    linear_acceleration = np.array([lin_acc.y, -lin_acc.x, lin_acc.z])
                                else:
                                    # Fallback to zeros
                                    print(f"Warning: Could not find angular velocity and linear acceleration in MLAImu message")
                                    angular_velocity = np.zeros(3)
                                    linear_acceleration = np.zeros(3)

                            # Print coordinate transformation info on first message
                            if len(imu_data) == 0:
                                print("Note: MLAImu data converted from right-handed (forward Y, right X, up Z) to left-handed (X forward, Y left, Z up) coordinate system")
                        except Exception as e:
                            print(f"Error processing MLAImu: {e}")
                            # Fallback to zeros
                            angular_velocity = np.zeros(3)
                            linear_acceleration = np.zeros(3)
                    else:
                        # Standard sensor_msgs/Imu
                        angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                        linear_acceleration = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

                    imu_data.append(ImuData(ts, angular_velocity, linear_acceleration))
                except Exception as e:
                    print(f"Error processing IMU message: {e}")

            # Process wheel encoder messages
            for connection, timestamp, rawdata in reader.messages(connections=wheel_connections):
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)

                    # Convert timestamp to seconds
                    ts = timestamp / 1e9  # nanoseconds to seconds

                    # Extract wheel speeds based on message type
                    if 'WheelReport' in connection.msgtype:
                        # Handle endpoint_msgs/WheelReport
                        try:
                            # Print message structure for debugging on first message
                            if len(wheel_data) == 0:
                                print(f"WheelReport message attributes: {dir(msg)}")

                            # Based on the provided message structure, we need to extract wheel data
                            # We have multiple options: wheel_position_report, wheel_speed_report, wheel_angle_report

                            # Initialize wheel_speeds and wheel_angles as None
                            wheel_speeds = None
                            wheel_angles = None

                            # First, try to get wheel speed data from wheel_speed_report (preferred method)
                            if hasattr(msg, 'wheel_speed_report') and hasattr(msg.wheel_speed_report, 'wheel_speed_report_data'):
                                speed_data = msg.wheel_speed_report.wheel_speed_report_data
                                # Check if we have the mps (meters per second) fields - these are preferred
                                if hasattr(speed_data, 'front_left_mps') and hasattr(speed_data, 'front_right_mps') and \
                                   hasattr(speed_data, 'rear_left_mps') and hasattr(speed_data, 'rear_right_mps'):
                                    wheel_speeds = np.array([speed_data.front_left_mps, speed_data.front_right_mps,
                                                           speed_data.rear_left_mps, speed_data.rear_right_mps])
                                    print(f"Using wheel speed data (mps): {wheel_speeds}")
                                # Otherwise use the regular speed fields
                                elif hasattr(speed_data, 'front_left') and hasattr(speed_data, 'front_right') and \
                                     hasattr(speed_data, 'rear_left') and hasattr(speed_data, 'rear_right'):
                                    wheel_speeds = np.array([speed_data.front_left, speed_data.front_right,
                                                           speed_data.rear_left, speed_data.rear_right])
                                    print(f"Using wheel speed data: {wheel_speeds}")

                            # If wheel_speeds is still None, try to get wheel position data (encoder values)
                            if wheel_speeds is None and hasattr(msg, 'wheel_position_report') and hasattr(msg.wheel_position_report, 'wheel_position_report_data'):
                                pos_data = msg.wheel_position_report.wheel_position_report_data
                                if hasattr(pos_data, 'front_left') and hasattr(pos_data, 'front_right') and \
                                   hasattr(pos_data, 'rear_left') and hasattr(pos_data, 'rear_right'):
                                    # Extract wheel position values (encoder ticks)
                                    wheel_positions = np.array([pos_data.front_left, pos_data.front_right,
                                                              pos_data.rear_left, pos_data.rear_right])
                                    # Use wheel positions as our primary data
                                    wheel_speeds = wheel_positions
                                    print(f"Using wheel position data: {wheel_positions}")

                            # Get wheel angle data from wheel_angle_report
                            if hasattr(msg, 'wheel_angle_report') and hasattr(msg.wheel_angle_report, 'wheel_angle_report_data'):
                                angle_data = msg.wheel_angle_report.wheel_angle_report_data
                                if hasattr(angle_data, 'front_left') and hasattr(angle_data, 'front_right') and \
                                   hasattr(angle_data, 'rear_left') and hasattr(angle_data, 'rear_right'):
                                    wheel_angles = np.array([angle_data.front_left, angle_data.front_right,
                                                           angle_data.rear_left, angle_data.rear_right])
                                    print(f"Using wheel angle data: {wheel_angles}")

                            # If wheel_speeds is still None, try using wheel angle data as a last resort
                            if wheel_speeds is None and wheel_angles is not None:
                                wheel_speeds = np.zeros(4)  # Use zeros for speeds if we only have angles
                                print("Warning: Using zeros for wheel speeds, only angle data available")

                            # If we still don't have wheel data, try generic approaches
                            if 'wheel_speeds' not in locals() or wheel_speeds is None:
                                # Check for common wheel speed field names
                                if hasattr(msg, 'wheel_speeds'):
                                    wheel_speeds = msg.wheel_speeds
                                elif hasattr(msg, 'speeds'):
                                    wheel_speeds = msg.speeds
                                elif hasattr(msg, 'velocity'):
                                    wheel_speeds = msg.velocity
                                else:
                                    # Try to find any array field that might contain wheel speeds
                                    for field_name in dir(msg):
                                        if field_name.startswith('_'):
                                            continue
                                        field = getattr(msg, field_name)
                                        # Check if field is a list or array
                                        if isinstance(field, (list, tuple, np.ndarray)) and len(field) >= 4:
                                            wheel_speeds = field
                                            break

                            # If we found the wheel data, ensure it's in the right format
                            if 'wheel_speeds' in locals() and wheel_speeds is not None:
                                # Convert to numpy array if it's not already
                                if not isinstance(wheel_speeds, np.ndarray):
                                    wheel_speeds = np.array(wheel_speeds)

                                # Ensure we have 4 wheel values (front-left, front-right, rear-left, rear-right)
                                if len(wheel_speeds) < 4:
                                    # Pad with zeros if needed
                                    wheel_speeds = np.pad(wheel_speeds, (0, 4 - len(wheel_speeds)))
                                elif len(wheel_speeds) > 4:
                                    # Truncate if needed
                                    wheel_speeds = wheel_speeds[:4]
                            else:
                                # Fallback to zeros
                                print(f"Warning: Could not find wheel data in WheelReport message")
                                wheel_speeds = np.zeros(4)
                        except Exception as e:
                            print(f"Error processing WheelReport: {e}")
                            # Fallback to zeros
                            wheel_speeds = np.zeros(4)
                    elif 'Odometry' in connection.msgtype:
                        # For Odometry messages, convert linear and angular velocities to wheel speeds
                        linear_x = msg.twist.twist.linear.x
                        angular_z = msg.twist.twist.angular.z

                        # Simplified conversion to wheel speeds (this would depend on your robot's kinematics)
                        # For a differential drive robot with 4 wheels
                        wheel_speeds = np.array([linear_x + angular_z, linear_x - angular_z, linear_x + angular_z, linear_x - angular_z])
                    elif 'TwistStamped' in connection.msgtype:
                        # Similar conversion for TwistStamped messages
                        linear_x = msg.twist.linear.x
                        angular_z = msg.twist.angular.z

                        # Simplified conversion
                        wheel_speeds = np.array([linear_x + angular_z, linear_x - angular_z, linear_x + angular_z, linear_x - angular_z])
                    else:
                        # Default placeholder
                        wheel_speeds = np.zeros(4)

                    wheel_data.append(WheelEncoderData(ts, wheel_speeds, wheel_angles))
                except Exception as e:
                    print(f"Error processing wheel encoder message: {e}")

        print(f"Successfully read {len(images)} images, {len(imu_data)} IMU samples, and {len(wheel_data)} wheel encoder samples.")

    except Exception as e:
        print(f"Error reading rosbag: {e}")

    return images, imu_data, wheel_data


def load_and_synchronize_data(data_path: str, load_images: bool = True) -> Tuple[List[ImageData], List[ImuData], List[WheelEncoderData]]:
    """
    Loads sensor data from storage and aligns it based on timestamps.

    Args:
        data_path: Path to the directory containing sensor data files or a rosbag file.
        load_images: Whether to load and process image data. Set to False to skip image processing
                    and save time when only IMU and wheel encoder data are needed.

    Returns:
        A tuple containing lists of synchronized image, IMU, and wheel encoder data.
        Synchronization might involve interpolation or selecting nearest neighbors in time.
        If load_images is False, the images list will be empty.
    """
    print(f"Loading and synchronizing data from: {data_path}")

    # Check if the path is a file or directory
    path = Path(data_path)

    # If it's a file and has a .bag extension, try to read it as a rosbag
    if path.is_file() and path.suffix == '.bag':
        if ROSBAGS_AVAILABLE:
            images, imu_data, wheel_data = read_rosbag(data_path, load_images=load_images)
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
            images, imu_data, wheel_data = read_rosbag(data_path, load_images=load_images)
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

    # Generate dummy wheel data with wheel angles
    wheel_timestamps = np.linspace(0, 10, 501) # 50 Hz Wheel Encoders
    wheel_data = []
    for ts in wheel_timestamps:
        # Generate random wheel speeds
        wheel_speeds = np.random.rand(4)*5 + 10
        # Generate wheel angles: front wheels have small random angles, rear wheels are ~0
        wheel_angles = np.array([np.sin(ts*0.5)*0.1, np.sin(ts*0.5)*0.1, 0.0, 0.0])
        wheel_data.append(WheelEncoderData(ts, wheel_speeds, wheel_angles))

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
