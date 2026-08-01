# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors
# raghul@Raghul-s-TUF:~$ ros2 run b3rb_ros_line_follower runner
# Traceback (most recent call last):
#   File "/home/raghul/cognipilot/cranium/install/b3rb_ros_line_follower/lib/b3rb_ros_line_follower/runner", line 33, in <module>
#     sys.exit(load_entry_point('b3rb-ros-line-follower==0.0.0', 'console_scripts', 'runner')())
#   File "/home/raghul/cognipilot/cranium/install/b3rb_ros_line_follower/lib/b3rb_ros_line_follower/runner", line 25, in importlib_load_entry_point
#     return next(matches).load()
#   File "/usr/lib/python3.10/importlib/metadata/__init__.py", line 171, in load
#     module = import_module(match.group('module'))
#   File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
#     return _bootstrap._gcd_import(name[level:], package, level)
#   File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
#   File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
#   File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
#   File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
#   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
#   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#   File "/home/raghul/cognipilot/cranium/install/b3rb_ros_line_follower/lib/python3.10/site-packages/b3rb_ros_line_follower/b3rb_ros_line_follower.py", line 20, in <module>
#     from synapse_msgs.msg import EdgeVectors, Steering, ServerCommunication
# ImportError: cannot import name 'Steering' from 'synapse_msgs.msg' (/home/raghul/cognipilot/cranium/install/synapse_msgs/local/lib/python3.10/dist-packages/synapse_msgs/msg/__init__.py)
# [ros2run]: Process exited with failure 1

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# Analyzing lower 35% of the frame for immediate lane orientation ahead of the buggy
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.35
VECTOR_MAGNITUDE_MINIMUM = 5.0

class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def compute_vectors_from_image(self, image, thresh):
        """
        Analyzes the binary threshold image and extracts left and right lane edge vectors.
        """
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        vectors = []
        rover_point = [self.image_width / 2.0, self.lower_image_height]

        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]
            if len(coordinates) < 5:
                continue

            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = coordinates[coordinates[:, 1] == min_y_value]
            max_y_coords = coordinates[coordinates[:, 1] == max_y_value]

            min_y_coord = np.array(min_y_coords[0], dtype=float)
            max_y_coord = np.array(max_y_coords[0], dtype=float)

            # Calculate contour vector magnitude
            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if magnitude > VECTOR_MAGNITUDE_MINIMUM:
                middle_point = (min_y_coord + max_y_coord) / 2.0
                distance = np.linalg.norm(middle_point - rover_point)

                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = float(np.max(min_y_coords[:, 0]))
                else:
                    max_y_coord[0] = float(np.max(max_y_coords[:, 0]))

                vectors.append([list(min_y_coord), list(max_y_coord), distance])

                # Draw raw candidate vectors in blue
                cv2.line(
                    image, 
                    (int(min_y_coord[0]), int(min_y_coord[1])), 
                    (int(max_y_coord[0]), int(max_y_coord[1])), 
                    BLUE_COLOR, 2
                )

        return vectors, image

    def process_image_for_edge_vectors(self, image):
        """
        Applies HSV thresholding and extracts left/right lane vectors cleanly.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Convert to HSV color space for better resilience against shadow/lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Thresholding for dark black lane border stripes
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        thresh = cv2.inRange(hsv, lower_black, upper_black)

        # Clean noise using morphological opening/closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 3. Crop ROI to lower portion of the image
        thresh_cropped = thresh[self.upper_image_height:].copy()
        image_cropped = image[self.upper_image_height:].copy()

        # 4. Compute vectors from binary contours
        vectors, debug_img = self.compute_vectors_from_image(image_cropped, thresh_cropped)

        # 5. Sort vectors by proximity to the rover
        vectors = sorted(vectors, key=lambda x: x[2])

        # 6. Separate left and right vectors
        half_width = self.image_width / 2.0
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2.0) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2.0) >= half_width]

        final_vectors = []
        for side_vectors in [vectors_left, vectors_right]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                
                # Draw key lane boundary in green
                p1 = (int(best_vector[0][0]), int(best_vector[0][1]))
                p2 = (int(best_vector[1][0]), int(best_vector[1][1]))
                cv2.line(debug_img, p1, p2, GREEN_COLOR, 2)

                # Map y-coordinates back to global frame space
                v_copy = [
                    [best_vector[0][0], best_vector[0][1] + self.upper_image_height],
                    [best_vector[1][0], best_vector[1][1] + self.upper_image_height]
                ]
                final_vectors.append(v_copy)

        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        vectors = self.process_image_for_edge_vectors(image)

        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (Left Boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (Right Boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)

def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()