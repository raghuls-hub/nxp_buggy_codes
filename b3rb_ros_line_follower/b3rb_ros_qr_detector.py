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
from std_msgs.msg import String

import cv2
import numpy as np

QOS_PROFILE_DEFAULT = 10

class QRDetector(Node):
    """
    ROS 2 Node that receives compressed camera feeds, detects and decodes QR codes,
    and publishes detected string payloads onto /qr_detection.
    """
    def __init__(self):
        super().__init__('qr_detector')

        # Last detected payload to prevent duplicate continuous flooding
        self.last_qr_data = ""

        # Publisher for detected QR code strings
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            QOS_PROFILE_DEFAULT)

        # Subscriber for camera compressed images
        self.subscription_image = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            QOS_PROFILE_DEFAULT)

        # OpenCV QR Code Detector Instance
        self.qr_detector = cv2.QRCodeDetector()

        self.get_logger().info("QR Detector Node Initialized and listening to camera.")

    def image_callback(self, message):
        """Processes compressed image frames and searches for readable QR codes."""
        try:
            # Convert compressed image bytes to OpenCV BGR image
            np_arr = np.frombuffer(message.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                return

            # Detect and decode QR Code
            qr_data, points, _ = self.qr_detector.detectAndDecode(cv_image)

            if qr_data:
                # Publish newly detected QR string payload
                qr_msg = String()
                qr_msg.data = str(qr_data)
                self.publisher_qr.publish(qr_msg)

                if qr_data != self.last_qr_data:
                    self.get_logger().info(f"🔍 Scanned QR Code: '{qr_data}'")
                    self.last_qr_data = qr_data

        except Exception as e:
            self.get_logger().error(f"Error processing QR image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()