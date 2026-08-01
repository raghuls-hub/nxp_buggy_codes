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
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10

# Driving Bounds for Joy Interface [-1.0, 1.0]
CRUISE_SPEED = 0.25      # Base forward speed (Range: 0.0 to 1.0)
MIN_SPEED = 0.10         # Turning speed (Range: 0.0 to 1.0)
MAX_STEER = 0.8          # Steering angle limit (Range: -1.0 to 1.0)
KP_STEER = 0.0035        # Proportional Steering Gain
SINGLE_LANE_OFFSET = 120 # Pixel offset when only 1 lane is detected

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy that receives vision, LIDAR, and server feedback 
    and outputs motor commands via sensor_msgs/Joy on /cerebri/in/joy.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Driving Targets ------------------
        self.target_speed = CRUISE_SPEED
        self.target_turn = 0.0

        # ------------------ Publishers ------------------
        # Joy publisher required by Cerebri motor controller
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ Subscriptions ------------------
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # Timer to publish drive commands at 10Hz (100ms)
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info("B3RB Line Follower initialized & listening on /cerebri/in/joy.")

    def publish_drive_commands(self):
        """Timer callback that periodically sends motor control packets to Cerebri."""
        msg = Joy()
        # Buttons array keeps buggy in autonomous mode
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  
        # axes = [0.0, speed, 0.0, turn]
        msg.axes = [0.0, float(self.target_speed), 0.0, float(self.target_turn)]
        self.publisher_joy.publish(msg)

    def edge_vectors_callback(self, message):
        """
        Calculates horizontal error and updates target steering and speed.
        """
        img_width = message.image_width
        if img_width == 0:
            return

        rover_center_x = img_width / 2.0
        target_center_x = rover_center_x
        vector_count = message.vector_count

        if vector_count == 2:
            # Dual lines: Calculate middle target
            left_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            right_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
            target_center_x = (left_x + right_x) / 2.0

        elif vector_count == 1:
            # Single line fallback
            detected_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            if detected_x < rover_center_x:
                target_center_x = detected_x + SINGLE_LANE_OFFSET
            else:
                target_center_x = detected_x - SINGLE_LANE_OFFSET
        else:
            target_center_x = rover_center_x

        # Calculate pixel deviation
        error_x = target_center_x - rover_center_x

        # Proportional Steering Calculation
        raw_turn = -KP_STEER * error_x
        self.target_turn = max(-MAX_STEER, min(MAX_STEER, raw_turn))

        # Adaptive Speed: Slow down on turns
        turn_severity = abs(self.target_turn) / MAX_STEER
        self.target_speed = CRUISE_SPEED * (1.0 - 0.5 * turn_severity)
        self.target_speed = max(MIN_SPEED, self.target_speed)

    def lidar_callback(self, message):
        """Placeholder for Phase 3 Obstacle Avoidance."""
        pass

    def server_communication_callback(self, message):
        """Placeholder for Phase 2 Server Communication."""
        pass

    def qr_detection_callback(self, message):
        """Placeholder for Phase 2 QR Detection."""
        pass

    def sign_board_callback(self, message):
        """Placeholder for Phase 3 Sign Recognition."""
        pass

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop on exit
        stop_msg = Joy()
        stop_msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        stop_msg.axes = [0.0, 0.0, 0.0, 0.0]
        node.publisher_joy.publish(stop_msg)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()