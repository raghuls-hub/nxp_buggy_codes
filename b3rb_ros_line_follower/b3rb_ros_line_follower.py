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
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10

# Driving & Centering Constants
CRUISE_SPEED = 0.4
MIN_SPEED = 0.12
MAX_STEER = 0.8
KP_STEER = 0.0035
SINGLE_LANE_OFFSET = 120

BUGGY_ID = 1
SERVER_ID = 2

# Turn Execution & Bias Constants
TARGET_TURN_ANGLE = math.radians(85.0)  # 85 degrees relative yaw target
TURN_BIAS = 0.35                        # Directional steering offset during turns

class LineFollower(Node):
    """
    Core Controller Node for B3RB Buggy:
    - Vector-Aware Junction Navigation & Closed-Loop Lane Centering Controller.
    - Blends dynamic edge vector tracking with relative yaw progress to eliminate boundary breaches.
    - Manages destination stops, QR disappearance, and server handshakes.
    """
    def __init__(self):
        super().__init__('line_follower')

        # Motion Output
        self.target_speed = 0.0
        self.target_turn = 0.0
        self.is_driving_enabled = False

        # Orientation & Junction Turn State
        self.current_yaw = 0.0
        self.is_turning = False
        self.turn_direction = None          # "LEFT" or "RIGHT"
        self.initial_yaw = 0.0              # Heading theta_0 at turn entry

        # QR Tracking State
        self.detected_qr_buffer = ""
        self.has_scanned_active_qr = False

        # Server Communication State
        self.msg_uid = 10
        self.current_target = None
        self.awaiting_ack_for_uid = None

        # ------------------ Publishers ------------------
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ Subscriptions ------------------
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/cerebri/out/odometry',
            self.odometry_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
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

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)
        self.get_logger().info("🏎️ Line Follower with Vector-Guided Centering & Turn Controller Initialized.")

    # ------------------ STAGE A: Heading & Orientation Helper Methods ------------------

    def normalize_angle(self, angle):
        """Wraps angle strictly to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def odometry_callback(self, msg):
        """Extracts current Euler yaw angle from Odometry quaternion."""
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ------------------ Command Publisher & Safety Enforcement ------------------

    def publish_drive_commands(self):
        """Publishes control commands with steering strictly clamped within [-1.0, 1.0]."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Software override flag
        clamped_turn = max(-1.0, min(1.0, float(self.target_turn)))
        clamped_speed = float(self.target_speed)
        msg.axes = [0.0, clamped_speed, 0.0, clamped_turn]
        self.publisher_joy.publish(msg)

    # ------------------ STAGE B: Dynamic Vector-Guided Junction Navigation ------------------

    def sign_board_callback(self, message):
        """Initializes vector-guided turn controller upon receiving command."""
        if not self.is_driving_enabled or self.is_turning:
            return

        direction = message.data
        if direction in ["LEFT", "RIGHT"]:
            self.is_turning = True
            self.turn_direction = direction
            self.initial_yaw = self.current_yaw
            self.get_logger().info(f"🔄 Vector-Guided Turn Triggered: {direction} (Entry Yaw: {math.degrees(self.initial_yaw):.1f}°)")

    def edge_vectors_callback(self, message):
        """Processes continuous edge vectors for line-centering and vector-guided turn execution."""
        if not self.is_driving_enabled:
            self.target_speed = 0.0
            self.target_turn = 0.0
            return

        img_width = message.image_width
        if img_width == 0:
            return

        rover_center_x = img_width / 2.0
        target_center_x = rover_center_x
        vector_count = message.vector_count

        # 1. Evaluate Path Centroid based on visible edge vectors
        if vector_count == 2:
            left_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            right_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
            target_center_x = (left_x + right_x) / 2.0
        elif vector_count == 1:
            detected_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            if self.is_turning:
                # Mid-turn single vector bias: keep distance relative to inner curve boundary
                if self.turn_direction == "LEFT":
                    target_center_x = detected_x + SINGLE_LANE_OFFSET
                else:
                    target_center_x = detected_x - SINGLE_LANE_OFFSET
            else:
                target_center_x = detected_x + SINGLE_LANE_OFFSET if detected_x < rover_center_x else detected_x - SINGLE_LANE_OFFSET

        # Proportional centering error
        error_x = target_center_x - rover_center_x
        base_centering_turn = -KP_STEER * error_x

        # 2. Check Junction Turn Completion Criteria
        if self.is_turning:
            turn_progress = abs(self.normalize_angle(self.current_yaw - self.initial_yaw))
            steer_sign = 1.0 if self.turn_direction == "LEFT" else -1.0

            # Exit Condition: Reached target yaw (85 degrees) OR locked onto two straight vectors ahead late in turn
            if turn_progress >= TARGET_TURN_ANGLE or (turn_progress > math.radians(45.0) and vector_count == 2):
                self.get_logger().info(f"✅ Turn Complete at {math.degrees(turn_progress):.1f}°. Returning to standard line tracking.")
                self.is_turning = False
                self.target_turn = max(-MAX_STEER, min(MAX_STEER, base_centering_turn))
                self.target_speed = CRUISE_SPEED
                return

            # Blend turn bias into line-centering controller during turn
            blended_turn = base_centering_turn + (steer_sign * TURN_BIAS)
            self.target_turn = max(-MAX_STEER, min(MAX_STEER, blended_turn))
            self.target_speed = MIN_SPEED
            return

        # 3. Standard Straight Proportional Line Following
        self.target_turn = max(-MAX_STEER, min(MAX_STEER, base_centering_turn))
        turn_severity = abs(self.target_turn) / MAX_STEER
        self.target_speed = max(MIN_SPEED, CRUISE_SPEED * (1.0 - 0.5 * turn_severity))

    # ------------------ Destination Handshake & Server Protocol ------------------

    def qr_detection_callback(self, message):
        if not self.is_driving_enabled and not self.has_scanned_active_qr:
            return

        qr_text = message.data

        if qr_text != "":
            self.detected_qr_buffer = qr_text
            self.has_scanned_active_qr = True
        elif qr_text == "" and self.has_scanned_active_qr:
            final_scanned_payload = self.detected_qr_buffer
            self.get_logger().info(f"🛑 Target reached! QR code disappeared: '{final_scanned_payload}'")

            self.is_driving_enabled = False
            self.target_speed = 0.0
            self.target_turn = 0.0

            self.has_scanned_active_qr = False
            self.detected_qr_buffer = ""

            self.send_server_message(text_payload=final_scanned_payload, ack_flag=0)

    def send_server_message(self, text_payload, ack_flag=0, custom_uid=None):
        packet = ServerCommunication()
        packet.src = BUGGY_ID
        packet.dest = SERVER_ID
        packet.uid = custom_uid if custom_uid is not None else self.msg_uid
        packet.ack = ack_flag
        packet.msg = text_payload

        self.publisher_server.publish(packet)

        if ack_flag == 0:
            self.awaiting_ack_for_uid = packet.uid
            self.msg_uid = (self.msg_uid + 1) % 256

    def server_communication_callback(self, message):
        if message.dest != BUGGY_ID:
            return

        if message.ack == 1:
            if message.uid == self.awaiting_ack_for_uid:
                self.awaiting_ack_for_uid = None
            return

        if message.msg != "":
            incoming_text = message.msg
            self.send_server_message(text_payload="", ack_flag=1, custom_uid=message.uid)

            if incoming_text == "OK":
                self.is_driving_enabled = False
                self.target_speed = 0.0
                self.target_turn = 0.0
                self.get_logger().info("🎉 Mission Complete!")
            elif incoming_text == "INVALID":
                self.get_logger().warn("⚠️ Server returned INVALID.")
            else:
                self.current_target = incoming_text
                self.is_driving_enabled = True
                self.get_logger().info(f"🚀 Driving enabled for target: '{self.current_target}'")

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Joy()
        stop_msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        stop_msg.axes = [0.0, 0.0, 0.0, 0.0]
        node.publisher_joy.publish(stop_msg)

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()