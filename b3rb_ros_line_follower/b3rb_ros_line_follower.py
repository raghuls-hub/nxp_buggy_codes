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
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, Steering, ServerCommunication

import math

QOS_PROFILE_DEFAULT = 10

# Driving Parameters
CRUISE_SPEED = 1.2       # Base speed on straight paths (m/s)
MIN_SPEED = 0.5          # Speed during tight turns (m/s)
MAX_STEER_ANGLE = 0.5    # Maximum steering angle limit (radians)
KP_STEER = 0.0035        # Proportional Gain for steering control
SINGLE_LANE_OFFSET = 120 # Pixel offset from a single lane boundary to road center

class LineFollower(Node):
    """
    ROS 2 Node that receives lane vectors, LIDAR, server commands, and traffic signs
    to control the buggy's velocity and steering.
    """
    def __init__(self):
        super().__init__('line_follower')

        # Steering control variables
        self.target_speed = 0.0
        self.target_turn = 0.0

        # State Machine Variable (Prepared for Phase 4)
        self.current_state = "LANE_FOLLOWING"

        # Publishers
        self.publisher_cmd_vel = self.create_publisher(
            Steering,
            '/cmd_vel',
            QOS_PROFILE_DEFAULT)

        self.publisher_server_comm = self.create_publisher(
            ServerCommunication,
            '/server_data_ack',
            QOS_PROFILE_DEFAULT)

        # Subscribers
        self.subscription_edge_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_object_recog = self.create_subscription(
            String,
            '/sign_board_detection',
            self.object_recog_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_qr_detector = self.create_subscription(
            String,
            '/qr_code_detection',
            self.qr_code_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_server_comm = self.create_subscription(
            ServerCommunication,
            '/server_data_receive',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # Command Loop Timer (Runs at 20 Hz / 50ms)
        self.timer = self.create_timer(0.05, self.control_loop)

    def edge_vectors_callback(self, message):
        """
        Calculates steering error and adjusts speed based on incoming edge vectors.
        """
        img_width = message.image_width
        if img_width == 0:
            return

        rover_center_x = img_width / 2.0
        target_center_x = rover_center_x

        vector_count = message.vector_count

        if vector_count == 2:
            # Both left and right lane boundaries detected -> calculate midpoint
            left_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            right_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
            target_center_x = (left_x + right_x) / 2.0

        elif vector_count == 1:
            # Single lane boundary detected -> apply fixed horizontal offset
            detected_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            if detected_x < rover_center_x:
                # Left lane detected -> aim to the right of it
                target_center_x = detected_x + SINGLE_LANE_OFFSET
            else:
                # Right lane detected -> aim to the left of it
                target_center_x = detected_x - SINGLE_LANE_OFFSET

        else:
            # No lane detected -> maintain current direction safely
            target_center_x = rover_center_x

        # Calculate horizontal pixel deviation from road center
        error_x = target_center_x - rover_center_x

        # Proportional Steering Control
        raw_turn = -KP_STEER * error_x
        self.target_turn = max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, raw_turn))

        # Adaptive Speed: Slow down on sharp turns, accelerate on straight paths
        turn_severity = abs(self.target_turn) / MAX_STEER_ANGLE
        self.target_speed = CRUISE_SPEED * (1.0 - 0.6 * turn_severity)
        self.target_speed = max(MIN_SPEED, self.target_speed)

    def lidar_callback(self, message):
        """Placeholder for LIDAR processing (Phase 3)."""
        pass

    def object_recog_callback(self, message):
        """Placeholder for sign board handling (Phase 3)."""
        pass

    def qr_code_callback(self, message):
        """Placeholder for QR code parsing (Phase 2)."""
        pass

    def server_communication_callback(self, message):
        """Placeholder for Municipality Server messages (Phase 2)."""
        pass

    def control_loop(self):
        """Publishes the computed speed and turn angle to the vehicle's actuators."""
        steering_msg = Steering()
        steering_msg.solar_panel_ctl = 0.0

        # Assign targets computed from vision/navigation callbacks
        steering_msg.speed = float(self.target_speed)
        steering_msg.steering_angle = float(self.target_turn)

        self.publisher_cmd_vel.publish(steering_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop on exit
        stop_msg = Steering()
        stop_msg.speed = 0.0
        stop_msg.steering_angle = 0.0
        node.publisher_cmd_vel.publish(stop_msg)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()