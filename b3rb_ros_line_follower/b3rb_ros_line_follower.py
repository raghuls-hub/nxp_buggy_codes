# Copyright 2024-2026 NXP
# Dual-Mode Line Follower with Alongside-Lane Cyclic Distance Turning Mechanism

import rclpy
from rclpy.node import Node
import math
import json
import cv2
import numpy as np
import sys
from enum import Enum

from sensor_msgs.msg import Joy, CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10

# Speeds
CRUISE_SPEED            = 0.38   # Fast straight cruise speed
TURN_CRAWL_SPEED        = 0.20   # Crawl speed while turning to reduce delta distance
TURN_CURVE_TRACK_SPEED  = 0.30   # Speed when moving forward alongside lane

# Control & Steering Constants
KP_STEER_DIST           = 0.0075
AUTO_STEER_MAGNITUDE    = 0.55   # Turning steer command magnitude (+ = Left, - = Right)
DELTA_TOLERANCE_PX      = 15.0   # Hysteresis band (pixels) for d_cached distance
SINGLE_LANE_OFFSET      = 120.0  # Fallback offset if d_cached is uninitialized

TARGET_TURN_ANGLE       = math.radians(85.0)  # Stop turning cycle at 85° (80° - 90°)

BUGGY_ID                = 1
SERVER_ID               = 2


class DrivingMode(Enum):
    DUAL_CENTERING = 1
    TURNING = 2


class LineFollower(Node):
    """
    ROS 2 Dual-Mode Line Follower Node with Alongside-Lane Cyclic Distance Turning Mechanism.
    """
    def __init__(self):
        super().__init__('line_follower')

        self.current_mode = DrivingMode.DUAL_CENTERING
        self.is_driving_enabled = False

        self.target_speed = 0.0
        self.target_turn = 0.0

        # Yaw and Cached Distance Memory
        self.current_yaw = 0.0
        self.initial_yaw = 0.0
        self.turn_direction = None
        self.d_cached = SINGLE_LANE_OFFSET

        # Parabola Curve Data from Edge Vectors Node
        self.parabola_detected = False
        self.parabola_A = 0.0
        self.parabola_x_lane = 0.0

        self.latest_camera_frame = None

        # Server Communication State
        self.detected_qr_buffer = ""
        self.has_scanned_active_qr = False
        self.msg_uid = 10
        self.current_target = None
        self.awaiting_ack_for_uid = None

        # Publishers
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)
        self.publisher_vector_debug = self.create_publisher(CompressedImage, '/debug_images/vector_images', QOS_PROFILE_DEFAULT)
        self.publisher_mode = self.create_publisher(String, '/driving_mode', QOS_PROFILE_DEFAULT)

        # Subscriptions
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(Odometry, '/cerebri/out/odometry', self.odometry_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/turning_parabola', self.turning_parabola_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)
        self.get_logger().info("🏎️ Cyclic Alongside-Lane Line Follower Active.")

    def publish_mode_status(self):
        msg = String()
        msg.data = self.current_mode.name
        self.publisher_mode.publish(msg)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def camera_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_camera_frame = frame
        except Exception as e:
            self.get_logger().error(f"Error decoding image: {e}")

    def odometry_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def turning_parabola_callback(self, msg):
        """Receives live parabola curve data from /turning_parabola topic."""
        try:
            data = json.loads(msg.data)
            self.parabola_detected = data.get("lane_detected", False)
            if self.parabola_detected:
                self.parabola_A = data.get("A", 0.0)
                self.parabola_x_lane = data.get("target_x_lane", 0.0)
        except Exception as e:
            self.get_logger().error(f"Error parsing parabola topic: {e}")

    def publish_drive_commands(self):
        self.publish_mode_status()
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        clamped_steer = float(max(-1.0, min(1.0, self.target_turn)))
        msg.axes = [0.0, float(self.target_speed), 0.0, clamped_steer]
        self.publisher_joy.publish(msg)

    def sign_board_callback(self, message):
        # Gated until server start signal is received
        if not self.is_driving_enabled:
            return

        direction = message.data.strip().upper()
        if direction in ["LEFT", "RIGHT"]:
            self.turn_direction = direction
            self.current_mode = DrivingMode.TURNING
            self.initial_yaw = self.current_yaw
            self.publish_mode_status()

            banner = (
                "\n" + "░" * 65 + "\n"
                f"  🚀 RESULTANT DIRECTION RECEIVED: [{direction}]\n"
                f"  🔄 MODE TOGGLED: [DUAL_CENTERING] -> [TURNING]\n"
                f"  📏 Cached Lane Offset (d_cached): {self.d_cached:.1f} px\n"
                f"  📐 Initial Yaw Recorded: {math.degrees(self.initial_yaw):.2f}°\n"
                + "░" * 65 + "\n"
            )
            print(banner, flush=True)
            sys.stdout.flush()

    def edge_vectors_callback(self, message):
        img_w = message.image_width if message.image_width > 0 else 640
        img_h = message.image_height if message.image_height > 0 else 480
        rover_center_x = img_w / 2.0
        vector_count = message.vector_count

        # Extract Left/Right Vectors
        vec_left, vec_right = None, None
        if vector_count >= 1:
            v1_mid = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            if vector_count >= 2:
                v2_mid = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
                vec_left = message.vector_1 if v1_mid < v2_mid else message.vector_2
                vec_right = message.vector_2 if v1_mid < v2_mid else message.vector_1
            else:
                if v1_mid < rover_center_x:
                    vec_left = message.vector_1
                else:
                    vec_right = message.vector_1

        mode_str = ""

        if self.is_driving_enabled:

            # ── MODE 1: DUAL CENTERING ────────────────────────────────────────
            if self.current_mode == DrivingMode.DUAL_CENTERING:
                mode_str = "MODE: DUAL_CENTERING"

                if vec_left and vec_right:
                    x_l = vec_left[0].x
                    x_r = vec_right[0].x
                    self.d_cached = (x_r - x_l) / 2.0  # Continuously update cached lane offset
                    target_center_x = (x_l + x_r) / 2.0
                elif vec_left:
                    target_center_x = vec_left[0].x + self.d_cached
                elif vec_right:
                    target_center_x = vec_right[0].x - self.d_cached
                else:
                    target_center_x = rover_center_x

                dist_error = target_center_x - rover_center_x
                self.target_turn = -KP_STEER_DIST * dist_error
                self.target_speed = CRUISE_SPEED

            # ── MODE 2: TURNING WITH ALONGSIDE-LANE CYCLIC MECHANISM ───────────
            elif self.current_mode == DrivingMode.TURNING:
                angle_turned = abs(self.normalize_angle(self.current_yaw - self.initial_yaw))

                # Step 1: Calculate Delta Distance between Buggy Center and Parabola Lane End
                if not self.parabola_detected:
                    delta_d = float('inf')  # Set delta to INFINITY if no parabola exists
                else:
                    delta_d = abs(self.parabola_x_lane - rover_center_x)

                d_target = self.d_cached
                delta_threshold = d_target + DELTA_TOLERANCE_PX

                # Step 2: Cyclic Turning & Forward Logic
                if delta_d == float('inf'):
                    # State A: No Parabola -> Turn vehicle towards left/right direction
                    self.target_speed = TURN_CRAWL_SPEED
                    self.target_turn = +AUTO_STEER_MAGNITUDE if self.turn_direction == "LEFT" else -AUTO_STEER_MAGNITUDE
                    mode_str = f"TURNING [{self.turn_direction}] Delta=INF -> Searching Parabola..."
                    self.get_logger().info(f"🔄 Parabola Missing (delta=INF) -> Turning towards {self.turn_direction}. Angle: {math.degrees(angle_turned):.1f}°")

                elif delta_d > delta_threshold:
                    # State B: Distance increased above d_cached -> Turn to reduce delta back to d_cached
                    self.target_speed = TURN_CRAWL_SPEED
                    self.target_turn = +AUTO_STEER_MAGNITUDE if self.turn_direction == "LEFT" else -AUTO_STEER_MAGNITUDE
                    mode_str = f"TURNING [{self.turn_direction}] Delta ({delta_d:.1f}px) > Target -> Turning to reduce delta..."
                    self.get_logger().info(f"📐 Delta ({delta_d:.1f}px) > Target ({d_target:.1f}px) -> Turning to reduce distance. Angle: {math.degrees(angle_turned):.1f}°")

                else:
                    # State C: Reached d_cached distance -> Stop turning hard & Move Forward Alongside Lane!
                    self.target_speed = TURN_CURVE_TRACK_SPEED

                    # Smooth forward tracking along target offset
                    desired_x = self.parabola_x_lane + (self.d_cached if self.turn_direction == "LEFT" else -self.d_cached)
                    dist_error = desired_x - rover_center_x
                    self.target_turn = -KP_STEER_DIST * dist_error

                    mode_str = f"ALONGSIDE LANE [{self.turn_direction}] Delta ({delta_d:.1f}px) <= Target -> Moving Forward!"
                    self.get_logger().info(f"🚀 ALONGSIDE LANE: Delta ({delta_d:.1f}px) <= Target ({d_target:.1f}px) -> Moving Forward! Angle: {math.degrees(angle_turned):.1f}°")

                # Step 3: Termination Check (80° to 90° Turn Reached)
                if angle_turned >= TARGET_TURN_ANGLE:
                    self.get_logger().info(
                        f"✅ TURN COMPLETE! Turned: {math.degrees(angle_turned):.1f}°. Re-engaging [DUAL_CENTERING].")

                    self.current_mode = DrivingMode.DUAL_CENTERING
                    self.turn_direction = None
                    self.parabola_detected = False
                    self.publish_mode_status()

        else:
            self.target_speed, self.target_turn = 0.0, 0.0
            mode_str = "STANDBY"

        self.render_debug_vectors(img_w, img_h, rover_center_x, vec_left, vec_right, mode_str)

    def render_debug_vectors(self, img_w, img_h, rover_center_x, vec_l, vec_r, mode_str):
        if self.latest_camera_frame is not None:
            canvas = cv2.resize(self.latest_camera_frame.copy(), (img_w, img_h))
        else:
            canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        cv2.line(canvas, (int(rover_center_x), 0), (int(rover_center_x), img_h), (128, 128, 128), 1)

        status_color = (0, 165, 255) if self.current_mode == DrivingMode.TURNING else (0, 255, 0)
        cv2.putText(canvas, mode_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2)

        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', canvas)[1]).tobytes()
            self.publisher_vector_debug.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug vector image: {e}")

    # Server Communication Handlers
    def qr_detection_callback(self, message):
        if not self.is_driving_enabled and not self.has_scanned_active_qr:
            return
        qr_text = message.data
        if qr_text != "":
            self.detected_qr_buffer = qr_text
            self.has_scanned_active_qr = True
        elif qr_text == "" and self.has_scanned_active_qr:
            final_scanned = self.detected_qr_buffer
            self.get_logger().info(f"🛑 Destination reached! Scanned: '{final_scanned}'")
            self.is_driving_enabled = False
            self.target_speed, self.target_turn = 0.0, 0.0
            self.has_scanned_active_qr = False
            self.send_server_message(text_payload=final_scanned, ack_flag=0)

    def send_server_message(self, text_payload, ack_flag=0, custom_uid=None):
        packet = ServerCommunication()
        packet.src, packet.dest = BUGGY_ID, SERVER_ID
        packet.uid = custom_uid if custom_uid is not None else self.msg_uid
        packet.ack, packet.msg = ack_flag, text_payload
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
            self.send_server_message(text_payload="", ack_flag=1, custom_uid=message.uid)
            if message.msg == "OK":
                self.is_driving_enabled = False
                self.target_speed, self.target_turn = 0.0, 0.0
            elif message.msg != "INVALID":
                self.current_target = message.msg
                self.is_driving_enabled = True


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()