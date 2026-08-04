# Copyright 2024-2026 NXP
# Parabola & Vector Line Follower with Sequential State Machine & Junction Turn Fallback

import rclpy
from rclpy.node import Node
import json
import cv2
import numpy as np

from sensor_msgs.msg import Joy, CompressedImage
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10

# ── Driving & Control Constants ───────────────────────────────────────────────
CRUISE_FORWARD_SPEED       = 0.35   # Normal driving speed
JUNCTION_TURN_SPEED        = 0.25   # Speed during blind junction recovery turn
JUNCTION_STEER_MAGNITUDE   = 0.50   # Open-loop turn steering magnitude

KP_STEER_DIST              = 0.0075 # Steering P gain for centering/parabola offset
SINGLE_LANE_OFFSET         = 120.0  # Default half-lane width (pixels)
DELTA_TOLERANCE            = 15.0   # Boundary tolerance for parabola distance control

MAX_LOST_FRAMES_THRESHOLD  = 5      # Consecutive frames without lane lines before blind turn recovery

BUGGY_ID                   = 1
SERVER_ID                  = 2

# Mission States
STATE_IDLE                 = "IDLE"
STATE_EN_ROUTE_PATIENT     = "EN_ROUTE_PATIENT"
STATE_EN_ROUTE_HOSPITAL    = "EN_ROUTE_HOSPITAL"


class LineFollower(Node):
    """
    ROS 2 Line Follower Node featuring:
    1. Sequential Mission State Machine (Server -> Patient -> Hospital).
    2. Junction blind-turn recovery using result direction when lane lines disappear.
    """
    def __init__(self):
        super().__init__('line_follower')

        # Mission State Machine
        self.mission_state = STATE_IDLE
        self.is_driving_enabled = False

        # Motion Output State
        self.target_speed = 0.0
        self.target_turn = 0.0

        # Perception & Navigation State
        self.turn_direction = "LEFT"  # "LEFT" or "RIGHT"
        self.d_cached = SINGLE_LANE_OFFSET
        self.delta_d = float('inf')
        self.parabola_detected = False
        self.no_lane_frames = 0        # Counter for missing lane frames

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

        # Subscriptions
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/turning_parabola', self.turning_parabola_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/sign_board_detection', self.sign_callback, QOS_PROFILE_DEFAULT)

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)
        self.get_logger().info("🏎️ State Machine Line Follower Node Initialized.")

    def sign_callback(self, msg):
        direction = msg.data.strip().upper()
        if direction in ["LEFT", "RIGHT", "STRAIGHT"]:
            self.turn_direction = direction
            self.get_logger().info(f"🧭 Result Direction Updated: [{self.turn_direction}]")

    def camera_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_camera_frame = frame
        except Exception as e:
            self.get_logger().error(f"Error decoding image: {e}")

    def turning_parabola_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.parabola_detected = data.get("lane_detected", False)
            if self.parabola_detected:
                self.delta_d = abs(data.get("delta_d", float('inf')))
                if "direction" in data:
                    self.turn_direction = str(data.get("direction")).strip().upper()
            else:
                self.delta_d = float('inf')
        except Exception as e:
            self.get_logger().error(f"Error parsing parabola payload: {e}")

    def apply_parabola_distance_control(self, resultant_dir):
        upper_bound = self.d_cached + DELTA_TOLERANCE
        lower_bound = max(0.0, self.d_cached - DELTA_TOLERANCE)

        self.target_speed = CRUISE_FORWARD_SPEED

        if self.delta_d > upper_bound:
            dist_error = self.delta_d - upper_bound
            self.target_turn = resultant_dir * min(0.45, KP_STEER_DIST * dist_error)
            return f"PARABOLA: Delta {self.delta_d:.1f} > {upper_bound:.1f} -> Steer Inward"
        elif self.delta_d < lower_bound:
            dist_error = lower_bound - self.delta_d
            self.target_turn = -resultant_dir * min(0.45, KP_STEER_DIST * dist_error)
            return f"PARABOLA: Delta {self.delta_d:.1f} < {lower_bound:.1f} -> Steer Outward"
        else:
            self.target_turn = 0.0
            return f"PARABOLA: Matched ({self.delta_d:.1f} ~= {self.d_cached:.1f}) -> Straight"

    def edge_vectors_callback(self, message):
        img_w = message.image_width if message.image_width > 0 else 640
        img_h = message.image_height if message.image_height > 0 else 480
        rover_center_x = img_w / 2.0
        vector_count = message.vector_count

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

        status_str = ""

        if self.is_driving_enabled:
            resultant_dir = 1.0 if self.turn_direction == "LEFT" else (-1.0 if self.turn_direction == "RIGHT" else 0.0)

            # 1. Parabola Steering Mode
            if self.parabola_detected and self.delta_d != float('inf'):
                self.no_lane_frames = 0
                status_str = self.apply_parabola_distance_control(resultant_dir)

            # 2. Vector Centering Mode
            elif vec_left or vec_right:
                self.no_lane_frames = 0
                if vec_left and vec_right:
                    x_l = vec_left[0].x
                    x_r = vec_right[0].x
                    self.d_cached = (x_r - x_l) / 2.0
                    target_center_x = (x_l + x_r) / 2.0
                elif vec_left:
                    target_center_x = vec_left[0].x + self.d_cached
                else:
                    target_center_x = vec_right[0].x - self.d_cached

                dist_error = target_center_x - rover_center_x
                self.target_turn = -KP_STEER_DIST * dist_error
                self.target_speed = CRUISE_FORWARD_SPEED
                status_str = f"VECTOR TRACKING: Err={dist_error:.1f}px"

            # 3. Blind Junction Turn Fallback (No vectors & No Parabola)
            else:
                self.no_lane_frames += 1
                if self.no_lane_frames >= MAX_LOST_FRAMES_THRESHOLD:
                    self.target_speed = JUNCTION_TURN_SPEED
                    self.target_turn = resultant_dir * JUNCTION_STEER_MAGNITUDE
                    status_str = f"⚠️ JUNCTION BLIND TURN ({self.turn_direction}): Steer={self.target_turn:.2f}"
                else:
                    self.target_speed = CRUISE_FORWARD_SPEED
                    self.target_turn = 0.0
                    status_str = f"COASTING: Searching for lane ({self.no_lane_frames}/{MAX_LOST_FRAMES_THRESHOLD})"

        else:
            self.target_speed, self.target_turn = 0.0, 0.0
            status_str = f"STANDBY ({self.mission_state})"

        self.render_debug_vectors(img_w, img_h, rover_center_x, status_str)

    def publish_drive_commands(self):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        clamped_steer = float(max(-1.0, min(1.0, self.target_turn)))
        msg.axes = [0.0, float(self.target_speed), 0.0, clamped_steer]
        self.publisher_joy.publish(msg)

    def render_debug_vectors(self, img_w, img_h, rover_center_x, status_str):
        if self.latest_camera_frame is not None:
            canvas = cv2.resize(self.latest_camera_frame.copy(), (img_w, img_h))
        else:
            canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        cv2.line(canvas, (int(rover_center_x), 0), (int(rover_center_x), img_h), (128, 128, 128), 1)
        cv2.putText(canvas, f"STATE: {self.mission_state} | {status_str}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', canvas)[1]).tobytes()
            self.publisher_vector_debug.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    # ── SEQUENTIAL MISSION STATE MACHINE (Server -> Patient -> Hospital) ──────
    def qr_detection_callback(self, message):
        if not self.is_driving_enabled and not self.has_scanned_active_qr:
            return

        qr_text = message.data.strip()
        if qr_text != "":
            self.detected_qr_buffer = qr_text
            self.has_scanned_active_qr = True

        elif qr_text == "" and self.has_scanned_active_qr:
            scanned_payload = self.detected_qr_buffer
            self.has_scanned_active_qr = False

            # Stage 1: Arrival at Patient Place
            if self.mission_state == STATE_EN_ROUTE_PATIENT:
                self.get_logger().info(f"📍 ARRIVED AT PATIENT PLACE: '{scanned_payload}'")
                self.send_server_message(text_payload=scanned_payload, ack_flag=0)

                # Transition state machine to head toward hospital
                self.mission_state = STATE_EN_ROUTE_HOSPITAL
                self.get_logger().info("🏥 MISSION UPDATE: Moving to Hospital Place...")
                self.is_driving_enabled = True

            # Stage 2: Arrival at Hospital Place
            elif self.mission_state == STATE_EN_ROUTE_HOSPITAL:
                self.get_logger().info(f"🏥 ARRIVED AT HOSPITAL PLACE: '{scanned_payload}'")
                self.send_server_message(text_payload=scanned_payload, ack_flag=0)

                # Stop vehicle motion & return to Idle
                self.mission_state = STATE_IDLE
                self.is_driving_enabled = False
                self.target_speed, self.target_turn = 0.0, 0.0
                self.get_logger().info("🏁 MISSION COMPLETE! Vehicle stopped.")

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
                self.mission_state = STATE_IDLE

            elif message.msg != "INVALID":
                # Start Mission Sequence: Server sends initial target
                self.current_target = message.msg
                self.mission_state = STATE_EN_ROUTE_PATIENT
                self.is_driving_enabled = True
                self.get_logger().info(f"🟢 SERVER INSTRUCTION RECEIVED: En route to Patient ({self.current_target})")


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