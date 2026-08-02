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
from synapse_msgs.msg import ServerCommunication

import cv2
import numpy as np
from collections import deque, Counter

QOS_PROFILE_DEFAULT = 10
BUGGY_ID = 1

WARP_SIZE = 200                  # 200x200 pixels normalized top-down ROI
MIN_WHITE_AREA_THRESHOLD = 300   # Min cumulative white area to reject REAR_FACE boards

class ObjectRecognizer(Node):
    """
    ROS 2 Node for Upgraded Signboard Detection:
    - Stage A: Perspective Normalization to 200x200 canvas.
    - Stage B: Rear-Facing Signboard Rejection via cumulative white pixel area.
    - Stage C: Sub-panel Geometric Arrow Classification (Centroid Skewness).
    - Stage D: Temporal Voting Buffer (N=10 sliding window, >= 60% majority threshold).
    - Publishes debug visuals to /debug_images/sign_board_contours.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        self.current_target_letter = None
        self.last_emitted_direction = ""

        # Stage D: Temporal Voting Buffer setup
        self.voting_buffer_size = 10
        self.voting_threshold = 0.60
        self.direction_buffer = deque(maxlen=self.voting_buffer_size)

        # ------------------ Publishers ------------------
        self.publisher_signs = self.create_publisher(
            String,
            '/sign_board_detection',
            QOS_PROFILE_DEFAULT)

        self.publisher_debug_img = self.create_publisher(
            CompressedImage,
            '/debug_images/sign_board_contours',
            QOS_PROFILE_DEFAULT)

        # ------------------ Subscriptions ------------------
        self.subscription_image = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_callback,
            QOS_PROFILE_DEFAULT)

        self.get_logger().info("Upgraded Signboard Recognizer (Rear Rejection + Perspective + Voting) Initialized.")

    def server_callback(self, message):
        """Extracts active target letter from server messages."""
        if message.dest == BUGGY_ID and message.msg != "" and message.ack == 0:
            if message.msg not in ["OK", "INVALID", "PARKED"]:
                payload = message.msg.strip()
                self.current_target_letter = payload.split('_')[-1]
                self.get_logger().info(f"🎯 Target letter updated: '{self.current_target_letter}'")

    # ------------------ STAGE A: Perspective Normalization ------------------

    def order_points(self, pts):
        """Orders points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_signboard_roi(self, image, cnt):
        """Applies perspective transformation to produce a 200x200 top-down view."""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            rect_rot = cv2.minAreaRect(cnt)
            pts = cv2.boxPoints(rect_rot)

        rect = self.order_points(pts)

        dst = np.array([
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))
        return warped

    # ------------------ STAGE C: Geometric Arrow Classification ------------------

    def classify_arrow_direction(self, arrow_cnt, w_box, h_box):
        """Classifies arrow direction using geometric centroid skewness."""
        if h_box > 1.2 * w_box:
            return "STRAIGHT"

        if w_box >= h_box:
            M = cv2.moments(arrow_cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                if cx < (w_box / 2.0):
                    return "LEFT"
                else:
                    return "RIGHT"

        return "STRAIGHT"

    def process_warped_signboard(self, warped_roi):
        """Processes 200x200 ROI for rear rejection (Stage B) and arrow classification (Stage C)."""
        gray = cv2.cvtColor(warped_roi, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # STAGE B: Rear-Facing Signboard Rejection
        total_white_area = cv2.countNonZero(white_mask)
        if total_white_area < MIN_WHITE_AREA_THRESHOLD:
            return "REAR_FACE", total_white_area

        # STAGE C: Lower Sub-Panel Arrow Analysis (y >= 100)
        white_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrows = []
        for w_cnt in white_cnts:
            if cv2.contourArea(w_cnt) > 60:
                wx, wy, ww, wh = cv2.boundingRect(w_cnt)
                if wy >= (WARP_SIZE * 0.5):
                    direction = self.classify_arrow_direction(w_cnt, ww, wh)
                    arrows.append({'dir': direction, 'x': wx})

        if arrows:
            arrows.sort(key=lambda a: a['x'])
            return arrows[0]['dir'], total_white_area

        return None, total_white_area

    # ------------------ MAIN CALLBACK (STAGES A - D) ------------------

    def image_callback(self, message):
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                return

            debug_img = cv_image.copy()

            # Green Mask Isolation
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            raw_frame_direction = None
            is_rear_facing = False

            for cnt in contours:
                if cv2.contourArea(cnt) > 5000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                    # Stage A: Warp ROI
                    warped_roi = self.warp_signboard_roi(cv_image, cnt)

                    # Stage B & C: Process ROI
                    direction_res, white_area = self.process_warped_signboard(warped_roi)

                    if direction_res == "REAR_FACE":
                        is_rear_facing = True
                        cv2.putText(debug_img, "REAR FACE REJECTED", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        raw_frame_direction = direction_res

                    # Draw Warped ROI on debug canvas
                    debug_img[0:WARP_SIZE, 0:WARP_SIZE] = warped_roi
                    cv2.rectangle(debug_img, (0, 0), (WARP_SIZE, WARP_SIZE), (255, 255, 0), 2)
                    break

            # Stage D: Temporal Voting Buffer Logic
            if raw_frame_direction and not is_rear_facing:
                self.direction_buffer.append(raw_frame_direction)
            elif is_rear_facing or raw_frame_direction is None:
                if len(self.direction_buffer) > 0:
                    self.direction_buffer.clear()
                    self.last_emitted_direction = ""

            # Majority Voting Consensus Check
            if len(self.direction_buffer) >= (self.voting_buffer_size // 2):
                counts = Counter(self.direction_buffer)
                most_common, freq = counts.most_common(1)[0]
                consensus_ratio = freq / len(self.direction_buffer)

                cv2.putText(debug_img, f"Vote: {most_common} ({freq}/{len(self.direction_buffer)})",
                            (WARP_SIZE + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if consensus_ratio >= self.voting_threshold and most_common != self.last_emitted_direction:
                    self.last_emitted_direction = most_common
                    msg = String()
                    msg.data = most_common
                    self.publisher_signs.publish(msg)
                    self.get_logger().info(f"🚦 Majority Vote Triggered -> Direction: '{most_common}' ({consensus_ratio*100:.1f}%)")

            # Publish Debug Frame
            self.publish_debug_image(debug_img)

        except Exception as e:
            self.get_logger().error(f"Error in signboard processing pipeline: {e}")

    def publish_debug_image(self, cv_img):
        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', cv_img)[1]).tobytes()
            self.publisher_debug_img.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()