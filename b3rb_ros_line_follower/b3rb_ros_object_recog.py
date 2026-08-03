# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import cv2
import numpy as np
import sys
from collections import Counter

QOS_PROFILE_DEFAULT = 10

WARP_SIZE = 200                  # Normalized ROI canvas size
MIN_WHITE_AREA_THRESHOLD = 250   # Min white pixel area to classify as FRONT board
MIN_GREEN_CONTOUR_AREA = 1000    # Green contour area threshold
REQUIRED_CONTINUOUS_FRAMES = 8   # Frames required to lock onto a front board

# Spatial ROI Boundaries (Fraction of Frame Dimensions)
ROI_ENTRY_FRAC  = 0.05           # Top Y boundary
ROI_EXIT_FRAC   = 0.90           # Bottom Y boundary
ROI_X_LEFT_FRAC = 0.22           # Left X boundary (Ignores left-lane signboards)
ROI_X_RIGHT_FRAC= 0.78           # Right X boundary (Ignores right-lane signboards)


class ObjectRecognizer(Node):
    """
    ROS 2 Node for Junction Signboard Detection:
    - Constrained X/Y ROI bounding box to focus exclusively on current lane signboards.
    - Pauses detection automatically when buggy enters TURNING mode.
    - Locks onto front-facing boards and publishes direction payload upon crossing.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Mode Tracking
        self.current_driving_mode = "DUAL_CENTERING"

        # Detection State Machine Flags
        self.continuous_frame_count = 0
        self.is_board_locked = False
        self.last_centroid_y = 0.0

        # Internal Direction Storage Buffer
        self.direction_votes = []

        # Publishers
        self.publisher_signs = self.create_publisher(
            String, '/sign_board_detection', QOS_PROFILE_DEFAULT)
        self.publisher_debug_img = self.create_publisher(
            CompressedImage, '/debug_images/sign_board_contours', QOS_PROFILE_DEFAULT)

        # Subscriptions
        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/driving_mode', self.driving_mode_callback, QOS_PROFILE_DEFAULT)

        self.get_logger().info("🚦 ROI-Constrained Signboard Recognizer Operational.")

    def driving_mode_callback(self, msg):
        """Disables/Enables signboard detection depending on driving mode."""
        new_mode = msg.data.strip().upper()
        if new_mode != self.current_driving_mode:
            self.current_driving_mode = new_mode
            if self.current_driving_mode == "TURNING":
                self.get_logger().info("⏸️ TURNING mode active: Signboard detection PAUSED.")
                self.reset_detection_state()
            else:
                self.get_logger().info("▶️ DUAL_CENTERING mode active: Signboard detection RESUMED.")

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_signboard_roi(self, image, cnt):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        pts = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(cnt))
        rect = self.order_points(pts)

        dst = np.array([
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))

    def classify_arrow_direction(self, arrow_cnt, w_box, h_box):
        if h_box > 1.2 * w_box:
            return "STRAIGHT"

        if w_box >= h_box:
            M = cv2.moments(arrow_cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                return "LEFT" if cx < (w_box / 2.0) else "RIGHT"

        return "STRAIGHT"

    def inspect_and_extract_board(self, cv_image, cnt):
        try:
            warped_roi = self.warp_signboard_roi(cv_image, cnt)
            gray = cv2.cvtColor(warped_roi, cv2.COLOR_BGR2GRAY)
            _, white_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

            total_white_area = cv2.countNonZero(white_mask)

            if total_white_area < MIN_WHITE_AREA_THRESHOLD:
                return False, None, warped_roi

            white_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            arrows = []
            for w_cnt in white_cnts:
                if cv2.contourArea(w_cnt) > 40:
                    wx, wy, ww, wh = cv2.boundingRect(w_cnt)
                    if wy >= (WARP_SIZE * 0.35):
                        direction = self.classify_arrow_direction(w_cnt, ww, wh)
                        arrows.append({'dir': direction, 'x': wx})

            if arrows:
                arrows.sort(key=lambda a: a['x'])
                return True, arrows[0]['dir'], warped_roi

            return True, "STRAIGHT", warped_roi

        except Exception as e:
            return False, None, None

    def publish_resultant_direction(self):
        if not self.direction_votes:
            self.reset_detection_state()
            return

        counts = Counter(self.direction_votes)
        most_common, freq = counts.most_common(1)[0]
        confidence = (freq / len(self.direction_votes)) * 100.0

        msg = String()
        msg.data = most_common
        self.publisher_signs.publish(msg)

        banner = (
            "\n" + "═" * 68 + "\n"
            f"  🎯 FRONT SIGNBOARD PASSED -> RESULTANT DIRECTION PUBLISHED\n"
            f"  🧭 Direction Payload : [{most_common}]\n"
            f"  📊 Voting Consensus  : {freq}/{len(self.direction_votes)} samples ({confidence:.1f}%)\n"
            + "═" * 68 + "\n"
        )
        print(banner, flush=True)
        sys.stdout.flush()
        self.get_logger().info(f"Published direction [{most_common}] to topic /sign_board_detection")

        self.reset_detection_state()

    def reset_detection_state(self):
        self.continuous_frame_count = 0
        self.is_board_locked = False
        self.direction_votes.clear()

    def image_callback(self, message):
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            h_full, w_full = cv_image.shape[:2]
            debug_img = cv_image.copy()

            # Define Spatial ROI Window (Center-Lane Focused)
            roi_top_y   = int(h_full * ROI_ENTRY_FRAC)
            roi_bot_y   = int(h_full * ROI_EXIT_FRAC)
            roi_left_x  = int(w_full * ROI_X_LEFT_FRAC)
            roi_right_x = int(w_full * ROI_X_RIGHT_FRAC)

            # Draw ROI Bounding Window
            cv2.rectangle(debug_img, (roi_left_x, roi_top_y), (roi_right_x, roi_bot_y), (255, 255, 0), 2)

            # ── IF BUGGY IS TURNING: SUPPRESS DETECTION ──
            if self.current_driving_mode == "TURNING":
                cv2.putText(debug_img, "DETECTION PAUSED (TURNING MODE)", (roi_left_x + 10, roi_top_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                self.publish_debug_image(debug_img)
                return

            # Color Thresholding
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_GREEN_CONTOUR_AREA]

            front_boards = []
            rear_board_count = 0

            for cnt in valid_contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = x + (bw / 2.0)
                cy = y + (bh / 2.0)

                # Filter strictly within current lane X and Y spatial bounds
                if (roi_top_y <= cy <= roi_bot_y) and (roi_left_x <= cx <= roi_right_x):
                    is_front, dir_extracted, warped_roi = self.inspect_and_extract_board(cv_image, cnt)

                    if is_front:
                        front_boards.append((cnt, dir_extracted, warped_roi, cy, (x, y, bw, bh)))
                    else:
                        rear_board_count += 1
                        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), 1)

            if front_boards:
                best_front = max(front_boards, key=lambda item: cv2.contourArea(item[0]))
                cnt, dir_res, warped_roi, cy, (bx, by, bw, bh) = best_front
                self.last_centroid_y = cy

                cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)

                if not self.is_board_locked:
                    self.continuous_frame_count += 1
                    self.get_logger().info(
                        f"STAGE 1: Front Board Locked ({self.continuous_frame_count}/{REQUIRED_CONTINUOUS_FRAMES})")

                    if self.continuous_frame_count >= REQUIRED_CONTINUOUS_FRAMES:
                        self.is_board_locked = True
                        self.get_logger().info("🔒 STAGE 1 LOCKED: Front signboard verified!")

                if self.is_board_locked:
                    if dir_res is not None:
                        self.direction_votes.append(dir_res)
                        cv2.putText(debug_img, f"FRONT: {dir_res} ({len(self.direction_votes)} votes)",
                                    (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if warped_roi is not None:
                        debug_img[0:WARP_SIZE, 0:WARP_SIZE] = warped_roi

            else:
                if not self.is_board_locked and self.continuous_frame_count > 0:
                    self.continuous_frame_count -= 1

                if self.is_board_locked:
                    self.get_logger().info("🚪 STAGE 3 TRIGGERED: Front signboard crossed!")
                    self.publish_resultant_direction()

            status_text = f"LOCK: {self.continuous_frame_count}/{REQUIRED_CONTINUOUS_FRAMES}" if not self.is_board_locked else f"LOCKED ({len(self.direction_votes)} votes)"
            cv2.putText(debug_img, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if self.is_board_locked else (0, 165, 255), 2)

            self.publish_debug_image(debug_img)

        except Exception as e:
            self.get_logger().error(f"Error in signboard recognizer: {e}")

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