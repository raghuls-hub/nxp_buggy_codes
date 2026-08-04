# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
# Parabola perception restricted to bottom 50% image ROI with midpoint partition check

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

import cv2
import numpy as np
import json

QOS_PROFILE_DEFAULT = 10

# ── ROI Parameters (Restricted to Lower 50% Image) ────────────────────────────
ROI_TOP_FRAC       = 0.50   # Top limit: Start at 50% of full image height
ROI_BOTTOM_FRAC    = 0.98   # Bottom limit: Cut off near bumper (98%)
BORDER_STRIP_PX    = 5      # Side padding clip

SLIDING_STRIPS     = 10     # Number of horizontal strips within lower 50%
MIN_CONTOUR_AREA   = 50     # Contour area threshold for lower frame extraction
MIN_CONTOUR_SPAN   = 8      # Vertical span threshold
MIN_POLY_POINTS    = 4      # Minimum centroid points required to fit quadratic curve

# Color Segmentation Thresholds for Black Lines
LANE_BLACK_HSV_LOWER = np.array([0, 0, 0])
LANE_BLACK_HSV_UPPER = np.array([180, 255, 100])
LAB_L_MAX_THRESHOLD  = 90

BUGGY_ID = 1


def contour_vertical_span(contour):
    y = contour[:, 0, 1]
    return int(np.max(y)) - int(np.min(y))


def get_centroid_x(contour):
    return float(np.mean(contour[:, 0, 0]))


def extract_parabola_centroids(contour, n_bins: int):
    """
    Splits lower-ROI contours into horizontal strips to extract clean curve points (x, y).
    """
    pts_x = contour[:, 0, 0].astype(float)
    pts_y = contour[:, 0, 1].astype(float)
    y_min, y_max = pts_y.min(), pts_y.max()
    span = y_max - y_min
    if span < 1:
        return []
    bin_h = span / n_bins
    centroids = []
    for i in range(n_bins):
        lo, hi = y_min + i * bin_h, y_min + (i + 1) * bin_h
        mask = (pts_y >= lo) & (pts_y < hi)
        if mask.sum() > 0:
            centroids.append((float(np.mean(pts_x[mask])), float(np.mean(pts_y[mask]))))
    centroids.sort(key=lambda p: p[1])
    return centroids


class EdgeVectorsNode(Node):
    """
    ROS 2 Node enforcing:
    1. Parabola extraction ONLY on bottom 50% of image frame.
    2. Strict frame partitioning: Left lane start < mid_x, Right lane start >= mid_x.
    """
    def __init__(self):
        super().__init__('edge_vectors')

        # Control & State
        self.is_driving_enabled = False
        self.current_mode = "DUAL_CENTERING"
        self.turn_direction = "LEFT"

        # Publishers
        self.publisher_vectors = self.create_publisher(EdgeVectors, '/edge_vectors', QOS_PROFILE_DEFAULT)
        self.publisher_curves = self.create_publisher(String, '/edge_curves', QOS_PROFILE_DEFAULT)
        self.publisher_parabola = self.create_publisher(String, '/turning_parabola', QOS_PROFILE_DEFAULT)
        self.publisher_thresh = self.create_publisher(CompressedImage, '/debug_images/thresh_image', QOS_PROFILE_DEFAULT)
        self.publisher_debug = self.create_publisher(CompressedImage, '/debug_images/vector_image', QOS_PROFILE_DEFAULT)

        # Subscriptions
        self.subscription_image = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.image_callback, QOS_PROFILE_DEFAULT)
        self.subscription_mode = self.create_subscription(String, '/driving_mode', self.mode_callback, QOS_PROFILE_DEFAULT)
        self.subscription_sign = self.create_subscription(String, '/sign_board_detection', self.sign_callback, QOS_PROFILE_DEFAULT)
        self.subscription_server = self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_callback, QOS_PROFILE_DEFAULT)

        self.get_logger().info("🏎️ Lower 50% Parabola Lane Perception Node Active")

    def server_callback(self, msg):
        if msg.dest == BUGGY_ID and msg.msg not in ["", "OK", "INVALID"]:
            if not self.is_driving_enabled:
                self.is_driving_enabled = True
                self.get_logger().info("🟢 SERVER START: Perception Enabled!")
        elif msg.dest == BUGGY_ID and msg.msg == "OK":
            self.is_driving_enabled = False
            self.get_logger().info("🔴 SERVER STOP: Perception Standby.")

    def mode_callback(self, msg):
        raw_mode = msg.data.strip().upper()
        if "TURNING" in raw_mode:
            self.current_mode = "TURNING"
        else:
            self.current_mode = "DUAL_CENTERING"

    def sign_callback(self, msg):
        if not self.is_driving_enabled:
            return
        direction = msg.data.strip().upper()
        if direction in ["LEFT", "RIGHT"]:
            self.turn_direction = direction

    def _publish_compressed(self, publisher, cv_img):
        try:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', cv_img)[1]).tobytes()
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Image publish error: {e}")

    def image_callback(self, message):
        if not self.is_driving_enabled:
            return

        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            h_full, w_full = cv_image.shape[:2]
            frame_center_x = w_full / 2.0

            # ── 1. Restrict ROI to Lower 50% of Frame ────────────────────────
            x_lo, x_hi = BORDER_STRIP_PX, w_full - BORDER_STRIP_PX
            roi_top_y = int(h_full * ROI_TOP_FRAC)       # 50% y-level
            roi_bot_y = int(h_full * ROI_BOTTOM_FRAC)    # 98% y-level

            roi = cv_image[roi_top_y:roi_bot_y, x_lo:x_hi].copy()
            roi_h, roi_w = roi.shape[:2]
            roi_mid_x = roi_w / 2.0

            # ── 2. Segmentation (HSV + LAB Black Line Thresholding) ───────────
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv, LANE_BLACK_HSV_LOWER, LANE_BLACK_HSV_UPPER)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            _, mask_lab = cv2.threshold(lab[:, :, 0], LAB_L_MAX_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

            thresh = cv2.bitwise_and(mask_hsv, mask_lab)
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_CONTOUR_AREA and contour_vertical_span(c) >= MIN_CONTOUR_SPAN]

            curve_data = {'left': {}, 'right': {}}
            vectors = []
            debug_canvas = cv_image.copy()

            # Draw visual boundaries on canvas
            cv2.line(debug_canvas, (0, roi_top_y), (w_full, roi_top_y), (255, 255, 0), 1)  # ROI 50% Line
            cv2.line(debug_canvas, (int(frame_center_x), roi_top_y), (int(frame_center_x), h_full), (0, 255, 255), 1)  # Mid partition

            # ── 3. Parabola Fitting in Lower 50% Frame ────────────────────────
            for side in ['left', 'right']:
                side_cnts = [c for c in valid_cnts if (get_centroid_x(c) < roi_mid_x if side == 'left' else get_centroid_x(c) >= roi_mid_x)]

                if not side_cnts:
                    continue

                best_cnt = max(side_cnts, key=cv2.contourArea)
                centroids = extract_parabola_centroids(best_cnt, SLIDING_STRIPS)

                if len(centroids) < MIN_POLY_POINTS:
                    continue

                # Map ROI centroids back to full frame Y and X pixel values
                ys = np.array([pt[1] + roi_top_y for pt in centroids], dtype=np.float64)
                xs = np.array([pt[0] + x_lo for pt in centroids], dtype=np.float64)

                # Parabola equation: x = A*y^2 + B*y + C
                coeffs = np.polyfit(ys, xs, 2)
                A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

                # Evaluate bottom (start) & top endpoints within lower ROI
                eval_y_bot = float(roi_bot_y)
                eval_y_top = float(roi_top_y)
                eval_x_bot = (A * (eval_y_bot ** 2)) + (B * eval_y_bot) + C
                eval_x_top = (A * (eval_y_top ** 2)) + (B * eval_y_top) + C

                # ── Spatial Partition Validation ──────────────────────────────
                if side == 'left' and eval_x_bot >= frame_center_x:
                    self.get_logger().warn("Rejected Left Parabola: Start point crossed right partition.")
                    continue

                if side == 'right' and eval_x_bot < frame_center_x:
                    self.get_logger().warn("Rejected Right Parabola: Start point crossed left partition.")
                    continue
                # ──────────────────────────────────────────────────────────────

                vec_bot = (eval_x_bot, eval_y_bot)
                vec_top = (eval_x_top, eval_y_top)
                vectors.append((side, vec_bot, vec_top))

                curve_data[side] = {'A': A, 'B': B, 'C': C}

                # Render Parabola inside bottom 50%
                y_samples = np.linspace(roi_top_y, roi_bot_y, 25)
                x_samples = (A * (y_samples ** 2)) + (B * y_samples) + C
                pts = np.column_stack((x_samples, y_samples)).astype(np.int32)
                pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w_full)]
                if len(pts) > 1:
                    line_color = (255, 0, 0) if side == 'left' else (0, 0, 255)
                    cv2.polylines(debug_canvas, [pts], isClosed=False, color=line_color, thickness=3)

                # Draw verified starting point marker
                cv2.circle(debug_canvas, (int(eval_x_bot), int(eval_y_bot)), 5, (0, 255, 0), -1)

                # Publish Parabola metadata for turning logic
                if side == self.turn_direction.lower():
                    eval_y_lookahead = roi_top_y + (roi_h * 0.4)
                    target_x_lane = (A * (eval_y_lookahead ** 2)) + (B * eval_y_lookahead) + C
                    delta_d = abs(target_x_lane - frame_center_x)

                    parabola_payload = json.dumps({
                        "lane_detected": True,
                        "A": A,
                        "B": B,
                        "C": C,
                        "target_x_lane": float(target_x_lane),
                        "delta_d": float(delta_d),
                        "eval_y": int(eval_y_lookahead),
                        "direction": self.turn_direction
                    })
                    self.publisher_parabola.publish(String(data=parabola_payload))

            # ── 4. Publish EdgeVectors ────────────────────────────────────────
            vm = EdgeVectors()
            vm.image_height, vm.image_width = h_full, w_full
            vm.vector_count = len(vectors)

            for i, vec in enumerate(vectors):
                target_vec = vm.vector_1 if i == 0 else vm.vector_2
                target_vec[0].x, target_vec[0].y = map(float, vec[1])  # Bottom start
                target_vec[1].x, target_vec[1].y = map(float, vec[2])  # Top end (at 50% image level)

            self.publisher_vectors.publish(vm)

            cm = String()
            cm.data = json.dumps(curve_data)
            self.publisher_curves.publish(cm)

            # Debug Visualizations
            self._publish_compressed(self.publisher_thresh, thresh)
            self._publish_compressed(self.publisher_debug, debug_canvas)

        except Exception as e:
            self.get_logger().error(f"Error in lower 50% image processing: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()