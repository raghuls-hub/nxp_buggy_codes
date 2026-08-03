# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

import cv2
import numpy as np
import json

QOS_PROFILE_DEFAULT = 10

# ── ROI & Partition Parameters ────────────────────────────────────────────────
ROI_TOP_FRAC       = 0.40  # Horizon start for straight vectors
ROI_BOTTOM_FRAC    = 0.90  # Horizon end for straight vectors

# Parabola Lower ROI Parameters (Filters top noise)
PARABOLA_ROI_TOP   = 0.55  # Lower 45% of image for curve extraction
PARABOLA_ROI_BOT   = 0.92  # Bottom limit
BORDER_STRIP_PX    = 5

MIN_CONTOUR_AREA   = 150   # Minimum px² contour area
MIN_CONTOUR_HEIGHT = 12    # Minimum vertical span in pixels

# Color Segmentation Thresholds for Black Lines
LANE_BLACK_HSV_LOWER = np.array([0, 0, 0])
LANE_BLACK_HSV_UPPER = np.array([180, 255, 95])
LAB_L_MAX_THRESHOLD  = 85

# Polyline & Parabola Parameters
SLIDING_STRIPS       = 10
MIN_POLY_POINTS      = 4
POLY_SAMPLE_POINTS   = 10
LOOKAHEAD_Y_PIXEL    = 340 # Y evaluation point for turning parabola

BUGGY_ID = 1


def contour_vertical_span(contour):
    y = contour[:, 0, 1]
    return int(np.max(y)) - int(np.min(y))


def get_centroid_x(contour):
    return float(np.mean(contour[:, 0, 0]))


def contour_centroids_by_strip(contour, n_bins: int):
    """Extracts strip centroids along a contour for clean curve fitting."""
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
    ROS 2 Perception Node:
    - Dual-lane edge vector extraction for straight driving (/edge_vectors, /edge_curves)
    - Dedicated Parabola Extractor using lower ROI contours (/turning_parabola)
    - Fully gated by Server Communication start state.
    """
    def __init__(self):
        super().__init__('edge_vectors')

        # Control & State Flags
        self.is_driving_enabled = False
        self.current_mode = "DUAL_CENTERING"
        self.turn_direction = "LEFT"

        # Publishers
        self.publisher_vectors = self.create_publisher(EdgeVectors, '/edge_vectors', QOS_PROFILE_DEFAULT)
        self.publisher_curves = self.create_publisher(String, '/edge_curves', QOS_PROFILE_DEFAULT)
        self.publisher_parabola = self.create_publisher(String, '/turning_parabola', QOS_PROFILE_DEFAULT)
        self.publisher_thresh = self.create_publisher(CompressedImage, '/debug_images/thresh_image', QOS_PROFILE_DEFAULT)
        self.publisher_debug = self.create_publisher(CompressedImage, '/debug_images/vector_image', QOS_PROFILE_DEFAULT)
        self.publisher_parabola_debug = self.create_publisher(CompressedImage, '/debug_images/parabola_curve', QOS_PROFILE_DEFAULT)

        # Subscriptions
        self.subscription_image = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.image_callback, QOS_PROFILE_DEFAULT)
        self.subscription_mode = self.create_subscription(String, '/driving_mode', self.mode_callback, QOS_PROFILE_DEFAULT)
        self.subscription_sign = self.create_subscription(String, '/sign_board_detection', self.sign_callback, QOS_PROFILE_DEFAULT)
        self.subscription_server = self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_callback, QOS_PROFILE_DEFAULT)

        self.get_logger().info("🏎️ Edge Vector & Parabola Node Active [Awaiting Server Start...]")

    def server_callback(self, msg):
        """Gates all perception until active server command is received."""
        if msg.dest == BUGGY_ID and msg.msg not in ["", "OK", "INVALID"]:
            if not self.is_driving_enabled:
                self.is_driving_enabled = True
                self.get_logger().info("🟢 SERVER START RECEIVED: Perception & Detection Enabled!")
        elif msg.dest == BUGGY_ID and msg.msg == "OK":
            self.is_driving_enabled = False
            self.get_logger().info("🔴 SERVER STOP RECEIVED: Perception Paused.")

    def mode_callback(self, msg):
        self.current_mode = msg.data.strip().upper()

    def sign_callback(self, msg):
        # Ignore sign boards if server has not enabled driving
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

    def process_turning_parabola(self, cv_image, h_full, w_full):
        """Extracts parabolic curve equation using LOWER ROI contours during TURNING mode."""
        if self.current_mode != "TURNING":
            payload = json.dumps({"lane_detected": False, "reason": "NOT_IN_TURNING_MODE"})
            self.publisher_parabola.publish(String(data=payload))
            return

        # Terminal Log: Currently Active Topic
        self.get_logger().info("📡 [ACTIVE TOPIC: /turning_parabola] Processing Parabolic Curve...")

        try:
            # 1. Focus strictly on lower ROI to eliminate horizon noise
            p_top_y = int(h_full * PARABOLA_ROI_TOP)
            p_bot_y = int(h_full * PARABOLA_ROI_BOT)
            p_roi = cv_image[p_top_y:p_bot_y, :].copy()
            roi_h, roi_w = p_roi.shape[:2]

            # 2. Thresholding for Black Lane Line
            hsv = cv2.cvtColor(p_roi, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv, LANE_BLACK_HSV_LOWER, LANE_BLACK_HSV_UPPER)
            lab = cv2.cvtColor(p_roi, cv2.COLOR_BGR2LAB)
            _, mask_lab = cv2.threshold(lab[:, :, 0], LAB_L_MAX_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.bitwise_and(mask_hsv, mask_lab)

            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3)

            # 3. Partition Mask based on turn direction (Left 60% vs Right 60%)
            dir_mask = np.zeros_like(thresh)
            if self.turn_direction == "LEFT":
                dir_mask[:, 0:int(roi_w * 0.65)] = 255
            else:
                dir_mask[:, int(roi_w * 0.35):roi_w] = 255

            active_thresh = cv2.bitwise_and(thresh, dir_mask)

            # 4. Find Lane Contours
            cnts, _ = cv2.findContours(active_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_CONTOUR_AREA and contour_vertical_span(c) >= MIN_CONTOUR_HEIGHT]

            if not valid_cnts:
                self.get_logger().warn("⚠️ [/turning_parabola] No valid curve contours found in lower ROI.")
                payload = json.dumps({"lane_detected": False, "reason": "NO_VALID_CONTOURS"})
                self.publisher_parabola.publish(String(data=payload))
                return

            # Pick largest contour on turning side
            best_contour = max(valid_cnts, key=cv2.contourArea)

            # Extract clean strip centroids along the contour
            centroids = contour_centroids_by_strip(best_contour, SLIDING_STRIPS)

            if len(centroids) < MIN_POLY_POINTS:
                self.get_logger().warn("⚠️ [/turning_parabola] Insufficient contour strip points for polyfit.")
                payload = json.dumps({"lane_detected": False, "reason": "INSUFFICIENT_POINTS"})
                self.publisher_parabola.publish(String(data=payload))
                return

            # Convert to absolute full image coordinates (y, x)
            ys = np.array([pt[1] + p_top_y for pt in centroids])
            xs = np.array([pt[0] for pt in centroids])

            # Fit x = A*y^2 + B*y + C
            poly_coeffs = np.polyfit(ys, xs, 2)
            A, B, C = float(poly_coeffs[0]), float(poly_coeffs[1]), float(poly_coeffs[2])

            eval_y = min(max(LOOKAHEAD_Y_PIXEL, p_top_y), p_bot_y)
            target_x_lane = (A * (eval_y ** 2)) + (B * eval_y) + C

            payload = json.dumps({
                "lane_detected": True,
                "A": A,
                "B": B,
                "C": C,
                "target_x_lane": float(target_x_lane),
                "eval_y": int(eval_y),
                "direction": self.turn_direction
            })
            self.publisher_parabola.publish(String(data=payload))
            self.get_logger().info(f"✅ [/turning_parabola] Parabola FIT OK! A={A:.6f} | Target X={target_x_lane:.1f}")

            # 5. Draw Fitted Parabola Curve on Overlay
            parabola_debug_img = cv_image.copy()
            y_pts = np.linspace(p_top_y, p_bot_y, 40)
            x_pts = (A * (y_pts ** 2)) + (B * y_pts) + C

            curve_points = []
            for x_val, y_val in zip(x_pts, y_pts):
                if 0 <= x_val < w_full:
                    curve_points.append([int(x_val), int(y_val)])

            if len(curve_points) > 1:
                cv2.polylines(parabola_debug_img, [np.array(curve_points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=4)

            # Draw evaluated point and text
            cv2.circle(parabola_debug_img, (int(target_x_lane), eval_y), 8, (0, 0, 255), -1)
            cv2.putText(parabola_debug_img, f"TOPIC: /turning_parabola | A: {A:.6f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            self._publish_compressed(self.publisher_parabola_debug, parabola_debug_img)

        except Exception as e:
            self.get_logger().error(f"Error in parabola processing: {e}")

    def image_callback(self, message):
        # Server Gating Check
        if not self.is_driving_enabled:
            return

        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            h_full, w_full = cv_image.shape[:2]

            # ── 1. TURNING MODE: Process Parabola Curve ──────────────────────
            if self.current_mode == "TURNING":
                self.process_turning_parabola(cv_image, h_full, w_full)

            # ── 2. DUAL CENTERING MODE: Process Straight Edge Vectors ─────────
            else:
                self.get_logger().info("📡 [ACTIVE TOPIC: /edge_vectors] Processing Straight Dual Lanes...")

                x_lo, x_hi = BORDER_STRIP_PX, w_full - BORDER_STRIP_PX
                roi_top_y = int(h_full * ROI_TOP_FRAC)
                roi_bot_y = int(h_full * ROI_BOTTOM_FRAC)

                roi = cv_image[roi_top_y:roi_bot_y, x_lo:x_hi].copy()
                roi_h, roi_w = roi.shape[:2]
                mid_x = roi_w / 2.0
                roi_mid_y = int(roi_h / 2.0)

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_hsv = cv2.inRange(hsv, LANE_BLACK_HSV_LOWER, LANE_BLACK_HSV_UPPER)
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                _, mask_lab = cv2.threshold(lab[:, :, 0], LAB_L_MAX_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

                thresh = cv2.bitwise_and(mask_hsv, mask_lab)
                k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3)

                bottom_thresh = thresh[roi_mid_y:roi_h, :]
                top_thresh = thresh[0:roi_mid_y, :]

                cnts_bot, _ = cv2.findContours(bottom_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts_top, _ = cv2.findContours(top_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                curve_data = {'left': {}, 'right': {}}
                vectors = []

                for side in ['left', 'right']:
                    bot_partition = [c for c in cnts_bot if (get_centroid_x(c) < mid_x if side == 'left' else get_centroid_x(c) >= mid_x)]
                    top_partition = [c for c in cnts_top if (get_centroid_x(c) < mid_x if side == 'left' else get_centroid_x(c) >= mid_x)]

                    bot_centroids = []
                    if bot_partition:
                        c_bot_best = max(bot_partition, key=contour_vertical_span)
                        bot_centroids = [(p[0], p[1] + roi_mid_y) for p in contour_centroids_by_strip(c_bot_best, 6)]

                    top_centroids = []
                    if top_partition:
                        c_top_best = max(top_partition, key=contour_vertical_span)
                        top_centroids = contour_centroids_by_strip(c_top_best, 6)

                    top_point_count = len(top_centroids)
                    total_centroids = top_centroids + bot_centroids

                    if len(total_centroids) >= MIN_POLY_POINTS:
                        ys = np.array([p[1] for p in total_centroids])
                        xs = np.array([p[0] for p in total_centroids])

                        coeffs = np.polyfit(ys, xs, 2)
                        sample_ys = np.linspace(0, roi_h - 1, POLY_SAMPLE_POINTS)
                        sample_xs = np.polyval(coeffs, sample_ys)

                        is_valid = all(x < mid_x if side == 'left' else x >= mid_x for x in sample_xs)

                        if is_valid:
                            vec_bot = (sample_xs[-1], sample_ys[-1] + roi_top_y)
                            vec_top = (sample_xs[0], sample_ys[0] + roi_top_y)
                            vectors.append((side, vec_bot, vec_top))

                            curve_data[side] = {
                                'A': float(coeffs[0]),
                                'B': float(coeffs[1]),
                                'C': float(coeffs[2]),
                                'top_points': top_point_count,
                                'bot_points': len(bot_centroids)
                            }

                vm = EdgeVectors()
                vm.image_height, vm.image_width = h_full, w_full
                vm.vector_count = len(vectors)

                for i, vec in enumerate(vectors):
                    target_vec = vm.vector_1 if i == 0 else vm.vector_2
                    target_vec[0].x, target_vec[0].y = map(float, vec[1])
                    target_vec[1].x, target_vec[1].y = map(float, vec[2])

                self.publisher_vectors.publish(vm)

                cm = String()
                cm.data = json.dumps(curve_data)
                self.publisher_curves.publish(cm)

                # Debug Overlay
                debug_img = roi.copy()
                cv2.line(debug_img, (0, roi_mid_y), (roi_w, roi_mid_y), (0, 255, 255), 1)
                cv2.line(debug_img, (int(mid_x), 0), (int(mid_x), roi_h), (255, 255, 255), 1)
                cv2.putText(debug_img, "TOPIC: /edge_vectors", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                debug_full = cv_image.copy()
                debug_full[roi_top_y:roi_bot_y, x_lo:x_hi] = debug_img
                self._publish_compressed(self.publisher_thresh, thresh)
                self._publish_compressed(self.publisher_debug, debug_full)

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")


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