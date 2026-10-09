import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class FollowGapNode(Node):
    """Reactive LiDAR planner for F1TENTH-style autonomous driving."""

    def __init__(self):
        super().__init__('follow_gap_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('forward_fov_deg', 190.0)
        self.declare_parameter('max_lidar_range', 10.0)
        self.declare_parameter('smoothing_window', 5)
        self.declare_parameter('gap_threshold', 1.15)
        self.declare_parameter('min_gap_points', 8)
        self.declare_parameter('car_width', 0.32)
        self.declare_parameter('safety_margin', 0.18)
        self.declare_parameter('disparity_threshold', 0.45)
        self.declare_parameter('bubble_radius', 0.65)
        self.declare_parameter('safety_front_angle_deg', 18.0)
        self.declare_parameter('emergency_stop_distance', 0.55)
        self.declare_parameter('caution_distance', 1.65)
        self.declare_parameter('max_speed', 5.5)
        self.declare_parameter('min_speed', 1.0)
        self.declare_parameter('corner_speed', 2.2)
        self.declare_parameter('max_steering_angle', 0.40)
        self.declare_parameter('steering_smoothing', 0.35)
        self.declare_parameter('centerline_bias', 0.18)
        self.declare_parameter('target_distance', 4.0)
        self.declare_parameter('recovery_speed', 0.7)
        self.declare_parameter('recovery_reverse_speed', -0.6)
        self.declare_parameter('reverse_clearance', 0.38)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.forward_fov = math.radians(self.get_parameter('forward_fov_deg').value)
        self.max_lidar_range = self.get_parameter('max_lidar_range').value
        self.smoothing_window = int(self.get_parameter('smoothing_window').value)
        self.gap_threshold = self.get_parameter('gap_threshold').value
        self.min_gap_points = int(self.get_parameter('min_gap_points').value)
        self.car_width = self.get_parameter('car_width').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.disparity_threshold = self.get_parameter('disparity_threshold').value
        self.bubble_radius = self.get_parameter('bubble_radius').value
        self.safety_front_angle = math.radians(self.get_parameter('safety_front_angle_deg').value)
        self.emergency_stop_distance = self.get_parameter('emergency_stop_distance').value
        self.caution_distance = self.get_parameter('caution_distance').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value
        self.corner_speed = self.get_parameter('corner_speed').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.steering_smoothing = self.get_parameter('steering_smoothing').value
        self.centerline_bias = self.get_parameter('centerline_bias').value
        self.target_distance = self.get_parameter('target_distance').value
        self.recovery_speed = self.get_parameter('recovery_speed').value
        self.recovery_reverse_speed = self.get_parameter('recovery_reverse_speed').value
        self.reverse_clearance = self.get_parameter('reverse_clearance').value

        self.prev_steering = 0.0
        self.last_status_time = 0.0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'Follow-the-Gap planner ready: {self.scan_topic} -> {self.drive_topic}'
        )

    def scan_callback(self, scan_msg):
        ranges, angles = self._usable_scan(scan_msg)
        if ranges.size == 0:
            self._publish_drive(0.0, 0.0)
            return

        front_clearance = self._front_clearance(ranges, angles)

        planned_ranges = ranges.copy()
        self._extend_disparities(planned_ranges, scan_msg.angle_increment)
        self._apply_safety_bubble(planned_ranges, scan_msg.angle_increment)

        gap_start, gap_end = self._best_gap(planned_ranges)
        if gap_start is None:
            self._recover(ranges, angles, front_clearance, 'NO GAP')
            return

        if front_clearance < self.emergency_stop_distance:
            self._recover(ranges, angles, front_clearance, 'CLOSE')
            return

        target_idx = self._best_point(planned_ranges, angles, gap_start, gap_end)
        steering = float(np.clip(
            angles[target_idx],
            -self.max_steering_angle,
            self.max_steering_angle,
        ))
        steering = self._smooth_steering(steering)
        speed = self._speed_for(steering, front_clearance, planned_ranges[target_idx])

        self._publish_drive(speed, steering)
        self._log_status('RUN', speed, steering, front_clearance)

    def _usable_scan(self, scan_msg):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        angles = scan_msg.angle_min + np.arange(ranges.size) * scan_msg.angle_increment

        finite_ranges = np.where(np.isfinite(ranges), ranges, self.max_lidar_range)
        finite_ranges = np.clip(finite_ranges, scan_msg.range_min, self.max_lidar_range)

        half_fov = self.forward_fov / 2.0
        forward_mask = np.abs(angles) <= half_fov
        ranges = finite_ranges[forward_mask]
        angles = angles[forward_mask]

        if self.smoothing_window > 1 and ranges.size >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window, dtype=np.float32) / self.smoothing_window
            pad_width = self.smoothing_window // 2
            padded_ranges = np.pad(ranges, pad_width, mode='edge')
            ranges = np.convolve(padded_ranges, kernel, mode='valid')[:ranges.size]

        return ranges, angles

    def _front_clearance(self, ranges, angles):
        front_mask = np.abs(angles) <= self.safety_front_angle
        if not np.any(front_mask):
            return self.max_lidar_range
        return float(np.min(ranges[front_mask]))

    def _apply_safety_bubble(self, ranges, angle_increment):
        closest_idx = int(np.argmin(ranges))
        closest_range = max(float(ranges[closest_idx]), 0.05)
        if closest_range > self.caution_distance:
            return
        bubble_angle = math.atan2(self.bubble_radius, closest_range)
        bubble_points = max(1, int(math.ceil(bubble_angle / abs(angle_increment))))
        start = max(0, closest_idx - bubble_points)
        end = min(ranges.size, closest_idx + bubble_points + 1)
        ranges[start:end] = 0.0

    def _extend_disparities(self, ranges, angle_increment):
        inflation_width = self.car_width / 2.0 + self.safety_margin
        original = ranges.copy()

        for idx in range(ranges.size - 1):
            left = original[idx]
            right = original[idx + 1]
            if abs(left - right) < self.disparity_threshold:
                continue

            closer_range = max(min(left, right), 0.05)
            extend_angle = math.atan2(inflation_width, closer_range)
            extend_points = max(1, int(math.ceil(extend_angle / abs(angle_increment))))

            if left < right:
                start = idx + 1
                end = min(ranges.size, idx + 1 + extend_points)
            else:
                start = max(0, idx - extend_points + 1)
                end = idx + 1

            ranges[start:end] = np.minimum(ranges[start:end], closer_range)

    def _best_gap(self, ranges):
        for threshold in (self.gap_threshold, self.gap_threshold * 0.75, self.gap_threshold * 0.5):
            gap = self._largest_gap(ranges, threshold)
            if gap[0] is not None:
                return gap
        return None, None

    def _largest_gap(self, ranges, threshold):
        free_space = ranges > threshold
        best_start = None
        best_end = None
        best_score = -1.0
        start = None

        for idx, is_free in enumerate(free_space):
            if is_free and start is None:
                start = idx
            elif not is_free and start is not None:
                best_start, best_end, best_score = self._score_gap(
                    ranges,
                    start,
                    idx - 1,
                    best_start,
                    best_end,
                    best_score,
                )
                start = None

        if start is not None:
            best_start, best_end, best_score = self._score_gap(
                ranges,
                start,
                ranges.size - 1,
                best_start,
                best_end,
                best_score,
            )

        return best_start, best_end

    def _score_gap(self, ranges, start, end, best_start, best_end, best_score):
        if end - start + 1 < self.min_gap_points:
            return best_start, best_end, best_score

        score = self._gap_score(ranges, start, end)
        if score > best_score:
            return start, end, score
        return best_start, best_end, best_score

    def _gap_score(self, ranges, start, end):
        width = end - start + 1
        mean_depth = float(np.mean(np.minimum(ranges[start:end + 1], self.target_distance)))
        return width * mean_depth

    def _best_point(self, ranges, angles, start, end):
        gap_ranges = ranges[start:end + 1]
        gap_angles = angles[start:end + 1]

        capped_range = np.minimum(gap_ranges, self.target_distance)
        clearance_score = capped_range / max(self.target_distance, 0.01)
        heading_score = np.cos(gap_angles)
        continuity_score = 1.0 - np.minimum(
            np.abs(gap_angles - self.prev_steering) / max(self.max_steering_angle * 2.0, 0.01),
            1.0,
        )
        center_score = 1.0 - np.minimum(np.abs(gap_angles) / max(self.forward_fov / 2.0, 0.01), 1.0)

        scores = (
            2.4 * clearance_score
            + 1.0 * heading_score
            + 0.8 * continuity_score
            + self.centerline_bias * center_score
        )
        return start + int(np.argmax(scores))

    def _smooth_steering(self, steering):
        alpha = float(np.clip(self.steering_smoothing, 0.0, 1.0))
        smoothed = alpha * self.prev_steering + (1.0 - alpha) * steering
        self.prev_steering = smoothed
        return smoothed

    def _speed_for(self, steering, front_clearance, target_clearance):
        steer_ratio = min(abs(steering) / self.max_steering_angle, 1.0)
        speed = self.max_speed - steer_ratio * (self.max_speed - self.corner_speed)

        if front_clearance < self.caution_distance:
            caution_ratio = max(
                0.0,
                (front_clearance - self.emergency_stop_distance)
                / (self.caution_distance - self.emergency_stop_distance),
            )
            speed = min(speed, self.min_speed + caution_ratio * (self.corner_speed - self.min_speed))

        if target_clearance < self.caution_distance:
            speed = min(speed, self.corner_speed)

        return float(np.clip(speed, 0.0, self.max_speed))

    def _recover(self, ranges, angles, front_clearance, mode):
        left_mask = angles > math.radians(20.0)
        right_mask = angles < math.radians(-20.0)
        left_clearance = self._mean_clearance(ranges, left_mask)
        right_clearance = self._mean_clearance(ranges, right_mask)

        if left_clearance >= right_clearance:
            steering = self.max_steering_angle
        else:
            steering = -self.max_steering_angle

        if front_clearance < self.reverse_clearance:
            speed = self.recovery_reverse_speed
            steering = -steering
        else:
            speed = self.recovery_speed

        steering = self._smooth_steering(steering)
        self._publish_drive(speed, steering)
        self._log_status(mode, speed, steering, front_clearance)

    def _mean_clearance(self, ranges, mask):
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.minimum(ranges[mask], self.target_distance)))

    def _publish_drive(self, speed, steering):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(drive_msg)

    def _log_status(self, mode, speed, steering, clearance):
        now = time.monotonic()
        if now - self.last_status_time < 1.0:
            return
        self.last_status_time = now
        self.get_logger().info(
            f'{mode} speed={speed:.2f} steer={steering:+.3f} front={clearance:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FollowGapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
