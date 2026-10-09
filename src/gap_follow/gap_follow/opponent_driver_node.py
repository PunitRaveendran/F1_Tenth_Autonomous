import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
import numpy as np
import csv
import os
from ament_index_python.packages import get_package_share_directory


class OpponentDriverNode(Node):
    """
    Drives the opponent car along the raceline at a configurable fraction
    of the optimal speed.  Publishes to /opp_drive so the sim bridge
    steps the second agent.

    This is a simplified Pure Pursuit — no interpolation, no lookahead
    boost — just enough to keep the car cleanly on track at moderate speed.
    """

    def __init__(self):
        super().__init__('opponent_driver_node')

        # ── Parameters ────────────────────────────────────────
        _default_raceline = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps', 'raceline.csv')
        self.declare_parameter(
            'raceline_path',
            _default_raceline,
        )
        self.declare_parameter('speed_scale', 0.5)     # fraction of raceline speed
        self.declare_parameter('lookahead_min', 0.8)
        self.declare_parameter('lookahead_max', 1.8)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('steering_max', 0.41)

        self.raceline_path = self.get_parameter('raceline_path').value
        self.speed_scale   = self.get_parameter('speed_scale').value
        self.L_min         = self.get_parameter('lookahead_min').value
        self.L_max         = self.get_parameter('lookahead_max').value
        self.wheelbase     = self.get_parameter('wheelbase').value
        self.steer_max     = self.get_parameter('steering_max').value
        # ──────────────────────────────────────────────────────

        # Load raceline
        self.waypoints = self._load_raceline(self.raceline_path)
        self.n = len(self.waypoints)
        self.get_logger().info(
            f'Opponent driver loaded {self.n} waypoints '
            f'(speed_scale={self.speed_scale:.0%})'
        )

        # Publishers / Subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/opp_drive', 10
        )
        self.marker_pub = self.create_publisher(
            Marker, '/opponent_driver/target', 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/opp_racecar/odom', self.odom_callback, 10
        )

        self.get_logger().info('Opponent driver node ready — waiting for /opp_racecar/odom')

    # ── Load raceline ─────────────────────────────────────────
    def _load_raceline(self, path):
        waypoints = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                try:
                    x  = float(row[1])
                    y  = float(row[2])
                    vx = float(row[5])
                    waypoints.append([x, y, vx])
                except (ValueError, IndexError):
                    continue
        return np.array(waypoints)

    # ── Odom callback — Pure Pursuit ──────────────────────────
    def odom_callback(self, msg):
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))

        speed = abs(msg.twist.twist.linear.x)

        # Adaptive lookahead
        L = np.clip(
            self.L_min + (speed / 8.0) * (self.L_max - self.L_min),
            self.L_min, self.L_max,
        )

        # Nearest waypoint
        dists = np.linalg.norm(
            self.waypoints[:, :2] - np.array([cx, cy]), axis=1
        )
        nearest_idx = int(np.argmin(dists))

        # Find lookahead target — first waypoint beyond L that is in front
        target = None
        for i in range(nearest_idx, nearest_idx + self.n):
            idx = i % self.n
            wp  = self.waypoints[idx]
            d   = np.linalg.norm(wp[:2] - np.array([cx, cy]))
            if d >= L:
                dx_t = wp[0] - cx
                dy_t = wp[1] - cy
                local_x = np.cos(yaw) * dx_t + np.sin(yaw) * dy_t
                if local_x > 0.0:
                    target = wp
                    break

        if target is None:
            target = self.waypoints[(nearest_idx + 10) % self.n]

        # Transform to vehicle frame
        dx = target[0] - cx
        dy = target[1] - cy
        local_x =  np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        # Pure Pursuit steering
        alpha = np.arctan2(local_y, local_x)
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), L)
        steering = np.clip(steering, -self.steer_max, self.steer_max)

        # Speed — use raceline speed scaled down
        target_speed = target[2] * self.speed_scale

        # Safety: slow down if pointing sideways
        if abs(alpha) > np.pi / 4:
            target_speed *= 0.5

        target_speed = np.clip(target_speed, 0.5, 8.0 * self.speed_scale)

        # Publish drive
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering)
        drive_msg.drive.speed = float(target_speed)
        self.drive_pub.publish(drive_msg)

        # Publish target marker for RViz
        self._publish_marker(target[0], target[1])

    def _publish_marker(self, tx, ty):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns = 'opponent_driver'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = tx
        m.pose.position.y = ty
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = 0.25
        m.scale.y = 0.25
        m.scale.z = 0.25
        # Orange color to distinguish from ego (red)
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 1.0
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
