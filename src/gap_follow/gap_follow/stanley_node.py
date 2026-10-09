import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import csv
import os
from ament_index_python.packages import get_package_share_directory

class StanleyNode(Node):
    """
    Stanley controller for precise raceline tracking.
    """
    def __init__(self):
        super().__init__('stanley_node')

        # ── Parameters ───────────────────────────────────────
        _default_raceline = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps', 'raceline.csv')
        self.declare_parameter('raceline_path', _default_raceline)
        self.declare_parameter('k_gain',        3.7769)   
        self.declare_parameter('k_soft',        0.3439)   
        self.declare_parameter('speed_scale',   1.2401)   
        self.declare_parameter('wheelbase',     0.33)
        self.declare_parameter('steering_max',  0.41)
        self.declare_parameter('lookahead',     3)

        self.raceline_path = self.get_parameter('raceline_path').value
        self.k             = self.get_parameter('k_gain').value
        self.k_soft        = self.get_parameter('k_soft').value
        self.speed_scale   = self.get_parameter('speed_scale').value
        self.wheelbase     = self.get_parameter('wheelbase').value
        self.steer_max     = self.get_parameter('steering_max').value
        self.lookahead     = self.get_parameter('lookahead').value
        # ─────────────────────────────────────────────────────

        # Load raceline
        self.waypoints = self.load_raceline(self.raceline_path)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {self.raceline_path}')

        # Publishers / Subscribers
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/stanley/markers', 10)
        self.odom_sub   = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.get_logger().info('Stanley Controller node ready')

    def load_raceline(self, path):
        waypoints = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not row or row[0].startswith('#') or row[0] == 's_m' or len(row) < 6:
                    continue
                try:
                    # s, x, y, psi, kappa, vx, ax
                    x     = float(row[1])
                    y     = float(row[2])
                    kappa = float(row[4])
                    vx    = float(row[5])
                    waypoints.append([x, y, 0.0, vx, kappa])  # Placeholder for psi at idx 2
                except ValueError:
                    continue
                    
        waypoints = np.array(waypoints)
        # Compute track heading (psi) dynamically to avoid CSV rotation issues
        n = len(waypoints)
        for i in range(n):
            next_idx = (i + 1) % n
            dx = waypoints[next_idx, 0] - waypoints[i, 0]
            dy = waypoints[next_idx, 1] - waypoints[i, 1]
            waypoints[i, 2] = np.arctan2(dy, dx)
            
        return waypoints

    def normalize_angle(self, angle):
        """Normalize an angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def odom_callback(self, msg):
        # Current pose (center of gravity / rear axle)
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        # Quaternion to yaw
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))

        # Current speed
        vx = msg.twist.twist.linear.x
        speed = abs(vx)

        # 1. Compute Front Axle Position
        fx = cx + self.wheelbase * np.cos(yaw)
        fy = cy + self.wheelbase * np.sin(yaw)

        # 2. Find nearest waypoint to the front axle
        n = len(self.waypoints)
        dists = np.linalg.norm(self.waypoints[:, :2] - np.array([fx, fy]), axis=1)
        nearest_idx = np.argmin(dists)

        # Target waypoint
        wp_x, wp_y, wp_psi, wp_v, wp_kappa = self.waypoints[nearest_idx]

        # 3. Calculate Heading Error (theta_e)
        # Difference between track heading and car heading
        theta_e = self.normalize_angle(wp_psi - yaw)

        # 4. Calculate Cross-Track Error (e_fa)
        # Vector from path point to front axle
        dx = fx - wp_x
        dy = fy - wp_y
        
        # Cross product to determine sign of the error
        e_fa = np.sin(wp_psi) * dx - np.cos(wp_psi) * dy

        # 5. Stanley Control Law with Feedforward Steering
        # delta_ff accounts for the track's natural curvature
        delta_ff = np.arctan(self.wheelbase * wp_kappa)
        
        # delta = theta_e + arctan(k * e_fa / (v + k_soft)) + delta_ff
        delta = theta_e + np.arctan2(self.k * e_fa, speed + self.k_soft) + delta_ff
        steering = np.clip(delta, -self.steer_max, self.steer_max)

        # 6. Speed Control
        # Extract target speed from a point slightly ahead to account for latency
        latency_idx = (nearest_idx + self.lookahead) % n
        target_speed = self.waypoints[latency_idx, 3] * self.speed_scale

        # Emergency slowdown if we are somehow extremely misaligned
        if abs(theta_e) > np.pi / 3:
            target_speed *= 0.5

        target_speed = np.clip(target_speed, 0.5, 8.0 * self.speed_scale)

        # 7. Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering)
        drive_msg.drive.speed          = float(target_speed)
        self.drive_pub.publish(drive_msg)

        # 8. Publish visualization
        self.publish_markers(fx, fy, wp_x, wp_y)

    def publish_markers(self, fx, fy, tx, ty):
        markers = MarkerArray()

        # Front axle point
        axle_marker = Marker()
        axle_marker.header.frame_id = 'map'
        axle_marker.ns   = 'stanley'
        axle_marker.id   = 0
        axle_marker.type = Marker.SPHERE
        axle_marker.action = Marker.ADD
        axle_marker.pose.position.x = fx
        axle_marker.pose.position.y = fy
        axle_marker.pose.position.z = 0.1
        axle_marker.pose.orientation.w = 1.0
        axle_marker.scale.x = 0.2
        axle_marker.scale.y = 0.2
        axle_marker.scale.z = 0.2
        axle_marker.color.r = 0.0
        axle_marker.color.g = 1.0
        axle_marker.color.b = 0.0
        axle_marker.color.a = 1.0
        markers.markers.append(axle_marker)

        # Target point on path
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.ns   = 'stanley'
        target_marker.id   = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = tx
        target_marker.pose.position.y = ty
        target_marker.pose.position.z = 0.1
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        target_marker.color.r = 1.0
        target_marker.color.g = 0.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        markers.markers.append(target_marker)

        # Line connecting them (showing cross-track error)
        line = Marker()
        line.header.frame_id = 'map'
        line.ns = 'stanley'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0
        from geometry_msgs.msg import Point
        p1 = Point(); p1.x = fx; p1.y = fy; p1.z = 0.1
        p2 = Point(); p2.x = tx; p2.y = ty; p2.z = 0.1
        line.points.append(p1)
        line.points.append(p2)
        markers.markers.append(line)

        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = StanleyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
