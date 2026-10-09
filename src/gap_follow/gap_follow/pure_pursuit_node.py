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

class PurePursuitNode(Node):
    """
    Robust Pure Pursuit controller for raceline tracking.
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ── Parameters ───────────────────────────────────────
        _default_raceline = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps', 'raceline.csv')
        self.declare_parameter('raceline_path', _default_raceline)
        self.declare_parameter('lookahead_min', 0.7)   # Reduced from 0.8 to prevent corner cutting
        self.declare_parameter('lookahead_max', 1.8)   # Reduced from 2.5
        self.declare_parameter('speed_scale',   1.0)
        self.declare_parameter('wheelbase',     0.33)
        self.declare_parameter('steering_gain', 1.1)   # Slightly sharper steering (was 1.0)
        self.declare_parameter('steering_max',  0.41)

        self.raceline_path = self.get_parameter('raceline_path').value
        self.L_min         = self.get_parameter('lookahead_min').value
        self.L_max         = self.get_parameter('lookahead_max').value
        self.speed_scale   = self.get_parameter('speed_scale').value
        self.wheelbase     = self.get_parameter('wheelbase').value
        self.steer_gain    = self.get_parameter('steering_gain').value
        self.steer_max     = self.get_parameter('steering_max').value
        # ─────────────────────────────────────────────────────

        # Load raceline
        self.waypoints = self.load_raceline(self.raceline_path)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {self.raceline_path}')

        # Publishers / Subscribers
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pure_pursuit/markers', 10)
        self.odom_sub   = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.get_logger().info('Pure Pursuit node ready')

    def load_raceline(self, path):
        waypoints = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                try:
                    # s, x, y, psi, kappa, vx, ax
                    x    = float(row[1])
                    y    = float(row[2])
                    vx   = float(row[5])
                    waypoints.append([x, y, vx])
                except (ValueError, IndexError):
                    continue
        return np.array(waypoints)

    def odom_callback(self, msg):
        # Current pose
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        # Quaternion to yaw
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))

        # Current speed for adaptive lookahead
        vx = msg.twist.twist.linear.x
        speed = abs(vx)

        # Adaptive lookahead: scales linearly with speed
        L = np.clip(self.L_min + (speed / 8.0) * (self.L_max - self.L_min),
                    self.L_min, self.L_max)

        # Find nearest waypoint
        dists = np.linalg.norm(self.waypoints[:, :2] - np.array([cx, cy]), axis=1)
        nearest_idx = np.argmin(dists)

        # Find lookahead point — first waypoint beyond distance L
        # Looking strictly forward along the track
        target_idx = nearest_idx
        n = len(self.waypoints)
        target = None
        for i in range(nearest_idx, nearest_idx + n):
            idx = i % n
            wp  = self.waypoints[idx]
            d   = np.linalg.norm(wp[:2] - np.array([cx, cy]))
            if d >= L:
                # Transform point to vehicle frame to check if it's IN FRONT
                dx_t = wp[0] - cx
                dy_t = wp[1] - cy
                local_x_t = np.cos(yaw) * dx_t + np.sin(yaw) * dy_t
                if local_x_t > 0.0:
                    target_idx = idx
                    # Interpolate exact point on segment using circle-line intersection
                    prev_idx = (i - 1) % n
                    prev_wp = self.waypoints[prev_idx]
                    
                    V = wp[:2] - prev_wp[:2]
                    W = prev_wp[:2] - np.array([cx, cy])
                    
                    a = np.dot(V, V)
                    b = 2 * np.dot(V, W)
                    c = np.dot(W, W) - L**2
                    
                    discriminant = b**2 - 4*a*c
                    if a > 1e-6 and discriminant >= 0:
                        t = (-b + np.sqrt(discriminant)) / (2*a)
                        target_x = prev_wp[0] + t * V[0]
                        target_y = prev_wp[1] + t * V[1]
                        target_v = prev_wp[2] + t * (wp[2] - prev_wp[2])
                        target = np.array([target_x, target_y, target_v])
                    else:
                        target = wp
                    break

        if target is None:
            # Fallback if no suitable point found (e.g. sharp hairpin)
            target = self.waypoints[(nearest_idx + 10) % n]

        # Transform target to vehicle frame
        dx = target[0] - cx
        dy = target[1] - cy
        local_x =  np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        # Pure Pursuit steering angle
        # alpha = angle to target in vehicle frame
        alpha = np.arctan2(local_y, local_x)
        steering = self.steer_gain * np.arctan2(
            2.0 * self.wheelbase * np.sin(alpha), L
        )
        steering = np.clip(steering, -self.steer_max, self.steer_max)

        # Target speed from raceline
        # The raceline generator already computes a backward-integrated braking profile.
        # If we use the speed at the pure-pursuit lookahead distance (L ~ 1.5-2m), the car will brake too early.
        # Instead, we look at the speed just slightly ahead of the car (e.g., 3 waypoints = ~0.6m) to account for actuator latency.
        latency_idx = (nearest_idx + 3) % n
        target_speed = self.waypoints[latency_idx, 2] * self.speed_scale
        
        # Extremely mild safety limit if we are somehow perpendicular to the track
        # (Helps car recover if it spun out)
        if abs(alpha) > np.pi / 4:
            target_speed *= 0.5

        target_speed = np.clip(target_speed, 0.5, 8.0 * self.speed_scale)

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering)
        drive_msg.drive.speed          = float(target_speed)
        self.drive_pub.publish(drive_msg)

        # Publish visualisation markers
        self.publish_markers(cx, cy, target[0], target[1], L)

    def publish_markers(self, cx, cy, tx, ty, L):
        markers = MarkerArray()

        # Lookahead circle
        circle = Marker()
        circle.header.frame_id = 'map'
        circle.ns   = 'pure_pursuit'
        circle.id   = 0
        circle.type = Marker.CYLINDER
        circle.action = Marker.ADD
        circle.pose.position.x = cx
        circle.pose.position.y = cy
        circle.pose.position.z = 0.0
        circle.pose.orientation.w = 1.0
        circle.scale.x = L * 2
        circle.scale.y = L * 2
        circle.scale.z = 0.01
        circle.color.r = 0.0
        circle.color.g = 0.5
        circle.color.b = 1.0
        circle.color.a = 0.2
        markers.markers.append(circle)

        # Target point
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.ns   = 'pure_pursuit'
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

        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
