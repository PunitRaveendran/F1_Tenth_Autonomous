import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import csv
import math

class AdvancedPurePursuit(Node):
    def __init__(self):
        super().__init__('advanced_pure_pursuit')
        
        # ── 1. CONFIGURABLE PARAMETERS ──
        self.declare_parameter('speed_scale', 0.70)      # 0.70 gives ~5.6 m/s on straights and ~2.8-3.0 m/s in hairpins
        self.declare_parameter('min_lookahead', 1.2)    # meters (tight corners)
        self.declare_parameter('max_lookahead', 2.4)    # meters (straights)
        self.declare_parameter('max_lat_accel', 4.0)    # m/s^2 tire grip limit
        
        self.wheelbase = 0.33                           # F1TENTH wheelbase
        self.max_steer = 0.41                           # Physical servo limit (rad)
        self.max_steer_rate = 3.2                       # Slew rate limit (rad/sec)
        
        self.last_steering = 0.0
        self.last_time = None
        
        # Dynamically detect active map from sim.yaml or parameter
        import yaml, os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sim_yaml_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'config', 'sim.yaml')
        default_raceline = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'raceline.csv')
        
        if os.path.exists(sim_yaml_path):
            try:
                with open(sim_yaml_path, 'r') as yf:
                    sdata = yaml.safe_load(yf)
                    mpath = sdata.get('bridge', {}).get('ros__parameters', {}).get('map_path', '')
                    if 'Nuerburgring' in mpath:
                        default_raceline = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Nuerburgring_map_raceline.csv')
                    elif 'Spielberg' in mpath:
                        default_raceline = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Spielberg_map_raceline.csv')
            except Exception:
                pass

        self.declare_parameter('raceline_path', default_raceline)
        self.raceline_path = self.get_parameter('raceline_path').value
        self.waypoints = self.load_raceline(self.raceline_path)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {os.path.basename(self.raceline_path)} for Advanced Pure Pursuit!')

        # Publishers & Subscribers
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pure_pursuit/markers', 10)
        self.odom_sub   = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

    def load_raceline(self, path):
        waypoints = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not row or row[0].startswith('#') or row[0] == 's_m' or len(row) < 6:
                    continue
                try:
                    # x, y, speed, curvature (kappa)
                    x = float(row[1])
                    y = float(row[2])
                    kappa = float(row[4])
                    speed = float(row[5])
                    waypoints.append([x, y, speed, kappa])
                except ValueError:
                    continue
        return np.array(waypoints)

    def odom_callback(self, msg):
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        
        # Heading (yaw) from quaternion
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        current_speed = abs(msg.twist.twist.linear.x)

        # Dynamic parameters
        speed_scale   = self.get_parameter('speed_scale').value
        min_lookahead = self.get_parameter('min_lookahead').value
        max_lookahead = self.get_parameter('max_lookahead').value
        max_lat_accel = self.get_parameter('max_lat_accel').value

        # 1. ADAPTIVE LOOKAHEAD DISTANCE
        # Stretches lookahead smoothly as speed increases
        lookahead = min_lookahead + (current_speed / 8.0) * (max_lookahead - min_lookahead)
        lookahead = float(np.clip(lookahead, min_lookahead, max_lookahead))

        # 2. CLOSEST WAYPOINT ON TRACK
        n = len(self.waypoints)
        dists = np.linalg.norm(self.waypoints[:, :2] - np.array([cx, cy]), axis=1)
        closest_idx = int(np.argmin(dists))

        # 3. FORWARD LOOKAHEAD TARGET SEARCH (with Circle-Segment Interpolation)
        target = None
        for i in range(closest_idx, closest_idx + 120):
            idx = i % n
            wp  = self.waypoints[idx]
            d   = np.linalg.norm(wp[:2] - np.array([cx, cy]))
            
            # Check vehicle-relative coordinate to ensure point is strictly in front
            dx_t = wp[0] - cx
            dy_t = wp[1] - cy
            local_x_t = math.cos(yaw) * dx_t + math.sin(yaw) * dy_t
            
            if d >= lookahead and local_x_t > 0.0:
                # Interpolate exact point on segment for silky-smooth tracking
                prev_idx = (i - 1) % n
                prev_wp = self.waypoints[prev_idx]
                
                v_seg = wp[:2] - prev_wp[:2]
                w_seg = prev_wp[:2] - np.array([cx, cy])
                
                a = np.dot(v_seg, v_seg)
                b = 2.0 * np.dot(v_seg, w_seg)
                c = np.dot(w_seg, w_seg) - lookahead**2
                
                discriminant = b**2 - 4.0 * a * c
                if a > 1e-6 and discriminant >= 0.0:
                    t = (-b + math.sqrt(discriminant)) / (2.0 * a)
                    t = np.clip(t, 0.0, 1.0)
                    target = prev_wp[:2] + t * v_seg
                else:
                    target = wp[:2]
                break

        # Fallback if vehicle is facing away or in rare transition
        if target is None:
            fallback_idx = (closest_idx + max(5, int(lookahead / 0.2))) % n
            target = self.waypoints[fallback_idx, :2]

        # 4. TRANSFORM TARGET INTO VEHICLE LOCAL FRAME
        dx = target[0] - cx
        dy = target[1] - cy
        local_x =  math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # 5. PURE PURSUIT CONTROL LAW
        alpha = math.atan2(local_y, local_x)
        raw_steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead)
        raw_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))

        # Steering Slew Rate Limiter (eliminates violent twitches)
        now = self.get_clock().now()
        if self.last_time is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0.0:
                max_change = self.max_steer_rate * dt
                delta = raw_steering - self.last_steering
                if abs(delta) > max_change:
                    raw_steering = self.last_steering + math.copysign(max_change, delta)
        self.last_steering = raw_steering
        self.last_time = now
        steering_angle = raw_steering

        # 6. PREDICTIVE BRAKING HORIZON (Solves the corner crashing bug!)
        # Instead of waiting until we are in the corner, scan ahead over the braking distance
        # Braking distance at ~6 m/s is ~4-5 meters (~20 to 25 waypoints)
        brake_window = int(18 + current_speed * 1.5)
        future_indices = [(closest_idx + k) % n for k in range(brake_window)]
        future_speeds = self.waypoints[future_indices, 2]
        
        # Anticipate upcoming corner speed:
        anticipated_corner_speed = float(np.min(future_speeds)) * speed_scale

        # Physical grip boundary: v <= sqrt(a_lat / curvature)
        curr_kappa = abs(self.waypoints[closest_idx, 3])
        tire_grip_speed = math.sqrt(max_lat_accel / max(curr_kappa, 0.05))

        # Dynamic steering angle penalty (derate speed if steering hard)
        steer_penalty = 1.0 - 0.35 * (abs(steering_angle) / self.max_steer)

        # Final safe commanded speed
        drive_speed = min(anticipated_corner_speed, tire_grip_speed * steer_penalty)
        drive_speed = float(np.clip(drive_speed, 1.5, 8.0 * speed_scale))

        # Safety recovery: if car somehow spun > 60 degrees off course, crawl to rejoin
        if abs(alpha) > 1.0:
            drive_speed = 1.5

        # 7. PUBLISH DRIVE MESSAGE
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = now.to_msg()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = drive_speed
        self.drive_pub.publish(drive_msg)

        # 8. RVIZ VISUALIZATION
        self.publish_markers(target[0], target[1], lookahead)

    def publish_markers(self, tx, ty, lookahead):
        markers = MarkerArray()
        
        # Target Point Sphere (Cyan)
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.ns = 'pure_pursuit'
        target_marker.id = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = tx
        target_marker.pose.position.y = ty
        target_marker.pose.position.z = 0.3
        target_marker.scale.x = 0.35
        target_marker.scale.y = 0.35
        target_marker.scale.z = 0.35
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 1.0
        target_marker.color.a = 1.0
        markers.markers.append(target_marker)

        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedPurePursuit()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
