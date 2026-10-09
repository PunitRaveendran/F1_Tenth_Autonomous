import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import csv
import time
import os
from ament_index_python.packages import get_package_share_directory


# ── State definitions ─────────────────────────────────────────
RACELINE  = 'RACELINE'
TRAILING  = 'TRAILING'
OVERTAKE  = 'OVERTAKE'
RETURN    = 'RETURN'
EMERGENCY = 'EMERGENCY'


class StateMachineNode(Node):
    """
    5-state FSM controller for F1TENTH head-to-head racing.

    Absorbs Pure Pursuit logic internally and switches the waypoint
    source depending on the current state:
      RACELINE  → follow raceline waypoints (normal racing)
      TRAILING  → follow raceline but speed-match the opponent
      OVERTAKE  → follow the spliner-generated overtake path
      RETURN    → blend back from current position to the raceline
      EMERGENCY → hard brake + steer toward largest LiDAR gap
    """

    def __init__(self):
        super().__init__('state_machine_node')

        # ── Parameters ────────────────────────────────────────
        _default_raceline = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps', 'raceline.csv')
        self.declare_parameter('raceline_path', _default_raceline)
        self.declare_parameter('lookahead_min', 0.7)
        self.declare_parameter('lookahead_max', 1.8)
        self.declare_parameter('speed_scale', 1.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('steering_gain', 1.1)
        self.declare_parameter('steering_max', 0.41)

        # State machine thresholds
        self.declare_parameter('detect_range', 4.0)      # m — enter TRAILING
        self.declare_parameter('overtake_commit', 3.0)    # m — enter OVERTAKE
        self.declare_parameter('clear_behind', 2.0)       # m — opponent behind → RETURN
        self.declare_parameter('return_tolerance', 0.3)   # m — close to raceline → RACELINE
        self.declare_parameter('emergency_range', 0.8)    # m — any forward beam < this
        self.declare_parameter('trailing_speed_delta', 0.5)  # m/s above opp speed
        self.declare_parameter('lost_timeout', 1.0)       # s — opponent lost → back to RACELINE

        self.L_min      = self.get_parameter('lookahead_min').value
        self.L_max      = self.get_parameter('lookahead_max').value
        self.speed_scale = self.get_parameter('speed_scale').value
        self.wheelbase  = self.get_parameter('wheelbase').value
        self.steer_gain = self.get_parameter('steering_gain').value
        self.steer_max  = self.get_parameter('steering_max').value

        self.detect_range    = self.get_parameter('detect_range').value
        self.overtake_commit = self.get_parameter('overtake_commit').value
        self.clear_behind    = self.get_parameter('clear_behind').value
        self.return_tol      = self.get_parameter('return_tolerance').value
        self.emergency_range = self.get_parameter('emergency_range').value
        self.trail_delta     = self.get_parameter('trailing_speed_delta').value
        self.lost_timeout    = self.get_parameter('lost_timeout').value

        # Load raceline [x, y, vx]
        rl_path = self.get_parameter('raceline_path').value
        self.waypoints = self._load_raceline(rl_path)
        self.n_wp = len(self.waypoints)
        self.get_logger().info(f'State machine loaded {self.n_wp} waypoints')

        # ── State ─────────────────────────────────────────────
        self.state = RACELINE
        self.prev_state = None

        # Ego
        self.ego_x = self.ego_y = self.ego_yaw = self.ego_speed = 0.0

        # Opponent
        self.opp_x = self.opp_y = 0.0
        self.opp_detected = False
        self.opp_last_seen = 0.0  # monotonic time

        # Overtake path
        self.overtake_path = None    # Nx3 array [x, y, speed]
        self.has_overtake_path = False

        # LiDAR (for emergency)
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0

        # ── Publishers ────────────────────────────────────────
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.state_pub  = self.create_publisher(String, '/state_machine/state', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/state_machine/markers', 10)

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(Odometry, '/ego_racecar/odom', self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/opponent/pose', self._opp_pose_cb, 10)
        self.create_subscription(Bool, '/opponent/detected', self._opp_det_cb, 10)
        self.create_subscription(Path, '/overtake/path', self._path_cb, 10)
        self.create_subscription(LaserScan, '/ego_racecar/scan', self._scan_cb, 10)

        self.get_logger().info('State machine node ready — starting in RACELINE')

    # ── Raceline loader ───────────────────────────────────────
    def _load_raceline(self, path):
        rows = []
        with open(path) as f:
            for row in csv.reader(f, delimiter=';'):
                if not row or row[0].startswith('#'):
                    continue
                try:
                    rows.append([float(row[1]), float(row[2]), float(row[5])])
                except (ValueError, IndexError):
                    continue
        return np.array(rows)  # x, y, vx

    # ── Callbacks ─────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.ego_yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.ego_speed = abs(msg.twist.twist.linear.x)
        self._update()

    def _opp_pose_cb(self, msg):
        self.opp_x = msg.pose.position.x
        self.opp_y = msg.pose.position.y

    def _opp_det_cb(self, msg):
        self.opp_detected = msg.data
        if msg.data:
            self.opp_last_seen = time.monotonic()

    def _path_cb(self, msg):
        if not msg.poses:
            self.has_overtake_path = False
            return
        arr = []
        for p in msg.poses:
            arr.append([p.pose.position.x, p.pose.position.y, p.pose.position.z])
        self.overtake_path = np.array(arr)
        self.has_overtake_path = True

    def _scan_cb(self, msg):
        self.scan_ranges = np.array(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    # ── Pure Pursuit core ─────────────────────────────────────
    def _pure_pursuit(self, waypoints, speed_override=None, speed_cap=None):
        """
        Run Pure Pursuit on the given waypoints array (Nx3: x, y, speed).
        Returns (steering, target_speed, target_x, target_y).
        """
        n = len(waypoints)
        L = np.clip(
            self.L_min + (self.ego_speed / 8.0) * (self.L_max - self.L_min),
            self.L_min, self.L_max)

        # Nearest waypoint
        dists = np.linalg.norm(waypoints[:,:2] - [self.ego_x, self.ego_y], axis=1)
        nearest = int(np.argmin(dists))

        # Find lookahead target
        target = None
        for i in range(nearest, nearest + n):
            idx = i % n
            wp = waypoints[idx]
            d = np.linalg.norm(wp[:2] - [self.ego_x, self.ego_y])
            if d >= L:
                dx_t = wp[0] - self.ego_x
                dy_t = wp[1] - self.ego_y
                lx = np.cos(self.ego_yaw)*dx_t + np.sin(self.ego_yaw)*dy_t
                if lx > 0:
                    # Circle-line interpolation
                    prev = (i-1) % n
                    V = wp[:2] - waypoints[prev,:2]
                    W = waypoints[prev,:2] - [self.ego_x, self.ego_y]
                    a = np.dot(V,V)
                    b = 2*np.dot(V,W)
                    c = np.dot(W,W) - L**2
                    disc = b**2 - 4*a*c
                    if a > 1e-6 and disc >= 0:
                        t = (-b + np.sqrt(disc)) / (2*a)
                        tx = waypoints[prev,0] + t*V[0]
                        ty = waypoints[prev,1] + t*V[1]
                        tv = waypoints[prev,2] + t*(wp[2]-waypoints[prev,2])
                        target = np.array([tx, ty, tv])
                    else:
                        target = wp
                    break

        if target is None:
            target = waypoints[(nearest+10)%n]

        # Steering
        dx = target[0] - self.ego_x
        dy = target[1] - self.ego_y
        lx =  np.cos(self.ego_yaw)*dx + np.sin(self.ego_yaw)*dy
        ly = -np.sin(self.ego_yaw)*dx + np.cos(self.ego_yaw)*dy
        alpha = np.arctan2(ly, lx)
        steer = self.steer_gain * np.arctan2(2*self.wheelbase*np.sin(alpha), L)
        steer = np.clip(steer, -self.steer_max, self.steer_max)

        # Speed
        if speed_override is not None:
            tgt_speed = speed_override
        else:
            # Lookahead boost (same as original PP)
            la_speeds = [waypoints[(nearest+j)%n, 2] for j in range(15)]
            base = target[2]
            boost = max(la_speeds)
            tgt_speed = max(base, boost * 0.9) * self.speed_scale

        if abs(alpha) > np.pi / 4:
            tgt_speed *= 0.5

        if speed_cap is not None:
            tgt_speed = min(tgt_speed, speed_cap)

        tgt_speed = np.clip(tgt_speed, 0.5, 8.0 * self.speed_scale)

        return steer, tgt_speed, target[0], target[1]

    # ── Distance helpers ──────────────────────────────────────
    def _opp_distance(self):
        return np.sqrt((self.ego_x-self.opp_x)**2 + (self.ego_y-self.opp_y)**2)

    def _opp_is_ahead(self):
        """Check if opponent is in front of ego car."""
        dx = self.opp_x - self.ego_x
        dy = self.opp_y - self.ego_y
        lx = np.cos(self.ego_yaw)*dx + np.sin(self.ego_yaw)*dy
        return lx > 0

    def _dist_to_raceline(self):
        d = np.linalg.norm(self.waypoints[:,:2] - [self.ego_x, self.ego_y], axis=1)
        return float(np.min(d))

    def _forward_min_range(self):
        """Min LiDAR range in the forward ±30° cone."""
        if self.scan_ranges is None:
            return 999.0
        n = len(self.scan_ranges)
        angles = self.scan_angle_min + np.arange(n) * self.scan_angle_inc
        fwd_mask = np.abs(angles) < np.radians(30)
        fwd = self.scan_ranges[fwd_mask]
        fwd = fwd[np.isfinite(fwd) & (fwd > 0.05)]
        return float(np.min(fwd)) if len(fwd) > 0 else 999.0

    # ── Emergency gap-follow ──────────────────────────────────
    def _emergency_steer(self):
        """Find largest gap in forward scan and steer toward it."""
        if self.scan_ranges is None:
            return 0.0
        n = len(self.scan_ranges)
        angles = self.scan_angle_min + np.arange(n) * self.scan_angle_inc
        fwd_mask = np.abs(angles) < np.radians(90)
        ranges = np.where(
            np.isfinite(self.scan_ranges) & (self.scan_ranges > 0.05),
            self.scan_ranges, 0.0)
        ranges = ranges * fwd_mask
        # Simple: steer toward the beam with max range
        best = int(np.argmax(ranges))
        return float(np.clip(angles[best] * 0.8, -self.steer_max, self.steer_max))

    # ── State machine update ──────────────────────────────────
    def _update(self):
        opp_dist  = self._opp_distance() if self.opp_detected else 999.0
        opp_ahead = self._opp_is_ahead() if self.opp_detected else False
        fwd_min   = self._forward_min_range()
        time_since_seen = time.monotonic() - self.opp_last_seen

        new_state = self.state

        # ── Transitions ───────────────────────────────────────
        if self.state == RACELINE:
            if fwd_min < self.emergency_range:
                new_state = EMERGENCY
            elif self.opp_detected and opp_ahead and opp_dist < self.detect_range:
                new_state = TRAILING

        elif self.state == TRAILING:
            if fwd_min < self.emergency_range:
                new_state = EMERGENCY
            elif not self.opp_detected and time_since_seen > self.lost_timeout:
                new_state = RACELINE
            elif self.has_overtake_path and opp_dist < self.overtake_commit:
                new_state = OVERTAKE

        elif self.state == OVERTAKE:
            if fwd_min < self.emergency_range:
                new_state = EMERGENCY
            elif not opp_ahead or opp_dist > self.clear_behind:
                new_state = RETURN
            elif not self.has_overtake_path:
                new_state = TRAILING

        elif self.state == RETURN:
            if fwd_min < self.emergency_range:
                new_state = EMERGENCY
            elif self._dist_to_raceline() < self.return_tol:
                new_state = RACELINE
            elif self.opp_detected and opp_ahead and opp_dist < self.detect_range:
                new_state = TRAILING

        elif self.state == EMERGENCY:
            if fwd_min > self.emergency_range * 1.5:
                new_state = RACELINE

        # Log transitions
        if new_state != self.state:
            self.get_logger().info(f'State: {self.state} → {new_state}')
            self.prev_state = self.state
            self.state = new_state

        # Publish state
        s = String()
        s.data = self.state
        self.state_pub.publish(s)

        # ── Execute current state ─────────────────────────────
        steer = 0.0
        speed = 0.5
        tx = self.ego_x
        ty = self.ego_y

        if self.state == RACELINE:
            steer, speed, tx, ty = self._pure_pursuit(self.waypoints)

        elif self.state == TRAILING:
            # Approach at raceline speed, blend to opponent speed as distance closes
            # Assume opponent is going ~1.5 - 2.5 m/s if we don't know
            opp_est_speed = max(self.ego_speed * 0.5, 1.5)
            
            if opp_dist > self.detect_range * 0.6:
                # Still far away within detection range, keep normal speed mostly
                cap = 8.0 
            elif opp_dist > self.overtake_commit:
                # Approaching, blend speed based on distance
                blend = (opp_dist - self.overtake_commit) / (self.detect_range * 0.6 - self.overtake_commit)
                cap = opp_est_speed + (4.0 - opp_est_speed) * blend
            else:
                # Very close, match speed closely + a small delta to close the gap
                cap = opp_est_speed + self.trail_delta
                
            steer, speed, tx, ty = self._pure_pursuit(
                self.waypoints, speed_cap=cap)

        elif self.state == OVERTAKE:
            if self.has_overtake_path:
                steer, speed, tx, ty = self._pure_pursuit(self.overtake_path)
            else:
                steer, speed, tx, ty = self._pure_pursuit(self.waypoints)

        elif self.state == RETURN:
            steer, speed, tx, ty = self._pure_pursuit(self.waypoints)

        elif self.state == EMERGENCY:
            steer = self._emergency_steer()
            speed = 2.0

        # Publish drive
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

        # Markers
        self._publish_markers(tx, ty, opp_dist)

    # ── Visualization ─────────────────────────────────────────
    def _publish_markers(self, tx, ty, opp_dist):
        ma = MarkerArray()

        # State text above car
        txt = Marker()
        txt.header.frame_id = 'map'
        txt.header.stamp = self.get_clock().now().to_msg()
        txt.ns = 'state_machine'; txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
        txt.pose.position.x = self.ego_x
        txt.pose.position.y = self.ego_y
        txt.pose.position.z = 0.7
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.3
        colors = {
            RACELINE:  (0.0, 1.0, 0.5),
            TRAILING:  (1.0, 1.0, 0.0),
            OVERTAKE:  (0.0, 0.5, 1.0),
            RETURN:    (0.5, 0.0, 1.0),
            EMERGENCY: (1.0, 0.0, 0.0),
        }
        r, g, b = colors.get(self.state, (1,1,1))
        txt.color.r = r; txt.color.g = g; txt.color.b = b; txt.color.a = 1.0
        txt.text = f'{self.state} ({opp_dist:.1f}m)'
        ma.markers.append(txt)

        # Lookahead target
        tgt = Marker()
        tgt.header.frame_id = 'map'
        tgt.header.stamp = self.get_clock().now().to_msg()
        tgt.ns = 'state_machine'; tgt.id = 1
        tgt.type = Marker.SPHERE; tgt.action = Marker.ADD
        tgt.pose.position.x = tx; tgt.pose.position.y = ty; tgt.pose.position.z = 0.1
        tgt.pose.orientation.w = 1.0
        tgt.scale.x = tgt.scale.y = tgt.scale.z = 0.25
        tgt.color.r = r; tgt.color.g = g; tgt.color.b = b; tgt.color.a = 1.0
        ma.markers.append(tgt)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
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
