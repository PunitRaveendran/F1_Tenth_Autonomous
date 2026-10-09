import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import csv
import yaml
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt
from ament_index_python.packages import get_package_share_directory


class SplinerNode(Node):
    """
    Generates cubic-spline overtake trajectories around a detected opponent.

    Uses a precomputed distance transform of the track map to determine
    which side has more wall clearance, then generates a 5-point spline
    that smoothly deviates from the raceline, passes the opponent, and
    returns.  Speeds are encoded in pose.position.z of the published Path.
    """

    def __init__(self):
        super().__init__('spliner_node')

        # ── Parameters ────────────────────────────────────────
        _maps_dir = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps')
        self.declare_parameter('raceline_path',
            os.path.join(_maps_dir, 'raceline.csv'))
        self.declare_parameter('map_yaml_path',
            os.path.join(_maps_dir, 'Spielberg_map.yaml'))
        self.declare_parameter('lateral_offset', 0.6)
        self.declare_parameter('approach_distance', 2.0)
        self.declare_parameter('exit_distance', 4.0)
        self.declare_parameter('speed_scale_overtake', 0.7)
        self.declare_parameter('max_curvature', 0.8)
        self.declare_parameter('spline_sample_interval', 0.1)
        self.declare_parameter('safety_margin', 0.3)

        self.lat_offset    = self.get_parameter('lateral_offset').value
        self.approach_dist = self.get_parameter('approach_distance').value
        self.exit_dist     = self.get_parameter('exit_distance').value
        self.speed_scale   = self.get_parameter('speed_scale_overtake').value
        self.max_kappa     = self.get_parameter('max_curvature').value
        self.sample_iv     = self.get_parameter('spline_sample_interval').value
        self.safety_margin = self.get_parameter('safety_margin').value

        # Load raceline [s, x, y, psi, kappa, vx]
        self.rl = self._load_raceline(self.get_parameter('raceline_path').value)
        self.n_wp = len(self.rl)
        self.total_s = float(self.rl[-1, 0])
        self.get_logger().info(f'Spliner loaded {self.n_wp} waypoints, track={self.total_s:.0f}m')

        # Distance transform for wall clearance
        self._load_distance_transform(self.get_parameter('map_yaml_path').value)

        # State
        self.ego_x = self.ego_y = self.ego_yaw = 0.0
        self.opp_x = self.opp_y = 0.0
        self.has_odom = self.opp_detected = False

        # Pub / Sub
        self.path_pub   = self.create_publisher(Path, '/overtake/path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/overtake/markers', 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/opponent/pose', self._opp_pose_cb, 10)
        self.create_subscription(Bool, '/opponent/detected', self._opp_det_cb, 10)
        self.create_timer(0.1, self._plan)  # 10 Hz
        self.get_logger().info('Spliner node ready')

    # ── Loaders ───────────────────────────────────────────────
    def _load_raceline(self, path):
        rows = []
        with open(path) as f:
            for row in csv.reader(f, delimiter=';'):
                if not row or row[0].startswith('#'):
                    continue
                try:
                    rows.append([float(row[i]) for i in range(6)])
                except (ValueError, IndexError):
                    continue
        return np.array(rows)  # columns: s, x, y, psi, kappa, vx

    def _load_distance_transform(self, yaml_path):
        self.has_map = False
        try:
            with open(yaml_path) as f:
                meta = yaml.safe_load(f)
            self.map_res = float(meta['resolution'])
            self.map_origin = [float(v) for v in meta['origin']]
            img_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
            flip = getattr(Image, 'Transpose', Image).FLIP_TOP_BOTTOM
            occ = np.array(Image.open(img_path).transpose(flip)).astype(np.float32)
            if occ.ndim == 3:
                occ = occ[:, :, 0]
            self.dt = distance_transform_edt(occ > 128) * self.map_res
            self.dt_shape = self.dt.shape
            self.has_map = True
            self.get_logger().info(f'Distance transform ready, max clearance {self.dt.max():.1f}m')
        except Exception as e:
            self.get_logger().warn(f'Map DT failed: {e} — defaulting to left-side pass')

    def _wall_dist(self, wx, wy):
        if not self.has_map:
            return 1.0
        c = int((wx - self.map_origin[0]) / self.map_res)
        r = int((wy - self.map_origin[1]) / self.map_res)
        if 0 <= r < self.dt_shape[0] and 0 <= c < self.dt_shape[1]:
            return float(self.dt[r, c])
        return 0.0

    # ── Callbacks ─────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.ego_yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.has_odom = True

    def _opp_pose_cb(self, msg):
        self.opp_x = msg.pose.position.x
        self.opp_y = msg.pose.position.y

    def _opp_det_cb(self, msg):
        self.opp_detected = msg.data

    # ── Helper: find raceline index at arc-length ─────────────
    def _idx_at_s(self, s):
        s = s % self.total_s
        return int(np.argmin(np.abs(self.rl[:, 0] - s)))

    def _offset_pt(self, idx, amount, sign):
        """Offset raceline point perpendicular to heading."""
        x, y, psi = self.rl[idx, 1], self.rl[idx, 2], self.rl[idx, 3]
        nx, ny = -np.sin(psi) * sign, np.cos(psi) * sign
        return x + amount * nx, y + amount * ny

    # ── Main planning loop ────────────────────────────────────
    def _plan(self):
        if not self.has_odom or not self.opp_detected:
            self._clear_markers()
            return

        # 1. Project opponent onto raceline
        d = np.sqrt((self.rl[:,1]-self.opp_x)**2 + (self.rl[:,2]-self.opp_y)**2)
        opp_idx = int(np.argmin(d))
        s_opp = self.rl[opp_idx, 0]
        psi_opp = self.rl[opp_idx, 3]

        # 2. Determine passing side via distance transform
        probe = self.lat_offset + self.safety_margin
        nl_x, nl_y = -np.sin(psi_opp), np.cos(psi_opp)
        lx = self.rl[opp_idx,1] + probe*nl_x
        ly = self.rl[opp_idx,2] + probe*nl_y
        rx = self.rl[opp_idx,1] - probe*nl_x
        ry = self.rl[opp_idx,2] - probe*nl_y

        l_clear = self._wall_dist(lx, ly)
        r_clear = self._wall_dist(rx, ry)
        sign = 1.0 if l_clear >= r_clear else -1.0

        available = max(l_clear, r_clear)
        eff_offset = min(self.lat_offset, available - self.safety_margin)
        if eff_offset < 0.2:
            return  # not enough room

        # 3. Five control points
        i1 = self._idx_at_s(s_opp - self.approach_dist)
        i3 = self._idx_at_s(s_opp + self.exit_dist * 0.5)
        i4 = self._idx_at_s(s_opp + self.exit_dist)

        p0 = np.array([self.ego_x, self.ego_y])
        p1 = np.array(self._offset_pt(i1, eff_offset * 0.3, sign))
        p2 = np.array(self._offset_pt(opp_idx, eff_offset, sign))
        p3 = np.array(self._offset_pt(i3, eff_offset * 0.3, sign))
        p4 = np.array([self.rl[i4, 1], self.rl[i4, 2]])

        pts = np.array([p0, p1, p2, p3, p4])

        # 4. Cubic spline parameterized by cumulative distance
        seg = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        t = np.concatenate(([0], np.cumsum(seg)))
        if t[-1] < 0.5:
            return

        try:
            cx = CubicSpline(t, pts[:,0], bc_type='natural')
            cy = CubicSpline(t, pts[:,1], bc_type='natural')
        except Exception:
            return

        # 5. Sample
        ns = max(int(t[-1] / self.sample_iv), 10)
        ts = np.linspace(0, t[-1], ns)
        sx, sy = cx(ts), cy(ts)

        # 6. Speeds — sinusoidal blend: full at edges, reduced at apex
        speeds = np.zeros(ns)
        for i in range(ns):
            dd = np.sqrt((self.rl[:,1]-sx[i])**2 + (self.rl[:,2]-sy[i])**2)
            rl_speed = self.rl[int(np.argmin(dd)), 5]
            frac = ts[i] / t[-1]
            scale = 1.0 - (1.0 - self.speed_scale) * np.sin(np.pi * frac)
            speeds[i] = rl_speed * scale

        # 7. Curvature check
        dx, dy = cx(ts,1), cy(ts,1)
        ddx, ddy = cx(ts,2), cy(ts,2)
        kappa = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-9)**1.5
        mk = float(np.max(kappa))
        if mk > self.max_kappa:
            self.get_logger().warn(
                f'Spline curvature {mk:.2f} > {self.max_kappa}',
                throttle_duration_sec=1.0)

        # 8. Publish path (speed in z)
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(ns):
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = float(sx[i])
            p.pose.position.y = float(sy[i])
            p.pose.position.z = float(speeds[i])  # speed encoded here
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self.path_pub.publish(path)

        self._publish_markers(pts, sx, sy, speeds, mk)

    # ── Visualization ─────────────────────────────────────────
    def _clear_markers(self):
        ma = MarkerArray()
        c = Marker(); c.header.frame_id='map'; c.ns='spliner'; c.action=Marker.DELETEALL
        ma.markers.append(c)
        self.marker_pub.publish(ma)

    def _publish_markers(self, ctrl, sx, sy, speeds, max_k):
        ma = MarkerArray()
        c = Marker(); c.header.frame_id='map'; c.ns='spliner'; c.action=Marker.DELETEALL
        ma.markers.append(c)
        ts = self.get_clock().now().to_msg()

        # Spline as colored line strip
        line = Marker()
        line.header.frame_id = 'map'; line.header.stamp = ts
        line.ns = 'spliner'; line.id = 1
        line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.scale.x = 0.08; line.pose.orientation.w = 1.0
        line.lifetime.nanosec = 200_000_000
        smin, smax = float(speeds.min()), float(speeds.max())
        sr = max(smax - smin, 0.1)
        for i in range(len(sx)):
            line.points.append(Point(x=float(sx[i]), y=float(sy[i]), z=0.05))
            f = (speeds[i] - smin) / sr
            line.colors.append(ColorRGBA(r=1.0-f, g=f, b=0.0, a=1.0))
        ma.markers.append(line)

        # Control points
        for i, pt in enumerate(ctrl):
            m = Marker()
            m.header.frame_id = 'map'; m.header.stamp = ts
            m.ns = 'spliner'; m.id = 10+i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = float(pt[0]); m.pose.position.y = float(pt[1])
            m.pose.position.z = 0.15; m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
            m.lifetime.nanosec = 200_000_000
            ma.markers.append(m)

        # Info text
        txt = Marker()
        txt.header.frame_id = 'map'; txt.header.stamp = ts
        txt.ns = 'spliner'; txt.id = 20
        txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
        txt.pose.position.x = float(ctrl[2,0])
        txt.pose.position.y = float(ctrl[2,1])
        txt.pose.position.z = 0.6; txt.pose.orientation.w = 1.0
        txt.scale.z = 0.2
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f'k={max_k:.2f}'
        txt.lifetime.nanosec = 200_000_000
        ma.markers.append(txt)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = SplinerNode()
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
