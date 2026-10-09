import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import csv
import os
from ament_index_python.packages import get_package_share_directory


class OpponentDetectorNode(Node):
    """
    Detects the opponent car from LiDAR data using breakpoint-based
    clustering and geometric filtering.

    Algorithm:
      1. Convert LiDAR polar → Cartesian in car-local frame
      2. Split scan into clusters at range discontinuities (breakpoints)
      3. Filter clusters by size, distance, and bounding-box geometry
      4. Validate against raceline: reject clusters too far from track
      5. Transform best candidate to map frame and publish

    Publishes:
      /opponent/detected  (Bool)   — True when an opponent is in range
      /opponent/pose      (PoseStamped) — opponent centroid in map frame
      /opponent/markers   (MarkerArray) — RViz debug visualization
    """

    def __init__(self):
        super().__init__('opponent_detector_node')

        # ── Parameters ────────────────────────────────────────
        _default_raceline = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps', 'raceline.csv')
        self.declare_parameter(
            'raceline_path',
            _default_raceline,
        )
        self.declare_parameter('breakpoint_threshold', 0.5)   # m — range jump to split clusters
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_cluster_points', 40)
        self.declare_parameter('detection_range', 5.0)         # m — max distance to consider
        self.declare_parameter('car_width_min', 0.10)          # m — cluster width bounds
        self.declare_parameter('car_width_max', 0.50)          # m
        self.declare_parameter('raceline_lateral_tolerance', 2.0)  # m — reject if too far from track

        self.breakpoint_thresh = self.get_parameter('breakpoint_threshold').value
        self.min_pts  = self.get_parameter('min_cluster_points').value
        self.max_pts  = self.get_parameter('max_cluster_points').value
        self.det_range = self.get_parameter('detection_range').value
        self.car_w_min = self.get_parameter('car_width_min').value
        self.car_w_max = self.get_parameter('car_width_max').value
        self.lat_tol   = self.get_parameter('raceline_lateral_tolerance').value

        # Load raceline for track-proximity filtering
        raceline_path = self.get_parameter('raceline_path').value
        self.raceline_xy = self._load_raceline(raceline_path)
        self.get_logger().info(
            f'Opponent detector loaded {len(self.raceline_xy)} raceline waypoints'
        )

        # ── Ego state (updated by odom callback) ──────────────
        self.ego_x   = 0.0
        self.ego_y   = 0.0
        self.ego_yaw = 0.0
        self.has_odom = False

        # ── Temporal hysteresis ────────────────────────────────
        self.consecutive_detections = 0
        self.CONFIRM_THRESHOLD = 3   # require N consecutive frames
        self.consecutive_misses = 0
        self.MISS_THRESHOLD = 5      # require N misses to clear

        # ── Publishers ────────────────────────────────────────
        self.detected_pub = self.create_publisher(Bool, '/opponent/detected', 10)
        self.pose_pub     = self.create_publisher(PoseStamped, '/opponent/pose', 10)
        self.marker_pub   = self.create_publisher(MarkerArray, '/opponent/markers', 10)

        # ── Subscribers ───────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/ego_racecar/scan', self.scan_callback, 10
        )

        self.get_logger().info('Opponent detector node ready')

    # ── Load raceline ─────────────────────────────────────────
    def _load_raceline(self, path):
        waypoints = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                try:
                    x = float(row[1])
                    y = float(row[2])
                    waypoints.append([x, y])
                except (ValueError, IndexError):
                    continue
        return np.array(waypoints)

    # ── Odom callback ─────────────────────────────────────────
    def odom_callback(self, msg):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.ego_yaw = np.arctan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z),
        )
        self.has_odom = True

    # ── Scan callback — main detection pipeline ───────────────
    def scan_callback(self, msg):
        if not self.has_odom:
            return

        ranges = np.array(msg.ranges)
        n_beams = len(ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n_beams)

        # Replace inf/nan with max range
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges_clean = np.where(valid, ranges, msg.range_max)

        # ── Step 1: Convert to Cartesian (car-local frame) ────
        xs_local = ranges_clean * np.cos(angles)
        ys_local = ranges_clean * np.sin(angles)

        # ── Step 2: Breakpoint clustering ─────────────────────
        # Find indices where consecutive range values jump significantly
        diffs = np.abs(np.diff(ranges_clean))
        breakpoints = np.where(diffs > self.breakpoint_thresh)[0]

        # Build cluster index ranges: [start, end) for each cluster
        starts = np.concatenate(([0], breakpoints + 1))
        ends   = np.concatenate((breakpoints + 1, [n_beams]))

        # ── Step 3 & 4: Filter clusters ───────────────────────
        best_candidate = None
        best_distance  = float('inf')
        all_candidates = []

        for s, e in zip(starts, ends):
            n_pts = e - s

            # Size filter
            if n_pts < self.min_pts or n_pts > self.max_pts:
                continue

            cluster_x = xs_local[s:e]
            cluster_y = ys_local[s:e]

            # Only consider points in front of the car (local_x > 0)
            forward_mask = cluster_x > 0.0
            if forward_mask.sum() < self.min_pts:
                continue

            cluster_x = cluster_x[forward_mask]
            cluster_y = cluster_y[forward_mask]

            # Centroid in car-local frame
            cx_local = float(np.mean(cluster_x))
            cy_local = float(np.mean(cluster_y))
            dist = np.sqrt(cx_local**2 + cy_local**2)

            # Distance filter
            if dist > self.det_range or dist < 0.3:
                continue

            # Bounding box dimensions
            width  = float(np.max(cluster_y) - np.min(cluster_y))
            depth  = float(np.max(cluster_x) - np.min(cluster_x))

            # Geometric filter — car-shaped: width 0.1-0.5m, depth 0.05-0.5m
            if width < self.car_w_min or width > self.car_w_max:
                continue
            if depth > 0.50:
                continue  # too deep — likely a wall section

            # ── Transform centroid to map frame ───────────────
            cos_y = np.cos(self.ego_yaw)
            sin_y = np.sin(self.ego_yaw)
            cx_map = self.ego_x + cos_y * cx_local - sin_y * cy_local
            cy_map = self.ego_y + sin_y * cx_local + cos_y * cy_local

            # Raceline proximity check
            dists_to_raceline = np.sqrt(
                (self.raceline_xy[:, 0] - cx_map)**2 +
                (self.raceline_xy[:, 1] - cy_map)**2
            )
            min_dist_to_raceline = float(np.min(dists_to_raceline))

            if min_dist_to_raceline > self.lat_tol:
                continue

            # Valid candidate — track the closest one
            all_candidates.append({
                'cx_map': cx_map, 'cy_map': cy_map,
                'cx_local': cx_local, 'cy_local': cy_local,
                'dist': dist, 'width': width, 'n_pts': n_pts,
            })

            if dist < best_distance:
                best_distance = dist
                best_candidate = all_candidates[-1]

        # ── Temporal hysteresis ────────────────────────────────
        raw_detected = best_candidate is not None
        if raw_detected:
            self.consecutive_detections += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            if self.consecutive_misses >= self.MISS_THRESHOLD:
                self.consecutive_detections = 0

        confirmed = self.consecutive_detections >= self.CONFIRM_THRESHOLD

        # ── Publish results ───────────────────────────────────
        detected_msg = Bool()
        detected_msg.data = confirmed and raw_detected
        self.detected_pub.publish(detected_msg)

        if confirmed and best_candidate is not None:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'map'
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.position.x = best_candidate['cx_map']
            pose_msg.pose.position.y = best_candidate['cy_map']
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            self.pose_pub.publish(pose_msg)

        # ── RViz markers ─────────────────────────────────────
        self._publish_markers(best_candidate, all_candidates)

    # ── Visualization ─────────────────────────────────────────
    def _publish_markers(self, best, all_cands):
        markers = MarkerArray()

        # Clear old markers
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.ns = 'opponent_detector'
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        if best is not None:
            # Main detection — large green sphere
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'opponent_detector'
            m.id = 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = best['cx_map']
            m.pose.position.y = best['cy_map']
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.4
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.9
            m.lifetime.sec = 0
            m.lifetime.nanosec = 200_000_000  # 200ms
            markers.markers.append(m)

            # Text label with distance
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = self.get_clock().now().to_msg()
            t.ns = 'opponent_detector'
            t.id = 2
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = best['cx_map']
            t.pose.position.y = best['cy_map']
            t.pose.position.z = 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            t.text = f"OPP {best['dist']:.1f}m w={best['width']:.2f}m"
            t.lifetime.sec = 0
            t.lifetime.nanosec = 200_000_000
            markers.markers.append(t)

            # Detection range circle around ego
            ring = Marker()
            ring.header.frame_id = 'map'
            ring.header.stamp = self.get_clock().now().to_msg()
            ring.ns = 'opponent_detector'
            ring.id = 3
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x = self.ego_x
            ring.pose.position.y = self.ego_y
            ring.pose.position.z = 0.0
            ring.pose.orientation.w = 1.0
            ring.scale.x = self.det_range * 2
            ring.scale.y = self.det_range * 2
            ring.scale.z = 0.005
            ring.color.r = 0.0
            ring.color.g = 1.0
            ring.color.b = 0.5
            ring.color.a = 0.1
            ring.lifetime.sec = 0
            ring.lifetime.nanosec = 200_000_000
            markers.markers.append(ring)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentDetectorNode()
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
