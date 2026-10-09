import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import csv
import time
import os
from ament_index_python.packages import get_package_share_directory

class LapTimerNode(Node):
    """
    Robust Lap Timer for F1TENTH.
    
    Instead of relying on an x,y radius which can miss if the car takes a wide line,
    this node tracks the car's progress along the raceline waypoints.
    A lap is counted when the nearest waypoint index wraps around from the 
    end of the track (e.g. >90%) back to the start (e.g. <10%).
    
    The first lap timed is the "flying lap" that starts when the car 
    crosses the start/finish line for the first time.
    """

    def __init__(self):
        super().__init__('lap_timer_node')

        _default_raceline = os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'maps', 'raceline.csv')
        self.declare_parameter('raceline_path', _default_raceline)
        self.raceline_path = self.get_parameter('raceline_path').value

        # Load raceline
        self.waypoints = self.load_raceline(self.raceline_path)
        self.n_waypoints = len(self.waypoints)
        self.wp_xy = self.waypoints[:, :2]

        # ── State ─────────────────────────────────────────
        self.lap_count = 0
        self.lap_times = []
        self.lap_start_time = None
        
        self.last_nearest_idx = None
        self.started = False            # True once we cross start/finish for the first time
        
        self.lap_distance = 0.0
        self.last_pos = None
        # ──────────────────────────────────────────────────

        self.lap_pub = self.create_publisher(String, '/lap_timer/info', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )

        self.get_logger().info(
            f'Lap timer ready — tracking progress over {self.n_waypoints} waypoints. '
            f'Waiting for car to cross start/finish line (waypoint 0)...'
        )

    def load_raceline(self, path):
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

    def odom_callback(self, msg):
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y
        speed = abs(msg.twist.twist.linear.x)

        # Track distance traveled
        if self.last_pos is not None:
            self.lap_distance += np.sqrt(
                (cx - self.last_pos[0])**2 + (cy - self.last_pos[1])**2
            )
        self.last_pos = (cx, cy)

        # Find nearest waypoint
        dists = np.linalg.norm(self.wp_xy - np.array([cx, cy]), axis=1)
        nearest_idx = int(np.argmin(dists))

        # Detect wrap-around (car moved from end of track to start of track)
        wrap_around = False
        if self.last_nearest_idx is not None:
            # e.g., jumping from index 1600 back to index 5
            if self.last_nearest_idx > self.n_waypoints * 0.8 and nearest_idx < self.n_waypoints * 0.2:
                wrap_around = True

        self.last_nearest_idx = nearest_idx

        # Ignore everything until the car actually moves
        if speed < 0.1:
            return

        # ── State machine ─────────────────────────────────
        if not self.started:
            # We are on the out-lap. Wait until we cross the start/finish line
            if wrap_around:
                self.started = True
                self.lap_start_time = time.monotonic()
                self.lap_distance = 0.0
                self.get_logger().info('🏁 Crossed start/finish! Flying Lap 1 timing started...')
            return

        # Detect lap completion: wrap around happened AND at least 10 seconds passed
        # (The time check is a debounce to prevent double-triggering)
        if wrap_around and (time.monotonic() - self.lap_start_time) > 10.0:
            now = time.monotonic()
            lap_time = now - self.lap_start_time

            self.lap_count += 1
            self.lap_times.append(lap_time)

            best_time = min(self.lap_times)
            avg_time = sum(self.lap_times) / len(self.lap_times)
            avg_speed = self.lap_distance / lap_time if lap_time > 0 else 0

            is_best = lap_time <= best_time
            best_marker = ' ⚡ BEST' if is_best and self.lap_count > 1 else ''

            self.get_logger().info(
                f'🏁 Lap {self.lap_count} complete: {lap_time:.3f}s '
                f'(avg speed: {avg_speed:.2f} m/s){best_marker}'
            )
            if self.lap_count > 1:
                self.get_logger().info(
                    f'   Best: {best_time:.3f}s | Avg: {avg_time:.3f}s'
                )

            # Publish
            info_msg = String()
            info_msg.data = (
                f'lap={self.lap_count},time={lap_time:.3f},'
                f'best={best_time:.3f},avg={avg_time:.3f},'
                f'speed={avg_speed:.2f}'
            )
            self.lap_pub.publish(info_msg)

            # Reset for next lap
            self.lap_start_time = now
            self.lap_distance = 0.0

    def print_summary(self):
        if self.lap_times:
            print('─' * 45)
            print(f'SESSION SUMMARY')
            print(f'  Total laps:  {self.lap_count}')
            print(f'  Best lap:    {min(self.lap_times):.3f}s')
            print(f'  Worst lap:   {max(self.lap_times):.3f}s')
            print(f'  Average:     {sum(self.lap_times)/len(self.lap_times):.3f}s')
            print(f'  All laps:    {[round(t, 3) for t in self.lap_times]}')
            print('─' * 45)

def main(args=None):
    rclpy.init(args=args)
    node = LapTimerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
