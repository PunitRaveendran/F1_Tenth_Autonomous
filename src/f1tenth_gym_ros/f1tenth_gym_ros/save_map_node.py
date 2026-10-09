import argparse
import math
from pathlib import Path
import sys
import time

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


UNKNOWN_VALUE = 205
FREE_VALUE = 254
OCCUPIED_VALUE = 0


class MapSaver(Node):
    def __init__(self, output_prefix, topic, occupied_thresh, free_thresh):
        super().__init__('f1tenth_map_saver')
        self.output_prefix = Path(output_prefix)
        self.occupied_thresh = occupied_thresh
        self.free_thresh = free_thresh
        self.saved = False
        self.error = None

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.sub = self.create_subscription(OccupancyGrid, topic, self.map_callback, qos)
        self.get_logger().info(f'Waiting for OccupancyGrid on {topic}')

    def map_callback(self, msg):
        if self.saved:
            return

        try:
            self._save_map(msg)
            self.saved = True
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.get_logger().error(f'Failed to save map: {exc}')

    def _save_map(self, msg):
        if msg.info.width == 0 or msg.info.height == 0:
            raise ValueError('received empty map')

        if len(msg.data) != msg.info.width * msg.info.height:
            raise ValueError(
                f'map data has {len(msg.data)} cells, expected '
                f'{msg.info.width * msg.info.height}',
            )

        pgm_path = self.output_prefix.with_suffix('.pgm')
        yaml_path = self.output_prefix.with_suffix('.yaml')
        pgm_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_pgm(pgm_path, msg)
        self._write_yaml(yaml_path, pgm_path.name, msg)
        self.get_logger().info(
            f'Saved {msg.info.width}x{msg.info.height} map to {yaml_path}',
        )

    def _write_pgm(self, pgm_path, msg):
        width = msg.info.width
        height = msg.info.height

        with open(pgm_path, 'wb') as pgm_file:
            header = (
                'P5\n'
                '# CREATOR: f1tenth_gym_ros save_map_node\n'
                f'{width} {height}\n'
                '255\n'
            )
            pgm_file.write(header.encode())

            for y in range(height - 1, -1, -1):
                row = bytearray()
                for x in range(width):
                    occupancy = msg.data[x + y * width]
                    row.append(self._occupancy_to_pixel(occupancy))
                pgm_file.write(row)

    def _occupancy_to_pixel(self, occupancy):
        if occupancy < 0:
            return UNKNOWN_VALUE

        probability = occupancy / 100.0
        if probability >= self.occupied_thresh:
            return OCCUPIED_VALUE
        if probability <= self.free_thresh:
            return FREE_VALUE
        return UNKNOWN_VALUE

    def _write_yaml(self, yaml_path, image_name, msg):
        origin = msg.info.origin
        yaw = self._yaw_from_quaternion(origin.orientation)

        with open(yaml_path, 'w') as yaml_file:
            yaml_file.write(f'image: {image_name}\n')
            yaml_file.write('mode: trinary\n')
            yaml_file.write(f'resolution: {msg.info.resolution:.10g}\n')
            yaml_file.write(
                'origin: '
                f'[{origin.position.x:.10g}, {origin.position.y:.10g}, {yaw:.10g}]\n',
            )
            yaml_file.write('negate: 0\n')
            yaml_file.write(f'occupied_thresh: {self.occupied_thresh:.6f}\n')
            yaml_file.write(f'free_thresh: {self.free_thresh:.6f}\n')

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        siny_cosp = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='Output path without extension')
    parser.add_argument('--topic', default='/map', help='OccupancyGrid topic')
    parser.add_argument('--timeout', type=float, default=10.0, help='Seconds to wait')
    parser.add_argument('--occupied-thresh', type=float, default=0.65)
    parser.add_argument('--free-thresh', type=float, default=0.25)
    cli_args, _ = parser.parse_known_args(args if args is not None else sys.argv[1:])

    rclpy.init(args=args)
    node = MapSaver(
        cli_args.file,
        cli_args.topic,
        cli_args.occupied_thresh,
        cli_args.free_thresh,
    )

    deadline = time.monotonic() + cli_args.timeout
    try:
        while rclpy.ok() and not node.saved and node.error is None:
            if time.monotonic() >= deadline:
                node.error = TimeoutError(
                    f'timed out waiting for map on {cli_args.topic}',
                )
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        error = node.error
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if error is not None:
        raise SystemExit(str(error))


if __name__ == '__main__':
    main()
