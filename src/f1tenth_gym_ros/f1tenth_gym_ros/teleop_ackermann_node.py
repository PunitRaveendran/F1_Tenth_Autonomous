import select
import sys
import termios
import time
import tty

from ackermann_msgs.msg import AckermannDriveStamped
import rclpy
from rclpy.node import Node


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('max_forward_speed', 6.0)
        self.declare_parameter('max_reverse_speed', 3.0)
        self.declare_parameter('max_steering_angle', 0.40)
        self.declare_parameter('acceleration_rate', 2.5)
        self.declare_parameter('brake_rate', 4.5)
        self.declare_parameter('reverse_acceleration_rate', 2.0)
        self.declare_parameter('steering_rate', 0.75)
        self.declare_parameter('steering_return_rate', 1.8)
        self.declare_parameter('input_timeout', 0.10)
        self.declare_parameter('timer_period', 0.02)
        self.declare_parameter('status_period', 0.10)

        drive_topic = self.get_parameter('drive_topic').value
        self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.speed = 0.0
        self.angle = 0.0
        self.shutdown_requested = False

        self.max_forward_speed = self.get_parameter('max_forward_speed').value
        self.max_reverse_speed = self.get_parameter('max_reverse_speed').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.acceleration_rate = self.get_parameter('acceleration_rate').value
        self.brake_rate = self.get_parameter('brake_rate').value
        self.reverse_acceleration_rate = self.get_parameter(
            'reverse_acceleration_rate',
        ).value
        self.steering_rate = self.get_parameter('steering_rate').value
        self.steering_return_rate = self.get_parameter('steering_return_rate').value
        self.input_timeout = self.get_parameter('input_timeout').value
        self.timer_period = self.get_parameter('timer_period').value
        self.status_period = self.get_parameter('status_period').value

        self.longitudinal_intent = 0
        self.steering_intent = 0
        self.last_longitudinal_input = 0.0
        self.last_steering_input = 0.0
        self.last_update_time = time.monotonic()
        self.last_status_time = 0.0

        self._stdin_fd = None
        self._stdin_settings = None
        self._terminal_ready = False
        self._setup_terminal()

        print("""
Ackermann Game-Style Teleop
--------------------------
W / Up Arrow      : raise commanded speed
S / Down Arrow    : lower commanded speed / reverse
A / Left Arrow    : steer left
D / Right Arrow   : steer right
C                 : center steering
Space             : full stop
Q                 : quit

Speed is cruise-style: steering keys do not cancel forward motion.
Use S/down or Space when you actually want to slow down.
""")

        self.timer = self.create_timer(self.timer_period, self.control_loop)

    def _setup_terminal(self):
        if not sys.stdin.isatty():
            self.get_logger().warn(
                'teleop_ackermann_node requires a TTY for keyboard input',
            )
            return

        self._stdin_fd = sys.stdin.fileno()
        self._stdin_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._terminal_ready = True

    def _restore_terminal(self):
        if (
            self._terminal_ready
            and self._stdin_fd is not None
            and self._stdin_settings is not None
        ):
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_settings)
            self._terminal_ready = False

    def destroy_node(self):
        self._restore_terminal()
        super().destroy_node()

    def _read_key(self):
        if not self._terminal_ready:
            return None

        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None

        key = sys.stdin.read(1)
        if key != '\x1b':
            return key

        # Arrow keys arrive as multi-byte escape sequences.
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return key

        second = sys.stdin.read(1)
        if second != '[' or not select.select([sys.stdin], [], [], 0.0)[0]:
            return key

        third = sys.stdin.read(1)
        return {
            'A': 'UP',
            'B': 'DOWN',
            'C': 'RIGHT',
            'D': 'LEFT',
        }.get(third, key)

    def _consume_input(self, now):
        while True:
            key = self._read_key()
            if key is None:
                return
            self._handle_key(key, now)

    def _handle_key(self, key, now):
        lowered = key.lower() if len(key) == 1 else key

        if lowered in ('w', 'UP'):
            self.longitudinal_intent = 1
            self.last_longitudinal_input = now
        elif lowered in ('s', 'DOWN'):
            self.longitudinal_intent = -1
            self.last_longitudinal_input = now
        elif lowered in ('a', 'LEFT'):
            self.steering_intent = 1
            self.last_steering_input = now
        elif lowered in ('d', 'RIGHT'):
            self.steering_intent = -1
            self.last_steering_input = now
        elif lowered == 'c':
            self.angle = 0.0
            self.steering_intent = 0
            self.last_steering_input = 0.0
        elif lowered == ' ':
            self._hard_stop()
        elif lowered == 'q':
            self._hard_stop()
            self.publish_msg()
            self.shutdown_requested = True

    def _hard_stop(self):
        self.speed = 0.0
        self.angle = 0.0
        self.longitudinal_intent = 0
        self.steering_intent = 0
        self.last_longitudinal_input = 0.0
        self.last_steering_input = 0.0

    def _approach(self, value, target, step):
        if value < target:
            return min(value + step, target)
        return max(value - step, target)

    def _expire_inputs(self, now):
        if now - self.last_longitudinal_input > self.input_timeout:
            self.longitudinal_intent = 0
        if now - self.last_steering_input > self.input_timeout:
            self.steering_intent = 0

    def _update_speed(self, dt):
        if self.longitudinal_intent > 0:
            if self.speed < 0.0:
                self.speed = min(self.speed + self.brake_rate * dt, 0.0)
            else:
                self.speed = min(
                    self.speed + self.acceleration_rate * dt,
                    self.max_forward_speed,
                )
        elif self.longitudinal_intent < 0:
            if self.speed > 0.0:
                self.speed = max(self.speed - self.brake_rate * dt, 0.0)
            else:
                self.speed = max(
                    self.speed - self.reverse_acceleration_rate * dt,
                    -self.max_reverse_speed,
                )

    def _update_steering(self, dt):
        if self.steering_intent > 0:
            self.angle = min(
                self.angle + self.steering_rate * dt,
                self.max_steering_angle,
            )
        elif self.steering_intent < 0:
            self.angle = max(
                self.angle - self.steering_rate * dt,
                -self.max_steering_angle,
            )
        else:
            self.angle = self._approach(
                self.angle,
                0.0,
                self.steering_return_rate * dt,
            )

    def _print_status(self, now):
        if now - self.last_status_time < self.status_period:
            return

        self.last_status_time = now
        sys.stdout.write(
            f'\rSpeed {self.speed:5.2f} m/s | Steering {self.angle:+.3f} rad'
        )
        sys.stdout.flush()

    def control_loop(self):
        now = time.monotonic()
        dt = min(now - self.last_update_time, 0.1)
        self.last_update_time = now

        self._consume_input(now)
        self._expire_inputs(now)
        self._update_speed(dt)
        self._update_steering(dt)
        self.publish_msg()
        self._print_status(now)

    def publish_msg(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.angle
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()

    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._hard_stop()
        node.publish_msg()
    finally:
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
