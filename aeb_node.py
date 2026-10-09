import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import math

class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb_node')
        
        # We need to publish emergency braking commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # We need two Subscriptions for AEB: The LiDAR (Distance) and Odometry (Speed)
        self.scan_sub = self.create_subscription(LaserScan, '/ego_racecar/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        self.current_speed = 0.0
        self.get_logger().info('AEB System Armed! Monitoring Time-To-Collision.')

    def odom_callback(self, odom_msg):
        # Update our current forward speed (in meters per second)
        self.current_speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # Default behavior: Assume no crash is imminent
        crash_imminent = False
        
        # Only calculate TTC if we are actually moving
        if self.current_speed > 0.1:
            # Loop through every single laser beam in the array
            for i, distance in enumerate(scan_msg.ranges):
                if math.isinf(distance) or math.isnan(distance):
                    continue
                    
                angle = scan_msg.angle_min + (i * scan_msg.angle_increment)
                approach_speed = self.current_speed * math.cos(angle)
                
                if approach_speed <= 0.0:
                    continue
                    
                ttc = distance / approach_speed
                
                if ttc < 0.5:
                    crash_imminent = True
                    break # We found a crash point, stop checking

        if crash_imminent:
            self.get_logger().warn('CRASH IMMINENT! Braking!')
            brake_msg = AckermannDriveStamped()
            brake_msg.header.stamp = self.get_clock().now().to_msg()
            brake_msg.drive.speed = 0.0
            self.drive_pub.publish(brake_msg)
        else:
            # No crash imminent! Drive forward aggressively into the wall to test it!
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 6.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
