import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math

class BasicDriver(Node):
    def __init__(self):
        super().__init__('basic_driver')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/ego_racecar/scan', self.scan_callback, 10)
        
        # WEEK 5: LOCALIZATION (Tracking where we are on the map)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        self.has_left_start_line = False
        self.is_finished = False
        self.start_x = 0.0
        self.start_y = 0.0
        
        self.get_logger().info('ADVANCED Follow The Gap + LOCALIZATION Started!')

    def odom_callback(self, odom_msg):
        # Extract our exact global X and Y coordinates
        current_x = odom_msg.pose.pose.position.x
        current_y = odom_msg.pose.pose.position.y
        
        # Calculate how far we are from the absolute starting line (0,0)
        distance_from_start = math.sqrt((current_x - self.start_x)**2 + (current_y - self.start_y)**2)
        
        # 1. We must drive far away from the start line first so it doesn't instantly trigger
        if distance_from_start > 5.0:
            if not self.has_left_start_line:
                self.get_logger().info("Car is on the track! Lap officially started!")
            self.has_left_start_line = True
            
        # 2. If we are on the track, AND we come back to the starting coordinate, we finished!
        if self.has_left_start_line and distance_from_start < 2.0 and not self.is_finished:
            self.get_logger().info("🏁 FINISH LINE CROSSED! Applying Brakes! 🏁")
            self.is_finished = True

    def scan_callback(self, scan_msg):
        # If the car knows it has finished the race, ignore the LiDAR and hit the brakes!
        if self.is_finished:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            return

        # 1. Focus only on the front 180 degrees of the LiDAR
        ranges = list(scan_msg.ranges)
        start_idx = len(ranges) // 4
        end_idx = len(ranges) * 3 // 4
        front_ranges = ranges[start_idx:end_idx]
        
        # Clean up any bad LiDAR readings. CAPPING at 4.0m is the secret!
        # This flattens the "deepest" part of the track into a wide plateau.
        for i in range(len(front_ranges)):
            if front_ranges[i] > 4.0:
                front_ranges[i] = 4.0
                
        # 2. Find the absolute closest object
        closest_dist = float('inf')
        closest_idx = 0
        for i, dist in enumerate(front_ranges):
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
                
        # 3. Create the "Safety Bubble"
        # 0.8 was too big and was zeroing out the walls, causing zig-zagging. 0.5 is safe for the car width.
        bubble_radius = int(0.5 / scan_msg.angle_increment)
        bubble_start = max(0, closest_idx - bubble_radius)
        bubble_end = min(len(front_ranges), closest_idx + bubble_radius)
        
        for i in range(bubble_start, bubble_end):
            front_ranges[i] = 0.0
            
        # 4. FIND THE WIDEST CONTINUOUS GAP
        max_gap_start = 0
        max_gap_end = 0
        current_start = 0
        in_gap = False
        
        for i, dist in enumerate(front_ranges):
            if dist > 0.0:
                if not in_gap:
                    current_start = i
                    in_gap = True
            else:
                if in_gap:
                    if (i - current_start) > (max_gap_end - max_gap_start):
                        max_gap_start = current_start
                        max_gap_end = i
                    in_gap = False
                    
        if in_gap and (len(front_ranges) - current_start) > (max_gap_end - max_gap_start):
            max_gap_start = current_start
            max_gap_end = len(front_ranges)
            
        if max_gap_start == max_gap_end:
            # No gap found! Brake to avoid crashing.
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            return
            
        # 5. AIM FOR THE "CENTER OF THE MAX DEPTH"
        # Because we capped the LiDAR at 4.0m, there isn't just ONE deepest point anymore. 
        # There is a massive chunk of rays that are all 4.0m deep. We find all of them, and aim for the exact middle of them!
        max_depth = max(front_ranges[max_gap_start:max_gap_end])
        max_depth_indices = [i for i in range(max_gap_start, max_gap_end) if front_ranges[i] == max_depth]
        best_idx = max_depth_indices[len(max_depth_indices) // 2]
        
        actual_index = start_idx + best_idx
        target_angle = scan_msg.angle_min + (actual_index * scan_msg.angle_increment)
        
        # To calculate speed, we must look at the TRUE UNCAPPED distance of our target!
        true_target_depth = ranges[actual_index]
        
        # Send the drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Dynamic Speed Scaling! Go FAST on straightaways and slow down when obstacles are close!
        if true_target_depth > 8.0:
            drive_msg.drive.speed = 6.0
        elif true_target_depth > 5.0:
            drive_msg.drive.speed = 4.0
        elif true_target_depth > 3.0:
            drive_msg.drive.speed = 2.5
        else:
            drive_msg.drive.speed = 1.5
            
        # We need to steer sharply to dodge the boxes!
        drive_msg.drive.steering_angle = target_angle * 1.0
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BasicDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
