import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # Parameters
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('obstacle_distance', 0.4)
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('stuck_timeout', 10.0)  # seconds before declaring stuck

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Current target
        self.target_x = None
        self.target_y = None

        # Obstacle avoidance
        self.obstacle_ahead = False
        self.obstacle_left = False
        self.obstacle_right = False

        # Wall recovery state machine
        # States: 'none', 'backup', 'turn'
        self.recovery_state = 'none'
        self.recovery_start_time = None
        self.recovery_turn_dir = 1.0  # 1.0 = left, -1.0 = right

        # Stuck detection
        self.waypoint_start_time = None

        # Zone awareness
        self.current_zone = 'unknown'  # green, yellow, red, blue

        # --- Subscribers ---
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.waypoint_sub = self.create_subscription(
            PoseStamped, '/coverage_waypoint', self.waypoint_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.zone_sub = self.create_subscription(
            String, '/zone_color', self.zone_callback, 10
        )

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.reached_pub = self.create_publisher(String, '/waypoint_reached', 10)
        self.status_pub = self.create_publisher(String, '/control_status', 10)

        # Control loop at 10 Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Control node started')

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def odom_callback(self, msg):
        """Update robot pose from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def waypoint_callback(self, msg):
        """Receive new waypoint from planner."""
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.waypoint_start_time = self.get_clock().now()
        self.get_logger().info(
            f'New waypoint: ({self.target_x:.2f}, {self.target_y:.2f})'
        )

    def scan_callback(self, msg):
        """Check lidar for obstacles in front, left, and right sectors."""
        obstacle_dist = self.get_parameter('obstacle_distance').value

        if len(msg.ranges) == 0:
            return

        self.obstacle_ahead = False
        self.obstacle_left = False
        self.obstacle_right = False

        for i in range(len(msg.ranges)):
            r = msg.ranges[i]

            # Filter invalid readings
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue

            angle = msg.angle_min + i * msg.angle_increment

            if r < obstacle_dist:
                if -0.52 < angle < 0.52:       # front ±30°
                    self.obstacle_ahead = True
                elif 0.52 <= angle < 1.57:      # left 30°–90°
                    self.obstacle_left = True
                elif -1.57 < angle <= -0.52:    # right -90°– -30°
                    self.obstacle_right = True

    def zone_callback(self, msg):
        """Receive current zone color from perception node."""
        zone = msg.data.strip().lower()
        if zone in ('green', 'yellow', 'red', 'blue'):
            if zone != self.current_zone:
                self.get_logger().info(f'Zone changed: {self.current_zone} -> {zone}')
            self.current_zone = zone

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def publish_reached(self):
        """Tell the planner the current waypoint has been reached."""
        msg = String()
        msg.data = f'{self.target_x:.3f},{self.target_y:.3f}'
        self.reached_pub.publish(msg)
        self.get_logger().info(
            f'Waypoint reached: ({self.target_x:.2f}, {self.target_y:.2f})'
        )
        # Clear target so robot stops until the next waypoint arrives
        self.target_x = None
        self.target_y = None
        self.waypoint_start_time = None

    def publish_stuck(self):
        """Tell the planner the robot is stuck and cannot reach the waypoint."""
        msg = String()
        msg.data = f'stuck,{self.target_x:.3f},{self.target_y:.3f}'
        self.status_pub.publish(msg)
        self.get_logger().warn(
            f'Stuck trying to reach ({self.target_x:.2f}, {self.target_y:.2f})'
        )
        # Clear target so the planner can decide the next action
        self.target_x = None
        self.target_y = None
        self.waypoint_start_time = None

    # ------------------------------------------------------------------ #
    #  Control loop
    # ------------------------------------------------------------------ #

    def publish_cmd(self, linear_x, angular_z):
        """Wrap velocity into TwistStamped and publish."""
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'base_link'
        stamped.twist.linear.x = linear_x
        stamped.twist.angular.z = angular_z
        self.cmd_pub.publish(stamped)

    def control_loop(self):
        """Navigate to waypoint while avoiding obstacles and respecting zones."""

        # ---- No target → stop ---- #
        if self.target_x is None or self.target_y is None:
            self.publish_cmd(0.0, 0.0)
            return

        # ---- Red zone → skip immediately ---- #
        if self.current_zone == 'red':
            self.get_logger().info('In red zone (CCTV covered) — skipping waypoint')
            self.publish_reached()
            return

        max_linear = self.get_parameter('linear_speed').value
        max_angular = self.get_parameter('angular_speed').value
        goal_tol = self.get_parameter('goal_tolerance').value
        kp_lin = self.get_parameter('kp_linear').value
        kp_ang = self.get_parameter('kp_angular').value
        stuck_timeout = self.get_parameter('stuck_timeout').value

        # ---- Stuck detection ---- #
        if self.waypoint_start_time is not None:
            elapsed = (self.get_clock().now() - self.waypoint_start_time).nanoseconds / 1e9
            if elapsed > stuck_timeout:
                self.publish_stuck()
                return

        # ---- Distance and angle to target ---- #
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.robot_yaw)

        # ---- Goal reached ---- #
        if distance < goal_tol:
            self.publish_cmd(0.0, 0.0)  # stop
            self.publish_reached()
            return

        # ---- Wall recovery state machine ---- #
        if self.recovery_state != 'none':
            elapsed = (self.get_clock().now() - self.recovery_start_time).nanoseconds / 1e9

            if self.recovery_state == 'backup':
                # Reverse for 1.5 seconds
                if elapsed < 1.5:
                    self.publish_cmd(-0.15, 0.0)
                    return
                else:
                    # Switch to turning
                    self.recovery_state = 'turn'
                    self.recovery_start_time = self.get_clock().now()
                    return

            elif self.recovery_state == 'turn':
                # Turn for 2 seconds
                if elapsed < 2.0:
                    self.publish_cmd(0.0, max_angular * self.recovery_turn_dir)
                    return
                else:
                    # Recovery done
                    self.recovery_state = 'none'
                    self.recovery_start_time = None
                    return

        # ---- Obstacle avoidance ---- #
        if self.obstacle_ahead:
            # Always backup first, then turn to a clear direction
            self.recovery_state = 'backup'
            self.recovery_start_time = self.get_clock().now()

            # Pick turn direction: toward the clearer side
            if self.obstacle_left and not self.obstacle_right:
                self.recovery_turn_dir = -1.0  # turn right
            elif self.obstacle_right and not self.obstacle_left:
                self.recovery_turn_dir = 1.0   # turn left
            else:
                # Both blocked or neither — turn away from target heading
                # (since target is behind the wall)
                self.recovery_turn_dir = 1.0 if angle_error <= 0 else -1.0

            self.get_logger().info(
                f'Wall hit — backing up then turning '
                f'{"left" if self.recovery_turn_dir > 0 else "right"}'
            )
            self.publish_cmd(-0.15, 0.0)
            return

        # ---- Proportional navigation ---- #
        if abs(angle_error) > 0.3:
            # Turn in place to face the target
            self.publish_cmd(0.0, max(min(kp_ang * angle_error, max_angular), -max_angular))
        else:
            # Drive toward target with heading correction
            lin = max(min(kp_lin * distance, max_linear), 0.0)
            ang = max(min(kp_ang * angle_error, max_angular), -max_angular)
            self.publish_cmd(lin, ang)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()